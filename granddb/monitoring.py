"""
@file        monitoring.py
@brief       Read newly recorded files from the main database and process them to record monitoring informations
             in the monitoring database.
@details     For now process only GP80 files (wait for uniform naming to identify monitoring files).
             The program is designed to launch several processes in order to parallelize jobs (but limited by deadlocks
             when accessing the database).
@author      Fleg
@date        2025-07
@version     1.0.0
@project     GRAND
"""

from grand.dataio import TADC, TEfield, TVoltage, TRawVoltage
import psycopg2
from psycopg2.extras import execute_values
import glob
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from grand.aoi import *
from functools import wraps
import inspect
import random
import time
import grand.manage_log as mlg
import sqlite3
import argparse
import granddb.monitoring_dbconf as monitoring_dbconf



## Decorator to handle database connection (open, commit, close and retry in case of deadlock)
def with_db_cursor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries=5
        backoff_base=0.1
        jitter=0.1
        retries = 0
        while retries <= max_retries:
            try:
                with psycopg2.connect(**monitoring_dbconf.DB_CONFIG) as conn:
                    with conn.cursor() as cur:
                        sig = inspect.signature(func)
                        params = sig.parameters
                        expects_conn = 'conn' in params
                        return func(cur, conn, *args, **kwargs) if expects_conn else func(cur, *args, **kwargs)

            except psycopg2.errors.DeadlockDetected as e:
                conn.rollback()
                if retries >= max_retries:
                    logger.error(f"deadlock... end after {retries} retries")
                    raise
                else:
                    logger.info("deadlock: retry {retries}")
                sleep_time = backoff_base * (2 ** retries)
                sleep_time += random.uniform(0, jitter * sleep_time)
                time.sleep(sleep_time)
                retries += 1
                continue
            except Exception:
                # Immediately re-raise non-retryable exceptions
                logger.error("unexpected error")
                conn.rollback()
                raise

    return wrapper

## Function to read a monitoring file, extract usefull datas, calculate average over bucket time
# (every minute for Temp and Voltage and every hour for spectra) and save it into the DB
def monitor_file(rfile):
    trace_stats = defaultdict(lambda: {"sum": None, "count": 0, "ts_list":[]})
    mesures_stats = defaultdict(lambda: {"sum_temp": 0.0, "sum_batt": 0.0, "sum_stdev0" : 0.0 , "sum_stdev1" : 0.0 , "sum_stdev2" : 0.0 ,  "sum_stdev3" : 0.0 ,"count": 0,"ts_list":[]})

    if Path(rfile).is_file():
        run=TRun(rfile)
        tree=TRawVoltage(rfile)
        adctree = TADC(rfile)
    elif Path(rfile).is_dir():
        d = DataDirectory(rfile)
        run = d.trun
        tree = d.trawvoltage
        adctree = d.adc

    if (not next(iter(adctree)).enable_trigger_10s[0]):
        return
    
    run.get_entry(0)

    #print(f"t_bin_size = {run.t_bin_size[0]} ")

    #For this part we keep the same database connection open because it requests a lot of queries select and reopen a
    # new connection for each one would be too long
    with psycopg2.connect(**monitoring_dbconf.DB_CONFIG) as conn:
            with conn.cursor() as cur:
                for i, event in enumerate(tree):
                    for j in range(len(event.du_id)):
                        du=event.du_id[j]
                        gps_time = event.gps_time[j]
                        dt = datetime.fromtimestamp(gps_time, tz=timezone.utc)
                        #keys for env data (group by minutes) and spectra (group by hours)
                        hour_bin = dt.replace(minute=0, second=0, microsecond=0)
                        minute_bin = dt.replace(second=0, microsecond=0)
                        key_mesures = (minute_bin, du)
                        key_spec = (hour_bin, du)

                        # Fill env table
                        # Get mesures already registered into the dababase if they exists
                        if not mesures_stats[key_mesures]["ts_list"]:
                            mesures_stats[key_mesures]["ts_list"]=get_mesures_ts_list(cur, minute_bin,du)

                        if not trace_stats[key_spec]["ts_list"]:
                            trace_stats[key_spec]["ts_list"]=get_spectres_ts_list(cur ,hour_bin,du)

                        # If new measures are not yet registered then append them
                        # If measures exists for this exact timestamp (gps_time) then skip (means that we are reprocessing an already
                        # processed file or directory
                        if not mesures_stats[key_mesures]["ts_list"] or gps_time not in mesures_stats[key_mesures]["ts_list"]:
                            temp = event.gps_temp[j]
                            batt = event.battery_level[j]
                            mesures_stats[key_mesures]["ts_list"].append(gps_time)
                            mesures_stats[key_mesures]["sum_temp"] += temp
                            mesures_stats[key_mesures]["sum_batt"] += batt
                            mesures_stats[key_mesures]["count"] += 1

                        if not trace_stats[key_spec]["ts_list"] or gps_time not in trace_stats[key_spec]["ts_list"]:
                            trace_stats[key_spec]["ts_list"].append(gps_time)
                            trace = np.array(event.trace_ch[j])
                            #Fill spec table only for traces = 1024 long (so FFT is 512)
                            if trace[0].size == 1024 or trace[0].size == 512:
                                if trace_stats[key_spec]["sum"] is None:
                                    trace_stats[key_spec]["sum"] = trace.copy()
                                else:
                                    trace_stats[key_spec]["sum"] += trace
                                trace_stats[key_spec]["count"] += 1
                                mesures_stats[key_mesures]["sum_stdev0"] += float(np.std(trace[0]*8192/900000, axis=0))
                                mesures_stats[key_mesures]["sum_stdev1"] += float(np.std(trace[1]*8192/900000, axis=0))
                                mesures_stats[key_mesures]["sum_stdev2"] += float(np.std(trace[2]*8192/900000, axis=0))
                                mesures_stats[key_mesures]["sum_stdev3"] += float(np.std(trace[3]*8192/900000, axis=0))

    # Close the tree files
    run.close_file()
    tree.close_file()
    # Build the table of data to be recoreded in the database (so we will do the insert in a single request).
    # Calculate the average over the time buckets. First for temps and then for spectras.
    rows_temp = []
    for (minute_bin, du_id), stats in mesures_stats.items():
        if stats["count"] and stats["sum_temp"] and stats["sum_batt"]:
            rows_temp.append((
                minute_bin,
                du_id,
                stats["sum_temp"] / stats["count"],
                stats["sum_batt"] / stats["count"] - 90.825,
                stats["count"],
                mesures_stats[(minute_bin,du_id)]["ts_list"],
                stats["sum_stdev0"] / stats["count"],
                stats["sum_stdev1"] / stats["count"],
                stats["sum_stdev2"] / stats["count"],
                stats["sum_stdev3"] / stats["count"]
            ))

    # Save the temps/voltages into the database (here the connection will be managed directly by the function to keep it
    # as short as possible to limit locking and free database connections as soon as possible.
    save_temps(rows_temp)

    # Same as previous but for spectras
    rows_spec = []
    for (hour_bin, du_id), stats in trace_stats.items():
        if stats["count"]>0:
            # Calculate the FFT
            mean_traces = stats["sum"] / stats["count"]
            fft_vals = np.fft.fft(mean_traces, axis=1)
            #t_bin_size in nanosec (so 10⁹) and we want freqs in Mhz (10⁶) so need to divide by 1000
            freqs = np.fft.fftfreq(mean_traces.shape[1], d=run.t_bin_size[0]/1000.0 )
            positive_freqs = freqs[:mean_traces.shape[1] // 2]
            # If frequencies not yet in the database then save it.
            if (len(positive_freqs) not in frequency_list):
                save_freqs(positive_freqs)
            #magnitude = 2.0 / mean_trace.shape[1] * np.abs(fft_vals[:, :mean_trace.shape[1] // 2])
            # Keep only positives frequencies
            magnitude = np.abs(fft_vals[:, :mean_traces.shape[1] // 2])

            rows_spec.append((
                hour_bin,
                du_id,
                magnitude.shape[1],
                magnitude[0].tolist(),
                magnitude[1].tolist(),
                magnitude[2].tolist(),
                magnitude[3].tolist(),
                stats["count"],
                trace_stats[(hour_bin,du_id)]["ts_list"]
            ))

    save_spectres(rows_spec)

## Function to get the recorded mesures (temps, volts) from the DB at a timestamp bucket (ts) and for a du_id
def get_mesures_ts_list(cur, ts, du_id):
    cur.execute("""
        SELECT ts_list
        FROM mesures
        WHERE datetime = %s
        AND du_id = %s
    """, (ts, du_id))
    result = cur.fetchone()
    ts_list = result[0] if result and result[0] else []
    return ts_list

## Function to get the recorded spectras from the DB at a timestamp bucket (ts) and for a du_id
def get_spectres_ts_list(cur,ts,du_id):
    cur.execute("""
        SELECT ts_list
        FROM spectres
        WHERE datetime = %s
        AND du_id = %s
    """, (ts,du_id))
    result = cur.fetchone()
    ts_list = result[0] if result and result[0] else []
    return ts_list

## Function to write into the database the table of mesures
## If some mesures already exists in the database for this time bucket (e.g. comming from another file previously treated)
## then calculate the average between existing and new data
@with_db_cursor
def save_temps(cur, conn, rows_temp):
    if len(rows_temp) > 0:
        query = """
            INSERT INTO mesures (datetime, du_id, temperature, voltage, weight, ts_list, stdev0,stdev1,stdev2,stdev3)
            VALUES %s
            ON CONFLICT (datetime, du_id) DO UPDATE
            SET
                temperature = ((mesures.temperature * mesures.weight) + (EXCLUDED.temperature * EXCLUDED.weight)) / (mesures.weight + EXCLUDED.weight),
                voltage = ((mesures.voltage * mesures.weight) + (EXCLUDED.voltage * EXCLUDED.weight)) / (mesures.weight + EXCLUDED.weight),
                stdev0 = ((mesures.stdev0 * mesures.weight) + (EXCLUDED.stdev0 * EXCLUDED.weight)) / (mesures.weight + EXCLUDED.weight),
                stdev1 = ((mesures.stdev1 * mesures.weight) + (EXCLUDED.stdev1 * EXCLUDED.weight)) / (mesures.weight + EXCLUDED.weight),
                stdev2 = ((mesures.stdev2 * mesures.weight) + (EXCLUDED.stdev2 * EXCLUDED.weight)) / (mesures.weight + EXCLUDED.weight),
                stdev3 = ((mesures.stdev3 * mesures.weight) + (EXCLUDED.stdev3 * EXCLUDED.weight)) / (mesures.weight + EXCLUDED.weight),
                weight = mesures.weight + EXCLUDED.weight, 
                ts_list = (
                SELECT ARRAY(
                    SELECT DISTINCT val
                    FROM unnest(array_cat(mesures.ts_list, EXCLUDED.ts_list)) AS val
                    ORDER BY val
                    )
                ) 
            
            ;
        """
        execute_values(cur, query, rows_temp)
        conn.commit()

## Function to write into the database the table of spectras
## If some mesures already exists in the database for this time bucket (e.g. comming from another file previously treated)
## then calculate the average between existing and new data
@with_db_cursor
def save_spectres(cur, conn, rows_spec):
        query = """ 
        INSERT INTO spectres (datetime, du_id, len, powers_0, powers_1, powers_2, powers_3, weight,ts_list)
        VALUES %s 
        ON CONFLICT (datetime, du_id) DO UPDATE
        SET
            powers_0 = (
                SELECT ARRAY(
                    SELECT
                        (old * spectres.weight + new * EXCLUDED.weight) / (spectres.weight + EXCLUDED.weight)
                    FROM unnest(spectres.powers_0) WITH ORDINALITY AS old_vals(old, i)
                    JOIN unnest(EXCLUDED.powers_0) WITH ORDINALITY AS new_vals(new, j)
                    ON i = j
                )
            ),
            powers_1 = (
                SELECT ARRAY(
                    SELECT
                        (old * spectres.weight + new * EXCLUDED.weight) / (spectres.weight + EXCLUDED.weight)
                    FROM unnest(spectres.powers_1) WITH ORDINALITY AS old_vals(old, i)
                    JOIN unnest(EXCLUDED.powers_1) WITH ORDINALITY AS new_vals(new, j)
                    ON i = j
                )
            ),
            powers_2 = (
                SELECT ARRAY(
                    SELECT
                        (old * spectres.weight + new * EXCLUDED.weight) / (spectres.weight + EXCLUDED.weight)
                    FROM unnest(spectres.powers_2) WITH ORDINALITY AS old_vals(old, i)
                    JOIN unnest(EXCLUDED.powers_2) WITH ORDINALITY AS new_vals(new, j)
                    ON i = j
                    )
                ),
            powers_3 = (
                SELECT ARRAY(
                    SELECT
                        (old * spectres.weight + new * EXCLUDED.weight) / (spectres.weight + EXCLUDED.weight)
                    FROM unnest(spectres.powers_3) WITH ORDINALITY AS old_vals(old, i)
                    JOIN unnest(EXCLUDED.powers_3) WITH ORDINALITY AS new_vals(new, j)
                    ON i = j
                    )
                ),
            weight = spectres.weight + EXCLUDED.weight,
            ts_list = (
            SELECT ARRAY(
                SELECT DISTINCT val
                FROM unnest(array_cat(spectres.ts_list, EXCLUDED.ts_list)) AS val
                ORDER BY val
                )
            ) ;
        """
        execute_values(cur, query, rows_spec)
        conn.commit()

## Function to get the frequencies list from the database.
@with_db_cursor
def get_freqs(cur):
    cur.execute("SELECT len FROM frequences")
    rows = cur.fetchall()
    frequency_list = [row[0] for row in rows]
    return frequency_list

## Function to record the frequencies list into the database.
@with_db_cursor
def save_freqs(cur, conn, positive_freqs):
        query = """
        INSERT INTO frequences (len, freq)
        VALUES %s
        ON CONFLICT (len) DO NOTHING;
        """
        argfreq=[]
        argfreq.append((len(positive_freqs.tolist()),positive_freqs.tolist()))
        execute_values(cur, query, argfreq)
        conn.commit()


## Function to query the main database (MDB) to get the list of files (or directories) converted after a transfer.
## This is needed to interface the monitoring with the automatic conversion pipeline. This pipeline will call the
## monitoring program (passing the tag of the process).
## TODO: When gp80 and gaa will have an unique way to identify monitoring data then the filter will have to be adapted.
## For now, we use GP80%_MD_%-10s-%.root (so this programm will works only for gp80 files).
def get_files(tag):
    with psycopg2.connect(**monitoring_dbconf.MDB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT  DISTINCT regexp_replace(regexp_replace(t.target, 'raw', 'GrandRoot'), '/[^/]+$', '') || '/' || c.root_filename AS full_path
            FROM convertion c,transfer t
            WHERE c.id_raw_file = t.id_raw_file
            AND t.tag = %s
            AND retcode = 0
            AND root_filename LIKE %s
            """, (tag,'GP80%_MD_%.root'))
            results = cur.fetchall()
            #file_list = result #if result and result[0] else []

    return [result[0] for result in results]

## Process launcher
def process_file(filepath):
    logger.info(f"Found file: {filepath}")
    monitor_file(filepath)

if __name__ == "__main__":
    logger = mlg.get_logger_for_script(__name__)
    mlg.create_output_for_logger("info", log_stdout=True)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--tag", required=True, help="Tag for the files to register")
    args = argParser.parse_args()
    tag = args.tag

    #folder = "/sps/grand/data/gp80/GrandRoot/2025/07/"
    #pattern = os.path.join(folder, "GP80*_MD_*-10s-*.root")
    #filepaths = sorted(glob.glob(pattern))

    filepaths=get_files(tag)
    logger.info(f"Files to process : {filepaths}")
    # We shuffle the list to avoid processing concurrently 2 files concerning the same bucket of time (and limit
    # the risks of deadlock when accessing the database)
    random.shuffle(filepaths)

    frequency_list = get_freqs()

    # Limitation on the number of parallel processes. Must be the correct balance between performances and risks of
    # deadlocks or database max connexions
    MAX_PROCESSES = max(1,min(8, (os.cpu_count() or 1)) - 1)

    with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
        results = executor.map(process_file, filepaths)
        for result in results:  # This will raise any exceptions from workers
            pass  # or handle results if needed
    logger.info(f"End of monitoring")


