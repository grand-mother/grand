#!/usr/bin/env python
## Conversion of GRANDRaw ROOT files coming from simulators to GRANDROOT files
## by Lech Wiktor Piotrowski

import os
import sys
import argparse
from types import SimpleNamespace
import time
from pathlib import Path

import numpy as np
import glob
import datetime

from grand.dataio import TRun, TRunEfieldSim, TRunShowerSim, TEfield, TShower, TShowerSim
import raw_root_trees as RawTrees # this is here in Common
import grand.manage_log as mlg
from grand import Geodetic, GRANDCS
from grand.dataio.xmax_frame import propagation_direction, xmax_in_site_frame
# import matplotlib.pyplot as plt
# from scipy.ndimage.interpolation import shift  #to shift the time trance for the trigger simulation
# from scipy.ndimage import shift  #to shift the time trance for the trigger simulation

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)
logger.info("Converting rawroot to grandlib file")


#ToDo:latitude,longitude and altitude are available in ZHAireS .sry file, and could be added to the RawRoot file. Site too.
# Command line argument parsing
clparser = argparse.ArgumentParser(description="Convert simulation data in rawroot format into GRANDROOT format")
clparser.add_argument("file_dir_name", nargs='+', help="ROOT files containing GRANDRaw data TTrees, a directory with GRANDraw files or a .txt file with list of rawroot files")
clparser.add_argument("-o", "--output_parent_directory", help="Output parent directory", default="")
clparser.add_argument("-fo", "--forced_output_directory", help="Force this option as the output directory", default=None)
clparser.add_argument("--overwrite", action="store_true",
                      help="allow -fo to name a folder that already holds files (refused otherwise)")
clparser.add_argument("-s", "--site_name", help="The name of the site", default=None)
clparser.add_argument("-sl", "--site_layout", help="The layout of the site (eg. GP13, GP80, GAA)", default=None)
clparser.add_argument("-d", "--sim_date", help="The date of the simulation", default=None)
clparser.add_argument("-t", "--sim_time", help="The time of the simulation", default=None)
# clparser.add_argument("-d", "--sim_date", help="The date of the simulation", default="19000101")
# clparser.add_argument("-t", "--sim_time", help="The time of the simulation", default="000000")
clparser.add_argument("-e", "--extra", help="Extra information to store in the directory name", default="")
clparser.add_argument("-av", "--analysis_level", help="Analysis level of the data", default=0, type=int)
# clparser.add_argument("-se", "--serial", help="Serial number of the simulation", default=0)
clparser.add_argument("-la", "--latitude", help="Latitude of the site (deg)", default=None, type=float)
clparser.add_argument("-lo", "--longitude", help="Longitude of the site (deg)", default=None, type=float)
clparser.add_argument("-al", "--altitude", help="Altitude of the site (m)", default=None, type=float)
clparser.add_argument("-ru", "--run", help="Run number", default=None)
clparser.add_argument("-se", "--start_event", help="Starting event number", default=None)
clparser.add_argument("--target_duration_us",type=float,default=None,help="Adujust the trace lenght to the given duration, in us") 
clparser.add_argument("--trigger_time_ns",type=float,default=None,help="Adujust the trace so that the maximum is at given ns from the begining of the trace")
clparser.add_argument("--verbose", choices=["debug", "info", "warning", "error", "critical"],default="info", help="logger verbosity.")
clparser.add_argument("-ss", "--star_shape", help="For star-shapes: create a separate run for every event", action='store_true')
clparser.add_argument("-ef", "--events_per_file", help="How many events to hold in a single file", type=int, default=None)
clargs = clparser.parse_args()

mlg.create_output_for_logger(clargs.verbose, log_stdout=True)

logger.info("#################################################")

############################################################################################################################
# adjust the trace lenght to force the requested tpre and tpost  TODO:This should be properly codede into a tool on grandlib tools for trace manipulation
###########################################################################################################################
def adjust_trace(trace, t0s, CurrentTpre,CurrentTpost, DesiredTpre, DesiredTpost,TimeBinSize):

    #trace holds a list, needs to be converted into a numpy array
    trace=np.asarray(trace)
    #t0s is a vector with all the du's t0s

    #TODO: assert that the times are multiples of TimeBinSize, as we are not ready to handle fractional time bins
        
    logger.info("Adjusting trace so that t_pre is "+str(DesiredTpre)+" and t_post is "+str(DesiredTpost)+".Trace lenght will be "+str(DesiredTpre+DesiredTpost))
    logger.debug("Original t_pre is "+str(CurrentTpre)+" and t_post is "+str(CurrentTpost)+" .t_bin_size is "+str(TimeBinSize))
    #plt.plot(trace[1,1],label="before",linewidth=5)
    trace=adjust_trace_lenght(trace, DesiredTpre, DesiredTpost, CurrentTpre, CurrentTpost,TimeBinSize)
    #plt.plot(trace[1,1],label="after",linewidth=3)
    #plt.axvline(DesiredTpre/TimeBinSize,label="RequestedTriggerPosition",color="orange")
    #plt.axvline(CurrentTpre/TimeBinSize,label="OriginalTriggerPosition",color="blue")
    t0s,trace=adjust_trigger(trace, t0s, DesiredTpre, TimeBinSize)
    #plt.plot(trace[1,1],label="corrected t0")
    #plt.legend()
    #plt.show()    
    return t0s,trace
  

def adjust_trace_lenght(trace, DesiredTpre, DesiredTpost, CurrentTpre, CurrentTpost,TimeBinSize): 
    #we will assume that, on input, all traces have the same tpre and tpost
    #everything is in ns.
    #time window is defined as starting at t0-tpre and finishing at t0+tpost
    #If the DesiredTpre or DesiredTpost are, 0, the original value is kept
    #TimeBinSize is the size of the time bin
    #trace is a vector with shape   (n_du, 3,ntimebins)
    
    #TODO: assert that the times are multiples of TimeBinSize, as we are not ready to handle fractional time bins   
    if DesiredTpre!=0:
      DeltaTimePre=DesiredTpre-CurrentTpre
      DeltaBinsPre=int(np.round(DeltaTimePre/TimeBinSize))
    else:
      DeltaBinsPre=0                    
      DesiredTpre=CurrentTpre
      
    if DesiredTpost!=0:
      DeltaTimePost=DesiredTpost-CurrentTpost
      DeltaBinsPost=int(np.round(DeltaTimePost/TimeBinSize))
    else:
      DeltaBinsPost=0 
      DesiredTpost=CurrentTpost 

    #if the current value is larger than what we desire, we just remove the bins.                
    if DeltaBinsPre<0:
      trace=trace[:,:,-DeltaBinsPre:]
      logger.debug("We had to remove "+str(-DeltaBinsPre)+" bins at the start of trace")
      DeltaBinsPre=0
     
    if DeltaBinsPost<0 :
      trace=trace[:,:,:DeltaBinsPost]   
      logger.debug("We had to remove "+str(-DeltaBinsPost)+" bins at the end of trace")
      DeltaBinsPost=0
   
    #if the desired value is larger, we have to pad.
    if DeltaBinsPost>0 or DeltaBinsPre>0:
      npad = ((0,0),(0,0),(DeltaBinsPre, DeltaBinsPost))
      trace=np.pad(trace, npad, 'constant')          #TODO. Here I could pad using the "linear_ramp" mode, and pad wit a value that slowly aproached 0.
      logger.debug("We have to add "+str(DeltaBinsPre)+" bins at the start of the trace")
      logger.debug("We have to add "+str(DeltaBinsPost)+" bins at the end of the trace")

    #Check that we achieved the correct length (rounding errors can leave us one bin short or long (TODO:we lack a definition of what to do if the times are not a multiple of tbinsize.)
    DesiredTraceLenght=int((DesiredTpre+DesiredTpost)/TimeBinSize)

    if(len(trace[0,0])>DesiredTraceLenght):      
      trace=trace[:,:,:DesiredTraceLenght]
    elif(len(trace[0,0])<DesiredTraceLenght):
      delta=DesiredTraceLenght-len(trace[0,0])
      trace=np.pad(trace, ((0,0),(0,0),(0,delta)), 'constant')
      
    #now, we have traces that meet the desired tpre and tpost. No changes to t0 are needed (TODO: maybe changes to t0 are needed if we have a non-integer number of bins between the t_pre and the t_post)     
    return trace


def adjust_trigger(trace, CurrentT0s, TPre, TimeBinSize):
    # now lets process a "trigger" algorithm that will modify where the trace is located.
    # we asume trace is windowed between CurrentT0-Tpre and CurrentT0+tpost
    # trace will have dim (du,3 or 4,tbins)

    # totl will have dim du,tbins
    ttotal = np.linalg.norm(trace, axis=1)  # make the modulus (the 1 is to remove the time)
    # trigger_index will have dim du
    trigger_index = np.argmax(ttotal, axis=1)  # look where the maximum happens

    # this definition of the trigger times makes the trigger be at the begining of the bin where the maximum is, becouse the index starts at 0. This is compatible with the definition of the window that we give.
    trigger_time = trigger_index * TimeBinSize
    # If we need to shift the trigger time (the trigger time needs to be equal to tpre
    DeltaT = TPre - trigger_time
    ShiftBins = (DeltaT / TimeBinSize).astype(int, copy=False)

    # this is to assure that, if the maximum is found too late in the trace, we dont move outside of the original time window (normally, peaks are late in the time window, if you set the time window correctly).
    mask = ShiftBins < -TPre / TimeBinSize
    if mask.any():
        logger.error("some elements needed to be shifted only up to the limt, tpre was too small")
        ShiftBins[mask] = int(-TPre / TimeBinSize)

    # we cannot use np.roll, but roll makes re-appear the end of the trace at the begining if we roll to much
    # we cannot use scipy shift, that lets you state what value to put for the places you roll, on a 3D array

    # TODO: There must be a better way to do this without the for loop, but i lost a morning to it and i dont have the time to develop it now. Search for strided_indexing_roll on the web for inspiration.
    # for du_idx in range(trace.shape[0]):
    #     trace[du_idx] = shift(trace[du_idx], (0, ShiftBins[du_idx]), cval=0)
    for du_idx in range(trace.shape[0]):
        trace_shift(trace[du_idx], ShiftBins[du_idx])

    # we get the correct t0
    T0s = CurrentT0s - ShiftBins * TimeBinSize

    return T0s, trace

def is_no_antenna_event(trawefield):
    r"""Returns whether the current raw efield entry has no antenna at all.

    A shower can miss every antenna (issue #91).  The raw converters then
    write an efield entry with ``du_count`` 0 and empty per-antenna vectors.
    The traces decide, not ``du_count``: an entry that carries traces is
    never discarded, even if a converter left ``du_count`` at its default.

    Parameters
    ----------
    trawefield : RawEfieldTree
        The raw efield tree, positioned on the entry to test.

    Returns
    -------
    bool
        True if the entry has no antenna.
    """
    return len(trawefield.trace_x) == 0


def extract_trace(trawefield, event_number):
    r"""Stacks the raw x, y, z traces into one ``(n_du, 3, n_samples)`` array.

    Parameters
    ----------
    trawefield : RawEfieldTree
        The raw efield tree, positioned on the entry to read.
    event_number : int
        Used in the error message only.

    Returns
    -------
    ndarray or None
        The traces as float32, or None if the components cannot be stacked,
        in which case the error has been logged.
    """
    def ensure_2d_trace_component(component_trace_data):
        ct_arr = np.asarray(component_trace_data)
        if ct_arr.ndim == 1:
            # Reshape 1D array (n_samples,) to 2D array (1, n_samples)
            logger.warning(f"Trace component Shape: {ct_arr.shape}")
            print(f"Trace component Shape: {ct_arr.shape}")
            return ct_arr.reshape(1, -1)
        elif ct_arr.ndim == 0: # Should not happen for traces
            logger.error(f"Trace component is scalar! Shape: {ct_arr.shape}. This is unexpected.")
            return np.empty((1,0), dtype=ct_arr.dtype)
        # If 2D or more, assume it's (n_du, n_samples) or similar, return as is.
        # np.stack will later complain if shapes are not compatible.
        return ct_arr

    tx = ensure_2d_trace_component(trawefield.trace_x)
    ty = ensure_2d_trace_component(trawefield.trace_y)
    tz = ensure_2d_trace_component(trawefield.trace_z)

    # np.stack requires input arrays to have the same shape.
    try:
        return np.stack([tx, ty, tz], axis=1).astype(np.float32)
    except ValueError as e:
        logger.error(f"Failed to stack trace components for event {event_number}. Shapes - tx: {tx.shape}, ty: {ty.shape}, tz: {tz.shape}. Error: {e}")
        return None


def convert_date(date_str):
    # Convert input string to a struct_time object
    date_struct = time.strptime(date_str, "%Y-%m-%d")
    # Format the struct_time object as a string in YYYYMMDD format
    formatted_date = time.strftime("%Y%m%d", date_struct)
    return formatted_date


def desired_window(t_pre, t_post, trigger_time_ns=None, target_duration_us=None):
    r"""Returns the ``(t_pre, t_post)``, in ns, a trace is written with.

    ``--trigger_time_ns`` sets t_pre and keeps the duration; with
    ``--target_duration_us`` too, the duration is set as well.
    """
    if trigger_time_ns is not None:
        t_post = t_pre + t_post - trigger_time_ns
        t_pre = trigger_time_ns
    if target_duration_us is not None:
        t_post = target_duration_us * 1000 - t_pre
    return t_pre, t_post


def check_windows(file_list, trigger_time_ns=None, target_duration_us=None, star_shape=False):
    r"""Exits with a message if the requested windows cannot be written (#222).

    Checked before anything is written: the options must be positive, the
    trigger must fall inside the trace, and all the events of one run (one
    file with ``-ss``) must share one window, since the run tree stores only
    one.  Events that hit no antenna carry no trace and are not checked.
    """
    def fail(message):
        sys.exit("GRANDlib: sim2root: " + message)

    if trigger_time_ns is not None and not trigger_time_ns > 0:
        fail("--trigger_time_ns must be > 0, got %s" % trigger_time_ns)
    if target_duration_us is not None and not target_duration_us > 0:
        fail("--target_duration_us must be > 0, got %s" % target_duration_us)
    if (trigger_time_ns is not None and target_duration_us is not None
            and not trigger_time_ns < target_duration_us * 1000):
        fail("--trigger_time_ns (%s ns) must be shorter than --target_duration_us (%s ns)"
             % (trigger_time_ns, target_duration_us * 1000))

    run_window = None
    for filename in file_list:
        if star_shape:
            run_window = None
        try:
            trawefield = RawTrees.RawEfieldTree(filename)
        except OSError:
            continue
        for i in range(trawefield.get_entries()):
            trawefield.get_entry(i)
            if is_no_antenna_event(trawefield):
                continue
            window = desired_window(trawefield.t_pre, trawefield.t_post, trigger_time_ns, target_duration_us)
            where = "%s entry %d (t_pre %s ns, t_post %s ns)" % (filename, i, trawefield.t_pre, trawefield.t_post)
            if not window[1] > 0:
                fail("%s: the requested window ends %s ns before the trigger; give a longer "
                     "--target_duration_us or a shorter --trigger_time_ns" % (where, -window[1]))
            if run_window is None:
                run_window = window
            elif not np.allclose(window, run_window):
                fail("%s: its window (t_pre %s, t_post %s ns) differs from the run's (t_pre %s, "
                     "t_post %s ns), and the run stores only one.  Give --trigger_time_ns and "
                     "--target_duration_us to write every event with the same window%s"
                     % ((where,) + tuple(window) + tuple(run_window)
                        + ("" if star_shape else ", or convert the files separately with -ss",)))


def check_run_numbers(file_list, ext_run_number=None, star_shape=False):
    r"""Exits with a message if the inputs hold several runs and nothing says how to write them.

    The run trees were written for the first run only, while the event trees
    kept every event's own run number, so the later runs had no run entry
    (#223).  ``-ru`` writes them all as one run; ``-ss`` makes each input
    file its own run.
    """
    if ext_run_number is not None or star_shape:
        return
    runs = set()
    for filename in file_list:
        try:
            trawshower = RawTrees.RawShowerTree(filename)
        except OSError:
            continue
        for i in range(trawshower.get_entries()):
            trawshower.get_entry(i)
            runs.add(int(trawshower.run_number))
        trawshower.stop_using()
    if len(runs) > 1:
        runs = sorted(runs)
        sys.exit("GRANDlib: sim2root: the inputs hold runs %s and %s, and the run files describe "
                 "one run.  Give -ru N to write them all as run N, or convert each run separately"
                 % (", ".join(str(run) for run in runs[:-1]), runs[-1]))


def fail(message):
    r"""Exits with status 1 and `message` (#224: several failures exited 0)."""
    sys.exit("GRANDlib: sim2root: " + message)


# What this run wrote, so that a failure can remove it (#224): the folder,
# whether this run created it, and the files that were there before.
_output = {"dir": None, "created": False, "before": set()}


def _remove_partial_output():
    r"""Removes the files a failed conversion wrote, and its folder if it made it."""
    folder = _output["dir"]
    if folder is None or not folder.is_dir():
        return
    for path in folder.iterdir():
        if path.name not in _output["before"] and path.is_file():
            path.unlink()
    if _output["created"] and not any(folder.iterdir()):
        folder.rmdir()
    logger.error(f"The conversion failed; removed its partial output from {folder}")


def main():
    r"""Runs the conversion; on failure, removes what it had written (#224)."""
    try:
        convert()
    except BaseException as error:
        if not (isinstance(error, SystemExit) and error.code in (0, None)):
            _remove_partial_output()
        raise


def convert():
    r"""Converts the input files given on the command line."""
    # Check if the site layout is defined
    if not clargs.star_shape and not clargs.site_layout:
        fail("give the simulated site layout with -sl (e.g. -sl GP300)")

    # Initialise the run number if specified
    ext_run_number = None
    if clargs.run is not None:
        ext_run_number = int(clargs.run)

    # Initialise the starting event number
    ext_event_number = None
    if clargs.start_event is not None:
        ext_event_number = int(clargs.start_event)

    start_event_number = 0
    file_start_event_number = 0
    end_event_number = 0
    start_event_time = 0
    end_event_time = 0
    run_number = 0
    # Namespace for holding output trees
    gt = SimpleNamespace()

    # Check if a directory was given as input
    if Path(clargs.file_dir_name[0]).is_dir():
        # ToDo: add console flag for file order randomisation
        file_list = sorted(glob.glob(clargs.file_dir_name[0]+"/*.rawroot"))
    # Check if the first file is a list (of files, hopefully)
    elif Path(clargs.file_dir_name[0]).is_file() and Path(clargs.file_dir_name[0]).suffix==".txt":
        with open(clargs.file_dir_name[0]) as f:
            file_list = f.read().splitlines()
    else:
        file_list = clargs.file_dir_name

    if len(file_list)==0:
        fail("no .rawroot files in %s" % clargs.file_dir_name[0])
    # Opening a missing file for reading created it, and the run then failed
    # on an unbound variable (#224)
    missing = [name for name in file_list if not Path(name).is_file()]
    if missing:
        fail("no such input file: %s" % ", ".join(missing))
    if clargs.forced_output_directory is not None and not clargs.overwrite:
        forced = Path(clargs.output_parent_directory, clargs.forced_output_directory)
        if forced.is_dir() and any(forced.iterdir()):
            fail("the -fo folder %s already holds files, and a second file set next to them "
                 "breaks the later steps; give --overwrite to write there anyway" % forced)

    # Before anything is written: one window per run, and a valid one (#222)
    check_windows(file_list, clargs.trigger_time_ns, clargs.target_duration_us, clargs.star_shape)
    # ... and one run, unless -ru or -ss says what to do with several (#223)
    check_run_numbers(file_list, ext_run_number, clargs.star_shape)

    # How many events the open event files hold.  It started at 1 and the
    # first event was exempt from the check, so -ef 1 never split and an
    # exact multiple left an empty file set behind (#223).  A new set is now
    # opened only when an event is about to go into it.
    events_in_file = 0
    event_files_open = True     # init_all_trees opens the first set
    split_done = False

    # The name of the output directory
    out_dir_name = ""

    # Loop through the files specified on command line
    # for file_num, filename in enumerate(clargs.filename):
    for file_num, filename in enumerate(file_list):

        logger.info(f"Working on input file {filename}, {file_num+1}/{len(file_list)}")

        # Output filename for GRAND Trees
        # if clargs.output_filename is None:
        out_filename = os.path.join(os.path.split(filename)[0], "gr_"+os.path.split(filename)[1])
        # else:
        #     out_filename = clargs.output_filename

        # Read the raw trees from the file
        try:
            trawshower = RawTrees.RawShowerTree(filename)
            trawefield = RawTrees.RawEfieldTree(filename)
            trawmeta = RawTrees.RawMetaTree(filename)
        except OSError:
            print("Failed to read raw file", filename)
            continue

        nentries = trawshower.get_entries()

        # Skip files with no or empty trees
        if nentries==0 or trawefield.get_entries()==0 or trawmeta.get_entries()==0:
            logger.warning("No entries or one of the rawroot trees does not exist. Skipping this file.")
            continue

        # Loop through entries - assuming same number of entries in each tree
        # ToDo: this should be a tree iterator through one tree and getting the other through friends. Need to make friends working...
        for i in range(nentries):
            trawshower.get_entry(i)
            trawefield.get_entry(i)
            trawmeta.get_entry(i)

            OriginalTpre=trawefield.t_pre
            OriginalTpost=trawefield.t_post
            # Checked by check_windows before the loop (#222)
            DesiredTpre, DesiredTpost = desired_window(OriginalTpre, OriginalTpost,
                                                       clargs.trigger_time_ns, clargs.target_duration_us)
            #we modify this becouse it needs to be stored in the run file on the first event.
            trawefield.t_pre=DesiredTpre
            trawefield.t_post=DesiredTpost

            if events_in_file==0:
                if ext_event_number is not None:
                    file_start_event_number = end_event_number+1
                else:
                    file_start_event_number = trawshower.event_number

            # If the first entry on the first file, or dealing with star shape sim
            if (file_num==0 and i==0) or clargs.star_shape:

                # Overwrite the run number if specified on command line (only for the first event)
                if file_num==0 and i==0:
                    run_number = ext_run_number if ext_run_number is not None else trawshower.run_number
                    start_run_number = run_number
                # or increase it by one for star shapes
                elif clargs.star_shape: run_number += 1

                # Check if site name was not given as input, use the one from the trawshower
                if not clargs.site_name:
                    site = trawshower.site
                else:
                    site = clargs.site_name

                # Only for the tist entry of the first file
                if file_num==0 and i==0:
                    # Init output trees in the proper directory
                    out_dir_name = init_all_trees(clargs, trawshower.unix_date, run_number, site, gt)
                    # Set site_layout to the command line argument
                    if clargs.site_layout: gt.trun.site_layout = clargs.site_layout

                # Convert the RawShower entries
                rawshower2grandrootrun(trawshower, gt)
                # Convert the RawEfield entries
                rawefield2grandrootrun(trawefield, gt)

                #ToDo:latitude,longitude and altitude are available in ZHAireS .sry file, and could be added to the RawRoot file
                # JK: also available in Coreas!
                
                # Set the origin geoid
                gt.trun.origin_geoid = get_origin_geoid(clargs, trawshower)

                gt.trun.run_number = run_number
                gt.trunshowersim.run_number = run_number
                gt.trunefieldsim.run_number = run_number

                # The site, if none was recorded or -s gives one: -s changed
                # only trun and the folder name (#226)
                if trawshower.site == "" or clargs.site_name:
                    gt.trunshowersim.site = site
                if trawefield.site == "" or clargs.site_name:
                    gt.trunefieldsim.site = site

                gt.trun.site = site
                gt.trun.data_source = "simulation"

                # Fill the run trees and write
                # gt.trun.fill()
                gt.trunshowersim.fill()
                gt.trunefieldsim.fill()
                # gt.trun.write()

            if not event_files_open:
                logger.info("Creating new event files")
                init_event_trees(out_dir_name, gt)
                event_files_open = True

            # Convert the RawShowerTree entries
            rawshower2grandroot(trawshower, gt)
            # Convert the RawMetaTree entries - (this goes before the efield becouse the efield needs the info on the second and nanosecond)
            rawmeta2grandroot(trawmeta, gt)

            # A shower that hit no antenna (issue #91). It is kept: the shower,
            # showersim and efield trees all get the event, the efield entry
            # with du_count 0 and empty per-antenna arrays, so that the
            # simulation still counts in effective-area statistics.
            if is_no_antenna_event(trawefield):
                logger.warning(f"Event {trawshower.event_number} in {filename} hit no antenna "
                               f"(du_count 0): writing its shower information with an empty efield.")
                rawefield2grandroot(trawefield, gt, ext_trace=np.zeros((0, 3, 0), dtype=np.float32),
                                    ext_t_0=np.zeros(0, dtype=np.float64))
            else:
                trace = extract_trace(trawefield, trawshower.event_number)
                if trace is None:
                    # Skip this event: the trace components could not be stacked
                    continue

                # Change the trace lenght as specified in the comand line
                ext_t_0, trace = adjust_trace(trace, trawefield.t_0, OriginalTpre, OriginalTpost, DesiredTpre, DesiredTpost,trawefield.t_bin_size)

                # Convert the RawEfieldTree entries
                rawefield2grandroot(trawefield, gt, ext_trace=trace, ext_t_0=ext_t_0)

            # Overwrite the run number if specified on command line
            if ext_run_number is not None:
                gt.trun.run_number = ext_run_number
                gt.trunshowersim.run_number = ext_run_number
                gt.trunefieldsim.run_number = ext_run_number
                gt.tshower.run_number = ext_run_number
                gt.tshowersim.run_number = ext_run_number
                gt.tefield.run_number = ext_run_number
            # For starshape, update the event trees run numbers
            elif clargs.star_shape:
                gt.tshower.run_number = run_number
                gt.tshowersim.run_number = run_number
                gt.tefield.run_number = run_number

            # Overwrite the event number if specified on command line
            if ext_event_number is not None:
                gt.tshower.event_number = ext_event_number
                gt.tshowersim.event_number = ext_event_number
                gt.tefield.event_number = ext_event_number

            # store temporarily the first event number (with -ss, every file
            # is its own run, so its first event starts a new run: #220)
            if (file_num==0 or clargs.star_shape) and i==0:
                start_event_number = gt.tshower.event_number
                start_event_time = trawshower.unix_date
                if ext_event_number is not None:
                    start_event_number = ext_event_number
                    file_start_event_number = start_event_number

            # # This event in file names ordering is kept only for compatibility with DC2 release. It is meaningless and probably better to keep the first and last events as sims are giving them, because maybe... that can tell us some positions in the original sims event list
            # # ToDo: Remove this ordering all together
            # if clargs.events_per_file is None:
            #     # Correct the first/last event number for file naming
            #     if(gt.tshower.event_number<start_event_number):
            #         start_event_number = gt.tshower.event_number
            #
            #     if(gt.tshower.event_number>end_event_number):
            #         end_event_number = gt.tshower.event_number
            # # ToDo: this should be the standard case
            # else:

            end_event_number = gt.tshower.event_number
            end_event_time = trawshower.unix_date

            gt.tshowersim.input_name = Path(filename).stem

            # Fill the event trees
            gt.tshower.fill()
            gt.tshowersim.fill()
            gt.tefield.fill()
            events_in_file += 1

            # If filled max number of events in file
            if clargs.events_per_file and events_in_file == clargs.events_per_file:

                # tmp_start_event_number = gt.tshower.event_number

                # Save and close the current event files
                logger.info("Writing event trees")
                gt.tshower.write(force_close_file=True)
                gt.tshowersim.write(force_close_file=True)
                gt.tefield.write(force_close_file=True)
                gt.tshower.stop_using()
                gt.tshowersim.stop_using()
                gt.tefield.stop_using()

                # Move the saved event files to proper filenames
                logger.info("Renaming event files")
                rename_event_files(clargs, out_dir_name, file_start_event_number, end_event_number)

                event_files_open = False
                split_done = True
                events_in_file = 0

        # For the first file, get all the file's events du ids and pos
        if file_num==0:
            du_ids, du_xyzs, du_geoids = get_tree_du_id_xyz_geoid(trawefield, trawshower.shower_core_pos, gt.trun.origin_geoid)
            tdu_ids, tdu_xyzs, tdu_geoids = du_ids, du_xyzs, du_geoids
        # For other files, append du ids and pos to the ones already retrieved
        else:
            # ToDo: POSSIBLE BUG! Maybe nonunique du_ids are added, as the du_ids already in the list are not checked (?)
            tdu_ids, tdu_xyzs, tdu_geoids = get_tree_du_id_xyz_geoid(trawefield, trawshower.shower_core_pos, gt.trun.origin_geoid)
            du_ids = np.append(du_ids, tdu_ids)
            du_xyzs = np.vstack([du_xyzs, tdu_xyzs])
            du_geoids = np.vstack([du_geoids, tdu_geoids])

        # For star shapes, set the trun's du_id/xyz now and fill/write the tree
        if clargs.star_shape:
            gt.trun.du_id = tdu_ids
            gt.trun.du_xyz = np.array(tdu_xyzs)
            gt.trun.du_geoid = np.array(tdu_geoids)   # was "du_geoids", silently ignored (#220)

            # This run's units only (du_ids accumulates over all files)
            gt.trun.du_tilt = np.zeros(shape=(len(tdu_ids), 2), dtype=np.float32)

            # For now (and for the foreseeable future) all DU will have the same bin size at the level of the efield simulator.
            gt.trun.t_bin_size = np.array([trawefield.t_bin_size] * len(tdu_ids))

            # Set the site layout to star_shape if not overriden by a command line option
            if not clargs.site_layout: gt.trun.site_layout = "star_shape"

            # Fill the start and end events in trun
            fill_star_end_event_in_run(start_event_number, end_event_number, start_event_time, end_event_time, gt.trun)

            # Fill and write the TRun
            gt.trun.fill()

        # gt.tshower.first_interaction = trawshower.first_interaction

        trawmeta.close_file()

        trawshower.stop_using()
        trawefield.stop_using()
        trawmeta.stop_using()

        # Increment the event number if starting one specified on command line
        if ext_event_number is not None:
            ext_event_number += 1

    if out_dir_name == "":
        fail("no events in the input files (all were empty or unreadable); nothing written")

    # Fill the trun with antenna positions and ids from ALL the events (not for star shape, already done)
    # ToDo: this should be done with TChain in one loop over all the files... maybe (which would be faster?)
    if not clargs.star_shape:

        # Get indices of the unique du_ids
        unique_dus_idx = np.unique(du_ids, return_index=True)[1]
        # Leave only the unique du_ids
        du_ids = du_ids[unique_dus_idx]
        # Sort the DUs
        sorted_idx = np.argsort(du_ids)
        du_ids = du_ids[sorted_idx]
        # Stack x/y/z together and leave only the ones for unique du_ids, sort
        du_xyzs = du_xyzs[unique_dus_idx][sorted_idx]
        du_geoids = du_geoids[unique_dus_idx][sorted_idx]

        # Assign the du ids and positions to the trun tree
        gt.trun.du_id = du_ids
        gt.trun.du_xyz = du_xyzs
        gt.trun.du_geoid = du_geoids
        gt.trun.du_tilt = np.zeros(shape=(len(du_ids), 2), dtype=np.float32)

        #For now (and for the forseable future) all DU will have the same bin size at the level of the efield simulator.
        gt.trun.t_bin_size = [trawefield.t_bin_size]*len(du_ids)

        # Fill the start and end events in trun
        fill_star_end_event_in_run(start_event_number, end_event_number, start_event_time, end_event_time, gt.trun)

        # Fill and write the TRun
        gt.trun.fill()
        # gt.trun.write()
        # gt.trunshowersim.write()
        # gt.trunefieldsim.write()

    # Write the event trees, unless the last set was already written by -ef
    # (or one opened after it got no event)
    event_files_pending = event_files_open and not (split_done and events_in_file == 0)
    if event_files_pending:
        gt.tshower.write(force_close_file=True)
        gt.tshowersim.write(force_close_file=True)
        gt.tefield.write(force_close_file=True)
    elif event_files_open:
        for tree, name in ((gt.tshower, "shower"), (gt.tshowersim, "showersim"), (gt.tefield, "efield")):
            tree.stop_using()
            Path(out_dir_name, name + ".root").unlink(missing_ok=True)
    gt.trun.write(force_close_file=True)
    gt.trunshowersim.write(force_close_file=True)
    gt.trunefieldsim.write(force_close_file=True)

    # Rename the created files to appropriate names
    logger.info("Renaming files to proper file names")
    rename_all_files(clargs, out_dir_name, file_start_event_number, end_event_number, start_run_number,
                     event_files=event_files_pending)

# Initialise all output trees and their directory
def init_all_trees(clargs, unix_date, run_number, site, gt):

    # Use date/time from command line argument if specified, otherwise the unix time
    #date, time = datetime.datetime.utcfromtimestamp(unix_date).strftime('%Y%m%d_%H%M%S').split("_") #changed to comply with deprecation warning.
    date,time=datetime.datetime.fromtimestamp(unix_date, datetime.UTC).strftime('%Y%m%d_%H%M%S').split("_")

    if clargs.sim_date is not None:
        date = clargs.sim_date
    if clargs.sim_time is not None:
        time = clargs.sim_time

    # Create the appropriate output directory
    if clargs.forced_output_directory is None:
        out_dir_name = form_directory_name(clargs, date, time, run_number, site)
        logger.info(f"Storing files in directory {out_dir_name}")
        out_dir_name.mkdir(parents=True)  # -o may name a folder not yet made (#257)
        _output.update(dir=out_dir_name, created=True, before=set())
    # If another directory was forced as the output directory, create it
    else:
        out_dir_name = Path(clargs.output_parent_directory, clargs.forced_output_directory)
        existed = out_dir_name.is_dir()
        out_dir_name.mkdir(parents=True, exist_ok=True)
        _output.update(dir=out_dir_name, created=not existed,
                       before={path.name for path in out_dir_name.iterdir()})

    # Create appropriate GRANDROOT trees in temporary file names (event range not known until the end of the loop)
    # Init run trees only if requested
    gt.trun = TRun((out_dir_name / "run.root").as_posix())
    gt.trunshowersim = TRunShowerSim((out_dir_name / "runshowersim.root").as_posix())
    gt.trunefieldsim = TRunEfieldSim((out_dir_name / "runefieldsim.root").as_posix())
    init_event_trees(out_dir_name, gt)

    return out_dir_name

# Initialise event output trees
def init_event_trees(out_dir_name, gt):
    gt.tshower = TShower((out_dir_name / "shower.root").as_posix())
    gt.tshowersim = TShowerSim((out_dir_name / "showersim.root").as_posix())
    gt.tefield = TEfield((out_dir_name / "efield.root").as_posix())

## Convert the RawShowerTree first entry to run values
def rawshower2grandrootrun(trawshower, gt):
    gt.trunshowersim.run_number = trawshower.run_number
    # Name and version of the shower simulator
    # If there is a single space in sim_name, assume the second part is a version
    if trawshower.sim_name.count(" ") == 1:
        sim_name, sim_version = trawshower.sim_name.split()
        gt.trunshowersim.sim_name = sim_name
        gt.trunshowersim.sim_version = sim_version
    # If no single space or more spaces, don't fill the version, put everything as sim_name
    else:
        gt.trunshowersim.sim_name = trawshower.sim_name
        gt.trunshowersim.sim_version = trawshower.sim_version

    gt.trunshowersim.site = trawshower.site

    # Generate the simulator for trun
    if "aires" in trawshower.sim_name.lower():
        gt.trun.data_generator = "ZHAireS"
    elif "coreas" in trawshower.sim_name.lower():
        gt.trun.data_generator = "CoREAS"

    # ToDo: Add trun.data_generator_version when it becomes available in rawroot!
    gt.trun.data_generator_version = "unknown"

    #### ZHAireS/Coreas
    # * THINNING *
    # Thinning energy, relative to primary energy
    # this is EFRCTHN in Coreas (the 0th THIN value)
    gt.trunshowersim.rel_thin = trawshower.rel_thin

    # this is the maximum weight, computed in zhaires as PrimaryEnergy*RelativeThinning*WeightFactor/14.0 (see aires manual section 3.3.6 and 2.3.2) to make it mean the same as Corsika Wmax
    # this is WMAX in Coreas (the 1st THIN value) - Weight limit for thinning
    gt.trunshowersim.maximum_weight = trawshower.maximum_weight

    # this is the ratio of energy at wich thining starts in hadrons and electromagnetic particles. In Aires is always 1
    # this is THINRAT in Coreas (the 0th THINH value) - hadrons
    gt.trunshowersim.hadronic_thinning = trawshower.hadronic_thinning

    # this is the ratio of electromagnetic to hadronic maximum weights.
    # this is WEIRAT in Coreas (the 1st THINH value)
    gt.trunshowersim.hadronic_thinning_weight = trawshower.hadronic_thinning_weight

    # Maximum radius (in cm) at observation level within which all particles are subject to inner radius thinning. In corsika particles are sampled following a r^(-4) distribution
    # Aires has a similar feature, but the definition is much more complex...so this will be left empty for now.
    # this is RMAX in Coreas (the 2nd THIN value)
    # gt.trunshowersim.rmax = trawshower.rmax

    # * CUTS *
    # gamma energy cut (GeV)
    gt.trunshowersim.lowe_cut_gamma = trawshower.lowe_cut_gamma

    # electron/positron energy cut (GeV)
    gt.trunshowersim.lowe_cut_e = trawshower.lowe_cut_e

    # muons energy cut (GeV)
    gt.trunshowersim.lowe_cut_mu = trawshower.lowe_cut_mu

    # mesons energy cut (GeV)
    gt.trunshowersim.lowe_cut_meson = trawshower.lowe_cut_meson

    # nucleons energy cut (GeV)
    gt.trunshowersim.lowe_cut_nucleon = trawshower.lowe_cut_nucleon


# Convert the RawEfieldTree first entry to run values
def rawefield2grandrootrun(trawefield, gt):
    gt.trunefieldsim.run_number = trawefield.run_number

    ## Name and version of the electric field simulator
    gt.trunefieldsim.sim_name = trawefield.sim_name
    gt.trunefieldsim.sim_version = trawefield.sim_version

    gt.trunefieldsim.site = trawefield.site

    ## Name of the atmospheric index of refraction model
    gt.trunefieldsim.refractivity_model = trawefield.refractivity_model
    gt.trunefieldsim.refractivity_model_parameters = trawefield.refractivity_model_parameters

    # The TRun run number
    gt.trun.run_number = trawefield.run_number

    ## The antenna time window is defined around a t0 that changes with the antenna, starts on t0-t_pre (thus t_pre should be positive) and ends on t0+post
    gt.trunefieldsim.t_pre = trawefield.t_pre
    gt.trunefieldsim.t_post = trawefield.t_post


def get_tree_du_id_xyz_geoid(trawefield, shower_core, origin_geoid):
    # *** Store the DU's to run - they needed to be collected from all events ***
    # Get the ids and positions from all the events

    #trawefield has the antenna positions in array coordinates, cartesian. Origin is at the delcared latitude, longitude and altitude of the site.
    print("Warning: using flat earth approximation for coordinates!.Event:",trawefield.event_number," Core:",shower_core)
    count = trawefield.draw("du_id:du_x:du_y:du_z", "", "goff")
    # No antenna in any entry of this file (issue #91): nothing to add to the run's DU list
    if count <= 0:
        logger.info("No antenna in this file's efield tree; it adds no DU to the run.")
        return (np.zeros(0, dtype=np.int32), np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))
    du_ids =np.array(np.frombuffer(trawefield.get_v1(), dtype=np.float64, count=count)).astype(np.int32)
    du_xs = np.array(np.frombuffer(trawefield.get_v2(), dtype=np.float64, count=count)).astype(np.float32)
    du_ys = np.array(np.frombuffer(trawefield.get_v3(), dtype=np.float64, count=count)).astype(np.float32)
    du_zs = np.array(np.frombuffer(trawefield.get_v4(), dtype=np.float64, count=count)).astype(np.float32)
    
    # Get indices of the unique du_ids
    # ToDo: sort?
    unique_dus_idx = np.unique(du_ids, return_index=True)[1]
    # Leave only the unique du_ids
    du_ids = du_ids[unique_dus_idx]
    # Stack x/y/z together and leave only the ones for unique du_ids
    du_xyzs = np.column_stack([du_xs, du_ys, du_zs])[unique_dus_idx]

    # Get lat/lon/alt from xyz
    # From the unique positions, so that the rows match du_ids and du_xyzs:
    # computed from the full draw arrays, they did not as soon as a file held
    # several events with the same antennas (issue #220)
    origin = Geodetic(latitude=origin_geoid[0], longitude=origin_geoid[1], height=origin_geoid[2])
    grandcs = GRANDCS(x=du_xyzs[:, 0], y=du_xyzs[:, 1], z=du_xyzs[:, 2], location=origin)
    du_geoid = np.moveaxis(np.asarray(Geodetic(grandcs), dtype=np.float32), 0, 1)

    return np.asarray(du_ids, dtype=np.int32), np.asarray(du_xyzs, dtype=np.float32), du_geoid


# Convert the RawShowerTree entries
def rawshower2grandroot(trawshower, gt):
    ### Event name (the task name, can be usefull to track the original simulation)
    ## ToDo: not in TShowerSim - decide
    # gt.tshowersim.event_name = trawshower.event_name

    ## Run and event number
    gt.tshower.run_number = trawshower.run_number
    gt.tshower.event_number = trawshower.event_number
    gt.tshowersim.run_number = trawshower.run_number
    gt.tshowersim.event_number = trawshower.event_number

    ### Event Date  (used to define the atmosphere and/or the magnetic field)
    # ToDo: Shouldn't it be an epoch already in sims?
    gt.tshowersim.event_date = int(time.mktime(time.strptime(trawshower.event_date, "%Y-%m-%d")))

    ### Random seed
    gt.tshowersim.rnd_seed = trawshower.rnd_seed

    ### Energy in neutrinos generated in the shower (GeV). Useful for invisible energy computation
    # gt.tshower.energy_in_neutrinos = trawshower.energy_in_neutrinos

    ### Primary energy (GeV)
    # ToDo: it should be a scalar on sim side
    gt.tshower.energy_primary = trawshower.energy_primary[0]

    # Fill energy_em (GeV)
    gt.tshower.energy_em = trawshower.energy_em[0]

    ### Shower azimuth (deg, CR convention)
    gt.tshower.azimuth = trawshower.azimuth

    ### Shower zenith  (deg, CR convention)
    gt.tshower.zenith = trawshower.zenith

    ### Primary particle type (PDG)
    # ToDo: it should be a scalar on sim side
    gt.tshower.primary_type = trawshower.primary_type[0]

    # Primary injection point [m] in Shower coordinates
    gt.tshowersim.primary_inj_point_shc = trawshower.primary_inj_point_shc

    ### Primary injection altitude [m] in Shower Coordinates
    gt.tshowersim.primary_inj_alt_shc = trawshower.primary_inj_alt_shc

    # primary injection direction in Shower Coordinates
    gt.tshowersim.primary_inj_dir_shc = trawshower.primary_inj_dir_shc

    ### Atmospheric model name TODO:standardize
    gt.tshower.atmos_model = trawshower.atmos_model

    # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
    gt.tshower.atmos_model_param = trawshower.atmos_model_param

    # Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]
    gt.tshowersim.atmos_altitude = trawshower.atmos_altitude
    gt.tshowersim.atmos_density = trawshower.atmos_density
    gt.tshowersim.atmos_depth = trawshower.atmos_depth

    ### Magnetic field parameters: Inclination, Declination, Fmodulus.: In shower coordinates. Declination
    # The Earth's magnetic field, B, is described by its strength, Fmodulus = |B|; its inclination, I, defined
    # as the angle between the local horizontal plane and the field vector; and its declination, D, defined
    # as the angle between the horizontal component of B, H, and the geographical North (direction of
    # the local meridian). The angle I is positive when B points downwards and D is positive when H is
    # inclined towards the East.
    gt.tshower.magnetic_field = trawshower.magnetic_field

    ### Shower Xmax depth  (g/cm2 along the shower axis)
    gt.tshower.xmax_grams = trawshower.xmax_grams

    ### Shower Xmax position in shower coordinates [m]
    gt.tshower.xmax_pos_shc = trawshower.xmax_pos_shc

    ### Shower Xmax position [m] in the site frame (that of du_xyz and shower_core_pos):
    # the ground-relative xmax_pos_shc plus the core, the point the readers compute as
    # FIX_xmax_pos.  Raw files written before the ZHAireS converter subtracted the ground
    # altitude carry xmax_pos_shc above sea level; xmax_in_site_frame detects that from the
    # geometry.  NaN when the simulation does not know Xmax (issue #104).
    gt.tshower.xmax_pos, _ = xmax_in_site_frame(
        trawshower.xmax_pos_shc, trawshower.zenith, trawshower.azimuth,
        trawshower.site_alt, trawshower.shower_core_pos)

    ### Shower direction: unit vector along which the shower travels, in the site frame
    # (x North, y West, z Up).  zenith and azimuth say where it comes from, so this is the
    # opposite of the vector they name; u_z < 0 for a downgoing shower (issue #104).
    gt.tshower.direction = propagation_direction(trawshower.zenith, trawshower.azimuth)

    ### Distance of Xmax  [m] to the ground
    # gt.tshower.xmax_distance = trawshower.xmax_distance

    ### Altitude of Xmax  [m]. Its important for the computation of the index of refraction at maximum, and of the cherenkov cone
    # gt.tshower.xmax_alt = trawshower.xmax_alt

    ### high energy hadronic model (and version) used TODO: standarize
    gt.tshowersim.hadronic_model = trawshower.hadronic_model

    ### low energy model (and version) used TODO: standarize
    gt.tshowersim.low_energy_model = trawshower.low_energy_model

    ### Time it took for the simulation of the cascade (s). In the case shower and radio are simulated together, use TotalTime/(nant-1) as an approximation
    gt.tshowersim.cpu_time = trawshower.cpu_time

    ###META ZHAireS/Coreas

    ### Core position with respect to the antenna array (undefined for neutrinos)
    ## ToDo: conversion?
    gt.tshower.shower_core_pos = trawshower.shower_core_pos
    #print("THI IS THE CORE",gt.tshower.shower_core_pos,trawshower.shower_core_pos)

    ### Longitudinal Pofiles (those compatible between Coreas/ZHAires)

    ## Longitudinal Profile of vertical depth (g/cm2) #we remove these becouse is not easily available in CORSIKA 
    #gt.tshowersim.long_depth = trawshower.long_depth
    ## Longitudinal Profile of slant depth (g/cm2)
    #gt.tshowersim.long_pd_depth = trawshower.long_slantdepth
    gt.tshowersim.long_pd_depth = trawshower.long_pd_depth
    ## Longitudinal Profile of Number of Gammas
    # TShowerSim calls it long_pd_gamma: assigned as "long_pd_gammas", the gamma
    # profile was stored nowhere (found by the #202 field guard)
    gt.tshowersim.long_pd_gamma = trawshower.long_pd_gammas
    ## Longitudinal Profile of Number of e+
    gt.tshowersim.long_pd_eplus = trawshower.long_pd_eplus
    ## Longitudinal Profile of Number of e-
    gt.tshowersim.long_pd_eminus = trawshower.long_pd_eminus
    ## Longitudinal Profile of Number of mu+
    gt.tshowersim.long_pd_muplus = trawshower.long_pd_muplus
    ## Longitudinal Profile of Number of mu-
    gt.tshowersim.long_pd_muminus = trawshower.long_pd_muminus
    ## Longitudinal Profile of Number of Hadrons (TShowerSim: long_pd_hadron;
    ## assigned as "long_pd_hadr" it was stored nowhere, #202)
    gt.tshowersim.long_pd_hadron = trawshower.long_pd_hadr

    # Not written: TShowerSim has no field for them.  They were assigned under
    # these names and silently stored nowhere until the #202 field guard made
    # that an error.  The energy profiles may sit on their own depth grid
    # (long_ed_depth), which TShowerSim cannot record next to long_pd_depth;
    # mapping them onto long_*_elow / long_*_edep is a data-model decision.
    #   long_pd_allch, long_pd_nuclei, long_ed_neutrino, long_ed_*_cut,
    #   long_ed_*_ioniz, long_ed_depth

    # gt.tshower.first_interaction = trawshower.first_interaction

# Convert the RawEfieldTree entries
def rawefield2grandroot(trawefield, gt, ext_trace = None, ext_t_0 = None):
    ## Run and event number
    gt.tefield.run_number = trawefield.run_number
    gt.tefield.event_number = trawefield.event_number

    # atmos_refractivity has no field in the GRANDROOT trees; it was stored
    # nowhere (#202)
    # gt.tshowersim.atmos_refractivity = trawefield.atmos_refractivity

    # Per antenna things
    gt.tefield.du_id = trawefield.du_id
    # gt.tefield.du_name = trawefield.du_name
    ## Number of detector units in the event - basically the antennas count
    gt.tefield.du_count = trawefield.du_count

    # ToDo!!!
    # gt.tefield.t_0 = trawefield.t_0
    gt.tefield.p2p = trawefield.p2p

    # ToDo: this should be a single vector of xyz
    ## X position in shower referential
    # gt.tefield.du_x = trawefield.du_x  (TEfield has no du_x; stored nowhere, #202)
    ## Y position in shower referential
    # gt.tefield.du_y = trawefield.du_y  (TEfield has no du_y; stored nowhere, #202)
    ## Z position in shower referential
    # gt.tefield.du_z = trawefield.du_z  (TEfield has no du_z; stored nowhere, #202)

    ## Efield trace in X,Y,Z direction
    if ext_trace is None:
        gt.tefield.trace = np.moveaxis(np.array([trawefield.trace_x, trawefield.trace_y, trawefield.trace_z]), 0,1).astype(np.float32)
    else:
        gt.tefield.trace = ext_trace
        # gt.tefield.trace_x=ext_trace[:,0,:]
        # gt.tefield.trace_y=ext_trace[:,1,:]
        # gt.tefield.trace_z=ext_trace[:,2,:]

    if ext_t_0 is not None:
        t_0 = ext_t_0
    else:
        t_0 = trawefield.t_0

    # Generate trigger times from t0s
    tempseconds=np.zeros((len(t_0)), dtype=np.int64)
    tempseconds[:]=gt.event_seconds
    tempnanoseconds= np.int64(gt.event_nanoseconds + t_0)
    #rolling over the nanoseconds    
    maskplus= gt.event_nanoseconds + t_0 >=1e9
    maskminus= gt.event_nanoseconds + t_0 <0
    tempnanoseconds[maskplus]-=np.int64(1e9)
    tempseconds[maskplus]+=np.int64(1)   
    tempnanoseconds[maskminus]+=np.int64(1e9)
    tempseconds[maskminus]-=np.int64(1)
    gt.tefield.du_nanoseconds=tempnanoseconds.astype(np.uint32)
    gt.tefield.du_seconds=tempseconds.astype(np.uint32)
    
    #store tpre in samples is the expected trigger position in sims. 
    #This can be furter enforced with the --trigger_time_ns switch. All dus have the same value at the efield generator level
    gt.tefield.trigger_position= np.ushort([trawefield.t_pre]*trawefield.du_count/trawefield.t_bin_size)

# Convert the RawMetaTree entries
def rawmeta2grandroot(trawmeta, gt):
    #gt.tshower.shower_core_pos = trawmeta.shower_core_pos this is duplicated, using ithe one in shower for compatibility
    gt.tshowersim.event_weight = trawmeta.event_weight
    gt.tshowersim.tested_cores = trawmeta.tested_cores
    #event time    
    if(trawmeta.unix_second>0):
      gt.tshower.core_time_s = trawmeta.unix_second              #this will be filled by the reconstruction of the core position eventually?
      gt.event_seconds = trawmeta.unix_second
    else:
      gt.tshower.core_time_s = 200854852
      gt.event_seconds = 200854852
    gt.tshower.core_time_ns = trawmeta.unix_nanosecond         #this will be filled by the reconstruction of the core position eventually?
    gt.event_nanoseconds = trawmeta.unix_nanosecond
    
    

## Get origin geoid
def get_origin_geoid(clargs, trawshower):
    # "is not None": 0 is a valid latitude, longitude or altitude
    lat = clargs.latitude if clargs.latitude is not None else trawshower.site_lat
    lon = clargs.longitude if clargs.longitude is not None else trawshower.site_lon
    alt = clargs.altitude if clargs.altitude is not None else trawshower.site_alt
    return [lat, lon, alt]

# Form the proper output directory name from command line arguments
def form_directory_name(clargs, date, time, run_number, site):
    # Change possible underscores in extra into -
    extra = clargs.extra.replace("_", "-")

    # Go through serial numbers in directory names to find a one that does not exist
    for sn in range(5000):
        # Without -e, "CD_{extra}_" left a double underscore (#226)
        cd = f"CD_{extra}_" if extra else "CD_"
        dir_name = Path(clargs.output_parent_directory, f"sim_{site}_{date}_{time}_RUN{run_number}_{cd}{sn:0>4}")
        if not dir_name.exists():
            break
    # If directories with serial number up to 5000 already created
    else:
        fail("all output folder names up to serial number 5000 exist; clean some up")

    return dir_name

# Rename the created files to appropriate names
def rename_all_files(clargs, path, start_event_number, end_event_number, run_number, event_files=True):

    # Go through run output files
    for fn_start in ["run", "runshowersim", "runefieldsim"]:
        # Go through serial numbers in directory names to find a one that does not exist
        for sn in range(5000):
            fn_in = Path(path, f"{fn_start}.root")
            # Proper name of the file
            fn_out = Path(path, f"{fn_start}_{run_number}_L{clargs.analysis_level}_{sn:0>4}.root")
            # If the output file with the current serial number does not exist, rename to it
            if not fn_out.exists():
                fn_in.rename(fn_out)
                break
        else:
            fail(f"no free file name for {fn_in} up to serial number 5000; clean some up")

    # Rename the event files
    if event_files:
        rename_event_files(clargs, path, start_event_number, end_event_number)

def rename_event_files(clargs, path, start_event_number, end_event_number):

    # Go through event output files
    for fn_start in ["shower", "showersim", "efield"]:
        # Go through serial numbers in directory names to find a one that does not exist
        for sn in range(5000):
            fn_in = Path(path, f"{fn_start}.root")
            # Proper name of the file
            fn_out = Path(path, f"{fn_start}_{start_event_number}-{end_event_number}_L{clargs.analysis_level}_{sn:0>4}.root")
            # If the output file with the current serial number does not exist, rename to it
            if not fn_out.exists():
                fn_in.rename(fn_out)
                break
        else:
            fail(f"no free file name for {fn_in} up to serial number 5000; clean some up")

## Simple shifting of a single x,y,z trace
def trace_shift(arr, shift):
    # Shift the array right
    if shift>0:
        arr[:,shift:]=arr[:,:-shift]
        arr[:,:shift]=0
    # Shift the array left
    elif shift<0:
        arr[:,:shift]=arr[:,-shift:]
        arr[:,shift:]=0
    # No shift
    else:
        return arr

## Fill the start and end events in trun
def fill_star_end_event_in_run(start_event_number, end_event_number, start_event_time, end_event_time, trun):
    trun.first_event = start_event_number
    trun.last_event = end_event_number
    trun.first_event_time = start_event_time
    trun.last_event_time = end_event_time

if __name__ == '__main__':
    main()
