from grand.aoi import *
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import grand.analysis.signals.extraction as ext
import numpy as np
import grand.analysis.fitting as fit
import grand.analysis.constants as cons
import grand.analysis.geom as geom
import grand.analysis.energy_reco as en
import shutil
from pathlib import Path
import config as conf
import re
from datetime import datetime
import pandas as pd

# ---------------------------------------------------------------
# Load RTK antenna positions from text file
# Convert antenna IDs to int, store x (East), y (North), z (sea level)
# ---------------------------------------------------------------
file_path = conf.antenna_file
column_names = ['antenna_ID', 'x', 'y', 'z'] 
antenna_position = pd.read_csv(file_path, sep='\s+', names=column_names, header=None)
antenna_position['antenna_ID'] = antenna_position['antenna_ID'].astype(int) 

# ---------------------------------------------------------------
# Function: read flagged events from text file
# Skips comments and empty lines
# ---------------------------------------------------------------
def read_event_list(txt_file, start_line, stop_line=None):
    valid_line_count = 0
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            valid_line_count += 1
            if valid_line_count < start_line:
                continue
            if stop_line is not None and valid_line_count > stop_line:
                return

            parts = line.split()
            if len(parts) < 2:
                continue

            runfile, evt_str = parts[:2]
            yield runfile, int(evt_str)


# ---------------------------------------------------------------
# Prepare output folder
# Delete previous reconstruction results if exist
# ---------------------------------------------------------------
out_dir_root = "reconstructed_events_AOI"
if Path(out_dir_root).is_dir():
    shutil.rmtree(out_dir_root)
Path(out_dir_root).mkdir(parents=True, exist_ok=True)

# Initialize a list to store all reconstructed events
el_all = []

flagged_txt = conf.flagged_txt


# ---------------------------------------------------------------
# Loop over flagged events
# ---------------------------------------------------------------
for rootfile, ev_idx in read_event_list(flagged_txt, start_line=1, stop_line=10):

    # Extract year, month, and run number from ROOT file name
    match = re.search(r'GP80_(\d{4})(\d{2})\d{2}_\d+_RUN(\d+)_', rootfile)
    if not match:
        print(f"Skipping invalid ROOT file name: {rootfile}")
        continue
    year, month, run_number_str = match.groups()
    path_rootfile = f'{conf.path_rootfile}{year}/{month}/{rootfile}'
    run_number = int(run_number_str)

   # ---------------------------------------------------------------
    # Load the EventList from the ROOT file
    # We use TRawVoltage and select:
    #   channel 1 → X polarization
    #   channel 2 → Y polarization
    #   channel 3 → Z polarization
    # ---------------------------------------------------------------
    el = EventList(path_rootfile, use_trawvoltage=True, trawvoltage_channels=[1,2,3])

    # Retrieve the flagged event
    e = el.get_event(event_number=ev_idx, run_number=run_number)
    if e is None:
        print(f"Event {ev_idx} not found in {rootfile}")
        continue

    print(f"Processing Run #{e.run_number}, Event #{e.event_number}")

    # -------------------------------
    # Convert voltage traces to ADC counts
    # At this stage, v.trace already contains only three components:
    #   v.trace[0] → X
    #   v.trace[1] → Y
    #   v.trace[2] → Z
    # Therefore channels=[0,1,2] refers to X, Y, Z respectively
    # --------------------------------
    ADC_traces_list = []
    for v in e.voltages:
        ADC_traces_list.append(ext.convert_voltage_to_ADC(v.trace, channels=[0,1,2])) #now only the 3 channels are stored
    ADC_traces = np.array(ADC_traces_list)
    n_antennas = len(e.voltages)

    # ---------------------------------------------------------------
    # Compute peak amplitudes and times for each antenna
    # ---------------------------------------------------------------
    peak_amps = np.array([ext.get_peak_amplitude(ADC_traces[i], channels=[0,1,2])
                          for i in range(n_antennas)])

    t0_all = np.array([v.t0.astype('int64') for v in e.voltages])
    t0 = t0_all - t0_all.min()

    peak_times = np.array([ext.get_peak_time(ADC_traces[i], t0[i], channels=[0,1,2])
                           for i in range(n_antennas)])

    # -------------------------------
    # Antenna positions
    # -------------------------------
    #Xants = np.array([[ant.position.x[0], ant.position.y[0], ant.position.z[0] + 1231] for ant in e.antennas])

    # ----------------------------------------------------------------
    # Map DU IDs to their positions (X, Y, Z) in GRAND reference frame
    # ----------------------------------------------------------------
    du_ids = np.array([v.du_id for v in e.voltages]).astype(int)

    du_positions_ev = antenna_position[antenna_position['antenna_ID'].isin(du_ids)]
    du_positions_ev = du_positions_ev.set_index('antenna_ID').loc[du_ids]
    x_coords = du_positions_ev['y'].astype(float).values #North
    y_coords = -du_positions_ev['x'].astype(float).values #West
    z_coords = du_positions_ev['z'].astype(float).values + cons.groundAltitude #DAQ level

    Xants = np.column_stack((x_coords, y_coords, z_coords))

    # -------------------------------
    # Plane Wave Fit (PWF)
    # -------------------------------
    theta_pwf_rad, phi_pwf_rad = fit.PWF_semianalytical(Xants, peak_times, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=5e-9)
    chi2_pwf_reduced = fit.PWF_loss((theta_pwf_rad, phi_pwf_rad), Xants, peak_times, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=5e-9)/(n_antennas - 2)

    # -------------------------------
    # Spherical Wave Fit (SWF)
    # -------------------------------
    theta_swf_rad, phi_swf_rad, r_xmax, t_s = fit.recons_swf(theta_pwf_rad, phi_pwf_rad, peak_times, Xants, sigma=5e-9)
    Xsource = fit.compute_Xsource_cartesian_coords(theta_swf_rad, phi_swf_rad, r_xmax)

    # Xsource has shape (1, 3); extract the (3,) Cartesian coordinate vector
    Xsource = Xsource[0]

    l_ant = geom.distance_source_antenna(Xants, Xsource)
    chi2_swf_reduced = fit.SWF_loss(theta_swf_rad, phi_swf_rad, r_xmax, t_s, Xants, peak_times, sigma=5e-9)/(n_antennas - 4)

    # -------------------------------
    # Angular Distribution Function (ADF)
    # -------------------------------
    theta_adf, phi_adf, delta_omega, scaling_factor = fit.recons_ADF(theta_pwf_rad, phi_pwf_rad, peak_amps, Xants, Xsource)
    eta, omega, omega_cr, l_ant, amplitude_model = fit.ADF_parameters(theta_adf, phi_adf, delta_omega, scaling_factor, Xants, Xsource, groundAltitude=cons.groundAltitude, Bvec=cons.Bvec)
    best_params = theta_adf, phi_adf, delta_omega, scaling_factor
    chi2_adf_reduced = fit.ADF_loss(best_params, peak_amps, Xants, Xsource)/(n_antennas - 4)
    sin_alpha = geom.sin_geomag_angle(theta_adf, phi_adf, B=cons.Bn)
    energy_elm = en.recons_energy_from_voltage(scaling_factor, sin_alpha)

    # ---------------------------------------------------------------
    # Store reconstruction results in the Shower object
    # ---------------------------------------------------------------
    e.shower = Shower()
    e.shower.du_count = n_antennas
    e.shower.zenith_pwf = theta_pwf_rad
    e.shower.azimuth_pwf = phi_pwf_rad
    e.shower.chi2_pwf = chi2_pwf_reduced

    e.shower.zenith_swf = theta_swf_rad
    e.shower.azimuth_swf = phi_swf_rad
    e.shower.r_xmax = r_xmax
    e.shower.t_s = t_s
    e.shower.chi2_swf = chi2_swf_reduced
    e.shower.distance_source_antenna = l_ant
    e.shower.Xsource = Xsource

    e.shower.zenith_adf = theta_adf
    e.shower.azimuth_adf = phi_adf
    e.shower.width = delta_omega
    e.shower.scaling_factor = scaling_factor
    e.shower.chi2_adf = chi2_adf_reduced
    e.shower.omega = omega
    e.shower.eta = eta
    e.shower.omega_cr = omega_cr
    e.shower.adf_amplitude = amplitude_model
    e.shower.energy_elm_voltage = energy_elm

    # ---------------------------------------------------------------
    # Write reconstructed event to output folder
    # ---------------------------------------------------------------
    #Stores a timestamp for the event file creation
    # if not --> an error occurs: AttributeError: 'Event' object has no attribute 'files_creation_time'
    e.files_creation_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    e.shower.out_dir = out_dir_root
    e.write(out_dir=out_dir_root)

    el_all.append(e)

# --------------------------------------
# Reload reconstructed events to check
# --------------------------------------
for e in el_all:
    ev_no = e.event_number
    run_no = e.run_number
    e_sh = e.shower

    print(f"================ Event {ev_no} Run {run_no} ================")

    print("__________________PWF__________________")
    print(f"zenith={np.rad2deg(e_sh.zenith_pwf):.2f}°, phi={np.rad2deg(e_sh.azimuth_pwf):.2f}°, chi2_pwf={e_sh.chi2_pwf:.2f} "
        )

    print("__________________SWF__________________")
    print(f"zenith={np.rad2deg(e_sh.zenith_swf):.2f}°, phi={np.rad2deg(e_sh.azimuth_swf):.2f}°, "
        f"t_s={e_sh.t_s:.3e}s, r_xmax={e_sh.r_xmax:.2f} m, chi2_swf={e_sh.chi2_swf:.2f},"
        f"Xsource={e_sh.Xsource}"
    )

    print("__________________ADF__________________")
    print(f"zenith={np.rad2deg(e_sh.zenith_adf):.2f}°, phi={np.rad2deg(e_sh.azimuth_adf):.2f}°, "
        f"width={e_sh.width:.4f}, scaling={e_sh.scaling_factor:.2f}, "
        f"chi2_adf={e_sh.chi2_adf:.2f}, "
        f"energy_elm={e_sh.energy_elm_voltage:.2e} eV, "
        f"omega={np.rad2deg(e_sh.omega)}, "
        f"omega_cherenkov={np.rad2deg(e_sh.omega_cr)}, "
        f"eta={np.rad2deg(e_sh.eta)}, "
        f"adf_amplitude={e_sh.adf_amplitude}, "
        f"l_ant={e_sh.distance_source_antenna}")
