import pandas as pd
import os
import numpy as np
import re
import grand.analysis.signals as sig
import grand.analysis.fitting as fit
import grand.analysis.constants as cons
import grand.analysis.energy_reco as en
import grand.analysis.coords.array_shower as co
import grand.analysis.geom as geom
from grand.dataio import TShower, TRawVoltage
import config as conf


# ---------------------------------------------------------------
# Function: read flagged events from text file
# Skips comments and empty lines
# ---------------------------------------------------------------
def read_event_list(txt_file, start_line, stop_line='None'):
    """
    Parameters:
    txt_file (str): Path to the input text file (Flagged event containing root file and corresponding event).
    start_line (int): First line number to read (1-based index: begins with 1).
    stop_line (int or None): Last line number to read (1-based index). If None, read until the end.

    Yields:
    tuple: (runfile, event) for each line in the specified range.
    """
    valid_line_count = 0
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # ignore empty or comment lines

            valid_line_count += 1

            # Skip lines before start_line
            if valid_line_count < start_line:
                continue

            # Stop if we exceed stop_line, stop_line is included in the count
            if stop_line is not None and valid_line_count > stop_line:
                return

            parts = line.split()
            if len(parts) < 2:
                continue  # skip malformed lines

            runfile, evt_str = parts[:2]
            yield runfile, int(evt_str)



# ---------------------------------------------------------------
# Load RTK antenna positions from text file
# Convert antenna IDs to int, store x (East), y (North), z (sea level)
# ---------------------------------------------------------------
file_path = conf.antenna_file
column_names = ['antenna_ID', 'x', 'y', 'z'] #'x' is East, 'y' is North, 'z' DAQ level (1231m) 
antenna_position = pd.read_csv(file_path, sep=r'\s+', names=column_names, header=None)
antenna_position['antenna_ID'] = antenna_position['antenna_ID'].astype(int) 

# ---------------------------------------------------------------
# Initialize TShower object to store reconstructed events
# ---------------------------------------------------------------
trecons = TShower()  


# ---------------------------------------------------------------
# Loop over flagged events from text file
# Parse year/month from filename to construct full ROOT file path
# ---------------------------------------------------------------

flagged_txt = conf.flagged_txt

for rootfile, ev_idx in read_event_list(flagged_txt, start_line=5, stop_line=15):

    # Extract year/month from ROOT file name
    match = re.search(r'GP80_(\d{4})(\d{2})\d{2}_', rootfile)
    if match:
        year, month = match.groups()
        path_rootfile = f'{conf.path_rootfile}{year}/{month}/{rootfile}'
        t_rawvoltage = TRawVoltage(path_rootfile)
        print("Full ROOT file path:", path_rootfile)

    # Load the specified event    
    t_rawvoltage.get_entry(ev_idx)  

    print("Event number:", t_rawvoltage.event_number)
    print("Run number:", t_rawvoltage.run_number)
    #print("DU IDs:", t_rawvoltage.du_id)

    # Set run and event numbers in TRecons
    trecons.run_number = t_rawvoltage.run_number
    trecons.event_number = t_rawvoltage.event_number

    # -----------------------------------------------------------------------------
    # Process voltage traces: convert to ADC counts, extract peak amplitude/time
    # -----------------------------------------------------------------------------
    num_antennas = len(t_rawvoltage.du_id)

    voltage_trace = np.array([sig.convert_voltage_to_ADC(t_rawvoltage.trace_ch[i], channels=[1,2,3]) for i in range(num_antennas)])

    peak_amps = np.array([sig.get_peak_amplitude(voltage_trace[i], channels=[1,2,3]) 
                          for i in range(num_antennas)])
    
    print('peak amps', peak_amps)
    
    t0_all = sig.compute_t0(t_rawvoltage) 
    peak_time = np.array([sig.get_peak_time(voltage_trace[i], t0_all[i], channels=[1,2,3]) 
                          for i in range(num_antennas)])

    print('peak times', peak_time)
    # ----------------------------------------------------------------
    # Map DU IDs to their positions (X, Y, Z) in GRAND reference frame
    # ----------------------------------------------------------------
    du_ids = np.array(t_rawvoltage.du_id).astype(int)

    du_positions_ev = antenna_position[antenna_position['antenna_ID'].isin(du_ids)]
    du_positions_ev = du_positions_ev.set_index('antenna_ID').loc[du_ids]
    x_coords = du_positions_ev['y'].astype(float).values #North
    y_coords = -du_positions_ev['x'].astype(float).values #West
    z_coords = du_positions_ev['z'].astype(float).values + cons.groundAltitude #DAQ level

    Xants = np.column_stack((x_coords, y_coords, z_coords))

    # ---------------------------------------------------------------
    # Store peak times, amplitudes and antenna positions in TShower
    # ---------------------------------------------------------------
    trecons.peak_time = peak_time
    trecons.peak_amps = peak_amps
    trecons.Xants = Xants
    trecons.du_count = num_antennas

    Xants = np.asarray(trecons.Xants)
    peak_time = np.asarray(trecons.peak_time)
    peak_amps = np.asarray(trecons.peak_amps)


    # ---------------------------------------------------------------
    # Plane Wave Fit (PWF)
    # ---------------------------------------------------------------
    theta_pwf_rad, phi_pwf_rad = fit.PWF_semianalytical(
        Xants, peak_time, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=5e-9
    )

    chi2_pwf_reduced = fit.PWF_loss(
        (theta_pwf_rad, phi_pwf_rad), Xants, peak_time, verbose=False, c=cons.c_light, n=cons.n_atm, sigma=5e-9
    )/(num_antennas - 2)

    # Store PWF results
    trecons.zenith_pwf = theta_pwf_rad
    trecons.azimuth_pwf = phi_pwf_rad
    trecons.chi2_pwf = chi2_pwf

    # ---------------------------------------------------------------
    # CRB calculation for PWF
    # ---------------------------------------------------------------
    stds_pwf = crb.CRB_PWF(
        theta_pwf_rad, phi_pwf_rad, Xants
    )

    trecons.crb_zenith_pwf = stds_pwf[0]
    trecons.crb_azimuth_pwf = stds_pwf[1]

    # ---------------------------------------------------------------
    # Spherical Wave Fit (SWF)
    # ---------------------------------------------------------------
    theta_swf_rad, phi_swf_rad, r_xmax, t_s = fit.recons_swf(trecons.zenith_pwf, trecons.azimuth_pwf, peak_time, Xants, sigma = 5e-9)

    # Compute source position in cartesian coordinates in GRAND detector frame
    Xsource = fit.compute_Xsource_cartesian_coords(theta_swf_rad, phi_swf_rad, r_xmax)

    chi2_swf_reduced = fit.SWF_loss(theta_swf_rad, phi_swf_rad, r_xmax, t_s, Xants, peak_time, sigma = 5e-9)/(num_antennas - 4)

    # Store SWF results
    trecons.zenith_swf = theta_swf_rad
    trecons.azimuth_swf = phi_swf_rad
    trecons.r_xmax = r_xmax
    trecons.t_s = t_s
    trecons.Xsource = Xsource
    trecons.chi2_swf = chi2_swf_reduced
    Xsource = np.asarray(trecons.Xsource)[0]

    l_ant = geom.distance_source_antenna(Xants, Xsource)
   
    # ---------------------------------------------------------------
    # Angular Distribution Function (ADF) reconstruction
    # ---------------------------------------------------------------
    theta_adf, phi_adf, delta_omega, scaling_factor = fit.recons_ADF(theta_pwf_rad, phi_pwf_rad, peak_amps, Xants, Xsource)
    eta, omega, omega_cr, l_ant, amplitude_model = fit.ADF_parameters(theta_adf, phi_adf, delta_omega, scaling_factor, Xants, Xsource, groundAltitude=cons.groundAltitude, Bvec=cons.Bvec)
    best_params = theta_adf, phi_adf, delta_omega, scaling_factor
    chi2_adf_reduced = fit.ADF_loss(best_params, peak_amps, Xants, Xsource)/(num_antennas - 4)

    # ---------------------------------------------------------------
    # Electromagnetic energy reconstruction 
    # ---------------------------------------------------------------
    sin_alpha = geom.sin_geomag_angle(theta_adf, phi_adf, B=cons.Bn)
    energy_elm = en.recons_energy_from_voltage(scaling_factor, sin_alpha)

    # ---------------------------------------------------------------
    # CRB calculations for ADF and SWF
    # ---------------------------------------------------------------

    stds = crb.CRB_ADF_SWF(
        theta_swf_rad, phi_swf_rad, r_xmax, t_s,
        theta_adf, phi_adf, delta_omega, scaling_factor,
        Xants
    )

    # Store ADF, energy results and CRB in TRecons
    trecons.zenith_adf = theta_adf
    trecons.azimuth_adf = phi_adf
    trecons.width = delta_omega
    trecons.scaling_factor = scaling_factor
    trecons.chi2_adf = chi2_adf_reduced
    trecons.omega = omega
    trecons.eta = eta
    trecons.omega_cr = omega_cr
    trecons.adf_amplitude = amplitude_model
    trecons.distance_source_antenna = l_ant
    trecons.energy_elm_voltage = energy_elm

    # Store CRB results
    trecons.crb_zenith_adf = stds[4]
    trecons.crb_azimuth_adf = stds[5]
    trecons.crb_width = stds[6]
    trecons.crb_scaling_factor = stds[7]
    trecons.crb_zenith_swf = stds[0]
    trecons.crb_azimuth_swf = stds[1]
    trecons.crb_r_xmax = stds[2]
    trecons.crb_t_s = stds[3]


    # ---------------------------------------------------------------
    # Fill the reconstruction tree for this event
    # ---------------------------------------------------------------
    trecons.fill()  

# ---------------------------------------------------------------
# Write the reconstructed events to a new ROOT file
# ---------------------------------------------------------------
trecons.write(f"{conf.output}")

# ---------------------------------------------------------------
# Read back the reconstructed events to check results
# ---------------------------------------------------------------
t_recons = TShower(f"{conf.output}")

for ev_no, run_no in t_recons.get_list_of_events():
    t_recons.get_event(ev_no, run_no)
    
    # print("__________________PWF__________________")
    # print(f"Event {ev_no} Run {run_no}: zenith={np.rad2deg(t_recons.zenith_pwf)}, phi={np.rad2deg(t_recons.azimuth_pwf)}, chi2_pwf={t_recons.chi2_pwf}")

    # print("__________________SWF__________________")
    # print(f"Event {ev_no} Run {run_no}: zenith={np.rad2deg(t_recons.zenith_swf)}, phi={np.rad2deg(t_recons.azimuth_swf)}, " 
    #       f"t_s={t_recons.t_s}, r_xsource={t_recons.r_xmax}, chi2_swf={t_recons.chi2_swf},"
    #       f"Xsource={t_recons.Xsource}")
    
    # print("__________________ADF__________________")
    # print(f"Event {ev_no} Run {run_no}: zenith={np.rad2deg(t_recons.zenith_adf)}, phi={np.rad2deg(t_recons.azimuth_adf)}, " 
    #       f"width={t_recons.width}, scaling={t_recons.scaling_factor}, chi2_adf={t_recons.chi2_adf}, energy elm={t_recons.energy_elm_voltage},"
    #       f"omega={np.rad2deg(t_recons.omega)},"
    #       f"omega cherenkov={np.rad2deg(t_recons.omega_cr)},"
    #       f"eta={np.rad2deg(t_recons.eta)},"
    #       f"amplitude model={t_recons.adf_amplitude},"
    #       f"l_ant={t_recons.distance_source_antenna}"
    #   )
    
    print("\n\n____________________________________")
    print(f"Event {ev_no} Run {run_no}")
    print(f"\nPWF Values and uncertainties:\n Zenith: {np.rad2deg(t_recons.zenith_pwf)} ± {np.rad2deg(t_recons.crb_zenith_pwf)} deg\n Azimuth: {np.rad2deg(t_recons.azimuth_pwf)} ± {np.rad2deg(t_recons.crb_azimuth_pwf)} deg")
    print(f"\nSWF Values and uncertainties:\n Zenith: {np.rad2deg(t_recons.zenith_swf)} ± {np.rad2deg(t_recons.crb_zenith_swf)} deg\n Azimuth: {np.rad2deg(t_recons.azimuth_swf)} ± {np.rad2deg(t_recons.crb_azimuth_swf)} deg\n r_xmax: {t_recons.r_xmax} ± {t_recons.crb_r_xmax} m\n t_s: {t_recons.t_s} ± {t_recons.crb_t_s} s")
    print(f"\nADF Values and uncertainties:\n Zenith: {np.rad2deg(t_recons.zenith_adf)} ± {np.rad2deg(t_recons.crb_zenith_adf)} deg\n Azimuth: {np.rad2deg(t_recons.azimuth_adf)} ± {np.rad2deg(t_recons.crb_azimuth_adf)} deg\n Width: {t_recons.width} ± {t_recons.crb_width}\n Scaling factor: {t_recons.scaling_factor} ± {t_recons.crb_scaling_factor}")