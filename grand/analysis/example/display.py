import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import grand.analysis.fitting as fit
import grand.analysis.coords.array_shower as co
import grand.analysis.geom as geom
from grand.dataio import TRecons
import config as conf

# ---------------------------------------------------------------
# Load RTK antenna positions from text file
# x (East), y (North), z (sea level)
# Convert antenna IDs to int
# Exclude DUs with IDs between 0 and 15 (GP13: removed)
# ---------------------------------------------------------------
file_path = conf.antenna_file
column_names = ['antenna_ID', 'x', 'y', 'z'] #'x' is East, 'y' is North, 'z' sea level (0m)
antenna_position = pd.read_csv(file_path, sep='\s+', names=column_names, header=None)
antenna_position['antenna_ID'] = antenna_position['antenna_ID'].astype(int) 
antenna_position = antenna_position[~antenna_position['antenna_ID'].between(0, 15)]

# ---------------------------------------------------------------
# Load reconstructed events
# ---------------------------------------------------------------
t_recons = TRecons(f"{conf.output}")

# Loop over all events in the reconstruction file
for ev_no, run_no in t_recons.get_list_of_events():
    t_recons.get_event(ev_no, run_no)
    K = co.shower_direction_vector(t_recons.zenith_adf, t_recons.azimuth_adf)
    # Source position from the SWF reconstruction
    Xsource = np.asarray(t_recons.Xsource)[0]
    # Compute the core position on the ground 
    Xcore = geom.compute_core(K, Xsource)
    Xants = np.asarray(t_recons.Xants)
    # Mean distance from antennas to shower source
    l_ant_mean = np.mean(geom.distance_source_antenna(Xants,Xsource))
    npts = 100
    # Mean Cherenkov angle 
    omega_cr_mean = np.mean(t_recons.omega_cr)
    # Generate vectors along the Cherenkov cone surface
    v_cone = geom.generate_cone_surface_vectors(K, omega_cr_mean, npts)
    print("Computing footprint with opening angle w_c = ",np.rad2deg(omega_cr_mean),"deg")
    # Compute intersection of cone vectors with ground → Cherenkov ellipse
    x_ell = np.zeros((npts,3))
    for i, v in enumerate(v_cone):
        x_ell[i,:] = geom.compute_core(v, Xsource)
    
   
    # ---------------------------------------------------------------
    # Compute mean ADF model (averaged over all eta angles)
    # ---------------------------------------------------------------
    w ,adf_f = fit.ADF_fun(l_ant_mean,t_recons.scaling_factor,omega_cr_mean,t_recons.width)

    # Reduced chi2 for ADF fit
    chi2_adf_reduced = t_recons.chi2_adf/(t_recons.du_count-4)

    fsuptit = (
    f"Event {ev_no} Run {run_no}"
    )
    ftit_adf = (
    rf"$\Theta = {np.rad2deg(t_recons.zenith_adf):.1f}^\circ, "
    rf"\Phi = {np.rad2deg(t_recons.azimuth_adf):.1f}^\circ, "
    rf"\chi^2_\mathrm{{adf}} / \mathrm{{ndf}} = {chi2_adf_reduced:.2f}$"
)
    
    # ---------------------------------------------------------------
    # Plot ADC peak amplitude vs angle ω and ADF model
    # ---------------------------------------------------------------
    plt.figure()
    # Peak ADC values measured at each triggered DU
    plt.plot(np.rad2deg(t_recons.omega),np.asarray(t_recons.peak_amps),'ob',label="max ADC @ DU") 
    plt.errorbar(np.rad2deg(t_recons.omega),np.asarray(t_recons.peak_amps),0.075*np.asarray(t_recons.peak_amps),fmt='None',marker='o',markerfacecolor='b')
    # ADF amplitude at each DU
    plt.plot(np.rad2deg(t_recons.omega),t_recons.adf_amplitude,'+r',label="ADF fit @ DU")    
    # Mean Cherenkov angle for all eta values
    plt.plot([np.rad2deg(omega_cr_mean), np.rad2deg(omega_cr_mean)],[0,max(np.asarray(t_recons.peak_amps))*1.5],label="mean Cherenkov angle")
    # Mean ADF model
    plt.plot(np.rad2deg(w), adf_f,"--r",label="mean ADF model")
    plt.ylim([0, max(t_recons.adf_amplitude)*1.5]) 
    plt.xlim([0,1.6])
    plt.xlabel("$\omega$ (deg)")
    plt.legend(loc="best")
    plt.suptitle(fsuptit, fontsize = 10)
    plt.title(ftit_adf)
    plt.ylabel("Voltage (ADC)")
    plt.show()


    # ---------------------------------------------------------------
    # Plot 2D footprint on the ground
    # ---------------------------------------------------------------

    distm = 5000  # distance in meters
    xmin, xmax = -distm, distm
    ymin, ymax = -distm, distm

    plt.figure()
    # Triggered DUs: color and size proportional to amplitude
    sc = plt.scatter(-Xants[:,1], Xants[:,0], c=t_recons.adf_amplitude, cmap='viridis', s=np.asarray(t_recons.adf_amplitude), label="Triggered DUs") 
    # All DUs on site 
    plt.scatter(antenna_position['x'], antenna_position['y'], marker ='+', color ='red', label="DUs on site") 
    plt.colorbar(sc, label='Peak amplitude (ADC)')
    # Shower direction arrow
    plt.arrow(-Xcore[1]+K[1]*1e3,Xcore[0]-K[0]*1e3, -K[1]*1e3, K[0]*1e3, head_width=1, head_length=1, fc='black', ec='black')
    # Footprint: core position (black dot) and Cherenkov ellipse (dashed line)
    plt.plot(-Xcore[1],Xcore[0],'ko')
    plt.plot(-x_ell[:,1],x_ell[:,0],'--k',linewidth=2)
    plt.xlabel('Easting [m]')
    plt.ylabel('Northing [m]')
    plt.grid(True)
    plt.suptitle(fsuptit, fontsize = 10)
    plt.title(ftit_adf)
    plt.legend()
    plt.axis('equal')
    plt.ylim([xmin,xmax])
    plt.xlim([ymin,ymax])
    plt.subplots_adjust(left=0.15) 
    plt.show()

    



