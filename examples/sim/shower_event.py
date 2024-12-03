#from grand import LTP, CartesianRepresentation
from grand.geo.coordinates import (
    CartesianRepresentation,
    LTP,
)
from grand import grand_add_path_data
#from grand import AntennaModel, AntennaProcessing, ShowerEvent
from grand.sim.detector.antenna_model import  AntennaModel
from grand.sim.detector.process_ant import AntennaProcessing
from grand.sim.shower.gen_shower import ShowerEvent

import grand.dataio.root_trees as groot
from grand.basis.type_trace import ElectricField

import scipy.fft as sf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

params = {
    "legend.fontsize": 14,
    "axes.labelsize": 22,
    "axes.titlesize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.figsize": (10, 8),
    "axes.grid": False,
}
plt.rcParams.update(params)

# Load the radio shower simulation data
shower = ShowerEvent()
e_path = grand_add_path_data("test_efield.root")
tefield = groot.TEfield(e_path)
tshower = groot.TShower(e_path)
trun = groot.TRun(e_path)
event_list = tefield.get_list_of_events()
# Work on the first event
event_number, run_number = event_list[0][0], event_list[0][1]
tefield.get_event(event_number, run_number)
tshower.get_event(event_number, run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
trun.get_run(run_number)
# get shower info
shower.origin_geoid  = trun.origin_geoid  # [lat, lon, height]
shower.load_root(tshower)                 # calculates grand_ref_frame, shower_frame, Xmax in LTP etc

#shower = ShowerEvent.load("../../tests/simulation/data/zhaires")

print("---------------------------------")
print("Shower Frame:")
print(f"{vars(shower.frame)}")
print(f"origin_geoid: {shower.origin_geoid} # lat, lon, alt")
print(f"Core={shower.core.flatten()}")
print(f"obstime: {shower.frame.obstime}")
print(f"Zenith: {shower.zenith}")
print(f"Azimuth: {shower.azimuth}")
print(f"Xmax (shc): {shower.maximum}")
print("---------------------------------")

# Define an antenna model
#
# A tabulated model of the GP300 antenna is used. Note that all three arms of an
# can be accessed as antenna_model.leff_ew (leff_sn, leff_z)
antenna_model  = AntennaModel()

# Plot tabulated phi and zenith angle (wrt antenna frame) dependence of
# effective lengths of antenna in phi and zenith direction (e_phi, e_theta).
phi = antenna_model.leff_ew.phi
theta = antenna_model.leff_ew.theta
frequency = antenna_model.leff_ew.frequency
X, Y = np.meshgrid(phi, theta)

n = 10  # number of plots to print
z = (frequency.max() / 1e6) / n
z = int(z - (z % 10))  # pick multiple of 10
# plt.rcParams['axes.grid'] = False
for i in range(n):
    fig = plt.figure()
    gs = GridSpec(80, 100)
    ax1 = plt.subplot(gs[:40, :])
    ax2 = plt.subplot(gs[40:, :])
    f1 = ax1.imshow(
        np.abs(antenna_model.leff_sn.leff_phi_reim[z * i, :, :]).T,
        cmap="Spectral_r",
        label="Leff_phi",
        extent=[phi.min(), phi.max(), theta.min(), theta.max()],
    )
    f2 = ax2.imshow(
        np.abs(antenna_model.leff_sn.leff_theta_reim[z * i, :, :]).T,
        cmap="Spectral_r",
        label="Leff_theta",
        extent=[phi.min(), phi.max(), theta.min(), theta.max()],
    )
    ax1.set_title("frequency: %.1f MHz" % (frequency[z * i] / 1e6))
    ax1.set_ylabel("theta [deg]")
    ax2.set_ylabel("theta [deg]")
    ax2.set_xlabel("phi [deg]")
    ax1.set_aspect("auto")
    ax2.set_aspect("auto")
    ax1.set_xticklabels([])
    ax1.grid(ls="--", alpha=0.4)
    ax2.grid(ls="--", alpha=0.4)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(f1, cax=cbar_ax, label="Effective Lenght [m]")
    # For legend
    ax1.text(300, 75, "Leff_phi", bbox={"facecolor": "white", "pad": 10}, fontsize=14)
    ax2.text(300, 75, "Leff_theta", bbox={"facecolor": "white", "pad": 10}, fontsize=14)
    
    plt.show()
