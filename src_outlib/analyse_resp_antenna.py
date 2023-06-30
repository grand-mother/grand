'''
Created on Sep 12, 2022

@author: root
'''
import os.path as osp

import numpy as np
from matplotlib import pyplot as plt

from grand import Geodetic, LTP, GRANDCS
from grand.simu.du.process_ant import AntennaProcessing
from grand.simu.shower.gen_shower import ShowerEvent
from grand.io.file_leff import TabulatedAntennaModel
from grand import grand_add_path_data, grand_get_path_root_pkg
import grand.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_file="log.txt", log_stdout=True)

G_path_ant_300 = grand_add_path_data("detector/GP300Antenna_EWarm_leff.npy")
G_path_ant = grand_add_path_data("detector/HorizonAntenna_EWarm_leff_loaded.npy")
G_r_ant = TabulatedAntennaModel.load(G_path_ant_300)


def study_file_resp_antenna():
    mod_300 = TabulatedAntennaModel.load(G_path_ant_300)
    mod_hor = TabulatedAntennaModel.load(G_path_ant)
    # print(antenna_model)
    logger.info(mod_300.table.frequency.shape)
    freq = mod_300.table.frequency.copy() / 1e6
    logger.info(f"freq 300  min:{freq[0]} max:{freq[-1]} delta:{freq[1]-freq[0]}")
    # logger.info(mod_300.table.frequency)
    logger.info(mod_hor.table.frequency.shape)
    freq = mod_hor.table.frequency.copy() / 1e6
    logger.info(f"freq Hori min:{freq[0]} max:{freq[-1]} delta:{freq[1]-freq[0]}")
    data = mod_300.table.phi
    logger.info(f"phi  min:{data[0]} max:{data[-1]} delta:{data[1]-data[0]}")
    data = mod_300.table.theta
    logger.info(f"theta  min:{data[0]} max:{data[-1]} delta:{data[1]-data[0]}")
    logger.info(f"module phi: {mod_300.table.leff_phi.shape}")


def plot_resp_antenna(phi_deg=0, theta_deg=0):
    r_ant = TabulatedAntennaModel.load(G_path_ant_300)
    # idx_freq = np.searchsorted(r_ant.table.frequency, freq_mhz*1e6)
    idx_phi = np.searchsorted(r_ant.table.phi, phi_deg)
    idx_theta = np.searchsorted(r_ant.table.theta, theta_deg)
    logger.info(f"idx: {idx_phi} ,{idx_theta} ")
    # logger.info(f"val: {idx_freq} ,{idx_phi} ,{idx_theta} ")
    plt.figure()
    plt.title(f"module response antenna: phi={phi_deg}deg, theta={theta_deg}deg")
    logger.info(f"{r_ant.table.frequency.shape} {r_ant.table.leff_phi.shape}")
    plt.plot(r_ant.table.frequency / 1e6, r_ant.table.leff_phi[:, idx_phi, idx_theta])
    # logger.info(r_ant.table.leff_phi[:][idx_phi][idx_theta])
    plt.grid()
    

def plot_resp_antenna_fix_theta(theta_deg=0, nb_phi=10):
    a_phi = np.linspace(0, 360, nb_phi, False)
    a_idx_phi = np.searchsorted(G_r_ant.table.phi, a_phi)
    idx_theta = np.searchsorted(G_r_ant.table.theta, theta_deg)
    plt.figure()
    plt.title(f"module response antenna theta={theta_deg}deg")
    for idx_phi in a_idx_phi:
        plt.plot(G_r_ant.table.frequency / 1e6, G_r_ant.table.leff_phi[:, idx_phi, idx_theta], label=f"phi={G_r_ant.table.phi[idx_phi]}")
    plt.legend()
    plt.grid()


def plot_kernel_resp_ant(): 
    # Load the radio shower simulation data
    showerdir = osp.join(grand_get_path_root_pkg(), "tests/simulation/data/zhaires")
    shower = ShowerEvent.load(showerdir)
    # print(shower.fields.keys())
    field = shower.fields[0]
    antpos_wrt_shower = field.electric.pos_xyz
    # RK: if antenna location was saved in LTP frame in zhaires.py, next step would not required.
    antenna_location = LTP(
        x=antpos_wrt_shower.x,
        y=antpos_wrt_shower.y,
        z=antpos_wrt_shower.z,
        frame=shower.frame,
    )
    antenna_frame = LTP(
        location=antenna_location,
        orientation="NWU",
        magnetic=True,
        obstime=shower.frame.obstime,
    )
    antenna = AntennaProcessing(G_r_ant, antenna_frame)
    
    logger.info(f"AntennaProcessing pos in shower frame {antpos_wrt_shower.flatten()}")
    logger.info(f"{vars(antenna_location)} {antenna_location.flatten()} \
                antenna pos LTP in shower frame")
    logger.info("---------------------------------")
    logger.info(f"{vars(antenna_frame)} antenna frame")
    logger.info("---------------------------------")
    
    # Compute the voltage on the antenna
    #
    # The electric field is assumed to be a plane-wave originating from the
    # shower axis at the depth of maximum development. Note that the direction
    # of observation and the electric field components are provided in the
    # shower frame. This is indicated by the `frame` named argument.
    Exyz = field.electric.e_xyz
    logger.info(mlg.chrono_start())
    # Xmax, Efield, and input frame are all in shower frame.
    logger.debug("compute_voltage")
    field.voltage = antenna.compute_voltage(shower.maximum, field.electric, frame=shower.frame)
    plt.figure()
    plt.plot(np.real(antenna.leff_frame_ant[0]), label="x real")
    #plt.plot(np.real(antenna.lx), label="lx real")
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(np.imag(antenna.leff_frame_ant[0]), label="x im")
    #plt.plot(np.imag(antenna.lx), label="lx im")
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(np.real(antenna.leff_frame_ant[1]), label="y")
    #plt.plot(antenna.ly, label="ly")
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(np.real(antenna.leff_frame_ant[2]), label="z")
    #plt.plot(antenna.lz, label="lz")
    plt.legend()
    plt.grid()



if __name__ == '__main__':
    # specific logger definition for script because __mane__ is "__main__" !
    logger = mlg.get_logger_for_script(__file__)
    logger.info(mlg.string_begin_script())
    
    plot_kernel_resp_ant()
    
    # study_file_resp_antenna()
    # plot_resp_antenna(90, 80)
    # plot_resp_antenna_fix_theta(80, 4)
    # plot_resp_antenna_fix_theta(88, 4)
    
    #########
    logger.info(mlg.string_end_script())
    plt.show()
