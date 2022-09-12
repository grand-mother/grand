'''
Created on Sep 12, 2022

@author: root
'''

import numpy as np
from matplotlib import pyplot as plt

from grand.simu import Antenna, ShowerEvent, TabulatedAntennaModel
from grand import grand_add_path_data, grand_get_path_root_pkg
import grand.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_file="log.txt", log_stdout=True)

logger.info(mlg.string_begin_script())

G_path_ant_300 = grand_add_path_data("detector/GP300Antenna_EWarm_leff.npy")
G_path_ant = grand_add_path_data("detector/HorizonAntenna_EWarm_leff_loaded.npy")


def study_file_resp_antenna():
    mod_300 = TabulatedAntennaModel.load(G_path_ant_300)
    mod_hor = TabulatedAntennaModel.load(G_path_ant)
    # print(antenna_model)
    logger.info(mod_300.table.frequency.shape)
    freq = mod_300.table.frequency.copy() / 1e6
    logger.info(f"freq 300  min:{freq[0]} max:{freq[-1]} delta:{freq[1]-freq[0]}")
    #logger.info(mod_300.table.frequency)
    logger.info(mod_hor.table.frequency.shape)
    freq = mod_hor.table.frequency.copy() / 1e6
    logger.info(f"freq Hori min:{freq[0]} max:{freq[-1]} delta:{freq[1]-freq[0]}")
    data = mod_300.table.phi
    logger.info(f"phi  min:{data[0]} max:{data[-1]} delta:{data[1]-data[0]}")
    data = mod_300.table.theta
    logger.info(f"theta  min:{data[0]} max:{data[-1]} delta:{data[1]-data[0]}")
    logger.info(f"module phi: {mod_300.table.leff_phi.shape}")

def plot_resp_antenna(freq_mhz=100, phi_deg=0, theta_deg=0):
    r_ant = TabulatedAntennaModel.load(G_path_ant_300)
    idx_freq = np.searchsorted(r_ant.table.frequency, freq_mhz*1e6)
    idx_phi = np.searchsorted(r_ant.table.phi, phi_deg)
    idx_theta = np.searchsorted(r_ant.table.theta, theta_deg)
    logger.info(f"idx: {idx_freq} ,{idx_phi} ,{idx_theta} ")
    #logger.info(f"val: {idx_freq} ,{idx_phi} ,{idx_theta} ")
    plt.figure()
    plt.title(f"module response antenna: f={freq_mhz}, phi={phi_deg}, theta={theta_deg}")
    plt.plot(r_ant.table.leff_phi[idx_freq][idx_phi][idx_theta])
    logger.info(r_ant.table.leff_phi[idx_freq][idx_phi][idx_theta])
    

if __name__ == '__main__':
    # specific logger definition for script because __mane__ is "__main__" !
    logger = mlg.get_logger_for_script(__file__)
    
    study_file_resp_antenna()
    plot_resp_antenna(100,90, 80)
    
    #########
    logger.info(mlg.string_end_script())
    plt.show()
