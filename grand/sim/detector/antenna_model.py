# Modified by SN in April 2024, in order to include the models for the effective length of the antennas produced by the codes NEC and matlab from HOU group. 
# The coresponding files are Light_GP300Antenna_nec_*_leff.npz and Light_GP300Antenna_mat_*_leff.npz for NEC and matlab respectively (* stands for the 3 arms X-NS, Y-EW and Z-vertical)
# The default effective length model were produced using HFSS simulation code by Xidian group. 
# The coresponding files of the default model are Light_GP300Antenna_EWarm_leff.npz, Light_GP300Antenna_SNarm_leff.npz and Light_GP300Antenna_Zarm_leff.npz.
# For details on how these simulations were produced using the different codes and comparisons between different model (HFSS, NEC matlab) please refer to the documents and presentations 
# posted in forge software forum https://forge.in2p3.fr/projects/software-forum/dmsf (for NEC and matlab) and https://forge.in2p3.fr/documents/1353, https://forge.in2p3.fr/documents/1354 for the HFSS simulations.


from logging import getLogger

from grand import grand_add_path_data

from dataclasses import dataclass, fields
from typing import Union, Any
from numbers import Number
import numpy as np
import os

logger = getLogger(__name__)

@dataclass
class DataTable:
    frequency: Union[Number, np.ndarray]
    theta: Union[Number, np.ndarray]
    phi: Union[Number, np.ndarray]
    leff_theta: Union[Number, np.ndarray] = None
    phase_theta: Union[Number, np.ndarray] = None
    leff_phi: Union[Number, np.ndarray] = None
    phase_phi: Union[Number, np.ndarray] = None
    leff_phi_reim: Union[Number, np.ndarray] = None
    leff_theta_reim: Union[Number, np.ndarray] = None

    #def __post_init__(self):
    #    logger.info(f"size phase {self.phase_theta.shape}")
    #    self.phase_theta_rad = np.deg2rad(self.phase_theta)
    #    self.phase_phi_rad = np.deg2rad(self.phase_phi)

def tabulated_antenna_model(filename):
    # RK: TODO: update this function after antenna model is finalized. Remove tag.
    #     1. GP300 model (~1GB/arm) 2. LP float32 model (~520MB/arm) 3. JM Light GP300 model (~120MB/arm)
    split_file = os.path.splitext(filename)
    if split_file[-1]==".npy": # for Horizon Antenna
        f, R, X, theta, phi, lefft, leffp, phaset, phasep = np.load(filename, mmap_mode="r")
        n_f = f.shape[0]
        n_theta = len(np.unique(theta[0, :]))
        n_phi = int(R.shape[1] / n_theta)
        shape = (n_f, n_phi, n_theta)
        logger.debug(f"shape freq, phi, theta: {f.shape} {phi.shape} {theta.shape}")
        logger.debug(f"shape R, X: {R.shape} {X.shape} {R.dtype} {X.dtype}")
        logger.debug(f"shape module tetha : {lefft.shape}")
        logger.debug(f"shape arg tetha : {phaset.shape}")
        logger.debug(f"type leff  : {lefft.dtype}")
        logger.debug(f"type f  : {f.dtype}")
        logger.debug(f"type phi  : {phi.dtype}")
        logger.debug(f"min max resistance  : {R.min()} {R.max()}")
        logger.debug(f"min max reactance  : {X.min()} {X.max()}")
        dtype = "f4"
        f = f[:, 0].astype(dtype) * 1.0e6  # MHz --> Hz
        theta = theta[0, :n_theta].astype(dtype)  # deg
        phi = phi[0, ::n_theta].astype(dtype)  # deg
        lefft = lefft.reshape(shape).astype(dtype)  # m
        leffp = leffp.reshape(shape).astype(dtype)  # m
        # RK TODO: Make sure going from rad to deg does not affect calculations somewhere else.
        phaset = phaset.reshape(shape).astype(dtype)  # deg
        phasep = phasep.reshape(shape).astype(dtype)  # deg
        t = DataTable(
            frequency=f,
            theta=theta,
            phi=phi,
            leff_theta=lefft,
            phase_theta=phaset,
            leff_phi=leffp,
            phase_phi=phasep,
        )
        return t


    if split_file[-1]==".npz": # for JM's file.
        logger.info(f"Using {filename}")
        f_leff = np.load(filename)
        if f_leff["version"][0] == "1.0":
            f = f_leff["freq_mhz"] * 1e6   # MHz --> Hz
            theta = np.arange(91).astype(float)
            phi = np.arange(361).astype(float)
            lefft = f_leff["leff_theta"]   # Real + j Imag. shape (phi, theta, freq) (361, 91, 221)
            leffp = f_leff["leff_phi"]     # Real + j Imag. shape (phi, theta, freq)
            lefft = np.moveaxis(lefft, -1, 0) # shape (phi, theta, freq) --> (freq, phi, theta)
            leffp = np.moveaxis(leffp, -1, 0) # shape (phi, theta, freq) --> (freq, phi, theta)

            logger.debug(f"shape freq, phi, theta: {f.shape} {phi.shape} {theta.shape}")
            logger.debug(f"shape module tetha : {lefft.shape}")
            logger.debug(f"type leff  : {lefft.dtype}")
            logger.debug(f"type f  : {f.dtype}")
            logger.debug(f"type phi  : {phi.dtype}")

            t = DataTable(
                frequency=f,
                theta=theta,
                phi=phi,
                leff_theta_reim=lefft,
                leff_phi_reim=leffp,
            )
            return t
        else:
            raise Exception(f"Provide a proper antenna model. Current input file is {filename}")

class AntennaModel:
    def __init__(self, du_type="GP300"):

        if du_type=="GP300":
            logger.info(f"Loading GP300 antenna model ...")
            
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_EWarm_leff.npz")
            self.leff_ew = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_SNarm_leff.npz")
            self.leff_sn = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_Zarm_leff.npz")
            self.leff_z = tabulated_antenna_model(path_ant)
            
        elif du_type=="GP300_nec":
            logger.info(f"Loading GP300 antenna model ...")
            
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_nec_Yarm_leff.npz")
            self.leff_ew = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_nec_Xarm_leff.npz")
            self.leff_sn = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_nec_Zarm_leff.npz")
            self.leff_z = tabulated_antenna_model(path_ant)
            
        elif du_type=="GP300_mat":
            logger.info(f"Loading GP300 antenna model ...")
            
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_mat_Yarm_leff.npz")
            self.leff_ew = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_mat_Xarm_leff.npz")
            self.leff_sn = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/Light_GP300Antenna_mat_Zarm_leff.npz")
            self.leff_z = tabulated_antenna_model(path_ant)
            
        elif du_type=='Horizon':
            path_ant = grand_add_path_data("detector/HorizonAntenna_EWarm_leff_loaded.npy")
            self.leff_ew = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/HorizonAntenna_SNarm_leff_loaded.npy")
            self.leff_sn = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/HorizonAntenna_Zarm_leff_loaded.npy")
            self.leff_z = tabulated_antenna_model(path_ant)

        self.d_leff = {"sn": self.leff_sn, "ew": self.leff_ew, "z": self.leff_z}

    def plot_effective_length(self):
        pass

