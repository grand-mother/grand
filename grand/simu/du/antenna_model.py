from logging import getLogger

from grand import grand_add_path_data
#from grand.io.file_leff import TabulatedAntennaModel

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
    leff_theta: Union[Number, np.ndarray]
    phase_theta: Union[Number, np.ndarray]
    leff_phi: Union[Number, np.ndarray]
    phase_phi: Union[Number, np.ndarray]
    leff_phi_cart: Any = None
    leff_theta_cart: Any = None

    def __post_init__(self):
        logger.info(f"size phase {self.phase_theta.shape}")
        self.phase_theta_rad = np.deg2rad(self.phase_theta)
        self.phase_phi_rad = np.deg2rad(self.phase_phi)

def tabulated_antenna_model(filename, tag=""):
    # RK: TODO: update this function after antenna model is finalized. Remove tag.
    #     1. GP300 model (~1GB/arm) 2. LP float32 model (~520MB/arm) 3. JM Light GP300 model (~120MB/arm)
    split_file = os.path.splitext(filename)

    if split_file[-1]==".npy" and tag=="": 
        f, R, X, theta, phi, lefft, leffp, phaset, phasep = numpy.load(path, mmap_mode="r")
        n_f = f.shape[0]
        n_theta = len(numpy.unique(theta[0, :]))
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

    if split_file[-1]==".npy" and tag=='LP':
        """As get_tabulated, but for files with reshaped arrays"""
        # mmap_mode reads only the necessary parts of the file from HD when needed
        # Readout uncached, cached by OS/HDD, rest of the voltage calc:
        # Original (float64): ~48 s, 0.7, ~2.1
        # Original, mmap: ~26.5 s, 0.25, ~2
        # Below, no dtype conversion in the script:
        # Reshaped, float64: ~48 s, ~0.5, ~2.4
        # Reshaped, float64, mmap: ~5.7 s, 0.025, ~2
        # Reshaped, float32: ~22.9 s, ~0.33, ~2.1
        # Reshaped, float32, mmap: ~2.9 s, ~0.017, ~2.0
        # Compressed, reshaped, float32: ~6.5 s
        # Speed reduction (cached by the OS) nommap->mmap 0.81 -> 0.27
        # Using arrays stored as float32 and avoiding dtype conversion ->0.02 (using float64 and avoiding dtype conversion does not change much? Not sure if I tested properly)
        # compressed is unaddected by mmap, always 3.87 s
        f, R, X, theta, phi, lefft, leffp, phaset, phasep = np.load(filename, mmap_mode="r")
        # f, R, X, theta, phi, lefft, leffp, phaset, phasep = np.load(filename)
        n_f = f.shape[1]
        # n_theta = len(np.unique(theta[0, :]))
        n_theta = len(np.unique(theta[:, 0]))
        # print(n_theta)
        # exit()
        # n_theta = 181
        n_phi = R.shape[0] // n_theta
        shape = (n_phi, n_theta, n_f)

        # dtype = "f4"
        dtype = np.float32
        # f = f[0, :].astype(dtype) * 1.0e6  # MHz --> Hz
        f = f[0, :] * 1.0e6  # MHz --> Hz
        theta = theta[:n_theta, 0]#.astype(dtype)  # deg
        phi = phi[::n_theta, 0]#.astype(dtype)  # deg
        # theta = np.arange(181).astype(dtype)
        # phi = np.arange(361).astype(dtype)
        # print(theta, phi)
        # exit()

        # Those are not needed, so don't read them from the mmaped file
        # R = R.reshape(shape).astype(dtype)  # Ohm
        # X = X.reshape(shape).astype(dtype)  # Ohm
        lefft = lefft.reshape(shape)  
        leffp = leffp.reshape(shape)  
        # RK TODO: Make sure going from rad to deg does not affect calculations somewhere else.
        phaset = phaset.reshape(shape)  # deg
        phasep = phasep.reshape(shape)  # deg

        # RK: added by me to use the same interpolation
        lefft  = np.moveaxis(lefft, 2, 0)    # (phi, theta, freq) --> (freq, phi, theta)
        leffp  = np.moveaxis(leffp, 2, 0)    # (phi, theta, freq) --> (freq, phi, theta)
        phaset = np.moveaxis(phaset, 2, 0)   # (phi, theta, freq) --> (freq, phi, theta)
        phasep = np.moveaxis(phasep, 2, 0)   # (phi, theta, freq) --> (freq, phi, theta)

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


class AntennaModel:
    def __init__(self, du_type="GP300"):

        if du_type=="GP300":
            logger.info(f"Loading GP300 antenna model ...")
            
            '''
            path_ant = grand_add_path_data("detector/GP300Antenna_EWarm_leff.npy")
            #path_ant = grand_add_path_data("detector/Light_GP300Antenna_EWarm_leff.npz")
            self.leff_ew = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/GP300Antenna_SNarm_leff.npy")
            #path_ant = grand_add_path_data("detector/Light_GP300Antenna_SNarm_leff.npz")
            self.leff_sn = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/GP300Antenna_Zarm_leff.npy")
            #path_ant = grand_add_path_data("detector/Light_GP300Antenna_Zarm_leff.npz")
            self.leff_z = tabulated_antenna(path_ant)
            '''

            self.leff_ew = tabulated_antenna_model("/home/data_challenge1_pm_lwp/PM_functions/PM_files/GP300Antenna_EWarm_leff_reshaped_float32.npy", tag='LP')
            self.leff_sn = tabulated_antenna_model("/home/data_challenge1_pm_lwp/PM_functions/PM_files/GP300Antenna_SNarm_leff_reshaped_float32.npy", tag='LP')
            self.leff_z = tabulated_antenna_model("/home/data_challenge1_pm_lwp/PM_functions/PM_files/GP300Antenna_Zarm_leff_reshaped_float32.npy", tag='LP')

        if du_type=='Horizon':
            path_ant = grand_add_path_data("detector/HorizonAntenna_EWarm_leff_loaded.npy")
            self.leff_ew = tabulated_antenna_model(path_ant)
            path_ant = grand_add_path_data("detector/HorizonAntenna_SNarm_leff_loaded.npy")
            self.leff_sn = tabulated_antenna(path_ant)
            path_ant = grand_add_path_data("detector/HorizonAntenna_Zarm_leff_loaded.npy")
            self.leff_z = tabulated_antenna_model(path_ant)

        self.d_leff = {"sn": self.leff_sn, "ew": self.leff_ew, "z": self.leff_z}

    def plot_effective_length(self):
        pass

