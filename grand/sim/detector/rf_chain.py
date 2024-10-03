#!/usr/bin/env python

"""
RF Chain (version 1) of detector units at the GP13 site in Dunhuang and in Europe.

This code includes: 
 * Antenna Impedance
 * Input Impedance  (computed after computing total_ABCD_matrix)
 * LNA
 * Balun after LNA (inside Nut)
 * cable + connector
 * VGA + Filter
 * Balun after VGA&filter + 200 ohm load + AD Chip

:Authors: PengFei Zhang, Xu Xin and Xidian University group, GRANDLIB adaptation Colley JM, Ramesh, and Sandra.

Reference document: 
  * "RF Chain simulation for GP300" by Xidian University, :Authors:  Pengfei Zhang and Pengxiong Ma, Chao Zhang, Rongjuan Wand, Xin Xu
  
Convention Port/direction:
 * 0 for X / North, South
 * 1 for Y / East/West
 * 2 for Z / Up
 
# Note: plotting has been moved to grand/scripts/plot_noise.py. Run: ./plot_noise.py -h for help.

Overview of calculations:
    Voc = E * Leff

    # S-parameters are measured using virtual network analyzer (VNA). A-parameters are computed from S-parameters.
    [A]    = [LNA]*[Balun A]*[Cable]*[VGA+Filter]     # total RF chain A-parameters without balun2
    Z_load = 50 * (1 + S11)/(1 - S11)                 # S11 for balun after VGA+Filter measured using VNA.
    Z_in   = (A11*Z_load + A12) / (A21*Z_load + A22)
    [A]    = [A] * [A balun2]                        # total RF chain A-parameters
    Z_ant  = antenna impedance computed from simulation

    # current and voltage at input of Balun
    I_in_BA = Voc / (Z_ant + Z_in)
    V_in_BA = I_in_BA * Z_in

    # Final voltage output at AD Chip in frequency domain.
    V_out = A11*V_in_BA + A12*I_in_BA
    I_out = A21*V_in_BA + A22*I_in_BA
"""

import os.path
import scipy.fft as sf
import numpy as np
import matplotlib.pyplot as plt

from grand import grand_add_path_data

from logging import getLogger
logger = getLogger(__name__)

def interpol_at_new_x(a_x, a_y, new_x):
    """
    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    :return: F(new_x) (float, (M)): interpolation of F at new_x
    # RK: scipy interpolate gave 0 values for S21 due to fill_values=(0,0)
    #.    which resulted in 'nan' values in A-parameters. Also final transfer
    #     function (TF) outside of the range of 10-300 MHz was weird. TF for Z-port produces a sharp peak around 10 MHz.
    #     So np.interp is used instead.
    """
    from scipy import interpolate
    assert a_x.shape[0] > 0
    #func_interpol = interpolate.interp1d(
    #    a_x, a_y, "cubic", bounds_error=False, fill_value=(1.0, 1.0)
    #)
    #return func_interpol(new_x)
    return np.interp(new_x, a_x, a_y)

def db2reim(dB, phase):
    """Convert quantity given in deciBel to a complex number.
    :param dB: input quantity in deciBel
    :param phase: phase of the input quantity in radians.
    """
    mag = 10 ** (dB / 20)

    re = mag * np.cos(phase)
    im = mag * np.sin(phase)
    
    return re, im

def s2abcd(s11, s21, s12, s22):
    """this is a normalized A-matrix represented by [a] in the document."""
    return np.asarray([
        [((1+s11)*(1-s22) + s12*s21)/(2*s21), ((1+s11)*(1+s22) - s12*s21)/(2*s21)],
        [((1-s11)*(1-s22) - s12*s21)/(2*s21), ((1-s11)*(1+s22) + s12*s21)/(2*s21)]
        ])

def matmul(A, B):
    """
    This function deals with 2x2 matrix multiplication.
    Input matrix shape in our case is (2,2,nports,nb_freqs)
    AxB = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
           [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    """
    assert A.shape[0]==2
    assert A.shape[1]==2
    assert A.shape[1]==B.shape[0]

    return np.asarray([
        [A[0,0]*B[0,0] + A[0,1]*B[1,0], A[0,0]*B[0,1] + A[0,1]*B[1,1]],
        [A[1,0]*B[0,0] + A[1,1]*B[1,0], A[1,0]*B[0,1] + A[1,1]*B[1,1]]
        ])


class GenericProcessingDU:
    """
    Define common attribut for frequencies for all DU effects processing
    """

    def __init__(self):
        """ """
        self.freqs_mhz = np.zeros(0)
        self.nb_freqs = 0
        self.size_sig = 0

    def _set_name_data_file(self, axis):
        """

        :param axis:
        """
        # fix a file version for processing by heritage
        pass

    ### SETTER

    def set_out_freq_mhz(self, freqs_mhz):
        """Define frequencies

        :param freqs_mhz: [MHz] given by scipy.fft.rfftfreq/1e6
        :type freqs_mhz: float (nb_freqs)
        """
        assert isinstance(freqs_mhz, np.ndarray)
        self.freqs_mhz = freqs_mhz
        self.nb_freqs = freqs_mhz.shape[0]
        self.size_sig = (self.nb_freqs - 1) * 2

class MatchingNetwork(GenericProcessingDU):    

    def __init__(self):
        """

        :param size_sig: size of the trace after
        """
        super().__init__()
        #self.data_lna = []
        self.sparams = []
        for axis in range(3):
            matcnet = np.loadtxt(self._set_name_data_file(axis), comments=['#', '!'])
            self.sparams.append(matcnet)
        self.freqs_in = matcnet[:, 0] / 1e6   # note: freqs_in for x and y ports is the same, but for z port is different.
        self.nb_freqs_in = len(self.freqs_in)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis):
        """

        ! Created Wed May 10 01:24:03 2023
        # hz S ma R 50
        ! 2 Port Network Data from SP1.SP block
        """
        axis_dict = {0:"X", 1:"Y", 2:"Z"}
        filename = os.path.join("detector", "RFchain_v2", "NewMatchingNetwork"f"{axis_dict[axis]}.s2p")

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        
        logger.debug(f"{self.sparams[0].shape}")
        self.set_out_freq_mhz(freqs_mhz)
        assert self.nb_freqs > 0

        # nb_freqs in __init__ is 0. nb_freqs changes after self.set_out_freq_mhz(freqs_mhz)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements dB, phase[deg]: S11, S21, S12, S22
        # Fill S-parameters from files obtained by measuring S-parameters using virtual network analyzer.
        for axis in range(3):
            freqs_in = self.sparams[axis][:, 0] / 1e6 # note: freqs_in for x and y ports is the same, but for z port is different.
            # ----- S11
            dbs11 = self.sparams[axis][:, 1]
            phs11 = np.deg2rad(self.sparams[axis][:, 2])
            #res11, ims11 = db2reim(dbs11, phs11)
            res11 = dbs11 * np.cos(phs11)
            ims11 = dbs11 * np.sin(phs11)
            self.dbs11[axis] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S21
            dbs21 = self.sparams[axis][:, 3]
            phs21 = np.deg2rad(self.sparams[axis][:, 4])
            #res21, ims21 = db2reim(dbs21, phs21)
            res21 = dbs21 * np.cos(phs21)
            ims21 = dbs21 * np.sin(phs21)
            self.dbs21[axis] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S12
            dbs12 = self.sparams[axis][:, 5]
            phs12 = np.deg2rad(self.sparams[axis][:, 6])
            #res12, ims12 = db2reim(dbs12, phs12)
            res12 = dbs12 * np.cos(phs12)
            ims12 = dbs12 * np.sin(phs12)
            self.dbs12[axis] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S22
            dbs22 = self.sparams[axis][:, 7]
            phs22 = np.deg2rad(self.sparams[axis][:, 8])
            #res22, ims22 = db2reim(dbs22, phs22)
            res22 = dbs22 * np.cos(phs22)
            ims22 = dbs22 * np.sin(phs22)
            self.dbs22[axis] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        xy_denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XY arms
        xy_denorm_factor = xy_denorm_factor[..., np.newaxis, np.newaxis]
        z_denorm_factor = np.array([[1, 50], [1/50., 1]])    # denormalizing factor for Z arms
        z_denorm_factor = z_denorm_factor[..., np.newaxis]

        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22) # this is a normalized A-matrix represented by [a] in the document.

        ABCD_matrix[..., :2, :] *= xy_denorm_factor # denormalizing factor for XY arms
        ABCD_matrix[..., 2, :] *= z_denorm_factor   # denormalizing factor for Z arm
        self.ABCD_matrix[:] = ABCD_matrix # this is an A-matrix represented by [A] in the document.

class LowNoiseAmplifier(GenericProcessingDU):
    """

    Class goals:
      * Perform the LNA filter on signal for each antenna
      * read only once LNA data files
      * pre_compute interpolation
    """

    def __init__(self):
        """

        :param size_sig: size of the trace after
        """
        super().__init__()
        #self.data_lna = []
        self.sparams = []
        for axis in range(3):
            lna = np.loadtxt(self._set_name_data_file(axis), comments=['#', '!'])
            self.sparams.append(lna)
        self.freqs_in = lna[:, 0] / 1e6   # note: freqs_in for x and y ports is the same, but for z port is different.
        self.nb_freqs_in = len(self.freqs_in)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis):
        """

        Ceyear Technologies,3672C,ZKL00189,2.1.5
        Calibration ON : 2P/1,2
        Sweep Type: lin Frequency Sweep
        S2P File: Measurements: S11, S21, S12, S22:
        Thursday, April 27, 2023
        Hz  S  dB  R 50.000
        """
        axis_dict = {0:"X", 1:"Y", 2:"Z"}
        filename = os.path.join("detector", "RFchain_v2", "NewLNA_"f"{axis_dict[axis]}.s2p")

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """
        compute s-parameters of LNA

        """
        logger.debug(f"{self.sparams[0].shape}")
        self.set_out_freq_mhz(freqs_mhz)
        assert self.nb_freqs > 0

        # nb_freqs in __init__ is 0. nb_freqs changes after self.set_out_freq_mhz(freqs_mhz)
        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements dB, phase[deg]: S11, S21, S12, S22
        # Fill S-parameters from files obtained by measuring S-parameters using virtual network analyzer.
        for axis in range(3):
            freqs_in = self.sparams[axis][:, 0] / 1e6 # note: freqs_in for x and y ports is the same, but for z port is different.
            # ----- S11
            dbs11 = self.sparams[axis][:, 1]
            phs11 = np.deg2rad(self.sparams[axis][:, 2])
            res11, ims11 = db2reim(dbs11, phs11)
            self.dbs11[axis] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s11[axis] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S21
            dbs21 = self.sparams[axis][:, 3]
            phs21 = np.deg2rad(self.sparams[axis][:, 4])
            res21, ims21 = db2reim(dbs21, phs21)
            self.dbs21[axis] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s21[axis] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S12
            dbs12 = self.sparams[axis][:, 5]
            phs12 = np.deg2rad(self.sparams[axis][:, 6])
            res12, ims12 = db2reim(dbs12, phs12)
            self.dbs12[axis] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s12[axis] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
            # ----- S22
            dbs22 = self.sparams[axis][:, 7]
            phs22 = np.deg2rad(self.sparams[axis][:, 8])
            res22, ims22 = db2reim(dbs22, phs22)
            self.dbs22[axis] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
            self.s22[axis] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        xy_denorm_factor = np.array([[1, 100], [1/100., 1]]) # denormalizing factor for XY arms
        xy_denorm_factor = xy_denorm_factor[..., np.newaxis, np.newaxis]
        z_denorm_factor = np.array([[1, 50], [1/50., 1]])    # denormalizing factor for Z arms
        z_denorm_factor = z_denorm_factor[..., np.newaxis]

        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22) # this is a normalized A-matrix represented by [a] in the document.

        ABCD_matrix[..., :2, :] *= xy_denorm_factor # denormalizing factor for XY arms
        ABCD_matrix[..., 2, :] *= z_denorm_factor   # denormalizing factor for Z arm
        self.ABCD_matrix[:] = ABCD_matrix # this is an A-matrix represented by [A] in the document.

class BalunAfterLNA(GenericProcessingDU):
    """
    Class goals:
      * deals with Balun after LNA (inside Nut). 
      * Note that balun is placed after LNA in version 1. The same type of balun is placed before matching-network and LNA in version 2.
      * Balun is used in X and Y ports only. No Balun in Z port.
      * Balun without matching network is used in version 1.
    """

    def __init__(self):
        """ """
        super().__init__()
        #self.data_cable = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis=0):
        """

        Created: May 4, 2023
        hz S ma R 50
        2 Port Network Data from SP1.SP block
        freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22         
        """
        #filename = os.path.join("detector", "RFchain_v2", "balun_after_LNA.s2p")
        #filename = os.path.join("detector", "RFchain_v2", "balun46in.s2p")
        filename = os.path.join("detector", "RFchain_v2", "balun_in_nut.s2p")
        
        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """Compute ABCD_matrix for frequency freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64) # shape = (2x2 matrix, 3 ports, nb_freqs)

        # freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22
        # ----- S11
        mags11 = self.sparams[:, 1]
        angs11 = np.deg2rad(self.sparams[:, 2])
        res11 = mags11 * np.cos(angs11)
        ims11 = mags11 * np.sin(angs11)
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        mags21 = self.sparams[:, 3]
        angs21 = np.deg2rad(self.sparams[:, 4])
        res21 = mags21 * np.cos(angs21)
        ims21 = mags21 * np.sin(angs21)
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        mags12 = self.sparams[:, 5]
        angs12 = np.deg2rad(self.sparams[:, 6])
        res12 = mags12 * np.cos(angs12)
        ims12 = mags12 * np.sin(angs12)
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        mags22 = self.sparams[:, 7]
        angs22 = np.deg2rad(self.sparams[:, 8])
        res22 = mags22 * np.cos(angs22)
        ims22 = mags22 * np.sin(angs22)
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis] # match with the shape of ABCD_matrix to broadcast.
        # for X and Y ports only. No Balun in Z port. shape of ABCD_matrix is (2, 2, 3, nb_freqs).     
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor
        # force components of ABCD_matrix for Z port to be identity matrix because there is no Balun in Z port.
        self.ABCD_matrix[:,:,2,:] = np.identity(2)[...,np.newaxis]  # add np.newaxis to broadcast to all frequencies.

class Cable(GenericProcessingDU):
    """

    Class goals:
      * pre_compute interpolation
    """

    def __init__(self):
        """ """
        super().__init__()
        #self.data_cable = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz

        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis=0):
        """

        :param axis:
        """
        filename = os.path.join("detector", "RFchain_v2", "cable+Connector.s2p")

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """Compute ABCD_matrix for frequency freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements: S11, S21, S12, S22
        # ----- S11
        dbs11 = self.sparams[:, 1]
        phs11 = np.deg2rad(self.sparams[:, 2])
        res11, ims11 = db2reim(dbs11, phs11)
        self.dbs11[:] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        dbs21 = self.sparams[:, 3]
        phs21 = np.deg2rad(self.sparams[:, 4])
        res21, ims21 = db2reim(dbs21, phs21)
        self.dbs21[:] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        dbs12 = self.sparams[:, 5]
        phs12 = np.deg2rad(self.sparams[:, 6])
        res12, ims12 = db2reim(dbs12, phs12)
        self.dbs12[:] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        dbs22 = self.sparams[:][:, 7]
        phs22 = np.deg2rad(self.sparams[:, 8])
        res22, ims22 = db2reim(dbs22, phs22)
        self.dbs22[:] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis] # match with the shape of ABCD_matrix.
        # for all three ports. shape of ABCD_matrix is (2, 2, ant ports, nb_freqs)   .     
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor

class VGAFilter(GenericProcessingDU):
    """

    Class goals:
      * pre_compute interpolation
    """

    def __init__(self, gain=0):
        """ 
        :param gain: gain setup for VGA in dB.
        """
        super().__init__()

        self.gain = gain
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz

        # shape = (nports, nfreqs). self.nb_freqs here is 0.
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self, axis=0):
        """

        :param axis:
        """
        assert self.gain in [-5, 0, 5, 20]
        logger.info(f"vga gain: {self.gain} dB")
        filename = os.path.join("detector", "RFchain_v2", "filter+"f"vga{self.gain}db+filter.s2p")

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """Compute ABCD_matrix for frequency freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.dbs11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.dbs22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

        # S2P File: Measurements: S11, S21, S12, S22
        # ----- S11
        dbs11 = self.sparams[:, 1]
        phs11 = np.deg2rad(self.sparams[:, 2])
        res11, ims11 = db2reim(dbs11, phs11)
        self.dbs11[:] = interpol_at_new_x(freqs_in, dbs11, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        dbs21 = self.sparams[:, 3]
        phs21 = np.deg2rad(self.sparams[:, 4])
        res21, ims21 = db2reim(dbs21, phs21)
        self.dbs21[:] = interpol_at_new_x(freqs_in, dbs21, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        dbs12 = self.sparams[:, 5]
        phs12 = np.deg2rad(self.sparams[:, 6])
        res12, ims12 = db2reim(dbs12, phs12)
        self.dbs12[:] = interpol_at_new_x(freqs_in, dbs12, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        dbs22 = self.sparams[:][:, 7]
        phs22 = np.deg2rad(self.sparams[:, 8])
        res22, ims22 = db2reim(dbs22, phs22)
        self.dbs22[:] = interpol_at_new_x(freqs_in, dbs22, self.freqs_mhz)     # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)        
        ABCD_matrix = s2abcd(self.s11, self.s21, self.s12, self.s22)
        ABCD_matrix *= denorm_factor # denormalizing factor for XYZ arms
        self.ABCD_matrix[:] = ABCD_matrix

class BalunBeforeADC(GenericProcessingDU):
    """Class goals:
      * Pass signal through Balun before Analog to Digitial Converter (ADC) for each antenna
      * Balun is used in x, y, and z ports
      * Same data is used for all three ports
      * read data files only once
      * pre_compute interpolation
      * this Balun is referred to as Balun1 Balun2
    """

    def __init__(self):
        """ 
        :param sparams: S-parameters data for x, y, and z ports. Same data is used for x, y, and z ports.
        :param freqs_in: frequencies corresponding to the S-parameters data for x, y, and z ports.
        :param s11, s21, s12, s22: S-parameters for x, y, and z ports. shape (3, nb_freqs).
        :param ABCD_matrix: not normalized ABCD matrix corresponding to S-parameters. shape (2, 2, nb_ports, nb_freqs)
        """
        super().__init__()
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64)

    def _set_name_data_file(self):
        """Created Mon May 15, 2023 11:21:18 2023
        hz S ma R 50
        2 Port Network Data from SP1.SP block
        freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22  
        """
        filename = os.path.join("detector", "RFchain_v2", "balun_before_ad.s2p")

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """compute s-parameters and ABCD matrix of Balun before AD chip for freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        # shape = (antenna_port, nb_freqs)
        self.s11 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s21 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s12 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.s22 = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.ABCD_matrix = np.zeros((2, 2, 3, self.nb_freqs), dtype=np.complex64) # shape = (2x2 matrix, 3 ports, nb_freqs)

        # freq  magS11  angS11  magS21  angS21  magS12  angS12  magS22  angS22
        # ----- S11
        mags11 = self.sparams[:, 1]
        angs11 = np.deg2rad(self.sparams[:, 2])
        res11 = mags11 * np.cos(angs11)
        ims11 = mags11 * np.sin(angs11)
        self.s11[:] = interpol_at_new_x(freqs_in, res11, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s11[:] += 1j * interpol_at_new_x(freqs_in, ims11, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S21
        mags21 = self.sparams[:, 3]
        angs21 = np.deg2rad(self.sparams[:, 4])
        res21 = mags21 * np.cos(angs21)
        ims21 = mags21 * np.sin(angs21)
        self.s21[:] = interpol_at_new_x(freqs_in, res21, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s21[:] += 1j * interpol_at_new_x(freqs_in, ims21, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S12
        mags12 = self.sparams[:, 5]
        angs12 = np.deg2rad(self.sparams[:, 6])
        res12 = mags12 * np.cos(angs12)
        ims12 = mags12 * np.sin(angs12)
        self.s12[:] = interpol_at_new_x(freqs_in, res12, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s12[:] += 1j * interpol_at_new_x(freqs_in, ims12, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.
        # ----- S22
        mags22 = self.sparams[:, 7]
        angs22 = np.deg2rad(self.sparams[:, 8])
        res22 = mags22 * np.cos(angs22)
        ims22 = mags22 * np.sin(angs22)
        self.s22[:] = interpol_at_new_x(freqs_in, res22, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s22[:] += 1j * interpol_at_new_x(freqs_in, ims22, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # for all three ports. shape should be (2, 2, ant ports, nb_freqs)
        denorm_factor = np.array([[1, 50], [1/50., 1]]) # denormalizing factor for XYZ arms
        denorm_factor = denorm_factor[..., np.newaxis, np.newaxis]
        self.ABCD_matrix[:] = s2abcd(self.s11, self.s21, self.s12, self.s22) * denorm_factor # this is an A-matrix represented by [A] in the document.

class Zload(GenericProcessingDU):
    """Class goals:
      * computes input impedance of load due to balun + 200ohm ADC.
    """

    def __init__(self):
        """Reflection coefficient (self.s) is measured using VNA.
        Same value is used for all ports.
        :param sparams: S-parameters data to compute Zload for x, y, and z ports. Same Zload is used for x, y, and z ports.
        :param freqs_in: frequencies corresponding to the S-parameters data for x, y, and z ports.
        :param s: reflection coefficient for x, y, and z ports. shape (nb_freqs,).
        :param Z_load: total impedance of the load that includes balun, 200 ohm resistor and AD chip.
        """
        super().__init__()
        self.sparams = np.loadtxt(self._set_name_data_file(), comments=['#', '!'])
        self.freqs_in = self.sparams[:, 0] / 1e6 # Hz to MHz
        self.s = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )
        self.Z_load = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )

    def _set_name_data_file(self, axis=0):
        """Ceyear Technologies,3672C, ZKL00189, 2.1.5
        Calibration ON : 2P/1,2
        Sweep Type: lin Frequency Sweep
        S1P File: Measurements: S22:
        Thursday, April 20, 2023
        Hz  S  RI  R 50
        """
        #filename = os.path.join("detector", "RFchain_v2", "zload_balun_200ohm.s1p")
        filename = os.path.join("detector", "RFchain_v2", "S_balun_AD.s1p")

        return grand_add_path_data(filename)

    def compute_for_freqs(self, freqs_mhz):
        """compute S-paramters and Zload for freqs_mhz

        :param freqs_mhz (float, (N)): [MHz] given by scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        freqs_in = self.freqs_in
        assert self.nb_freqs > 0

        self.s = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )
        self.Z_load = np.zeros(self.nb_freqs, dtype=np.complex64) # shape = (nb_freqs, )

        # S1P File: Measurements: S22
        #res = self.sparams[:, 1]
        #ims = self.sparams[:, 2]
        dbs = self.sparams[:, 1]
        phs = np.deg2rad(self.sparams[:, 2])
        res, ims = db2reim(dbs, phs)
        self.s[:] = interpol_at_new_x(freqs_in, res, self.freqs_mhz)       # interpolate s-parameters for self.freqs_mhz frequencies.
        self.s[:] += 1j * interpol_at_new_x(freqs_in, ims, self.freqs_mhz) # interpolate s-parameters for self.freqs_mhz frequencies.

        # Calculation of Zload (Zload = balun+200ohm + ADchip)
        self.Z_load[:] = 50 * (1 + self.s) / (1 - self.s)

class RFChain(GenericProcessingDU):
    """
    Facade for all elements in RF chain
    """

    def __init__(self, vga_gain=20):
        super().__init__()
        self.lna = LowNoiseAmplifier()
        self.balun1 = BalunAfterLNA()
        self.cable = Cable()
        self.vgaf = VGAFilter(gain=vga_gain)
        self.balun2 = BalunBeforeADC()
        self.zload = Zload()
        # Note: self.nb_freqs at this point is 0.
        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

    def compute_for_freqs(self, freqs_mhz):
        """Compute transfer function for frequency freqs_mhz

        :param freqs_mhz (float, (N)): return of scipy.fft.rfftfreq/1e6
        """
        self.set_out_freq_mhz(freqs_mhz)
        self.matcnet.compute_for_freqs(freqs_mhz)
        self.lna.compute_for_freqs(freqs_mhz)
        self.balun1.compute_for_freqs(freqs_mhz)
        self.cable.compute_for_freqs(freqs_mhz)
        self.vgaf.compute_for_freqs(freqs_mhz)
        self.balun2.compute_for_freqs(freqs_mhz)
        self.zload.compute_for_freqs(freqs_mhz)
        #self.balun_after_vga.compute_for_freqs(freqs_mhz)

        assert self.lna.nb_freqs > 0
        assert self.lna.ABCD_matrix.shape[-1] > 0
        assert self.lna.nb_freqs==self.balun1.nb_freqs

        assert self.matcnet.nb_freqs > 0
        assert self.matcnet.ABCD_matrix.shape[-1] > 0
        assert self.matcnet.nb_freqs==self.balun1.nb_freqs

        self.Z_ant = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.Z_in = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.V_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.I_out_RFchain = np.zeros((3, self.nb_freqs), dtype=np.complex64)
        self.total_ABCD_matrix = np.zeros(self.lna.ABCD_matrix.shape, dtype=np.complex64)

        # Note that components of ABCD_matrix for Z port of balun1 is set to 1 as no Balun is used. shape = (2,2,nports,nfreqs)
        # Note that this is a matrix multiplication
        # Associative property of matrix multiplication is used, ie. (AB)C = A(BC)
        # Make sure to multiply in this order: lna.ABCD_matrix * balun1.ABCD_matrix * cable.ABCD_matrix * vgaf.ABCD_matrix
        MM1 = matmul(self.balun1.ABCD_matrix, self.matcnet.ABCD_matrix)
        MM2 = matmul(MM1, self.lna.ABCD_matrix)
        MM3 = matmul(self.cable.ABCD_matrix, self.vgaf.ABCD_matrix)
        self.total_ABCD_matrix[:] = matmul(MM2, MM3)

        # Calculation of Z_in (this is the total impedence of the RF chain excluding antenna arm. see page 50 of the document.)
        self.Z_load = self.zload.Z_load[np.newaxis, :] # shape (nfreq) --> (1,nfreq) to broadcast with components of ABCD_matrix with shape (2,2,ports,nfreq).
        self.Z_in[:] = (self.total_ABCD_matrix[0,0] * self.Z_load + self.total_ABCD_matrix[0,1])/(self.total_ABCD_matrix[1,0] * self.Z_load + self.total_ABCD_matrix[1,1])

        # Once Z_in is calculated, calculate the final total_ABCD_matrix including Balun2.
        self.total_ABCD_matrix[:] = matmul(self.total_ABCD_matrix, self.balun2.ABCD_matrix) 

        # Antenna Impedance.
        filename = os.path.join("detector", "RFchain_v2", "Z_ant_3.2m.csv")
        filename = grand_add_path_data(filename)
        Zant_dat = np.loadtxt(filename, delimiter=",", comments=['#', '!'], skiprows=1)
        freqs_in = Zant_dat[:,0]  # MHz
        self.Z_ant[0] = interpol_at_new_x(freqs_in, Zant_dat[:,1], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[0] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,2], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] = interpol_at_new_x(freqs_in, Zant_dat[:,3], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[1] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,4], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] = interpol_at_new_x(freqs_in, Zant_dat[:,5], self.freqs_mhz)       # interpolate impedance for self.lna.freqs_mhz frequencies.
        self.Z_ant[2] += 1j * interpol_at_new_x(freqs_in, Zant_dat[:,6], self.freqs_mhz) # interpolate impedance for self.lna.freqs_mhz frequencies.

    def vout_f(self, voc_f):
        """ Compute final voltage after propagating signal through RF chain.
        Input: Voc_f (in frequency domain)
        Output: Voltage after RF chain in frequency domain.
        Make sure to run self.compute_for_freqs() before calling this method.
        RK Note: name 'vout_f' is a placeholder. Change it with something better. 
        """
        assert voc_f.shape==self.Z_in.shape  # shape = (nports, nfreqs)

        self.I_in_balunA = voc_f / (self.Z_ant + self.Z_in)
        self.V_in_balunA = self.I_in_balunA * self.Z_in

        # loop over three ports. shape of total_ABCD_matrix is (2,2,nports,nfreqs)
        for i in range(3):
            ABCD_matrix_1port = self.total_ABCD_matrix[:,:,i,:]
            ABCD_matrix_1port = np.moveaxis(ABCD_matrix_1port, -1, 0) # (2,2,nfreqs) --> (nfreqs,2,2), to compute inverse of ABCD_matrix using np.linalg.inv.
            ABCD_matrix_1port_inv = np.linalg.inv(ABCD_matrix_1port)
            V_out_RFchain = ABCD_matrix_1port_inv[:,0,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,0,1]*self.I_in_balunA[i]
            I_out_RFchain = ABCD_matrix_1port_inv[:,1,0]*self.V_in_balunA[i] + ABCD_matrix_1port_inv[:,1,1]*self.I_in_balunA[i]

            self.V_out_RFchain[i] = V_out_RFchain
            self.I_out_RFchain[i] = I_out_RFchain

        return self.V_out_RFchain

    def get_tf(self):
        """Return transfer function for all elements in RF chain
        total transfer function is the output voltage for input Voc of 1. It says by what factor the Voc will be multiplied by the RF chain.
        @return total TF (complex, (3,N)):
        """
        self._total_tf = self.vout_f(np.ones((3, self.nb_freqs)))

        return self._total_tf
