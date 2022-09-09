'''

'''
from logging import getLogger
import math
import random

from numpy.ma import log10, abs
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from grand import grand_add_path_data, grand_get_path_root_pkg
from grand import ECEF, Geodetic, LTP, GRANDCS
from grand.num.signal import fftget, ifftget, halfcplx_fullcplx
from grand.simu.elec_du import LNA_get, filter_get
from grand.simu.galaxy import galaxy_radio_signal
from grand.simu import Antenna, ShowerEvent, TabulatedAntennaModel
from grand.store.trace import TraceArrayDC1

logger = getLogger(__name__)

path_ant = grand_add_path_data("detector/GP300Antenna_EWarm_leff.npy")
G_antenna_model_ew = TabulatedAntennaModel.load(path_ant)    
path_ant = grand_add_path_data("detector/GP300Antenna_SNarm_leff.npy")
G_antenna_model_sn = TabulatedAntennaModel.load(path_ant)    
path_ant = grand_add_path_data("detector/GP300Antenna_Zarm_leff.npy")
G_antenna_model_z = TabulatedAntennaModel.load(path_ant)    


class SimuDetectorUnitEffect(object):
    '''
    Adaption of RF simulation chain for grandlib from
      * https://github.com/JuliusErv1ng/XDU-RF-chain-simulation/blob/main/XDU%20RF%20chain%20code.py
    '''
 
    def __init__(self):
        '''
        Constructor
        '''
        self.t_samp = 0.5  # Manually enter the same time interval as the .trace file
        self.show_flag = False
        self.noise_flag = False
        self.o_traces = TraceArrayDC1()
        self.o_shower_event = ShowerEvent()

# INTERNAL
    
    def _get_antenna(self, idx_ant):
        pos_ant = self.o_shower_event.fields[idx_ant].electric.r
        antenna_location = LTP(
            x=pos_ant.x,
            y=pos_ant.y,
            z=pos_ant.z,
            frame=self.o_shower_event.frame,
        )
        antenna_frame = LTP(
            location=antenna_location,
            orientation="NWU",
            magnetic=True
        )
        ant_3d = [1, 2, 3]
        # EW
        ant_3d[0] = Antenna(model=G_antenna_model_ew, frame=antenna_frame)
        # SN
        ant_3d[1] = Antenna(model=G_antenna_model_sn, frame=antenna_frame)
        # Z
        ant_3d[2] = Antenna(model=G_antenna_model_z, frame=antenna_frame)
        return ant_3d

    def _compute_antenna_response(self, idx_ant, e_theta, e_phi, N, f0, unit):
        # This Python file uses the following encoding: utf-8
    
        # from complex_expansion import expan
    
        # = == == == == This program is used as a subroutine to complete the calculation and expansion of the 30-250MHz complex equivalent length == == == == =
        #  ----------------------input- ---------------------------------- %
        # filename address, S1P file (delete the previous string of s1p file in advance, put the test results of the three ports in the antennaVSWR folder, and name them 1 2 3 in turn)
        # % show_flag :flag of showing picture
        # N is the extended length
        # e_theta, e_phi is the direction of incidence
        # If unit is 0, the test data is in the form of real and imaginary parts, and 1 is in the form of db and phase.
        # f0 is the frequency resolution,
        # % ----------------------output - ---------------------------------- %
        # f frequency sequence, the default unit is MHz
        # Lce_complex_expansion is the equivalent length of a specific incident direction
        # s11_complex is the antenna test data
    # for i in range(3):  # Polarization i = 1, 2, 3 respectively represent xyz polarization
    #         for p in range(3):
    #             # Xyz polarization of a single port
    #             Lce_complex_short[:, i, p] = e_radiation[:, p, i] / fenmu[:, p]
    #             [f, Lce_complex_expansion[:, i, p]] = expan(N, f0, f1, f2, Lce_complex_short[:, i, p])    
            # for p in range(Lcelie):
            #     Voc_shower_complex[:, p] = Lce_complex[:, 0, p] * E_shower_fft[:, 0]  
        Lce_complex_expansion = np.zeros((N, 3, 3), dtype=complex)
        ants = self._get_antenna(idx_ant)
        for idx_ant in range(3):
            ants[idx_ant].effective_length(self.o_shower_event.maximum, self.o_shower_event.fields[idx_ant].electric, self.o_shower_event.frame)
            for idx_axis in range(3):
                Lce_complex_expansion[:, idx_axis, idx_ant] = halfcplx_fullcplx(ants[idx_ant].dft_effv_len[idx_axis], (N % 2) == 0)
        
        # Interpolation: 30-250mhz interval 1mhz
        f_start = 30
        f_end = 250
        f_df = 1
        # +1 : f_end included
        f_size = (f_end - f_start) // f_df + 1
        s11_complex = np.zeros((f_size, 3), dtype=complex)
        for p in range(3):
            str_p = str(p + 1)
            # filename = ".//antennaVSWR//" + str_p + ".s1p"
            filename = grand_add_path_data(f"detector/antennaVSWR/{str_p}.s1p")
            freq = np.loadtxt(filename, usecols=0) / 1e6  # HZ to MHz
            if unit == 0:
                re = np.loadtxt(filename, usecols=1)
                im = np.loadtxt(filename, usecols=2)
                db = 20 * log10(abs(re + 1j * im))
            elif unit == 1:
                db = np.loadtxt(filename, usecols=1)
                deg = np.loadtxt(filename, usecols=2)
                mag = 10 ** (db / 20)
                re = mag * np.cos(deg / 180 * math.pi)
                im = mag * np.sin(deg / 180 * math.pi)
            if p == 0:
                dB = np.zeros((3, len(freq)))
            dB[p] = db
            #
            freqnew = np.arange(f_start, f_end + f_df, f_df)
            f_re = interpolate.interp1d(freq, re, kind="cubic")
            renew = f_re(freqnew)
            f_im = interpolate.interp1d(freq, im, kind="cubic")
            imnew = f_im(freqnew)
            s11_complex[:, p] = renew + 1j * imnew
        #
        return Lce_complex_expansion, s11_complex
    
    def _init_du_effect(self):
        lst = 18
        #===========================start calculating===================
        [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = self._time_data_get()

        # ======Galaxy noise power spectrum density, power, etc.=====================
        [self.galactic_v_complex_double, self.galactic_v_time] = galaxy_radio_signal(
            lst, N, f0, f1, self.o_trace.nb_det, self.show_flag
        )
        # =======================  cable  filter VGA balun=============================================
        [self.cable_coefficient, self.filter_coefficient] = filter_get(N, f0, 1, self.show_flag)        

    def _time_data_get(self):
        pass

# USER INTERFACE

    def set_file_efield(self, f_root):
        # TODO: ????
        self.o_traces = TraceArrayDC1(f_root)
        self._init_du_effect()

    def process_du_effect_all_events(self):
        pass
    
    def process_du_effect_traces(self, show_flag=False):
        '''
        input : Efield 3D for a set of trace
        output: Voltage 3D for a set of trace
        
        :param show_flag:
        '''
        nb_ant = self.o_trace.nb_det
        for idx_du in range(nb_ant):
            logger.info(f'Processing ============> {self.o_trace.a_du[idx_du]}')
            # =======Equivalent length================
            [Lce_complex, antennas11_complex_short] = self.compute_antenna_response(idx_du, e_theta, e_phi, N, f0, 1, show_flag)
            Lcehang = Lce_complex.shape[0]
            Lcelie = Lce_complex.shape[2]
            logger.debug(Lce_complex.shape)
            # =================== LNA=====================================================
            [rou1_complex, rou2_complex, rou3_complex] = LNA_get(
                antennas11_complex_short, N, f0, 1, show_flag
            )            
    
            # Output file path and name
            #  ===========================start calculating===========================================================
            [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = self._time_data_get()  
            Edata = ex_cut
            Edata = np.column_stack((Edata, ey_cut))
            Edata = np.column_stack((Edata, ez_cut))
            [E_shower_fft, E_shower_fft_m_single, E_shower_fft_p_single] = fftget(
                Edata, N, f1, False
            )  # Frequency domain signal
            # ======Open circuit voltage of air shower=================
            Voc_shower_complex = np.zeros((Lcehang, Lcelie), dtype=complex)
            # Frequency domain signal
            for p in range(Lcelie):
                Voc_shower_complex[:, p] = (
                    Lce_complex[:, 0, p] * E_shower_fft[:, 0]
                    +Lce_complex[:, 1, p] * E_shower_fft[:, 1]
                    +Lce_complex[:, 2, p] * E_shower_fft[:, 2]
                )
            # time domain signal
            [Voc_shower_t, Voc_shower_m_single, Voc_shower_p_single] = ifftget(
                Voc_shower_complex, N, f1, 2
            )
            # ===========Voltage with added noise=======================================
            Voc_noise_t = np.zeros((Lcehang, Lcelie))
            Voc_noise_complex = np.zeros((Lcehang, Lcelie), dtype=complex)
            for p in range(Lcelie):
                if self.noise_flag == 0:
                    Voc_noise_t[:, p] = Voc_shower_t[:, p]
                    Voc_noise_complex[:, p] = Voc_shower_complex[:, p]
                elif self.noise_flag == 1:
                    Voc_noise_t[:, p] = (
                        Voc_shower_t[:, p] + self.galactic_v_time[random.randint(a=0, b=175),:, p]
                    )
                    Voc_noise_complex[:, p] = (
                        Voc_shower_complex[:, p]
                        +self.galactic_v_complex_double[random.randint(a=0, b=175),:, p]
                    )
            [Voc_noise_t_ifft, Voc_noise_m_single, Voc_noise_p_single] = ifftget(
                Voc_noise_complex, N, f1, 2
            )
            # ==================Voltage after LNA=====================================================
            V_LNA_complex = np.zeros((N, Lcelie), dtype=complex)
            for p in range(Lcelie):
                V_LNA_complex[:, p] = rou1_complex[:, p] \
                                    * rou2_complex[:, p] \
                                    * rou3_complex[:, p] \
                                    * Voc_noise_complex[:, p]
                
            [V_LNA_t, V_LNA_m_single, V_LNA_p_single] = ifftget(V_LNA_complex, N, f1, 2)
            # ======================Voltage after  cable=============================================
            V_cable_complex = np.zeros((N, Lcelie), dtype=complex)
            for p in range(Lcelie):
                V_cable_complex[:, p] = V_LNA_complex[:, p] * self.cable_coefficient[:, p] + 0
            [V_cable_t, V_cable_m_single, V_cable_p_single] = ifftget(V_cable_complex, N, f1, 2)
            # ======================Voltage after filter=============================================
            V_filter_complex = np.zeros((N, Lcelie), dtype=complex)
            for p in range(Lcelie):
                V_filter_complex[:, p] = (
                    V_LNA_complex[:, p] * self.cable_coefficient[:, p] * self.filter_coefficient[:, p] + 0
                )
            [V_filter_t, V_filter_m_single, V_filter_p_single] = ifftget(V_filter_complex, N, f1, 2)
            # ====================Voltage after ADC======================================
            Length_AD = 14  # Significant bit of the value, plus a sign bit in addition to this
            Range_AD = 1.8 * 1e6  # Vpp,unit:uv
            delta = Range_AD / 2 / (2 ** (Length_AD - 1))
            V_ADC_t = np.sign(V_filter_t) * np.floor(abs(V_filter_t) / delta) * delta
            # result
            V_output3 = np.zeros((N, Lcelie + 1))
            V_output3[:, 0] = t_cut
            V_output3[:, 1:] = V_filter_t[:,:]
            # TODO: fleg code ici
            
    def write(self, f_name):
        pass
