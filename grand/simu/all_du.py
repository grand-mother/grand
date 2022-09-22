"""

"""

import math
import random
from logging import getLogger

import numpy as np
from scipy import interpolate
from numpy.ma import log10, abs
import matplotlib.pyplot as plt

from grand import grand_add_path_data, grand_get_path_root_pkg
from grand import ECEF, Geodetic, LTP, GRANDCS
from grand.num.signal import fftget, ifftget, halfcplx_fullcplx
from grand.simu.elec_du import LNA_get, filter_get
from grand.simu.galaxy import galaxy_radio_signal
from grand.simu import Antenna, ShowerEvent, TabulatedAntennaModel
import grand.geo.coordinates as coord
from grand.simu.antenna import ElectricField
from grand.io.root.file.simu_efield_gerb import FileSimuEfield


logger = getLogger(__name__)


class MasterSimuDetectorWithRootIo(object):
    def __init__(self, f_name_root):
        self.d_root = FileSimuEfield(f_name_root)
        self.simu_du = SimuDetectorUnitEffect()

    def _load_data_to_process_event(self, idx):
        logger.info("Compute du simulation for traces of event idx= {idx}")
        self.d_root.load_event_idx(idx)
        du_efield = self.d_root.traces
        du_pos = self.d_root.du_pos
        du_id = self.d_root.tt_efield.du_id
        du_t = self.d_root.get_time_trace_ns()
        self.simu_du.set_data_du(du_t, du_efield, du_pos, du_id)
        shower = ShowerEvent()
        shower.load_root(self.d_root.tt_shower)
        self.simu_du.set_data_gerb(shower)

    def compute_du_in_event(self, idx_du, idx_evt):
        self._load_data_to_process_event(idx_evt)
        return self.simu_du.process_du(idx_du)

    def compute_event(self, idx):
        self._load_data_to_process_event(idx)
        return self.simu_du.process_all_du()

    def compute_all_events(self):
        nb_events = self.d_root.get_nb_events()
        for idx in range(nb_events):
            self.compute_event(idx)


class SimuDetectorUnitEffect(object):
    """
    IO file data free, but not for data model

    Adaption of RF simulation chain for grandlib from
      * https://github.com/JuliusErv1ng/XDU-RF-chain-simulation/blob/main/XDU%20RF%20chain%20code.py
    """

    def __init__(self):
        """
        Constructor
        """
        self.t_samp = 0.5  # Manually enter the same time interval as the .trace file
        self.show_flag = False
        self.noise_flag = False
        self._set_antenna_leff_model()

    # INTERNAL

    def _get_ant_leff(self, idx_du):
        antenna_location = LTP(
            x=self.du_pos[idx_du, 0],
            y=self.du_pos[idx_du, 1],
            z=self.du_pos[idx_du, 2],
            frame=self.o_shower.frame,
        )
        logger.info(antenna_location)
        antenna_frame = LTP(location=antenna_location, orientation="NWU", magnetic=True)
        self.ant_leff = [1, 2, 3]
        self.ant_leff[0] = Antenna(model_leff=self.leff_sn, frame=antenna_frame)
        self.ant_leff[1] = Antenna(model_leff=self.leff_ew, frame=antenna_frame)
        ant_z = Antenna(model_leff=self.leff_z, frame=antenna_frame)
        self.ant_leff[2] = ant_z
        tt_a = ant_z.model_leff.table.leff_theta
        logger.info(f"ant_z : {np.min(tt_a):.3e} {np.max(tt_a):.3e}")

    def _init_du_effect(self):
        lst = 18
        # ===========================start calculating===================
        [t_cut, ex_cut, ey_cut, ez_cut, fs, f0, f, f1, N] = self._time_data_get()

        # ======Galaxy noise power spectrum density, power, etc.=====================
        [self.galactic_v_complex_double, self.galactic_v_time] = galaxy_radio_signal(
            lst, N, f0, f1, self.o_trace.nb_det, self.show_flag
        )
        # =======================  cable  filter VGA balun=============================================
        [self.cable_coefficient, self.filter_coefficient] = filter_get(N, f0, 1, self.show_flag)

    def _time_data_get(self):
        pass

    def _set_antenna_leff_model(self):
        path_ant = grand_add_path_data("detector/GP300Antenna_EWarm_leff.npy")
        self.leff_ew = TabulatedAntennaModel.load(path_ant)
        path_ant = grand_add_path_data("detector/GP300Antenna_SNarm_leff.npy")
        self.leff_sn = TabulatedAntennaModel.load(path_ant)
        path_ant = grand_add_path_data("detector/GP300Antenna_Zarm_leff.npy")
        self.leff_z = TabulatedAntennaModel.load(path_ant)

    # USER INTERFACE

    def set_data_du(self, du_time_efield, du_efield, du_pos, du_id):
        self.du_time_efield = du_time_efield
        self.du_efield = du_efield
        self.du_pos = du_pos
        self.du_id = du_id

    def set_data_gerb(self, shower):
        assert isinstance(shower, ShowerEvent)
        self.o_shower = shower

    def process_all_du(self):
        pass

    def process_du(self, idx_du):
        """
        Process trace of du idx_du
        """
        logger.info(f"==============>  Processing du with id: {self.du_id[idx_du]}")
        self._get_ant_leff(idx_du)
        logger.info(self.ant_leff[0].model_leff)
        d_efield = coord.CartesianRepresentation(
            x=self.du_efield[idx_du, 0], y=self.du_efield[idx_du, 1], z=self.du_efield[idx_du, 2]
        )
        o_efield = ElectricField(self.du_time_efield[idx_du] * 1e-9, d_efield)
        v_oc_sn = self.ant_leff[0].compute_voltage(
            self.o_shower.maximum, o_efield, self.o_shower.frame
        )
        v_oc_ew = self.ant_leff[1].compute_voltage(
            self.o_shower.maximum, o_efield, self.o_shower.frame
        )
        v_oc_z = self.ant_leff[2].compute_voltage(
            self.o_shower.maximum, o_efield, self.o_shower.frame
        )
        return v_oc_sn, v_oc_ew, v_oc_z
