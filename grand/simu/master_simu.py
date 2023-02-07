"""
Master module for the detector unit simulation GRAND
"""
import os
import os.path
from logging import getLogger
import time
import numpy as np
import scipy.fft as sf

import grand.geo.coordinates as coord
from grand.simu.du.process_ant import AntennaProcessing
from grand.simu.shower.gen_shower import ShowerEvent
from grand.io.root_files import FileSimuEfield
from grand.simu.du.model_ant_du import AntennaModelGp300
from grand.basis.type_trace import ElectricField
from grand.basis.traces_event import Handling3dTracesOfEvent
from grand.io.root_trees import VoltageEventTree
import grand.simu.du.rf_chain as grfc
import grand.num.signal as gsig
from grand.simu.galaxy import galaxy_radio_signal

logger = getLogger(__name__)


class MasterSimuDetectorWithRootIo:
    """
    Adapter to call SimuDetectorUnitEffect with IO GRANDROOT

    Goals:

      * Call simulator of detector units with ROOT data
      * Call on more than one event
      * Save output in ROOT format
    """

    ### INTERNAL

    def __init__(self, f_name_root):
        self.f_name_root = f_name_root
        self.d_root = FileSimuEfield(f_name_root)
        self.simu_du = SimuDetectorUnitEffect()
        self.o_traces = Handling3dTracesOfEvent()
        self.tt_volt = None
        self.file_out = ""

    def _load_data_to_process_event(self, idx):
        """
        Extract from ROOT file, data to process all Efield for event idx

        :param idx: index of detector in array data
        :type idx: int
        """
        logger.info(f"Compute du simulation for traces of event idx= {idx}")
        self.d_root.load_event_idx(idx)
        self.o_traces = self.d_root.get_obj_handling3dtraces()
        # for debug
        # self.o_traces.reduce_nb_du(5)
        assert isinstance(self.o_traces, Handling3dTracesOfEvent)
        self.simu_du.set_data_efield(self.o_traces)
        shower = ShowerEvent()
        shower.load_root(self.d_root.tt_shower)
        self.simu_du.set_data_shower(shower)

    ### OPERATION

    def compute_event_du_idx(self, idx_evt, idx_du):
        """
        Compute/simulate only one DU for index idx_du of event with index idx_evt

        :param idx_evt: index event
        :type idx_evt: int
        :param idx_du: index DU
        :type idx_du: int
        """
        self._load_data_to_process_event(idx_evt)
        return self.simu_du.compute_du_idx(idx_du)

    def compute_event_idx(self, idx):
        """
        Compute/simulate all DU in event with index idx

        :param idx: index event
        :type idx: int
        """
        self._load_data_to_process_event(idx)
        return self.simu_du.compute_du_all()

    def compute_event_all(self):
        """
        Compute/simulate all DU for all event in data file input
        """
        nb_events = self.d_root.get_nb_events()
        for idx in range(nb_events):
            self.compute_event_idx(idx)
            self.save_voltage()

    def set_output_file(self, file_out):
        """

        :param file_out: output path/file
        :type file_out: str
        """
        self.file_out = file_out

    def save_voltage(self, file_out="", append_file=True):
        """

        :param file_out: output path/file
        :type file_out: str
        :param append_file: use input file to add output
        :type append_file: bool
        """
        # delete file can be take time => start with this action
        if file_out == "":
            if self.file_out != "":
                file_out = self.file_out
            else:
                logger.error("No output file defined !")
                raise AssertionError
        if not append_file and os.path.exists(file_out):
            logger.info(f"save on new file option => remove file {file_out}")
            os.remove(file_out)
            time.sleep(1)
        logger.info(f"save result in {file_out}")
        self.tt_volt = VoltageEventTree(file_out)
        # now fill Voltage object
        freq_mhz = int(self.d_root.get_sampling_freq_mhz())
        self.tt_volt.du_count = self.o_traces.get_nb_du()
        logger.debug(f"We will add {self.tt_volt.du_count} DU")
        self.tt_volt.run_number = self.d_root.tt_efield.run_number
        self.tt_volt.event_number = self.d_root.tt_efield.event_number
        logger.debug(f"{type(self.tt_volt.run_number)} {type(self.tt_volt.event_number)}")
        logger.debug(f"{self.tt_volt.run_number} {self.tt_volt.event_number}")
        self.tt_volt.first_du = self.o_traces.du_id[0]
        self.tt_volt.time_seconds = self.d_root.tt_efield.time_seconds
        self.tt_volt.time_nanoseconds = self.d_root.tt_efield.time_nanoseconds
        self.o_traces.traces = self.simu_du.voc
        # self.tt_volt.event_size = 1999
        for idx in range(self.simu_du.o_efield.get_nb_du()):
            # trace = np.arange(self.tt_volt.event_size, dtype=np.float64)
            # self.o_traces.plot_trace_idx(idx)
            logger.debug(f"add DU {self.o_traces.du_id[idx]} in ROOT file")
            # logger.info(f"shape: {self.simu_du.voc[idx, 0].shape}")
            self.tt_volt.du_nanoseconds.append(self.d_root.tt_efield.du_nanoseconds[idx])
            self.tt_volt.du_seconds.append(self.d_root.tt_efield.du_seconds[idx])
            self.tt_volt.adc_sampling_frequency.append(freq_mhz)
            self.tt_volt.du_id.append(int(self.o_traces.du_id[idx]))
            # logger.info(f"du_id {type(self.o_traces.du_id[idx])}")
            self.tt_volt.trace_x.append(self.simu_du.v_out[idx, 0].astype(np.float64).tolist())
            self.tt_volt.trace_y.append(self.simu_du.v_out[idx, 1].astype(np.float64).tolist())
            self.tt_volt.trace_z.append(self.simu_du.v_out[idx, 2].astype(np.float64).tolist())
            # position
            self.tt_volt.pos_x.append(self.d_root.tt_efield.pos_x[idx])
            self.tt_volt.pos_y.append(self.d_root.tt_efield.pos_y[idx])
            self.tt_volt.pos_z.append(self.d_root.tt_efield.pos_z[idx])
        self.tt_volt.fill()
        self.tt_volt.write()


class SimuDetectorUnitEffect:
    """
    Simulator of antenna response, detector effect and galactic noise.

    Processing to do:

      * Convolution in time domain : (Efield*(l_eff + noise))*IR_rf_chain

        * '*' is convolution operator
        * l_eff : effective length of antenna, ie impulsional response of antenna
        * noise: galactic noise at local sideral time
        * IR_rf_chain :  impulsional response of electronic chain

    Processing performed:

      * Calculation in Fourier space: (F_Efield.(L_eff + F_noise)).TF_rf_chain

        * in Fourier domain convolution becomes multiplication
        * '.' multiplication term by term
        * L_eff : effective length of antenna in Fourier space, ie transfer function
        * F_noise: FFT of galactic noise at local sideral time
        * TF_rf_chain : transfer function of electronic chain

      * We used a common frequency definition for all calculation stored in freqs_mhz attribute
        and computed with function get_fastest_size_fft()

    .. note::
       * no IO, only memory processing
       * manage only one event
    """

    def __init__(self):
        """
        Constructor
        """
        # object contents Efield and network information
        self.o_efield = Handling3dTracesOfEvent()
        self.du_pos = None
        self.rf_chain = grfc.RfChainGP300()
        self.ant_model = AntennaModelGp300()
        # object of class ShowerEvent
        self.o_shower = None
        # object AntennaProcessing for SN arm
        self.ant_leff_sn = None
        # object AntennaProcessing for EW arm
        self.ant_leff_ew = None
        # object AntennaProcessing for Z arm
        self.ant_leff_z = None
        self.du_pos = None
        self.params = {"flag_add_noise": False, "lst": 18.0}
        # FFT info
        self.sig_size = 0
        self.fact_padding = 1.2
        #  fft_size ~= sig_size*fact_padding
        self.fft_size = 0
        # float (fft_size,) array of frequencies in MHz in Fourier domain
        self.freqs_mhz = 0
        # outputs
        self.fft_noise_gal_3d = None
        self.v_out = None
        self.voc = None

    ### INTERNAL

    def _get_ant_leff(self, idx_du):
        """
        Define for each antenna in DU idx_du an object AntennaProcessing according its position

        :param idx_du: index of DU
        :type idx_du: int
        """
        antenna_location = coord.LTP(
            x=self.du_pos[idx_du, 0],
            y=self.du_pos[idx_du, 1],
            z=self.du_pos[idx_du, 2],
            frame=self.o_shower.frame,
        )
        logger.debug(antenna_location)
        antenna_frame = coord.LTP(location=antenna_location, orientation="NWU", magnetic=True)
        self.ant_leff_sn = AntennaProcessing(model_leff=self.ant_model.leff_sn, pos=antenna_frame)
        self.ant_leff_ew = AntennaProcessing(model_leff=self.ant_model.leff_ew, pos=antenna_frame)
        self.ant_leff_z = AntennaProcessing(model_leff=self.ant_model.leff_z, pos=antenna_frame)
        # Set array frequency
        self.ant_leff_sn.set_out_freq_mhz(self.freqs_mhz)
        self.ant_leff_ew.set_out_freq_mhz(self.freqs_mhz)
        self.ant_leff_z.set_out_freq_mhz(self.freqs_mhz)

    ### SETTER

    def set_flag_add_noise(self, flag=True):
        """
        :param flag: True to add noise to antenna response
        :type flag: bool
        """
        self.params["flag_add_noise"] = flag

    def set_local_sideral_time(self, f_hour):
        """
        Define local sideral time

        :param file_out: between 0h and 24h
        :type file_out: float
        """
        logger.debug(f"{f_hour}")
        self.params["lst"] = f_hour

    def set_data_efield(self, tr_evt):
        """

        :param tr_evt: object contents Efield and network information
        :type tr_evt: Handling3dTracesOfEvent
        """
        assert isinstance(tr_evt, Handling3dTracesOfEvent)
        self.o_efield = tr_evt
        self.du_pos = tr_evt.network.du_pos
        self.voc = np.zeros_like(self.o_efield.traces)
        self.v_out = np.zeros_like(self.o_efield.traces)
        self.sig_size = self.o_efield.get_size_trace()
        # common frequencies for all processing in Fourier domain
        self.fft_size, self.freqs_mhz = gsig.get_fastest_size_fft(
            self.sig_size,
            self.o_efield.f_samp_mhz,
            self.fact_padding,
        )
        # precompute interpolation for all antennas with classmethod
        AntennaProcessing.init_interpolation(self.freqs_mhz, self.ant_model.leff_sn.table.frequency/1e6)
        # compute total transfer function of RF chain
        self.rf_chain.compute_for_freqs(self.freqs_mhz)
        if self.params["flag_add_noise"]:
            # lst: local sideral time, galactic noise max at 18h
            self.fft_noise_gal_3d = galaxy_radio_signal(
                self.params["lst"],
                self.fft_size,
                self.freqs_mhz,
                self.o_efield.get_nb_du(),
            )

    def set_data_shower(self, shower):
        """

        :param shower: object contents shower parameters
        :type shower: ShowerEvent
        """
        assert isinstance(shower, ShowerEvent)
        self.o_shower = shower

    ### GETTER / COMPUTER

    def compute_du_all(self):
        """
        Simulate all DU
        """
        for idx in range(self.o_efield.get_nb_du()):
            self.compute_du_idx(idx)

    def compute_du_idx(self, idx_du):
        """Simulate one DU
        Simulation DU effect computing for DU at idx

        Processing order:

          1. antenna responses
          2. add galactic noise
          3. RF chain effect

        :param idx_du: index of DU in array traces
        :type idx_du: int
        """
        logger.info(f"==============>  Processing DU with id: {self.o_efield.du_id[idx_du]}")
        self._get_ant_leff(idx_du)
        # logger.debug(self.ant_leff_sn.model_leff)
        # define E field at antenna position
        e_trace = coord.CartesianRepresentation(
            x=self.o_efield.traces[idx_du, 0],
            y=self.o_efield.traces[idx_du, 1],
            z=self.o_efield.traces[idx_du, 2],
        )
        efield_idx = ElectricField(self.o_efield.t_samples[idx_du] * 1e-9, e_trace)
        ########################
        # 1) antenna responses
        ########################
        self.voc[idx_du, 0] = self.ant_leff_sn.compute_voltage(
            self.o_shower.maximum, efield_idx, self.o_shower.frame
        ).V
        self.voc[idx_du, 1] = self.ant_leff_ew.compute_voltage(
            self.o_shower.maximum, efield_idx, self.o_shower.frame
        ).V
        self.voc[idx_du, 2] = self.ant_leff_z.compute_voltage(
            self.o_shower.maximum, efield_idx, self.o_shower.frame
        ).V
        fft_voc_3d = np.array(
            [
                self.ant_leff_sn.fft_resp_volt,
                self.ant_leff_ew.fft_resp_volt,
                self.ant_leff_z.fft_resp_volt,
            ]
        )
        ########################
        # 2) Add galactic noise
        ########################
        if self.params["flag_add_noise"]:
            noise_gal = sf.irfft(self.fft_noise_gal_3d[idx_du])[:, : self.sig_size]
            logger.debug(np.std(noise_gal, axis=1))
            self.voc[idx_du] += noise_gal
            fft_voc_3d += self.fft_noise_gal_3d[idx_du]
        ########################
        # 3) RF chain
        ########################
        fft_all_effect_3d = fft_voc_3d * self.rf_chain.get_tf_3d()
        # inverse FFT and remove zero-padding
        # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
        self.v_out[idx_du] = sf.irfft(fft_all_effect_3d)[:, : self.sig_size]
