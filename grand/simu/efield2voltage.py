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
import grand.io.root_trees as groot
from grand.basis.type_trace import ElectricField
from grand.simu.du.antenna_model import AntennaModel
from grand.simu.du.process_ant import AntennaProcessing
from grand.simu.shower.gen_shower import ShowerEvent
from grand.simu.noise.galaxy import galaxy_radio_signal
import grand.simu.noise.rf_chain as grfc

from functools import lru_cache

logger = getLogger(__name__)

def get_fastest_size_fft(sig_size, f_samp_mhz, padding_factor=1):
    """
    :param sig_size:         length of time traces
    :param f_samp_mhz:       sampling frequency in MHz. ex: 2000 MHz for dt_ns=0.5
    :param padding_factor:     factor to stretch length of time traces with zeros

    :return: size_fft (int,0), array freq (float,1) in MHz for rfft()
    """
    assert padding_factor >= 1
    dt_s      = 1e-6 / f_samp_mhz
    fast_size = sf.next_fast_len(int(padding_factor * sig_size + 0.5))
    freqs_mhz = sf.rfftfreq(fast_size, dt_s) * 1e-6
    return fast_size, freqs_mhz


class Efield2Voltage:
    """
    Class to compute voltage with GRANDROOT IO

    Goals:

      * Call simulator of detector units with ROOT data
      * Call on more than one event
      * Save output in ROOT format
    """

    def __init__(self, f_input, f_output="", seed=None, padding_factor=1.0):
        self.f_input    = f_input
        self.f_output   = f_output
        self.seed       = seed                                  # used to generate same set of random numbers. (gal noise)
        self.padding_factor = padding_factor
        self.events     = groot.TEfield(f_input)                # traces and du_pos are stored here
        self.run        = groot.TRun(f_input)                   # site_long, site_lat info is stored here. Used to define shower frame.
        self.shower     = groot.TShower(f_input)                # shower info (like energy, theta, phi, xmax etc) are stored here.
        self.events_list= self.events.get_list_of_events()      # [[evt0, run0], [evt1, run0], ...[evt0, runN], ...]
        self.rf_chain   = grfc.RfChainGP300()                   # loads RF chain. # RK: TODO: load this only if we want to add RF Chain.
        self.ant_model  = AntennaModel()                        # loads antenna models. time consuming. ant_type='GP300' (default), 'Horizon'
        self.params     = {"add_noise": True, "lst": 18.0, "add_rf_chain":True}
        self.last_run   = -1                                    # Not to load run info everytime event info is loaded.

    def get_event(self, event_idx=0):
        """
        Load data of event with index event_idx. Call this method everytime to load data for new events with different event_idx.
        :param: event_idx
        :    type: int
        """
        self.event_idx: int = event_idx  # index of events in the input file. 0 is for the first event and so on.
        if self.event_idx<len(self.events_list): 
            self.evt_nb = self.events_list[self.event_idx][0] 
            self.run_nb = self.events_list[self.event_idx][1]
        else:
            logger.warning(f"Event index {self.event_idx} is out of range. It must be less than {len(self.events_list)}.")
            return False

        logger.info(f"Running on event-number: {self.evt_nb}, run-number: {self.run_nb}")

        self.events.get_event(self.evt_nb, self.run_nb)           # update traces, du_pos etc for event with event_idx.
        self.shower.get_event(self.evt_nb, self.run_nb)           # update shower info (theta, phi, xmax etc) for event with event_idx.
        if self.last_run != self.run_nb:                          # load only for new run.
            self.run.get_run(self.run_nb)                         # update run info to get site latitude and longitude.
            self.last_run = self.run_nb

        # stack efield traces
        trace_shape   = np.asarray(self.events.trace).shape     # (nb_du, 3, tbins of a trace)
        self.du_id    = np.asarray(self.events.du_id)           # used for printing info and saving in voltage tree. 
        self.nb_du    = trace_shape[0]
        self.sig_size = trace_shape[-1]
        self.traces   = np.asarray(self.events.trace, dtype=np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)
        self.du_pos = np.asarray(self.run.du_xyz) # (nb_du, 3) antenna position wrt local grand coordinate

        # container to collect computed Voc and the final voltage in time domain for one event.
        self.voc   = np.zeros_like(self.traces)
        self.v_out = np.zeros_like(self.traces)

        # shower information like theta, phi, xmax etc for one event.
        shower = ShowerEvent()
        shower.origin_geoid  = self.run.origin_geoid    # [lat, lon, height]
        shower.load_root(self.shower)               # calculates grand_ref_frame, shower_frame, Xmax in LTP etc
        self.evt_shower = shower                    # Note that 'shower' is an instance of 'self.shower' for one event.
        self.dt_ns      = np.asarray(self.run.t_bin_size)   # sampling time in ns, sampling freq = 1e9/dt_ns.
        self.f_samp_mhz = 1e3/self.dt_ns            # MHz
        # comupte time samples in ns for all antennas in event with index event_idx.
        self.time_samples = self.get_time_samples() # t_samples.shape = (nb_du, self.sig_size)
        # common frequencies for all processing in Fourier domain.
        self.fft_size, self.freqs_mhz = get_fastest_size_fft(
            self.sig_size,
            self.f_samp_mhz,
            self.padding_factor,
        )
        logger.info(f"Length of frequency bins with padding factor {self.padding_factor} is {len(self.freqs_mhz)}.")
        # Compute galactic noise.
        if self.params["add_noise"]:
            # lst: local sideral time, galactic noise max at 18h
            self.fft_noise_gal_3d = galaxy_radio_signal(
                self.params["lst"],
                self.fft_size,
                self.freqs_mhz,
                self.nb_du,
                seed=self.seed
            )
        # compute total transfer function of RF chain. Can be computed only once in __init__ if length of time traces does not change between events.
        if self.params["add_rf_chain"]:
            self.rf_chain.compute_for_freqs(self.freqs_mhz)

    def get_leff(self, du_idx):
        """
        Define for each antenna in DU du_idx an object AntennaProcessing according its position

        :param du_idx: index of DU
        :    type du_idx: int
        """

        antenna_location = coord.LTP(
            x=self.du_pos[du_idx, 0], #self.du_pos[du_idx, 0],    # antenna position wrt local grand coordinate
            y=self.du_pos[du_idx, 1], #self.du_pos[du_idx, 1],    # antenna position wrt local grand coordinate
            z=self.du_pos[du_idx, 2], #self.du_pos[du_idx, 2],    # antenna position wrt local grand coordinate
            frame=self.evt_shower.grand_ref_frame
            )
        logger.debug(antenna_location)
        antenna_frame = coord.LTP(
            location=antenna_location, 
            orientation="NWU", 
            magnetic=True
            )
        self.ant_leff_sn = AntennaProcessing(model_leff=self.ant_model.leff_sn, pos=antenna_frame)
        self.ant_leff_ew = AntennaProcessing(model_leff=self.ant_model.leff_ew, pos=antenna_frame)
        self.ant_leff_z  = AntennaProcessing(model_leff=self.ant_model.leff_z , pos=antenna_frame)
        # Set array frequency
        self.ant_leff_sn.set_out_freq_mhz(self.freqs_mhz)
        self.ant_leff_ew.set_out_freq_mhz(self.freqs_mhz)
        self.ant_leff_z.set_out_freq_mhz(self.freqs_mhz)

    def get_time_samples(self):
        """
        Define time sample in ns for the duration of the trace
        t_samples.shape  = (nb_du, self.sig_size)
        t_start_ns.shape = (nb_du,)
        """
        t_start_ns = np.asarray(self.events.du_nanoseconds)[...,np.newaxis]   # shape = (nb_du, 1)
        t_samples = (
            np.outer(
                self.dt_ns * np.ones(self.nb_du), np.arange(0, self.sig_size, dtype=np.float64)
                ) + t_start_ns )
        logger.info(f"shape du_nanoseconds and t_samples =  {t_start_ns.shape}, {t_samples.shape}")

        return t_samples


    # compute voltage in one antenna of one event.
    def compute_du_idx(self, du_idx):
        """Simulate one DU. 
        This method is the base of computing voltage.
        All voltage computation is build on top of this method.
        Simulation DU effect computing for DU at idx

        Processing order:
          1. antenna responses
          2. add galactic noise
          3. RF chain effect

        :param du_idx: index of DU in array traces
        :    type du_idx: int
        """
        logger.info(f"==============>  Processing DU with id: {self.du_id[du_idx]}")
        self.get_leff(du_idx)
        logger.debug(self.ant_leff_sn.model_leff)
        # define E field at antenna position
        e_trace = coord.CartesianRepresentation(
            x=self.traces[du_idx, 0],
            y=self.traces[du_idx, 1],
            z=self.traces[du_idx, 2],
        )
        efield_idx = ElectricField(self.time_samples[du_idx] * 1e-9, e_trace)

        # ----- antenna responses -----
        # compute_voltage() --> return Voltage(t=t, V=volt_t)
        self.voc[du_idx, 0] = self.ant_leff_sn.compute_voltage(
            self.evt_shower.maximum, efield_idx, self.evt_shower.frame
        ).V
        self.voc[du_idx, 1] = self.ant_leff_ew.compute_voltage(
            self.evt_shower.maximum, efield_idx, self.evt_shower.frame
        ).V
        self.voc[du_idx, 2] = self.ant_leff_z.compute_voltage(
            self.evt_shower.maximum, efield_idx, self.evt_shower.frame
        ).V

        # only save Voc in frequency domain if you are adding either galactic noise or RF chain.
        if self.params["add_noise"] or self.params["add_rf_chain"]:
            fft_voc_3d = np.array(
                [
                    self.ant_leff_sn.fft_resp_volt,
                    self.ant_leff_ew.fft_resp_volt,
                    self.ant_leff_z.fft_resp_volt,
                ]
            )

        # ----- Add galactic noise -----
        if self.params["add_noise"]:
            noise_gal = sf.irfft(self.fft_noise_gal_3d[du_idx])[:, : self.sig_size]
            logger.debug(np.std(noise_gal, axis=1))
            self.voc[du_idx] += noise_gal
            fft_voc_3d += self.fft_noise_gal_3d[du_idx]

        # ----- Add RF chain -----
        if self.params["add_rf_chain"]:
            fft_voc_3d *= self.rf_chain.get_tf_3d()

        # Final voltage output for antenna with index du_idx
        if self.params["add_noise"] or self.params["add_rf_chain"]:
            # inverse FFT and remove zero-padding
            # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
            self.v_out[du_idx] = sf.irfft(fft_voc_3d)[:, : self.sig_size]
        else:
            self.v_out[du_idx] = self.voc[du_idx]

    # compute voltage in all antennas of one event.
    def compute_du_all(self):
        """
        Simulate all DU
        """
        for idx in range(self.nb_du):
            self.compute_du_idx(idx)

    # compute voltage in any one antennas of any one event.
    def compute_event_du_idx(self, event_idx, du_idx):
        """
        Compute/simulate only one DU for index du_idx of an event with index event_idx

        :param event_idx: index of event from self.events_list
        :    type idx_evt: int
        :param du_idx: index of DU (antenna) on which to compute voltage
        :    type du_idx: int
        """
        self.get_event(event_idx)
        return self.compute_du_idx(du_idx)

    # compute voltage in all antennas of any one event.
    def compute_event_idx(self, event_idx):
        """
        Compute/simulate all DU in event with index event_idx

        :param idx: index event
        :    type idx: int
        """
        self.get_event(event_idx)
        return self.compute_du_all()

    # compute voltage in all antennas from all events.
    def compute_event_all(self):
        """
        Compute/simulate all DU for all event in data file input
        """
        nb_events = len(self.events_list)
        for evt_idx in range(nb_events):
            self.compute_event_idx(evt_idx)
            self.save_voltage()

    def save_voltage(self, append_file=True):
        """
        : output path/file = self.f_output. It must be defined during instantiation of this class.
        :    type file_out: str
        :param append_file: use input file to add output
        :    type append_file: bool
        """
        # delete file can take time => start with this action
        if self.f_output == "":
            split_file = os.path.splitext(self.f_input)
            self.f_output   = split_file[0]+"_voltage.root"
            logger.info(f"No output file was defined. Output file is automatically defined as {f_output}")
        if not append_file and os.path.exists(f_output):
            logger.info(f"save on new file option => remove file {f_output}")
            os.remove(f_output)
            time.sleep(1)

        logger.info(f"save result in {self.f_output}")
        self.tt_volt = groot.TVoltage(self.f_output)

        # Fill voltage object. d_root = events
        self.tt_volt.du_count     = self.nb_du
        logger.debug(f"We will save voltage for {self.tt_volt.du_count} DUs.")

        self.tt_volt.run_number   = self.events.run_number
        self.tt_volt.event_number = self.events.event_number
        logger.debug(f"{type(self.tt_volt.run_number)} {type(self.tt_volt.event_number)}")
        logger.debug(f"{self.tt_volt.run_number} {self.tt_volt.event_number}")

        self.tt_volt.first_du         = self.du_id[0]
        self.tt_volt.time_seconds     = self.events.time_seconds
        self.tt_volt.time_nanoseconds = self.events.time_nanoseconds
        self.tt_volt.du_nanoseconds = self.events.du_nanoseconds
        self.tt_volt.du_seconds = self.events.du_seconds
        self.tt_volt.du_id = self.du_id
        self.tt_volt.trace = self.v_out

        self.tt_volt.fill()
        self.tt_volt.write()



