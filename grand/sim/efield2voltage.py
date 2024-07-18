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
import grand.dataio.root_trees as groot
from grand.basis.type_trace import ElectricField

from .detector.antenna_model import AntennaModel
from .detector.process_ant import AntennaProcessing
#from .detector.rf_chain import RFChain
from .detector.rf_chain2 import RFChain
from .detector.rf_chain2 import RFChainNut
from .shower.gen_shower import ShowerEvent
from .noise.galaxy import galactic_noise

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
    # ToDo: this function (or something higher) should properly handle different time bin for each trace
    freqs_mhz = sf.rfftfreq(fast_size, dt_s[0]) * 1e-6
    return fast_size, freqs_mhz


class Efield2Voltage:
    """
    Class to compute voltage with GRANDROOT IO

    Goals:

      * Call simulator of detector units with ROOT data
      * Call on more than one event
      * Save output in ROOT format
    """

    def __init__(self, f_input, f_output="", seed=None, padding_factor=1.0, du_type='GP300'):
        # du_type='GP300' (using hfss simulations), 'GP300_nec' (using nec simulations), 'GP300_mat' (using matlab simulations), 'Horizon'
        self.du_type = du_type
        self.f_input = f_input
        self.f_output = f_output
        self.seed = seed                                    # used to generate same set of random numbers. (gal noise)
        self.padding_factor = padding_factor
        self.events = groot.TEfield(f_input)                # traces and du_pos are stored here
        self.run = groot.TRun(f_input)                      # site_long, site_lat info is stored here. Used to define shower frame.
        self.shower = groot.TShower(f_input)                # shower info (like energy, theta, phi, xmax etc) are stored here.
        self.events_list = self.events.get_list_of_events() # [[evt0, run0], [evt1, run0], ...[evt0, runN], ...]
        #self.rf_chain = RFChain()                           # loads RF chain. # RK: TODO: load this only if we want to add RF Chain.
        self.rf_chain2 = RFChain()                         
        self.rf_chain2nut = RFChainNut()
        self.ant_model = AntennaModel(du_type)              # loads antenna models. time consuming. du_type='GP300' (default using hfss simulations), 'GP300_nec', 'GP300_mat', 'Horizon'
        self.params = {"add_noise": True, "lst": 18.0, "add_rf_chain":True, "add_rf_chain_nut":False}
        self.previous_run = -1                              # Not to load run info everytime event info is loaded.

    def get_event(self, event_idx=None, event_number=None, run_number=None):
        """
        Load data of event with index event_idx or with event_number and run_number. 
        Call this method everytime to load data for a new event with different event_idx, or with different event_number and run_number.
        :param: event_idx: index of event in events_list. It is a number from range(len(event_list)).
        :    type: int
        :param: event_number: event_number of an event. Combination of event_number and run_number must be unique.
        :    type: int
        :param: run_number: run_number of an event. Combination of event_number and run_number must be unique.
        :    type: int  
        Note: Either event_idx, or both event_number and run_number must be provided.      
        """
        self.event_idx = event_idx  # index of events. 0 is for the 1st event and so on. Just a placeholder if event_number and run_number are provided.
        if (event_number is not None) and (run_number is not None):
            self.event_number = event_number
            self.run_number = run_number
        elif (self.event_idx is not None) and (self.event_idx<len(self.events_list)): 
            self.event_number = self.events_list[self.event_idx][0]
            self.run_number = self.events_list[self.event_idx][1]
        else:
            message = f"Provide positive integer of either event_idx or both event_number and run_number. If event_idx is given, it must\
            be less than {len(self.events_list)}. If event_number and run_number are given, they must be from the list of (event_number, run_number)\
            {self.events_list}. Provided values are: event_idx={event_idx}, event_number={event_number}, run_number={run_number}."
            logger.exception(message)
            raise Exception(message)

        assert isinstance(self.event_number, int)
        assert isinstance(self.run_number, int)
        logger.info(f"Running on event_number: {self.event_number}, run_number: {self.run_number}")

        self.events.get_event(self.event_number, self.run_number)           # update traces, du_pos etc for event with event_idx.
        self.shower.get_event(self.event_number, self.run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
        if self.previous_run != self.run_number:                      # load only for new run.
            self.run.get_run(self.run_number)                         # update run info to get site latitude and longitude.
            self.previous_run = self.run_number

        # stack efield traces
        trace_shape = np.asarray(self.events.trace).shape  # (nb_du, 3, tbins of a trace)
        self.du_id = np.asarray(self.events.du_id)         # used for printing info and saving in voltage tree.
        self.nb_du = trace_shape[0]
        self.sig_size = trace_shape[-1]
        self.traces = np.asarray(self.events.trace, dtype=np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)
        self.du_pos = np.asarray(self.run.du_xyz) # (nb_du, 3) antenna position wrt local grand coordinate

        # shower information like theta, phi, xmax etc for one event.
        shower = ShowerEvent()
        shower.origin_geoid  = self.run.origin_geoid # [lat, lon, height]
        shower.load_root(self.shower)                # calculates grand_ref_frame, shower_frame, Xmax in shower_frame LTP etc
        self.evt_shower = shower                     # Note that 'shower' is an instance of 'self.shower' for one event.
        logger.info(f"shower origin in Geodetic: {self.run.origin_geoid}")

        self.dt_ns = np.asarray(self.run.t_bin_size) # sampling time in ns, sampling freq = 1e9/dt_ns.
        self.f_samp_mhz = 1e3/self.dt_ns             # MHz
        # comupte time samples in ns for all antennas in event with index event_idx.
        self.time_samples = self.get_time_samples()  # t_samples.shape = (nb_du, self.sig_size)
        # common frequencies for all processing in Fourier domain.
        self.fft_size, self.freqs_mhz = get_fastest_size_fft(
            self.sig_size,
            self.f_samp_mhz,
            self.padding_factor,
        )
        logger.info(f"Length of frequency bins with padding factor {self.padding_factor} is {len(self.freqs_mhz)}.")

        # container to collect computed Voc and the final voltage in time domain for one event.
        self.voc = np.zeros_like(self.traces) # time domain
        self.voc_f = np.zeros((trace_shape[0], trace_shape[1], len(self.freqs_mhz)), dtype=np.complex64) # frequency domain
        self.vout = np.zeros_like(self.traces) # final voltage in time domain
        self.vout_f = np.zeros_like(self.voc_f) # frequency domain. changes with addition of noise and signal propagation in rf chain.

        # initialize linear interpolation of Leff for self.freqs_mhz frequency. This is required once per event.
        AntennaProcessing.init_interpolation(
            self.ant_model.leff_sn.frequency/1e6, self.freqs_mhz
        )
        # Compute galactic noise.
        if self.params["add_noise"]:
            # lst: local sideral time, galactic noise max at 18h
            self.fft_noise_gal_3d = galactic_noise(
                self.params["lst"],
                self.fft_size,
                self.freqs_mhz,
                self.nb_du,
                seed=self.seed,
                du_type=self.du_type
            )
        # compute total transfer function of RF chain. Can be computed only once in __init__ if length of time traces does not change between events.
        if self.params["add_rf_chain"]:
            #self.rf_chain.compute_for_freqs(self.freqs_mhz)
            self.rf_chain2.compute_for_freqs(self.freqs_mhz)
            
        if self.params["add_rf_chain_nut"]:
        #    #self.rf_chain.compute_for_freqs(self.freqs_mhz)
            self.rf_chain2nut.compute_for_freqs(self.freqs_mhz)
            
    def get_leff(self, du_idx):
        """
        Define for each antenna in DU du_idx an object AntennaProcessing according its position

        :param du_idx: index of DU
        :    type du_idx: int
        """
        if self.du_pos[du_idx, 0]>22000000:
            raise ValueError("du_pos_x is too large for computing!")
        elif self.du_pos[du_idx, 1]>22000000:
            raise ValueError("du_pos_y is too large for computing!")
        elif self.du_pos[du_idx, 2]>22000000:
            raise ValueError("du_pos_z is too large for computing!")
        else:
            pass


        antenna_location = coord.LTP(
            x=self.du_pos[du_idx, 0], #self.du_pos[du_idx, 0],    # antenna position wrt local grand coordinate
            y=self.du_pos[du_idx, 1], #self.du_pos[du_idx, 1],    # antenna position wrt local grand coordinate
            z=self.du_pos[du_idx, 2], #self.du_pos[du_idx, 2],    # antenna position wrt local grand coordinate
            frame=self.evt_shower.grand_ref_frame
            )
        print("antenna_location", antenna_location)

        antenna_frame = coord.LTP(
            arg=antenna_location,
            location=antenna_location, 
            orientation="NWU", 
            magnetic=True
            )
        print("antenna_frame = " , antenna_frame)

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

    def add(self, addend):
        """
        Add addend on vout_f. Used to add noises manually.
        Make sure:
            addend is in frequency domain
            addend broadcasts with self.vout_f. self.vout_f.shape = (nb_du, 3, nb_freq_mhz)
            addend is computed/interpolated for self.freqs_mhz
        """
        assert self.vout_f.shape==addend.shape
        self.vout_f += addend

    def multiply(self, multiplier):
        """
        Multiply vout_f by multiplier. Used to manually provide RF chain.
        Make sure:
            multiplier is in frequency domain
            multiplier broadcasts with self.vout_f. self.vout_f.shape = (nb_du, 3, nb_freq_mhz)
            multiplier is computed/interpolated for self.freqs_mhz
        """
        assert self.vout_f.shape[-1]==multiplier.shape[-1]
        self.vout_f *= multiplier

    def final_voltage(self):
        """
        Return final voltage in time domain after adding noises and propagating signal through RF chain.
        """
        self.vout[:] = sf.irfft(self.vout_f)[..., :self.sig_size]

    # compute open circuit voltage in one antenna of one event.
    def compute_voc_du(self, du_idx):
        """Compute open circuit voltage for one DU of one event. 
        This method is the base of computing voltage.
        All voltage computation is build on top of this method.

        :param du_idx: index of DU in array traces
        :    type du_idx: int
        """
        logger.info(f"==============>  Processing DU with id: {self.du_id[du_idx]}")
        assert isinstance(du_idx, int)

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

        # Open circuit voltage in frequency domain
        self.voc_f[du_idx, 0] = self.ant_leff_sn.voc_f
        self.voc_f[du_idx, 1] = self.ant_leff_ew.voc_f
        self.voc_f[du_idx, 2] = self.ant_leff_z.voc_f

        # output voltage is time domain. At this stage, vout=voc.
        self.vout[du_idx] = self.voc[du_idx]

        # Use vout_f for further processing. Add noise and propagate signal through RF chain.
        # voc and voc_f is saved so that they can be used for testing or adding user defined noises and rf chain.
        self.vout_f[du_idx, 0] = self.ant_leff_sn.voc_f
        self.vout_f[du_idx, 1] = self.ant_leff_ew.voc_f
        self.vout_f[du_idx, 2] = self.ant_leff_z.voc_f

    def compute_voc_event(self, event_idx=None, event_number=None, run_number=None):
        """
        Compute/simulate all DU in event either with index event_idx or with event_number and run_number.
        Computes open circuit voltage in time domain (voc) and frequency domain (voc_f)
        voc.shape = (nb_du, 3, len of time traces)
        voc_f.shape = (nb_du, 3, len of freqs_mhz)

        :param: event_idx: index of event in events_list. It is a number from range(len(event_list)).
        :    type: int
        :param: event_number: event_number of an event. Combination of event_number and run_number must be unique.
        :    type: int
        :param: run_number: run_number of an event. Combination of event_number and run_number must be unique.
        :    type: int  
        Note: Either event_idx, or both event_number and run_number must be provided.      
        """
        # update event. Provide either integer event_idx, or event_number and run_number.
        self.get_event(event_idx, event_number, run_number)
        for du_idx in range(self.nb_du):
            self.compute_voc_du(du_idx)

    # compute voltage in one antenna of one event.
    def compute_voltage_du(self, du_idx):
        """Compute voltage output for one DU with index du_idx.
        This method can add noises and RF chain to Voc if requested.

        Processing order:
          1. compute Voc using antenna responses
          2. add galactic noise (if requested)
          3. RF chain effect    (if requested)

        :param: event_idx: index of event in events_list. It is a number from range(len(event_list)).
        :    type: int
        :param: event_number: event_number of an event. Combination of event_number and run_number must be unique.
        :    type: int
        :param: run_number: run_number of an event. Combination of event_number and run_number must be unique.
        :    type: int  
        Note: Either event_idx, or both event_number and run_number must be provided.      
        """
        assert isinstance(du_idx, int)
        self.compute_voc_du(du_idx)

        # ----- Add galactic noise -----
        if self.params["add_noise"]:
            # RK: I think irfft of galactic noise here is unnecessary.
            #noise_gal = sf.irfft(self.fft_noise_gal_3d[du_idx])[:, : self.sig_size]
            #logger.debug(np.std(noise_gal, axis=1))
            #self.voc[du_idx] += noise_gal
            self.vout_f[du_idx] += self.fft_noise_gal_3d[du_idx]

        # ----- Add RF chain -----
        if self.params["add_rf_chain"]:
            #self.vout_f[du_idx] *= self.rf_chain.get_tf()
            self.vout_f[du_idx] *= self.rf_chain2.get_tf()
        
        if self.params["add_rf_chain_nut"]:
            #self.vout_f[du_idx] *= self.rf_chain.get_tf()
            self.vout_f[du_idx] *= self.rf_chain2nut.get_tf()

        # Final voltage output for antenna with index du_idx
        if self.params["add_noise"] or self.params["add_rf_chain"]:
            # inverse FFT and remove zero-padding
            # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
            self.vout[du_idx] = sf.irfft(self.vout_f[du_idx])[:, : self.sig_size]
        
        if self.params["add_noise"] or self.params["add_rf_chain_nut"]:
            # inverse FFT and remove zero-padding
            # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
            self.vout[du_idx] = sf.irfft(self.vout_f[du_idx])[:, : self.sig_size]
        
    # compute voltage in all antennas of one event.
    def compute_voltage_event(self, event_idx=None, event_number=None, run_number=None):
        """
        Simulate all DU of an event either with index event_idx or with event_number and run_number.

        :param event_idx: index of event for which voltage is computed.
        This method is equivalent to the code below that takes longer.
        for du_idx in range(self.nb_du):
            self.compute_voltage_du(du_idx)

        :param: event_idx: index of event in events_list. It is a number from range(len(event_list)).
        :    type: int
        :param: event_number: event_number of an event. Combination of event_number and run_number must be unique.
        :    type: int
        :param: run_number: run_number of an event. Combination of event_number and run_number must be unique.
        :    type: int  
        Note: Either event_idx, or both event_number and run_number must be provided.      
        """
        # Provide either integer event_idx, or both event_number and run_number.
        self.compute_voc_event(event_idx, event_number, run_number)

        # ----- Add galactic noise -----
        if self.params["add_noise"]:
            self.add(self.fft_noise_gal_3d)

        # ----- Add RF chain -----
        if self.params["add_rf_chain"]:
            #self.multiply(self.rf_chain.get_tf())
            self.multiply(self.rf_chain2.get_tf())
            
        if self.params["add_rf_chain_nut"]:
            #self.multiply(self.rf_chain.get_tf())
            self.multiply(self.rf_chain2nut.get_tf())
        

        # Final voltage output for antenna with index du_idx
        if self.params["add_noise"] or self.params["add_rf_chain"]:
            # inverse FFT and remove zero-padding
            # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
            #self.vout = sf.irfft(self.vout_f)[..., :self.sig_size]
            self.final_voltage()   # inverse fourier transform. update self.vout.
        
        if self.params["add_noise"] or self.params["add_rf_chain_nut"]:
        #    # inverse FFT and remove zero-padding
        #    # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
        #    #self.vout = sf.irfft(self.vout_f)[..., :self.sig_size]
            self.final_voltage()   # inverse fourier transform. update self.vout.
        
    # Primary method to compute voltage. 
    # Compute voltage in any one antennas of any one event. If None, voltage for all DUs of all events is computed.
    def compute_voltage(self, 
        event_idx=None, 
        du_idx=None, 
        event_number=None, 
        run_number=None, 
        append_file=True
        ):
        """Primary method to compute voltage.  
        Compute/simulate voltage for any or all DUs for any or all events in input file.

        :param: event_idx: index of event in events_list. It is a number from range(len(event_list)). If None, all events in an input file is used.
        :    type: int, list, np.ndarray
        :param du_idx: index of DU for which voltage is computed. If None, all DUs of an event is used. du_idx can be used for only one event.
        :    type: int, list, np.ndarray
        :param: event_number: event_number of an event. Combination of event_number and run_number must be unique.  If None, all events in an input file is used.
        :    type: int, list, np.ndarray
        :param: run_number: run_number of an event. Combination of event_number and run_number must be unique.  If None, all events in an input file is used.
        :    type: int, list, np.ndarray  

        Note: Either event_idx, or both event_number and run_number must be provided, or all three must be None.      
              if du_idx is provided, voltage of the given DU of the given event is computed. 
              du_idx can be an integer or list/np.ndarray. du_idx can be used for only one event.
              If improper event_idx or (event_number and run_number) is used, an error is generated when self.get_event() is called.
              Selective events with either event_idx or both event_number and run_number can be given.
              If list/np.ndarray is provided, length of event_number and run_number must be equal.    
        """
        # compute voltage for all DUs of given event/s.
        if du_idx is None:
            # default case: compute voltage for all DUs of all events and all runs provided in the input file.
            if (event_idx is None) and (event_number is None) and (run_number is None):
                nb_events = len(self.events_list)
                # If there are no events in the file, exit
                if nb_events == 0:
                    message = "There are no events in the file! Exiting."
                    logger.error(message)
                    raise Exception(message)
                for evt_idx in range(nb_events):
                    self.compute_voltage_event(event_idx=evt_idx) # event_number and run_number is None
                    self.save_voltage(append_file)
            # compute voltage for one event with index event_idx or with event_number and run_number.
            elif isinstance(event_idx, int) or (isinstance(event_number, int) and isinstance(run_number, int)):
                self.compute_voltage_event(event_idx=event_idx, event_number=event_number, run_number=run_number)
                self.save_voltage(append_file)
            # compute voltage for a list of events given in event_idx. List can be given as 'list' or 'np.ndarray'.
            elif isinstance(event_idx, (list, np.ndarray)):
                for evt_idx in event_idx:
                    self.compute_voltage_event(event_idx=evt_idx)
                    self.save_voltage(append_file)
            # compute voltage for a list of events given in event_number and run_number. List can be given as 'list' or 'np.ndarray'.
            elif isinstance(event_number, (list, np.ndarray)) and isinstance(run_number, (list, np.ndarray)):
                assert len(event_number)==len(run_number)
                for i in range(len(event_number)):
                    self.compute_voltage_event(event_number=event_number[i], run_number=run_number[i])
                    self.save_voltage(append_file)
            else:
                message = f"Provide positive integer or list of either event_idx or both event_number and run_number. \
                Provided values are: event_idx={event_idx}, event_number={event_number}, run_number={run_number}."
                logger.exception(message)
                raise Exception(message)

        # Compute voltage of one DU of a given event. Note that this can be only done for one event.
        elif isinstance(du_idx, int):
            assert isinstance(event_idx, (int, type(None))), "event_index must be integer when du_idx is given. Can compute voltage for only one event."
            assert isinstance(event_number, (int, type(None))), "event_number must be integer when du_idx is given. Can compute voltage for only one event."
            assert isinstance(run_number, (int, type(None))), "run_number must be integer when du_idx is given. Can compute voltage for only one event."
            self.get_event(event_idx=event_idx, event_number=event_number, run_number=run_number) # update event
            self.compute_voltage_du(du_idx)

        # Compute voltage of list of DUs of a given event. Note that this can be only done for one event.
        elif isinstance(du_idx, (list, np.ndarray)):
            assert isinstance(event_idx, (int, type(None))), "event_index must be integer when du_idx is given. Can compute voltage for only one event."
            assert isinstance(event_number, (int, type(None))), "event_number must be integer when du_idx is given. Can compute voltage for only one event."
            assert isinstance(run_number, (int, type(None))), "run_number must be integer when du_idx is given. Can compute voltage for only one event."
            self.get_event(event_idx=event_idx, event_number=event_number, run_number=run_number) # update event
            for idx in du_idx:
                self.compute_voltage_du(idx)
        else:
            message = f"Provide positive integer or list of either event_idx or both event_number and run_number. \
            Provided values are: event_idx={event_idx}, event_number={event_number}, run_number={run_number}."
            logger.exception(message)
            raise Exception(message)

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
            logger.info(f"No output file was defined. Output file is automatically defined as {self.f_output}")
        if not append_file and os.path.exists(self.f_output):
            logger.info(f"save on a new file and remove existing file {self.f_output}")
            os.remove(self.f_output)
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
        self.tt_volt.trace = self.vout

        self.tt_volt.fill()
        self.tt_volt.write()

