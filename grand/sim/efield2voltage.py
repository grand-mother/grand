"""
Master module for the detector unit simulation GRAND
"""
import os
import os.path
from logging import getLogger
import time
import numpy as np
import scipy.fft as sf
from pathlib import Path

import grand.geo.coordinates as coord
import grand.dataio as groot
from grand.basis.type_trace import ElectricField

from .detector.antenna_model import AntennaModel
from .detector.process_ant import AntennaProcessing
from .detector.rf_chain import RFChain
from .detector.rf_chain import RFChainNut
from .detector.rf_chain import RFChain_gaa
from .shower.gen_shower import ShowerEvent
from .noise.galaxy import galactic_noise

logger = getLogger(__name__)

def get_fastest_size_fft(sig_size, f_samp_mhz, padding_factor=1):
    r"""Returns an FFT-friendly transform length and its frequency axis.

    Real FFTs are fastest at lengths whose prime factorisation is small, so
    rather than transforming ``sig_size`` samples directly this rounds up to
    the next such length via :func:`scipy.fft.next_fast_len` and returns the
    matching one-sided frequency axis for :func:`scipy.fft.rfft`.  Padding
    to a longer length also improves the frequency resolution of the result,
    which is what ``padding_factor`` is for.

    Parameters
    ----------
    sig_size : int
        Length of the time traces, in samples.
    f_samp_mhz : ndarray
        Sampling frequency in MHz, e.g. 2000 MHz for a 0.5 ns bin.  An
        array is expected and **only its first element is used**: the
        routine assumes every trace in the event shares one time binning.
    padding_factor : float, optional
        Factor by which to stretch the traces with zeros before
        transforming.  Must be at least 1; the default of 1 pads only as far
        as the next fast length.

    Returns
    -------
    fast_size : int
        The transform length actually to use, ``>= padding_factor*sig_size``.
    freqs_mhz : ndarray
        Frequency axis in MHz, of length ``fast_size//2 + 1``, matching the
        output of :func:`scipy.fft.rfft` at that length.

    Raises
    ------
    AssertionError
        If ``padding_factor`` is less than 1.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        from grand.sim.efield2voltage import get_fastest_size_fft

        n, freqs = get_fastest_size_fft(1000, np.array([2000.0]))
        print("padded length:", n)                 # 1000 is already 2^3 x 5^3
        print("frequency bins:", freqs.size, "| Nyquist:", freqs[-1], "MHz")

    Doubling the padding halves the bin spacing:

    .. jupyter-execute::

        n2, freqs2 = get_fastest_size_fft(1000, np.array([2000.0]),
                                          padding_factor=2)
        print("padded length:", n2)
        print("bin spacing: %.3f -> %.3f MHz" % (freqs[1], freqs2[1]))

    Notes
    -----
    That only ``f_samp_mhz[0]`` is read is a real limitation, not an
    oversight in this docstring: an event whose detection units record at
    different sampling rates would silently get the first unit's frequency
    axis applied to all of them.  The ``ToDo`` in the body marks the same
    point.
    """
    assert padding_factor >= 1
    dt_s      = 1e-6 / f_samp_mhz
    fast_size = sf.next_fast_len(int(padding_factor * sig_size + 0.5))
    # ToDo: this function (or something higher) should properly handle different time bin for each trace
    freqs_mhz = sf.rfftfreq(fast_size, dt_s[0]) * 1e-6
    #print(f"padding_factor {padding_factor} sig_size {sig_size} ({padding_factor * sig_size +0.5}) fast size {fast_size} freqs_mhz size {len(freqs_mhz)}")
    return fast_size, freqs_mhz


class Efield2Voltage:
    """
    Class to compute voltage with GRANDROOT IO

    Goals:
      * Call simulator of detector units with ROOT data
      * Call on more than one event
      * Call on some stations of some event (not tested, not sure it would work as is) #TODO:
      * Different models are availiable for the response of the Detectore units using different simulations packages. The availiable option are (according the du_type parameter) du_type='GP300' (using hfss simulations), 'GP300_nec' (using nec simulations), 'GP300_mat' (using matlab simulations), 'Horizon'
      * Save output in ROOT format
    """

    def __init__(self, d_input, f_output=None, output_directory=None, seed=None, padding_factor=1.0, du_type='GP300'):

        # If directory given, use DataDirectory
        r"""Opens the input and prepares the antenna and RF-chain models.

        Parameters
        ----------
        d_input : str
            Input ROOT file, or a directory of them.
        f_output : str, optional
            Output file.  Derived from the input name when omitted.
        output_directory : str, optional
            Directory to write into.
        seed : int, optional
            Seed for the noise generator.  ``None`` gives an independent
            realisation each run; a fixed value makes it reproducible.
        padding_factor : float, optional
            Zero-padding applied before the transform, which improves the
            frequency resolution.
        du_type : str, optional
            Which antenna model to use.

                Raises
                ------
                IOError
                    If `d_input` is neither a file nor a directory.

                Notes
                -----
                Construction reads the input file, so this object cannot be built
                without one.
        """
        if os.path.isdir(d_input):
            self.d_input = groot.DataDirectory(d_input)
            self.f_input = None
        # If file given, use DataFile
        elif os.path.isfile(d_input):
            self.d_input = groot.DataFile(d_input)
            self.f_input = d_input
        else:
            raise IOError("Input file/directory does not exist")

        f_input_TRun = self.d_input.trun
        f_input_TShower = self.d_input.tshower
        f_input_TEfield = self.d_input.tefield

        self.f_output = f_output

        # If output filename given, use it
        # if f_output:
        #     self.f_output = f_output
        # Otherwise, generate it from tefield filename
        # else:
        #     self.f_output = self.d_input.ftefield.filename.replace("efield", "voltage")

        # If output directory given, use it
        self.output_directory = ""
        if output_directory:
            self.output_directory = output_directory
            # self.f_output = output_directory + "/" + Path(self.f_output).name


        self.du_type = du_type                              # load antenna models
        self.seed = seed                                    # used to generate same set of random numbers. (gal noise)
        self.padding_factor = padding_factor               #
        self.events = f_input_TEfield        # traces and du_pos are stored here
        self.run = f_input_TRun                 # site_long, site_lat info is stored here. Used to define shower frame.
        self.shower = f_input_TShower        # shower info (like energy, theta, phi, xmax etc) are stored here.
        self.events_list = self.events.get_list_of_events() # [[evt0, run0], [evt1, run0], ...[evt0, runN], ...]
        self.rf_chain = RFChain()                           # loads RF chain for GP13
        self.rf_chainnut = RFChainNut()                      # loads RF chain for GP13 in the nut (output of LNA)
        self.rf_chaingaa = RFChain_gaa()                     # loads RF chain for G@Auger
        self.ant_model = AntennaModel(du_type)              # loads antenna models. time consuming. du_type='GP300' (default using hfss simulations), 'GP300_nec', 'GP300_mat', 'Horizon'
        # Every key the class reads must be present here.  Four of them --
        # resample_to_mhz, extend_to_us, calibration_smearing_sigma and
        # add_jitter_ns -- used to be set only by
        # scripts/convert_efield2voltage.py, so the command line worked while
        # the documented Python usage raised KeyError on the first call to
        # compute_voltage().  The defaults are the argparse defaults of that
        # script: zero, meaning the step is off.
        self.params = {
            "add_noise": True,
            "lst": 18.0,
            "add_rf_chain": True,
            "add_rf_chain_nut": False,
            "add_rf_chain_gaa": False,
            "resample_to_mhz": 0,            # 0: keep the input sampling rate
            "extend_to_us": 0,               # 0: keep the input trace length
            "calibration_smearing_sigma": 0, # 0: no calibration smearing
            "add_jitter_ns": 0,              # 0: no trigger-time jitter
        }
        self.previous_run = -1                              # Not to load run info everytime event info is loaded.

    def get_event(self, event_idx=None, event_number=None, run_number=None):
        r"""Loads the data of one event, selected by index or by number.

        Call this for every new event: it replaces the traces, positions and
        shower parameters the other methods work from.

        Parameters
        ----------
        event_idx : int, optional
            Index of the event in ``events_list``, from
            ``range(len(event_list))``.
        event_number : int, optional
            Event number.  Must be given together with `run_number`; the pair
            is unique.
        run_number : int, optional
            Run number.  Must be given together with `event_number`.

        Raises
        ------
        Exception
            If neither `event_idx` nor the ``(event_number, run_number)``
            pair identifies an event in the input.

        Notes
        -----
        Either `event_idx`, or both `event_number` and `run_number`, must be
        given.
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
        #self.traces = np.asarray(self.events.trace, dtype=np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins
        self.traces = self.events.trace.asnumpy().astype(np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)        
        trace_shape = self.traces.shape  # (nb_du, 3, tbins of a trace)
        self.du_id = np.asarray(self.events.du_id)         # used for printing info and saving in voltage tree.
        self.event_dus_indices = self.events.get_dus_indices_in_run(self.run)
        self.nb_du = trace_shape[0]
        self.sig_size = trace_shape[-1]

        # self.du_pos = np.asarray(self.run.du_xyz) # (nb_du, 3) antenna position wrt local grand coordinate
        self.du_pos = np.asarray(self.run.du_xyz)[self.event_dus_indices] # (nb_du, 3) antenna position wrt local grand coordinate

        # shower information like theta, phi, xmax etc for one event.
        shower = ShowerEvent()
        shower.origin_geoid  = self.run.origin_geoid # [lat, lon, height]
        shower.load_root(self.shower)                # calculates grand_ref_frame, shower_frame, Xmax in shower_frame LTP etc
        self.evt_shower = shower                     # Note that 'shower' is an instance of 'self.shower' for one event.
        logger.info(f"shower origin in Geodetic: {self.run.origin_geoid}")

        self.dt_ns = np.asarray(self.run.t_bin_size)[self.event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns.
        self.f_samp_mhz = 1e3/self.dt_ns             # MHz
        # comupte time samples in ns for all antennas in event with index event_idx.
        self.time_samples = self.get_time_samples()  # t_samples.shape = (nb_du, self.sig_size)

        self.target_sampling_rate_mhz = self.params["resample_to_mhz"]  # if differetn from 0, will resample the output to the required sampling rate in mhz
        if self.f_samp_mhz[0]==self.target_sampling_rate_mhz :
          self.target_sampling_rate_mhz=0  #no resampling needed

        assert  self.target_sampling_rate_mhz >= 0

        self.target_duration_us = self.params["extend_to_us"]        # if different from 0, will adjust padding factor to get a trace of this lenght in us
        assert self.target_duration_us >= 0

        if(self.target_duration_us>0):
          self.target_lenght= int(self.target_duration_us*self.f_samp_mhz[0])
          self.padding_factor=self.target_lenght/self.sig_size
          logger.debug(f"padding factor adjusted to {self.padding_factor} to reach a duration of {self.target_duration_us} us")
        else:
          self.target_lenght=int(self.padding_factor * self.sig_size + 0.5) #add 0.5 to avoid any rounding error for the int conversion
          self.target_duration_us = self.target_lenght/self.f_samp_mhz[0]

        assert self.padding_factor >= 1

        # common frequencies for all processing in Fourier domain.
        self.fft_size, self.freqs_mhz = get_fastest_size_fft(
            self.sig_size,
            self.f_samp_mhz,
            self.padding_factor,
        )

        #TODO: WARNING!. zero padding a signal that does not end in 0 will lead to spectral leakage. A treatment wit Windowing is recomended.
        #TODO: WARNING!. downsampling (decimation) will reduce the bandwidth of the system, and aliasing could ocurr. Formaly, the signal should be low-pass filtered before the downsampling
        # in our use case, we go from 2000Mhz to 500Mhz sampling rate, this means that bandwidth goes from 1000Mhz to 250Mhz.  a (causal and zero phase adusted!) Low pass filter should be aplied.
        # our RF chain already acts as a filter (the transfer function is 0 at 250Mhz) so if we apply the RF chain, we are safe. If you are not appling the rf chain, aliasing will ocurr.

        logger.debug(f"Electric field lenght is {self.sig_size} samples at {self.f_samp_mhz[0]}, spanning {self.sig_size/self.f_samp_mhz[0]} us.")
        logger.debug(f"With a padding factor of {self.padding_factor} we will take it to {self.target_lenght} samples, spanning {self.target_lenght/self.f_samp_mhz[0]} us.")
        logger.debug(f"However, optimal number of frequency bins to do a fast fft is {len(self.freqs_mhz)} giving traces of {self.fft_size} samples.")
        logger.debug(f"With this we will obtain traces spanning {self.fft_size/self.f_samp_mhz[0]} us, that we will then truncate if needed to get the requested trace duration.")


        # container to collect computed Voc and the final voltage in time domain for one event.
        #Matias: Since we now may want longer voltage traces, we can no longer use traces as referecne
        #self.voc = np.zeros_like(self.traces) # time domain
        self.voc = np.zeros((trace_shape[0], trace_shape[1], self.fft_size), dtype=float) # time domain
        self.voc_f = np.zeros((trace_shape[0], trace_shape[1], len(self.freqs_mhz)), dtype=np.complex64) # frequency domain
        self.vout = np.zeros_like(self.voc) # final voltage in time domain
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
            self.rf_chain.compute_for_freqs(self.freqs_mhz)

        if self.params["add_rf_chain_nut"]:
        #    #self.rf_chain.compute_for_freqs(self.freqs_mhz)
            self.rf_chainnut.compute_for_freqs(self.freqs_mhz)

        if self.params["add_rf_chain_gaa"]:
        #    #self.rf_chain.compute_for_freqs(self.freqs_mhz)
            self.rf_chaingaa.compute_for_freqs(self.freqs_mhz)

    def get_leff(self, du_idx):
        r"""Builds the antenna response for one detection unit.

        The effective length depends on the direction of the incoming signal
        in the antenna frame, so it is constructed per unit from that unit's
        position and the shower direction.

        Parameters
        ----------
        du_idx : int
            Index of the detection unit in the event arrays.

        Returns
        -------
        AntennaProcessing
            The response object for that unit's three arms.
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
        logger.debug(f"antenna_location = {antenna_location}")

        antenna_frame = coord.LTP(
            arg=antenna_location,
            location=antenna_location, 
            orientation="NWU", 
            magnetic=True
            )
        logger.debug(f"antenna_frame =  {antenna_frame}")

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

        Returns
        -------
        ndarray, shape (n_du, n_samples)
            Time axis of each unit, in nanoseconds.
        """
        t_start_ns = np.asarray(self.events.du_nanoseconds)[...,np.newaxis]   # shape = (nb_du, 1)
        t_samples = (
            np.outer(
                self.dt_ns * np.ones(self.nb_du), np.arange(0, self.sig_size, dtype=np.float64)
                ) + t_start_ns )
        logger.debug(f"shape du_nanoseconds and t_samples =  {t_start_ns.shape}, {t_samples.shape}")

        return t_samples

    def add(self, addend):
        r"""Adds `addend` to the output voltage spectrum, in place.

        Provided so that a caller can inject their own noise instead of, or
        in addition to, the built-in Galactic model.

        Parameters
        ----------
        addend : ndarray
            A frequency-domain quantity that broadcasts against ``vout_f``,
            whose shape is ``(n_du, 3, n_freqs)``.  It must already be
            evaluated on ``self.freqs_mhz``: nothing here interpolates it,
            and a mismatched axis will broadcast silently into the wrong
            frequencies.
        """
        assert self.vout_f.shape==addend.shape
        self.vout_f += addend

    def multiply(self, multiplier):
        r"""Multiplies the output voltage spectrum by `multiplier`, in place.

        Provided so that a caller can apply their own transfer function
        instead of the built-in RF chain.

        Parameters
        ----------
        multiplier : ndarray
            A frequency-domain quantity that broadcasts against ``vout_f``,
            whose shape is ``(n_du, 3, n_freqs)``, already evaluated on
            ``self.freqs_mhz``.
        """
        assert self.vout_f.shape[-1]==multiplier.shape[-1]
        self.vout_f *= multiplier

    #def final_voltage(self):
    #    """
    #    Return final voltage in time domain after adding noises and propagating signal through RF chain.
    #    """
    #    #self.vout[:] = sf.irfft(self.vout_f)[..., :self.sig_size] #MATIAS: here i will leave the padding, and later truncate to the requested lenght
    #    self.vout[:] = sf.irfft(self.vout_f)

    def final_resample(self):
        """
        after everything is done, change the sampling rate if needded and adjust to the desired target lenght:
        """

        if(self.target_sampling_rate_mhz>0): #if we need to resample
            #compute new number of points
            ratio=(self.target_sampling_rate_mhz/self.f_samp_mhz[0])
            m=int(self.fft_size*ratio)
            #now, since we resampled,  we have a new target_lenght
            self.target_lenght= int(self.target_duration_us*self.target_sampling_rate_mhz)
            logger.info(f"resampling the voltage from {self.f_samp_mhz[0]} to {self.target_sampling_rate_mhz} MHz, new trace lenght is {self.target_lenght} samples")
            #we use fourier interpolation, becouse its easy!
            self.vout = sf.irfft(self.vout_f, m)*ratio #renormalize the amplitudes
            #MATIAS: TODO: now, we are missing a place to store the new sampling rate!
        elif(self.params["add_noise"] or self.params["add_rf_chain"]): #we know we dont need to resample, but we might need to reproces the Voc (curently stored in vout by compute_voc_event) to take into acount the noise or the chain
            self.vout[:] = sf.irfft(self.vout_f)

        if(self.target_lenght<np.shape(self.vout)[2]):
            logger.info(f"truncating output to {self.target_lenght} samples")
            self.vout=self.vout[..., :self.target_lenght]


    # compute open circuit voltage in one antenna of one event.
    def compute_voc_du(self, du_idx):
        r"""Computes the open-circuit voltage for one detection unit.

        This is the base of every voltage computation: the others call it,
        directly or through :meth:`compute_voc_event`.

        Parameters
        ----------
        du_idx : int
            Index of the detection unit in the trace arrays.

        Notes
        -----
        Stores the result on the instance rather than returning it: ``voc``
        in the time domain and ``voc_f`` in the frequency domain.
        """
        logger.debug(f"==============>  Processing DU with id: {self.du_id[du_idx]}")
        assert isinstance(du_idx, int)

        self.get_leff(du_idx)
        #logger.debug(self.ant_leff_sn.model_leff)
        # define E field at antenna position

                    #add the calibration noise
        if(self.params["calibration_smearing_sigma"]>0):
          calfactor=np.random.normal(1,self.params["calibration_smearing_sigma"])
          logger.debug(f"Antenna {du_idx} smearing calibration factor {calfactor}")
        else:
          calfactor=1.0

        e_trace = coord.CartesianRepresentation(
            x=calfactor*self.traces[du_idx, 0],
            y=calfactor*self.traces[du_idx, 1],
            z=calfactor*self.traces[du_idx, 2],
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
        r"""Computes the open-circuit voltage for every unit in one event.

        Fills ``voc`` with shape ``(n_du, 3, n_samples)`` and ``voc_f`` with
        shape ``(n_du, 3, n_freqs)``.

        Parameters
        ----------
        event_idx : int, optional
            Index of the event in ``events_list``, from
            ``range(len(event_list))``.
        event_number : int, optional
            Event number.  Must be given together with `run_number`; the pair
            is unique.
        run_number : int, optional
            Run number.  Must be given together with `event_number`.

        Raises
        ------
        Exception
            If neither `event_idx` nor the ``(event_number, run_number)``
            pair identifies an event in the input.

        Notes
        -----
        Either `event_idx`, or both `event_number` and `run_number`, must be
        given.
        """
        # update event. Provide either integer event_idx, or event_number and run_number.
        self.get_event(event_idx, event_number, run_number)
        for du_idx in range(self.nb_du):
            self.compute_voc_du(du_idx)

    # compute voltage in one antenna of one event.
    def compute_voltage_du(self, du_idx):
        r"""Computes the output voltage for one detection unit.

        Applies, in order:

        1. the open-circuit voltage from the antenna response,
        2. Galactic noise, if ``params["add_noise"]``,
        3. the RF chain, if ``params["add_rf_chain"]``.

        Parameters
        ----------
        du_idx : int
            Index of the detection unit in the trace arrays.

        Notes
        -----
        Which stages run is taken from ``self.params``, not from arguments.
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
            self.vout_f[du_idx] *= self.rf_chain.get_tf()

        if self.params["add_rf_chain_nut"]:
            #self.vout_f[du_idx] *= self.rf_chain.get_tf()
            self.vout_f[du_idx] *= self.rf_chainnut.get_tf()

        if self.params["add_rf_chain_gaa"]:
            #self.vout_f[du_idx] *= self.rf_chain.get_tf()
            self.vout_f[du_idx] *= self.rf_chaingaa.get_tf()

        # Final voltage output for antenna with index du_idx
        if self.params["add_noise"] or self.params["add_rf_chain"]:
            # inverse FFT and remove zero-padding
            # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
            self.vout[du_idx] = sf.irfft(self.vout_f[du_idx])#[:, : self.sig_size]

        if self.params["add_noise"] or self.params["add_rf_chain_nut"]:
            # inverse FFT and remove zero-padding
            # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
            self.vout[du_idx] = sf.irfft(self.vout_f[du_idx])#[:, : self.sig_size]

        if self.params["add_noise"] or self.params["add_rf_chain_gaa"]:
            # inverse FFT and remove zero-padding
            # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
            self.vout[du_idx] = sf.irfft(self.vout_f[du_idx])#[:, : self.sig_size]

    # compute voltage in all antennas of one event.
    def compute_voltage_event(self, event_idx=None, event_number=None, run_number=None):
        r"""Computes the output voltage for every unit in one event.

        Equivalent to calling :meth:`compute_voltage_du` for each unit in
        turn, but vectorised over units and therefore much faster.

        Parameters
        ----------
        event_idx : int, optional
            Index of the event in ``events_list``, from
            ``range(len(event_list))``.
        event_number : int, optional
            Event number.  Must be given together with `run_number`; the pair
            is unique.
        run_number : int, optional
            Run number.  Must be given together with `event_number`.

        Raises
        ------
        Exception
            If neither `event_idx` nor the ``(event_number, run_number)``
            pair identifies an event in the input.

        Notes
        -----
        Either `event_idx`, or both `event_number` and `run_number`, must be
        given.
        """
        # Provide either integer event_idx, or both event_number and run_number.
        self.compute_voc_event(event_idx, event_number, run_number)

        # ----- Add galactic noise -----
        if self.params["add_noise"]:
            self.add(self.fft_noise_gal_3d)

        # ----- Add RF chain -----
        if self.params["add_rf_chain"]:
            self.multiply(self.rf_chain.get_tf())

        if self.params["add_rf_chain_nut"]:
            #self.multiply(self.rf_chain.get_tf())
            self.multiply(self.rf_chainnut.get_tf())

        if self.params["add_rf_chain_gaa"]:
            #self.multiply(self.rf_chain.get_tf())
            self.multiply(self.rf_chaingaa.get_tf())

        # # Final voltage output for antenna with index du_idx
        # if self.params["add_noise"] or self.params["add_rf_chain"]:
        #     # inverse FFT and remove zero-padding
        #     # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
        #     #self.vout = sf.irfft(self.vout_f)[..., :self.sig_size]
        #     self.final_voltage()   # inverse fourier transform. update self.vout.
        #
        # if self.params["add_noise"] or self.params["add_rf_chain_nut"]:
        # #    # inverse FFT and remove zero-padding
        # #    # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
        # #    #self.vout = sf.irfft(self.vout_f)[..., :self.sig_size]
        #     self.final_voltage()   # inverse fourier transform. update self.vout.
        #
        # if self.params["add_noise"] or self.params["add_rf_chain_gaa"]:
        # #    # inverse FFT and remove zero-padding
        # #    # WARNING: do not used sf.irfft(fft_vlna, self.sig_size) to remove padding
        # #    #self.vout = sf.irfft(self.vout_f)[..., :self.sig_size]
        #     self.final_voltage()   # inverse fourier transform. update self.vout.
        
    # Primary method to compute voltage. 
    # Compute voltage in any one antennas of any one event. If None, voltage for all DUs of all events is computed.
    def compute_voltage(self, 
        event_idx=None, 
        du_idx=None, 
        event_number=None, 
        run_number=None, 
        append_file=True
        ):
        r"""Computes voltages for any or all events, and saves them.

        The primary entry point.  With no arguments it processes every event
        in the input file.

        Parameters
        ----------
        event_idx : int, list or ndarray, optional
            Index or indices of events to process.  ``None`` processes all.
        du_idx : int, list or ndarray, optional
            Detection units to process.  ``None`` processes all of them.
            May be used for a single event only.
        event_number : int, list or ndarray, optional
            Event number or numbers, given with `run_number`.
        run_number : int, list or ndarray, optional
            Run number or numbers, given with `event_number`.
        append_file : bool, optional
            Append to the output file rather than replacing it.

        Notes
        -----
        Give either `event_idx`, or both `event_number` and `run_number`, or
        none of the three.  When lists are given, `event_number` and
        `run_number` must be the same length.

        The result is written to ``self.f_output`` as a side effect; the
        method returns nothing.
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
                    self.final_resample()
                    self.save_voltage(append_file)
            # compute voltage for one event with index event_idx or with event_number and run_number.
            elif isinstance(event_idx, int) or (isinstance(event_number, int) and isinstance(run_number, int)):
                self.compute_voltage_event(event_idx=event_idx, event_number=event_number, run_number=run_number)
                self.final_resample()
                self.save_voltage(append_file)
            # compute voltage for a list of events given in event_idx. List can be given as 'list' or 'np.ndarray'.
            elif isinstance(event_idx, (list, np.ndarray)):
                for evt_idx in event_idx:
                    self.compute_voltage_event(event_idx=evt_idx)
                    self.final_resample()
                    self.save_voltage(append_file)
            # compute voltage for a list of events given in event_number and run_number. List can be given as 'list' or 'np.ndarray'.
            elif isinstance(event_number, (list, np.ndarray)) and isinstance(run_number, (list, np.ndarray)):
                assert len(event_number)==len(run_number)
                for i in range(len(event_number)):
                    self.compute_voltage_event(event_number=event_number[i], run_number=run_number[i])
                    self.final_resample()
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
        r"""Writes the computed voltages to the output file.

        Parameters
        ----------
        append_file : bool, optional
            Append to an existing file instead of replacing it.

        Notes
        -----
        The destination is ``self.f_output``, fixed when the object was
        constructed.
        """
        # delete file can take time => start with this action
        # File name for DataDirecory
        if self.f_output is None and self.f_input is None:
            cur_file_name = Path(self.d_input.tefield.get_current_file().GetName()).name
            # Replace the efield in the file name (first occurence in the string) with voltage
            cur_f_output = str(Path(self.output_directory) / "voltage".join(cur_file_name.split("efield", 1)))
            logger.info(f"Output file is {cur_f_output}")
        # File name change in other cases
        elif self.f_output is None:
            split_file = os.path.splitext(self.f_input)
            self.f_output  = str(self.output_directory / split_file[0]+"_voltage.root")
            cur_f_output = self.f_output
            logger.info(f"No output file was defined. Output file is automatically defined as {cur_f_output}")
        else:
            cur_f_output = str(self.output_directory / Path(self.f_output))

        if not append_file and os.path.exists(self.output_directory / self.f_output):
            cur_f_output = str(self.output_directory / self.f_output)
            logger.info(f"save on a new file and remove existing file {cur_f_output}")
            os.remove(cur_f_output)
            time.sleep(1)

        logger.info(f"save result in {cur_f_output}")
        self.tt_volt = groot.TVoltage(cur_f_output)

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

        self.tt_volt.time_nanoseconds = self.events.time_nanoseconds


        #modify the trigger position if needed
        if(self.target_sampling_rate_mhz>0):
          originalsampling=1e3/self.dt_ns
          newsampling=self.f_samp_mhz
          ratio=originalsampling/newsampling
        else:
          ratio=1.0

        self.tt_volt.trigger_position=np.ushort(np.asarray(self.events.trigger_position)/ratio)

        #apply time jitter
        jitter= self.params["add_jitter_ns"]
        assert jitter >=0

        if(jitter>0):
           logger.info(f"adding {jitter} ns of time jitter to the trigger times.")
           #reinitialize the random number
           if(self.seed>0):
             np.random.seed(self.seed*(self.events.event_number+1))

           delays=np.round(np.random.normal(0,jitter,size=np.shape(self.events.du_nanoseconds)).astype(int))

           du_nanoseconds=np.asarray(self.events.du_nanoseconds)
           du_seconds=np.asarray(self.events.du_seconds)
           du_nanoseconds=self.events.du_nanoseconds+delays

           #now we have to roll the seconds
           maskplus= du_nanoseconds >=1e9
           maskminus= du_nanoseconds < 0
           du_nanoseconds[maskplus]-=int(1e9)
           du_seconds[maskplus]+=int(1)
           du_nanoseconds[maskminus]+=int(1e9)
           du_seconds[maskminus]-=int(1)

           self.events.du_nanoseconds=du_nanoseconds
           self.events.du_seconds=du_seconds



        self.tt_volt.du_nanoseconds = self.events.du_nanoseconds
        self.tt_volt.du_seconds = self.events.du_seconds
        self.tt_volt.du_id = self.du_id
        self.tt_volt.trace = self.vout

        self.tt_volt.fill()
        self.tt_volt.write()

