"""
Master module for the detector unit simulation GRAND
"""
import os
import warnings
import os.path
from logging import getLogger
import time

import numbers

import numpy as np

from grand.basis import validate as _validate
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
    if not padding_factor >= 1:
        raise ValueError(_validate.message(
            "get_fastest_size_fft", "'padding_factor' must be >= 1, got %s" % padding_factor))
    dt_s      = 1e-6 / f_samp_mhz
    fast_size = sf.next_fast_len(int(padding_factor * sig_size + 0.5))
    # ToDo: this function (or something higher) should properly handle different time bin for each trace
    freqs_mhz = sf.rfftfreq(fast_size, dt_s[0]) * 1e-6
    #print(f"padding_factor {padding_factor} sig_size {sig_size} ({padding_factor * sig_size +0.5}) fast size {fast_size} freqs_mhz size {len(freqs_mhz)}")
    return fast_size, freqs_mhz


def _grandlib_version():
    r"""Returns the installed GRANDlib version, or ``"unknown"``.

    See :func:`grand.provenance.package_version`.

    Returns
    -------
    str
    """
    from grand import provenance
    return provenance.package_version()



#: Below this distance from the core an Xmax position is taken as a placeholder,
#: not a measurement: real showers have Xmax kilometres away (issue #228).
_MIN_XMAX_DISTANCE_M = 100.0


def _trees_of_one_level(directory):
    r"""The efield, run and shower trees of `directory`, read at one level.

    ``DataDirectory`` picks the highest level of each tree type on its own, so
    a folder holding an L0 efield file and an L1 run file paired the L0 traces
    with the L1 sampling time, which silently doubled every voltage (issue
    #237).  The level is taken from the efield tree; the run tree must exist
    at that level, and the shower tree, whose content does not depend on the
    level, is taken at that level or the closest one below it.

    Parameters
    ----------
    directory : grand.dataio.DataDirectory

    Returns
    -------
    tuple
        ``(tefield, trun, tshower)``.

    Raises
    ------
    FileNotFoundError
        If the folder holds no efield tree, or no run tree at the efield's level.
    """
    def level_of(tree):
        levels = [level for level in range(10) if getattr(directory, "%s_l%d" % (tree, level), None) is not None]
        return levels

    efield_levels = level_of("tefield")
    if not efield_levels:
        raise FileNotFoundError(_validate.message(
            "Efield2Voltage", "%s holds no efield file (efield_*_L<level>_*.root)" % directory.dir_name))
    level = efield_levels[-1]
    trun = getattr(directory, "trun_l%d" % level, None)
    if trun is None:
        raise FileNotFoundError(_validate.message(
            "Efield2Voltage", "%s holds an efield file at level %d but no run file at that level "
            "(run_*_L%d_*.root); the run file carries the sampling time, so it must match the "
            "efield file" % (directory.dir_name, level, level)))
    shower_levels = [lvl for lvl in level_of("tshower") if lvl <= level]
    tshower = getattr(directory, "tshower_l%d" % shower_levels[-1]) if shower_levels else directory.tshower
    tefield = getattr(directory, "tefield_l%d" % level)
    logger.info("reading level %d: efield %s, run %s, shower %s"
                % (level, getattr(tefield, "file_name", "?"), getattr(trun, "file_name", "?"),
                   getattr(tshower, "file_name", "?")))
    return tefield, trun, tshower


#: The processing switches of `Efield2Voltage.params`, with their defaults.
PARAM_DEFAULTS = {
    "add_noise": True,
    "lst": 18.0,
    "add_rf_chain": True,
    "add_rf_chain_nut": False,
    "add_rf_chain_gaa": False,
    "resample_to_mhz": 0,            # 0: keep the input rate; other rates: in memory only, save_voltage refuses them (#229)
    "extend_to_us": 0,               # 0: keep the input trace length
    "calibration_smearing_sigma": 0, # 0: no calibration smearing
    "add_jitter_ns": 0,              # 0: no trigger-time jitter
}

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
            The simulation folder ``sim2root.py`` wrote, holding the
            ``efield_*``, ``run_*`` and ``shower_*`` files.  A single e-field
            file is not enough: the run and shower trees are read too.
        f_output : str, optional
            Output file name, relative to `output_directory`.  Derived from
            the input name when omitted, as ``voltage_*_L0_*.root``, which is
            what ``convert_voltage2adc.py`` looks for.
        output_directory : str, optional
            Directory to write into; the current directory when omitted
            (``convert_efield2voltage.py`` passes the input folder).
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
        # A single e-field file has no run or shower tree: it failed with
        # "'DataFile' object has no attribute 'trun'" (#185, #265)
        missing = [name for name in ("trun", "tshower", "tefield") if getattr(self.d_input, name, None) is None]
        if missing:
            raise ValueError(_validate.message(
                "Efield2Voltage", "%s holds no %s tree; give the folder sim2root.py wrote, with its "
                "efield_*, run_* and shower_* files" % (d_input, ", ".join(missing))))

        f_input_TRun = self.d_input.trun
        f_input_TShower = self.d_input.tshower
        f_input_TEfield = self.d_input.tefield
        if isinstance(self.d_input, groot.DataDirectory):
            f_input_TEfield, f_input_TRun, f_input_TShower = _trees_of_one_level(self.d_input)

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
            # A folder not yet made failed when writing (#182)
            Path(output_directory).mkdir(parents=True, exist_ok=True)
            # self.f_output = output_directory + "/" + Path(self.f_output).name


        self.du_type = du_type                              # load antenna models
        # None, or a non-negative integer: a negative seed was accepted and
        # silently meant "unseeded", and so did 0 (#230)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError(_validate.message(
                "Efield2Voltage", "'seed' must be None or a non-negative integer, got %r" % (seed,)))
        self.seed = seed                                    # used to generate same set of random numbers. (gal noise)
        # 0, negative, NaN or text failed later with a message about extend_to_us (#265)
        # Below 1 it read as "'extend_to_us' = 0 us is shorter than the traces" (#233)
        if not _validate.as_real(padding_factor, "padding_factor", "Efield2Voltage") >= 1:
            raise ValueError(_validate.message(
                "Efield2Voltage", "'padding_factor' must be at least 1 (the output cannot be shorter "
                "than the input), got %r" % (padding_factor,)))
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
        self.params = dict(PARAM_DEFAULTS)
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
        elif (self.event_idx is not None) and (0 <= self.event_idx < len(self.events_list)): 
            self.event_number = self.events_list[self.event_idx][0]
            self.run_number = self.events_list[self.event_idx][1]
        else:
            message = f"Provide positive integer of either event_idx or both event_number and run_number. If event_idx is given, it must\
            be less than {len(self.events_list)}. If event_number and run_number are given, they must be from the list of (event_number, run_number)\
            {self.events_list}. Provided values are: event_idx={event_idx}, event_number={event_number}, run_number={run_number}."
            logger.error(message)  # not in an except block: .exception logged 'NoneType: None' (#256)
            raise Exception(message)

        # The pair must be one the input holds: otherwise the trees below
        # keep the previously loaded event, which was then written under the
        # requested numbers (issue #238).
        for _name in ("event_number", "run_number"):
            _value = getattr(self, _name)
            if isinstance(_value, (bool, np.bool_)) or not isinstance(_value, numbers.Integral):
                raise TypeError(_validate.message(
                    "Efield2Voltage.get_event", "'%s' must be an integer, got %r" % (_name, _value)))
            setattr(self, _name, int(_value))
        if (self.event_number, self.run_number) not in {(int(e), int(r)) for e, r in self.events_list}:
            raise KeyError(_validate.message(
                "Efield2Voltage.get_event", "no event %d in run %d in the input; it holds (event, run) %s"
                % (self.event_number, self.run_number, [(int(e), int(r)) for e, r in self.events_list])))
        logger.info(f"Running on event_number: {self.event_number}, run_number: {self.run_number}")

        self.events.get_event(self.event_number, self.run_number)           # update traces, du_pos etc for event with event_idx.
        self.shower.get_event(self.event_number, self.run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
        if self.previous_run != self.run_number:                      # load only for new run.
            self.run.get_run(self.run_number)                         # update run info to get site latitude and longitude.
            self.previous_run = self.run_number
        # A lookup that finds nothing leaves the previous entry loaded: the
        # antenna response was then computed for another event's shower
        # (issue #247).  Refuse instead.
        if (int(self.shower.event_number), int(self.shower.run_number)) != (self.event_number, self.run_number):
            raise KeyError(_validate.message(
                "Efield2Voltage.get_event", "the shower tree has no entry for event %d of run %d; "
                "the efield and shower files of the input do not match" % (self.event_number, self.run_number)))
        if int(self.run.run_number) != self.run_number:
            self.previous_run = None
            raise KeyError(_validate.message(
                "Efield2Voltage.get_event", "the run tree has no entry for run %d; the efield and "
                "run files of the input do not match" % self.run_number))

        # stack efield traces
        #self.traces = np.asarray(self.events.trace, dtype=np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins
        self.traces = self.events.trace.asnumpy().astype(np.float32)  # x,y,z components are stored in events.trace. shape (nb_du, 3, tbins)        
        trace_shape = self.traces.shape  # (nb_du, 3, tbins of a trace)
        self.du_id = np.asarray(self.events.du_id)         # used for printing info and saving in voltage tree.
        # A NaN or inf sample spread through the FFT to the whole voltage trace,
        # and the ADC step then wrote it as the most negative integer (#239)
        if self.traces.size and not np.all(np.isfinite(self.traces)):
            bad = sorted({int(self.du_id[i]) for i in np.nonzero(~np.isfinite(self.traces))[0]})
            raise ValueError(_validate.message(
                "Efield2Voltage", "the e-field of event %s (run %s) has NaN or infinite samples, "
                "for units %s" % (self.event_number, self.run_number, bad)))
        # Calibration smearing draws from a generator seeded per event: it used
        # NumPy's global one, which the seed never reached, so two runs with
        # the same seed differed (#230)
        self._smearing_rng = np.random.default_rng(
            None if self.seed is None else [int(self.seed), int(self.event_number)])
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

        # The antenna response is evaluated in the direction of Xmax seen from
        # each antenna.  Without a usable Xmax that direction is undefined: a
        # NaN position crashed deep in the antenna lookup, and a position a few
        # centimetres from the core (a "-1 = unknown" read as a distance) gave
        # voltages near 1e-13 uV (issue #228).  Refuse instead.
        maximum = np.asarray(shower.maximum, dtype=float).ravel()
        if not np.all(np.isfinite(maximum)) or np.linalg.norm(maximum) < _MIN_XMAX_DISTANCE_M:
            raise ValueError(_validate.message(
                "Efield2Voltage.get_event", "event %d of run %d has no usable Xmax position "
                "(xmax_pos_shc %s, %.3g m from the core); the antenna response needs the "
                "direction of Xmax. Regenerate the simulation with the shower maximum filled in"
                % (self.event_number, self.run_number, np.asarray(self.shower.xmax_pos_shc).tolist(),
                   np.linalg.norm(maximum) if np.all(np.isfinite(maximum)) else float("nan"))))

        # A shower that hit no antenna (issue #91): there is nothing to
        # compute, but the event is still written, with du_count 0, so that it
        # keeps counting in the statistics downstream.
        if self.nb_du == 0:
            self._set_empty_event()
            return

        self.dt_ns = np.asarray(self.run.t_bin_size)[self.event_dus_indices] # sampling time in ns, sampling freq = 1e9/dt_ns.
        self.f_samp_mhz = 1e3/self.dt_ns             # MHz
        # comupte time samples in ns for all antennas in event with index event_idx.
        self.time_samples = self.get_time_samples()  # t_samples.shape = (nb_du, self.sig_size)

        self.target_sampling_rate_mhz = self.params["resample_to_mhz"]  # if differetn from 0, will resample the output to the required sampling rate in mhz
        if self.f_samp_mhz[0]==self.target_sampling_rate_mhz :
          self.target_sampling_rate_mhz=0  #no resampling needed

        _validate.non_negative(_validate.as_real(self.target_sampling_rate_mhz, "resample_to_mhz", "Efield2Voltage"),
                               "resample_to_mhz", "Efield2Voltage", "MHz")

        self.target_duration_us = self.params["extend_to_us"]        # if different from 0, will adjust padding factor to get a trace of this lenght in us
        _validate.non_negative(_validate.as_real(self.target_duration_us, "extend_to_us", "Efield2Voltage"),
                               "extend_to_us", "Efield2Voltage", "us")

        if(self.target_duration_us>0):
          self.target_lenght= int(self.target_duration_us*self.f_samp_mhz[0])
          self.padding_factor=self.target_lenght/self.sig_size
          logger.debug(f"padding factor adjusted to {self.padding_factor} to reach a duration of {self.target_duration_us} us")
        else:
          self.target_lenght=int(self.padding_factor * self.sig_size + 0.5) #add 0.5 to avoid any rounding error for the int conversion
          self.target_duration_us = self.target_lenght/self.f_samp_mhz[0]

        if not self.padding_factor >= 1:
            raise ValueError(_validate.message(
                "Efield2Voltage", "'extend_to_us' = %s us is shorter than the traces "
                "(%.4g us); the output cannot be shorter than the input"
                % (self.params["extend_to_us"], self.sig_size / self.f_samp_mhz[0])))

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

    def _set_empty_event(self):
        r"""Prepares the state for an event with no detection unit.

        Every per-unit array is set to its empty shape, ``(0, 3, 0)`` for the
        traces, so that :meth:`save_voltage` writes the event with
        ``du_count`` 0 and the steps in between have nothing to do.
        """
        logger.warning(
            f"Event {self.event_number} of run {self.run_number} has no antenna "
            "(du_count 0): no voltage to compute; it is written with du_count 0."
        )
        self.traces = np.zeros((0, 3, 0), dtype=np.float32)
        self.sig_size = 0
        self.du_pos = np.zeros((0, 3), dtype=np.float64)
        self.dt_ns = np.zeros(0, dtype=np.float64)
        self.f_samp_mhz = np.zeros(0, dtype=np.float64)
        self.time_samples = np.zeros((0, 0), dtype=np.float64)
        self.target_sampling_rate_mhz = 0
        self.target_lenght = 0
        self.fft_size = 0
        self.freqs_mhz = np.zeros(0, dtype=np.float64)
        self.voc = np.zeros((0, 3, 0), dtype=float)
        self.voc_f = np.zeros((0, 3, 0), dtype=np.complex64)
        self.vout = np.zeros_like(self.voc)
        self.vout_f = np.zeros_like(self.voc_f)

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
        # No antenna in this event (issue #91): the empty output stays as it is
        if self.nb_du == 0:
            return

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
        # No resampling, but anything applied in the frequency domain (noise or
        # any of the three chains) must be brought back: until then ``vout``
        # still holds the V_oc that compute_voc_event put there (issue #227).
        elif (self.params["add_noise"] or self.params["add_rf_chain"]
              or self.params["add_rf_chain_nut"] or self.params["add_rf_chain_gaa"]):
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
          calfactor=self._smearing_rng.normal(1,self.params["calibration_smearing_sigma"])
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
        self._check_params()
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

        # No antenna in this event (issue #91): no noise or RF chain to apply
        if self.nb_du == 0:
            return

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
    def _flush_voltage(self):
        r"""Writes and releases the output tree kept open by compute_voltage()."""
        batch = getattr(self, "_batch_volt", None)
        if batch and batch.get("tree") is not None:
            tree = batch["tree"]
            batch["tree"] = None
            tree.write()
            tree.stop_using()
            if batch.get("partial"):
                groot.data_tree.replace_output(batch.pop("partial"), batch["name"])

    def _discard_voltage(self):
        r"""Drops the output of a compute_voltage() that failed, writing nothing (#240)."""
        batch = getattr(self, "_batch_volt", None)
        if batch and batch.get("tree") is not None:
            tree = batch["tree"]
            batch["tree"] = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree.stop_using()
            if batch.get("partial") and os.path.exists(batch["partial"]):
                os.remove(batch.pop("partial"))

    def _check_params(self):
        r"""Checks the processing switches before any work is done.

        They were checked only when the first event was saved, after the whole
        computation, and the failed run left a stub output file (#240).
        """
        # A misspelt key was ignored and the default used; a flag was read by
        # truthiness, so the string 'no' turned the RF chain on (#265)
        unknown = sorted(set(self.params) - set(PARAM_DEFAULTS))
        if unknown:
            raise KeyError(_validate.message(
                "Efield2Voltage.params", "unknown key%s %s; the keys are %s"
                % ("s" if len(unknown) > 1 else "", ", ".join(map(repr, unknown)),
                   ", ".join(sorted(PARAM_DEFAULTS)))))
        for name in ("add_noise", "add_rf_chain", "add_rf_chain_nut", "add_rf_chain_gaa"):
            if not isinstance(self.params[name], (bool, np.bool_)):
                raise TypeError(_validate.message(
                    "Efield2Voltage.params", "%r must be True or False, got %r" % (name, self.params[name])))
        lst = _validate.as_real(self.params["lst"], "lst", "Efield2Voltage")
        if not 0 <= lst <= 24:
            raise ValueError(_validate.message("Efield2Voltage.params", "'lst' must be 0 to 24 h, got %r" % lst))
        for name, unit in (("resample_to_mhz", "MHz"), ("extend_to_us", "us"),
                           ("calibration_smearing_sigma", ""), ("add_jitter_ns", "ns")):
            _validate.non_negative(_validate.as_real(self.params[name], name, "Efield2Voltage"),
                                   name, "Efield2Voltage", unit)

    def compute_voltage(self, event_idx=None, du_idx=None, event_number=None, run_number=None,
                        append_file=True):
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
        self._check_params()
        self._batch_volt = {}
        try:
            self._compute_voltage(event_idx=event_idx, du_idx=du_idx, event_number=event_number,
                                  run_number=run_number, append_file=append_file)
        except BaseException:
            self._discard_voltage()
            self._batch_volt = None
            raise
        try:
            self._flush_voltage()
        finally:
            self._batch_volt = None

    def _compute_voltage(self, 
        event_idx=None, 
        du_idx=None, 
        event_number=None, 
        run_number=None, 
        append_file=True
        ):
        r"""Body of :meth:`compute_voltage`."""
        # NumPy integer scalars (from arrays, events_list...) are integers too
        def _plain(value):
            return int(value) if isinstance(value, np.integer) else value
        event_idx, event_number, run_number, du_idx = (
            _plain(event_idx), _plain(event_number), _plain(run_number), _plain(du_idx))

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
                _validate.same_length("Efield2Voltage.compute_voltage",
                                      event_number=event_number, run_number=run_number)
                for i in range(len(event_number)):
                    self.compute_voltage_event(event_number=event_number[i], run_number=run_number[i])
                    self.final_resample()
                    self.save_voltage(append_file)
            else:
                message = f"Provide positive integer or list of either event_idx or both event_number and run_number. \
                Provided values are: event_idx={event_idx}, event_number={event_number}, run_number={run_number}."
                logger.error(message)  # not in an except block: .exception logged 'NoneType: None' (#256)
                raise Exception(message)

        # Compute voltage of one DU of a given event. Note that this can be only done for one event.
        elif isinstance(du_idx, int):
            for _name, _value in (("event_idx", event_idx), ("event_number", event_number),
                                  ("run_number", run_number)):
                if not isinstance(_value, (int, type(None))):
                    raise TypeError(_validate.message(
                        "Efield2Voltage.compute_voltage", "'%s' must be a single integer when "
                        "'du_idx' is given (voltages of chosen units are computed for one event "
                        "at a time), got %s" % (_name, type(_value).__name__)))
            self.get_event(event_idx=event_idx, event_number=event_number, run_number=run_number) # update event
            self.compute_voltage_du(du_idx)

        # Compute voltage of list of DUs of a given event. Note that this can be only done for one event.
        elif isinstance(du_idx, (list, np.ndarray)):
            for _name, _value in (("event_idx", event_idx), ("event_number", event_number),
                                  ("run_number", run_number)):
                if not isinstance(_value, (int, type(None))):
                    raise TypeError(_validate.message(
                        "Efield2Voltage.compute_voltage", "'%s' must be a single integer when "
                        "'du_idx' is given (voltages of chosen units are computed for one event "
                        "at a time), got %s" % (_name, type(_value).__name__)))
            self.get_event(event_idx=event_idx, event_number=event_number, run_number=run_number) # update event
            for idx in du_idx:
                self.compute_voltage_du(idx)
        else:
            message = f"Provide positive integer or list of either event_idx or both event_number and run_number. \
            Provided values are: event_idx={event_idx}, event_number={event_number}, run_number={run_number}."
            logger.error(message)  # not in an except block: .exception logged 'NoneType: None' (#256)
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
        # A resampled voltage cannot be saved: TVoltage has no sampling-rate
        # field and this class writes no run tree, so every later step (such as
        # convert_voltage2adc.py) would read the input's t_bin_size and treat
        # the trace at the wrong rate, silently (issue #229).  Refused before
        # anything is written.  Resample the e-field instead, with
        # convert_efield2efield.py, which records the new rate in its run tree.
        if self.target_sampling_rate_mhz > 0:
            raise ValueError(_validate.message(
                "Efield2Voltage.save_voltage",
                "the voltage was resampled to %s MHz (params['resample_to_mhz']), but the "
                "voltage file cannot record a new sampling rate, so later steps would use the "
                "input's rate. Set resample_to_mhz to 0 and resample the e-field first "
                "(convert_efield2efield.py --target_sampling_rate_mhz), which writes the new "
                "rate to its run tree" % self.target_sampling_rate_mhz))

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
            self.f_output  = str(Path(self.output_directory) / (split_file[0]+"_voltage.root"))
            cur_f_output = self.f_output
            logger.info(f"No output file was defined. Output file is automatically defined as {cur_f_output}")
        else:
            cur_f_output = str(self.output_directory / Path(self.f_output))

        # Within compute_voltage() the output tree stays open across events and
        # is written once at the end: reopening and rewriting the file for every
        # event made each event slower than the last (#283), and with
        # append_file=False the file was deleted before every event, so only
        # the last one was kept
        batch = getattr(self, "_batch_volt", None)
        if batch is not None and batch.get("name") == cur_f_output and batch.get("tree") is not None:
            self.tt_volt = batch["tree"]
        elif batch is not None and (not append_file or not os.path.exists(cur_f_output)):
            # A new or replaced output is written under a temporary name and
            # moved into place when complete: a run that failed late left a
            # stub that broke every later run, and a re-run deleted the old
            # output before it had a new one (#240)
            self._flush_voltage()
            partial = groot.data_tree.partial_name(cur_f_output)
            if os.path.exists(partial):
                os.remove(partial)
            logger.info(f"save result in {cur_f_output}")
            self.tt_volt = groot.TVoltage(partial)
            batch.update(name=cur_f_output, tree=self.tt_volt, partial=partial)
        else:
            self._flush_voltage()
            # Path(): output_directory may be a string, which crashed here
            if not append_file and os.path.exists(Path(self.output_directory) / self.f_output):
                cur_f_output = str(Path(self.output_directory) / self.f_output)
                logger.info(f"save on a new file and remove existing file {cur_f_output}")
                os.remove(cur_f_output)
                time.sleep(1)

            logger.info(f"save result in {cur_f_output}")
            self.tt_volt = groot.TVoltage(cur_f_output)
            if batch is not None:
                batch.update(name=cur_f_output, tree=self.tt_volt)

        # Fill voltage object. d_root = events
        self.tt_volt.du_count     = self.nb_du
        logger.debug(f"We will save voltage for {self.tt_volt.du_count} DUs.")

        # Stamp the producing version. The simulated voltage depends on the
        # code as much as on the input -- the galactic-noise normalisation
        # moved by sqrt(2) on 2026-09-07 -- and without this there is nothing
        # in a file to say which side of such a change it came from.
        self.tt_volt.grandlib_version = _grandlib_version()
        # The level of the e-field it was made from, which the file name also
        # carries: it was left at 0, so a voltage_*_L1_* file said level 0 and
        # the next scan of the folder failed on the missing tvoltage_l1 (#240)
        self.tt_volt.analysis_level = int(self.events.analysis_level)

        self.tt_volt.run_number   = self.events.run_number
        self.tt_volt.event_number = self.events.event_number
        logger.debug(f"{type(self.tt_volt.run_number)} {type(self.tt_volt.event_number)}")
        logger.debug(f"{self.tt_volt.run_number} {self.tt_volt.event_number}")

        # An event with no antenna (issue #91) has no first DU; 0 is the default
        self.tt_volt.first_du         = self.du_id[0] if len(self.du_id) else 0
        self.tt_volt.time_seconds     = self.events.time_seconds
        self.tt_volt.time_nanoseconds = self.events.time_nanoseconds

        self.tt_volt.time_nanoseconds = self.events.time_nanoseconds


        # The trigger sample keeps the input's rate: a resampled voltage is
        # refused above (issue #229).
        self.tt_volt.trigger_position=np.ushort(np.asarray(self.events.trigger_position))

        #apply time jitter
        jitter= self.params["add_jitter_ns"]
        _validate.non_negative(_validate.as_real(jitter, "add_jitter_ns", "Efield2Voltage"),
                               "add_jitter_ns", "Efield2Voltage", "ns")

        if(jitter>0):
           logger.info(f"adding {jitter} ns of time jitter to the trigger times.")
           #reinitialize the random number
           # Seed 0 seeds too, and no seed no longer crashes (None > 0, #230)
           if self.seed is not None:
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
        if batch is None:
            self.tt_volt.write()

