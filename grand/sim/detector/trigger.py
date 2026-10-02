# -*- coding: utf-8 -*-
r"""Offline, DAQ-style first-level (T1) trigger on ADC traces.

This is the logic of ``scripts/T1_trigger_offline.py`` (snonis,
2024-10), moved here so that it can be imported, tested, and applied by
``scripts/convert_voltage2adc.py`` (issue #139).  The algorithm and the
default parameters are unchanged.

.. warning::

   The default parameters (:data:`DEFAULT_T1_CONFIG`) are copied verbatim
   from the offline script.  They have **not** been confirmed by the trigger
   group as the ones the DAQ uses.  See :func:`extract_trigger_parameters`
   for the points that need confirming.

The trigger, per channel
------------------------
All times are in ns and converted to samples assuming 2 ns per sample
(500 MHz, the GRAND ADC):

1. **T1 crossing.**  The first sample strictly above ``th1``.  It needs at
   least ``t_quiet/2`` samples before it in the trace, all at or below
   ``th1`` (the quiet time).  A first crossing earlier than that is rejected.
2. **Window.**  The ``t_period/2`` samples starting at the T1 crossing.
3. **T2 crossings.**  In the window, every upward crossing of ``+th2``
   (a sample above ``th2`` whose predecessor is not); the T1 crossing counts
   as the first one.  Negative crossings of ``-th2`` are *not* counted.
4. **Separation.**  Each T2 crossing must follow the previous one by less
   than ``t_sepmax`` ns; one that does not rejects the channel.
5. **Count.**  The number of crossings ``NC`` must be within
   ``[nc_min, nc_max]``.

``Q`` (peak / NC) is computed and returned, but, as in the offline script,
``q_min`` and ``q_max`` are not applied.

The trigger, per detector unit
------------------------------
A DU passes T1 if any of the channels in ``channels`` (by default 0, 1, 2,
the X, Y and Z ADC channels) passes.  :func:`t1_du_triggers` evaluates every
DU of an event.
"""

import numpy as np

__all__ = [
    "DEFAULT_T1_CONFIG",
    "DEFAULT_T1_CHANNELS",
    "T1_TRIGGER_FLAG",
    "extract_trigger_parameters",
    "t1_channel_trigger",
    "t1_du_triggers",
    "t1_trigger_flags",
]

#: The trigger parameters of ``scripts/T1_trigger_offline.py``, unchanged.
#: Times are in ns, thresholds in ADC counts.  ``t_pretrig``, ``t_overlap``
#: and ``t_posttrig`` configure the DAQ readout window and are not used by the
#: trigger itself.  To be confirmed by the trigger group.
DEFAULT_T1_CONFIG = {
    "t_quiet": 512,
    "t_period": 512,
    "t_sepmax": 10,
    "nc_min": 2,
    "nc_max": 8,
    "q_min": 0,
    "q_max": 255,
    "th1": 100,
    "th2": 50,
    # Configs of readout timewindow
    "t_pretrig": 960,
    "t_overlap": 64,
    "t_posttrig": 1024,
}

#: The ADC channels the offline script evaluates, (0, 1, 2) = (X, Y, Z).
DEFAULT_T1_CHANNELS = (0, 1, 2)

#: Value written in ``trigger_flag`` for a DU that passes T1; 0 otherwise.
#: ``trigger_flag`` is documented as "same as event_type", where 0x1000 is the
#: 10 s trigger, 0x8000 the random trigger and anything else a shower trigger;
#: the encoding of a T1 shower trigger is to be confirmed by the trigger group.
T1_TRIGGER_FLAG = 1


def _check_config(config, where="T1 trigger"):
    r"""Refuses T1 parameters that cannot describe a trigger.

    They were used as given (#267): a second threshold above the first, a
    coincidence range with ``nc_min > nc_max``, negative windows.

    Raises
    ------
    ValueError
        Naming the parameters and their values.
    """
    def fail(text):
        raise ValueError("GRANDlib: %s: %s" % (where, text))
    for key in ("t_quiet", "t_period", "t_sepmax", "nc_min", "nc_max", "th1", "th2"):
        if not config[key] >= 0:
            fail("%s must be >= 0, got %r" % (key, config[key]))
    if config["th2"] > config["th1"]:
        fail("th2 (%r) must not exceed th1 (%r): T2 crossings are counted after a T1 crossing"
             % (config["th2"], config["th1"]))
    if config["nc_min"] > config["nc_max"]:
        fail("nc_min (%r) must not exceed nc_max (%r)" % (config["nc_min"], config["nc_max"]))
    # At 2 ns per sample, a period under 2 ns is an empty window, which
    # failed with a bare IndexError (#289)
    if config["t_period"] < 2:
        fail("t_period must be at least 2 (ns, one sample), got %r" % (config["t_period"],))


def _config(trigger_config):
    r"""The defaults, updated with `trigger_config`."""
    config = dict(DEFAULT_T1_CONFIG)
    if trigger_config is not None:
        unknown = set(trigger_config) - set(DEFAULT_T1_CONFIG)
        if unknown:
            raise KeyError(f"Unknown T1 trigger parameters: {sorted(unknown)}")
        config.update(trigger_config)
        _check_config(config)
    return config


def extract_trigger_parameters(trace, trigger_config=None, baseline=0):
    r"""Extracts the T1 trigger information from one ADC channel.

    The algorithm of ``scripts/T1_trigger_offline.py``, unchanged; see the
    module docstring for a description.

    Parameters
    ----------
    trace : array_like
        One channel's trace, in ADC counts, sampled at 500 MHz (2 ns).
    trigger_config : dict, optional
        Trigger parameters overriding :data:`DEFAULT_T1_CONFIG`.
    baseline : float, optional
        Subtracted from the peak when computing ``Q``.  The thresholds are
        applied to the raw trace, so it must already be centred on 0.

    Returns
    -------
    dict
        ``index_T1_crossing`` (int), ``index_T2_crossing`` (array of int,
        indices in `trace`, the T1 crossing first), ``NC`` (int) and ``Q``.

    Raises
    ------
    ValueError
        If the channel does not trigger: no T1 crossing, not enough samples
        before it for the quiet time, or two T2 crossings more than
        ``t_sepmax`` apart.

    Notes
    -----
    The default parameters are the offline script's.  **The trigger group
    must confirm them**, and these points of the algorithm, before the result
    is taken as the DAQ's:

    - the thresholds ``th1 = 100`` and ``th2 = 50`` ADC counts, and that they
      apply to the positive polarity only (``-th2`` crossings are not counted,
      and a negative-only pulse never triggers);
    - ``t_sepmax = 10`` ns with a strict ``<``: at 2 ns per sample, T2
      crossings must be at most 8 ns apart, so an oscillation of period
      10 ns or longer (below 100 MHz) never gives NC >= 2;
    - that a crossing too far apart rejects the channel, rather than ending
      the count;
    - the quiet time: only the first crossing is tried, and it is rejected if
      it lies within the first ``t_quiet/2`` samples;
    - ``q_min``/``q_max``, which are not applied;
    - the channels used (0, 1, 2) and the "any channel" DU logic.
    """
    config = _config(trigger_config)
    trace = np.asarray(trace)
    th1, th2 = config["th1"], config["th2"]
    # ns to samples, at 2 ns per sample.
    half_quiet = config["t_quiet"] // 2

    crossings = np.flatnonzero(trace > th1)
    if crossings.size == 0:
        raise ValueError("No T1 crossing!")

    index_t1 = None
    for i in crossings:
        if i - half_quiet < 0:
            raise ValueError("Not enough data before T1 crossing!")
        if np.all(trace[max(0, i - half_quiet):i] <= th1):
            index_t1 = int(i)
            break
    if index_t1 is None:
        raise ValueError("No T1 crossing with Tquiet satified!")

    window = trace[index_t1:index_t1 + config["t_period"] // 2]
    above = (window > th2).astype(int)
    # The T1 crossing is the first T2 crossing; then every upward crossing.
    is_crossing = np.zeros(len(window), dtype=bool)
    is_crossing[0] = True
    is_crossing[1:] = np.diff(above) == 1
    index_t2 = np.flatnonzero(is_crossing)

    kept = [0]
    j = 1
    for i, j in zip(index_t2[:-1], index_t2[1:]):
        separation = (j - i) * 2  # ns
        if separation < config["t_sepmax"]:
            kept.append(int(j))
        else:
            raise ValueError(f"Violating Tsepmax, the separation is {separation} ns.")
    n_crossings = len(kept)

    return {
        "index_T1_crossing": index_t1,
        "index_T2_crossing": np.array(kept) + index_t1,
        "NC": n_crossings,
        # As in the offline script: the peak before the last crossing.
        "Q": (np.max(np.abs(window[:j])) - baseline) / n_crossings,
    }


def t1_channel_trigger(trace, trigger_config=None, baseline=0):
    r"""Whether one ADC channel passes T1.

    It passes if :func:`extract_trigger_parameters` finds a T1 crossing and
    the number of crossings is within ``[nc_min, nc_max]``.  The parameters
    are to be confirmed by the trigger group (see that function).

    Returns
    -------
    bool
    """
    config = _config(trigger_config)
    try:
        info = extract_trigger_parameters(trace, config, baseline)
    except ValueError:
        return False
    return config["nc_min"] <= info["NC"] <= config["nc_max"]


def t1_du_triggers(traces, trigger_config=None, channels=DEFAULT_T1_CHANNELS,
                   baseline=0):
    r"""Evaluates T1 on every detector unit of an event.

    Parameters
    ----------
    traces : array_like
        ADC traces with shape (N_du, N_channels, N_samples), e.g.
        ``TADC.trace_ch`` or the output of :meth:`grand.ADC.process`.
    trigger_config : dict, optional
        Trigger parameters overriding :data:`DEFAULT_T1_CONFIG`.  The defaults
        are the offline script's and are to be confirmed by the trigger group.
    channels : sequence of int, optional
        The channels evaluated; a DU triggers if any of them does.
    baseline : float, optional
        See :func:`extract_trigger_parameters`.

    Returns
    -------
    numpy.ndarray of bool
        One value per DU.
    """
    # One DU's (3, N) channels were taken for three DUs of one channel each,
    # and none triggered (#289)
    if np.ndim(traces) != 3:
        raise ValueError("GRANDlib: t1_du_triggers: traces must have shape (N_du, N_channels, "
                         "N_samples), got %s; for one unit's channels, pass traces[None]"
                         % (np.shape(traces),))
    # A NaN trace still triggered (#288)
    for index, du in enumerate(traces):
        if not np.all(np.isfinite(np.asarray(du, dtype=float))):
            raise ValueError("GRANDlib: t1_du_triggers: the trace of unit %d (index in the event) holds "
                             "NaN or inf" % index)
    return np.array(
        [any(t1_channel_trigger(du[ch], trigger_config, baseline) for ch in channels)
         for du in traces],
        dtype=bool,
    )


def t1_trigger_flags(traces, trigger_config=None, channels=DEFAULT_T1_CHANNELS,
                     baseline=0):
    r"""The ``trigger_flag`` values for every DU of an event.

    :data:`T1_TRIGGER_FLAG` for a DU that passes :func:`t1_du_triggers`, 0
    otherwise.  The encoding is to be confirmed by the trigger group.

    Returns
    -------
    numpy.ndarray of numpy.ushort
    """
    passed = t1_du_triggers(traces, trigger_config, channels, baseline)
    return np.where(passed, T1_TRIGGER_FLAG, 0).astype(np.ushort)


def t1_config_from_params(params):
    r"""The T1 trigger parameters, from ``KEY=VALUE`` strings.

    Parameters
    ----------
    params : list of str or None
        Overrides of :data:`grand.sim.detector.trigger.DEFAULT_T1_CONFIG`,
        e.g. ``['th1=120', 'nc_max=10']``.  The values are integers.

    Returns
    -------
    dict
        The full set of trigger parameters.

    Raises
    ------
    ValueError
        For a string that is not ``KEY=VALUE`` with an integer value, an
        unknown key, or values that cannot describe a trigger (``th2 > th1``,
        ``nc_min > nc_max``, a negative window; #267).
    """
    config = dict(DEFAULT_T1_CONFIG)
    for param in params or []:
        key, sep, value = param.partition('=')
        key = key.strip()
        if not sep or key not in DEFAULT_T1_CONFIG:
            raise ValueError(f'Bad --t1_param {param!r}: expected KEY=VALUE with KEY in {sorted(DEFAULT_T1_CONFIG)}')
        try:
            config[key] = int(value)
        except ValueError:
            raise ValueError(f'Bad --t1_param {param!r}: the value must be an integer') from None
    _check_config(config, "--t1_param")
    return config
