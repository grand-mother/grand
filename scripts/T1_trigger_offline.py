#!/usr/bin/env python3
r"""Offline DAQ-style T1 trigger on the ADC traces of a GrandRoot file.

Writes ``<file>.trigger.txt``, the entries in which at least one DU passes T1.

The algorithm and its default parameters now live in
:mod:`grand.sim.detector.trigger`; see there for what the trigger does and what
the trigger group still has to confirm.  Before issue #139 this script only
evaluated the first DU of each entry (``trace_ch[0]``); every DU is evaluated
now.

TO RUN:
    python T1_trigger_offline.py <adc.root>
"""
import sys

import numpy as np

import grand.dataio
from grand.sim.detector.trigger import (DEFAULT_T1_CHANNELS, DEFAULT_T1_CONFIG,
                                        extract_trigger_parameters,  # noqa: F401  (kept importable from here)
                                        t1_du_triggers)

# Kept under its old name for anyone importing it from this script.
dict_trigger_parameter = dict(DEFAULT_T1_CONFIG)


def triggered_entries(tadc, trigger_config=None, channels=DEFAULT_T1_CHANNELS):
    r"""The entries of `tadc` in which at least one DU passes T1."""
    trigger_index = []
    for k in range(tadc.get_number_of_entries()):
        tadc.get_entry(k)
        if np.any(t1_du_triggers(tadc.trace_ch, trigger_config, channels)):
            trigger_index.append(k)
    return trigger_index


if __name__ == "__main__":
    fname = sys.argv[1]
    file = grand.dataio.DataFile(fname)
    n_entries = file.tadc.get_number_of_entries()

    trigger_index = triggered_entries(file.tadc, dict_trigger_parameter)

    print(f"{fname}: {len(trigger_index)} out of {n_entries} triggered.")
    if len(trigger_index) > 0:
        np.savetxt(f"./{fname.split('/')[-1]}.trigger.txt", trigger_index, delimiter=', ', fmt='%d', header=str(dict_trigger_parameter))
