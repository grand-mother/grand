#!/usr/bin/env python3
r"""Offline DAQ-style T1 trigger on the ADC traces of a GrandRoot file.

Writes the entries in which at least one DU passes T1 to a text file, by
default ``<file>.trigger.txt`` in the current folder.

The algorithm and its default parameters now live in
:mod:`grand.sim.detector.trigger`; see there for what the trigger does and what
the trigger group still has to confirm.  Before issue #139 this script only
evaluated the first DU of each entry (``trace_ch[0]``); every DU is evaluated
now.

TO RUN:
    python T1_trigger_offline.py <adc.root> [-o <list.txt>] [--t1_param th1=120 ...]
"""
import argparse
import os

import numpy as np

from grand.sim.detector.trigger import (DEFAULT_T1_CHANNELS, DEFAULT_T1_CONFIG,
                                        extract_trigger_parameters,  # noqa: F401  (kept importable from here)
                                        t1_config_from_params, t1_du_triggers)

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


def manage_args(argv=None):
    r"""Parses the command line: it read ``sys.argv[1]`` and opened ``-h`` as a file (#183)."""
    parser = argparse.ArgumentParser(description="Offline T1 trigger on the ADC traces (TADC) of a GrandRoot file.")
    parser.add_argument("adc_file", help="ADC file (TADC) to scan")
    parser.add_argument("-o", "--out", default=None,
                        help="text file for the list of triggered entries (default: <adc_file>.trigger.txt "
                             "in the current folder)")
    parser.add_argument("--t1_param", action="append", metavar="KEY=VALUE",
                        help="override a T1 parameter (repeatable), e.g. --t1_param th1=120. Defaults: %s"
                             % DEFAULT_T1_CONFIG)
    return parser.parse_args(argv)


if __name__ == "__main__":
    import grand.dataio

    args = manage_args()
    try:
        config = t1_config_from_params(args.t1_param)
    except ValueError as error:
        raise SystemExit("GRANDlib: T1_trigger_offline: %s" % error)
    if not os.path.isfile(args.adc_file):
        raise SystemExit("GRANDlib: T1_trigger_offline: no such file: %s" % args.adc_file)
    file = grand.dataio.DataFile(args.adc_file)
    n_entries = file.tadc.get_number_of_entries()

    trigger_index = triggered_entries(file.tadc, config)

    print(f"{args.adc_file}: {len(trigger_index)} out of {n_entries} triggered.")
    if len(trigger_index) > 0:
        out = args.out or os.path.basename(args.adc_file) + ".trigger.txt"
        np.savetxt(out, trigger_index, delimiter=', ', fmt='%d', header=str(config))
