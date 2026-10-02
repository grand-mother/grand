#! /usr/bin/env python3

"""
Plots where in each trace its maximum falls, against the maximum.

JM Colley CNRS/IN2P3/LPNHE
"""
import argparse
import os

import numpy as np


def main(argv=None):
    # It read sys.argv directly: -h was taken for a file name, a bad index
    # silently became 0, and nothing was saved without a display (#246)
    parser = argparse.ArgumentParser(description="Plot the time and value of each trace's maximum.")
    parser.add_argument("efield_file", help="e-field file (TEfield), with its run and shower files beside it")
    parser.add_argument("event_index", nargs="?", type=int, default=0, help="event index (default 0)")
    parser.add_argument("--savefig", metavar="FILE", help="save the figure to FILE instead of showing it")
    args = parser.parse_args(argv)
    if not os.path.isfile(args.efield_file):
        raise SystemExit("GRANDlib: plot_tmax_vmax: no such file: %s" % args.efield_file)
    if args.event_index < 0:
        raise SystemExit("GRANDlib: plot_tmax_vmax: the event index must be >= 0, got %d" % args.event_index)

    import matplotlib
    if args.savefig:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import grand.dataio.root_files as froot

    try:
        ef3d = froot.get_handling3dtraces(args.efield_file, args.event_index)
    except Exception as error:      # the reader's own messages name the problem
        raise SystemExit("GRANDlib: plot_tmax_vmax: %s" % error)
    t_max, v_max = ef3d.get_tmax_vmax()

    plt.figure()
    plt.title(args.efield_file)
    t_tot = ef3d.get_size_trace() * ef3d.get_delta_t_ns()[0]
    t_max_rel = t_max - ef3d.t_samples[:, 0]
    plt.scatter(100 * t_max_rel / t_tot, v_max)
    plt.xlim([0, 100])
    plt.xlabel("Position of time of max in trace, % ")
    plt.ylabel(r"Max value of trace, $\mu$v/m ")
    plt.hlines(25, 0, 100, label="level of galaxy background", linestyles="-")
    v_inf = np.min(v_max) / 2
    v_sup = np.max(v_max)
    plt.vlines(15, v_inf, v_sup, label="inf border for max", linestyles="-.")
    plt.vlines(55, v_inf, v_sup, label="sup border for max", linestyles="--")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    if args.savefig:
        plt.savefig(args.savefig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
