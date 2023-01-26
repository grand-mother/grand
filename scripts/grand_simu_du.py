#! /usr/bin/env python3

import argparse

import numpy as np

import grand.manage_log as mlg
from grand.simu.master_simu import MasterSimuDetectorWithRootIo


# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)


def check_float_day_hour(s_hour):
    f_hour = float(s_hour)
    if f_hour < 0 or f_hour > 24:
        raise argparse.ArgumentTypeError(f"lts must be > 0h and < 24h.")
    return f_hour


def manage_args():
    parser = argparse.ArgumentParser(
        description="Calculation of DU response in volt for first event in Efield input file."
    )
    parser.add_argument(
        "file",
        help="Efield input data file in GRANDROOT format.",
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--no_noise",
        action="store_false",
        default=True,
        help="don't add galactic noise.",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        help="output file in GRANDROOT format. If the file exists it is overwritten.",
        required=True,
        # PB with option ???
        # type=argparse.FileType("w"),
    )
    parser.add_argument(
        "--verbose",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="logger verbosity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default="-1",
        help="Fix the random seed to reproduce same noise, must be positive integer",
    )
    parser.add_argument(
        "--lst",
        type=check_float_day_hour,
        default=18.0,
        help="lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h.",
    )
    # retrieve argument
    return parser.parse_args()


if __name__ == "__main__":
    args = manage_args()
    # define a handler for logger : standard only
    mlg.create_output_for_logger(args.verbose, log_stdout=True)
    logger.info(mlg.string_begin_script())
    # =============================================
    if args.seed > 0:
        logger.info(f"Fix numpy random seed")
        np.random.seed(args.seed)
    master = MasterSimuDetectorWithRootIo(args.file.name)
    master.simu_du.set_flag_add_noise(args.no_noise)
    master.simu_du.set_local_sideral_time(args.lst)
    master.compute_event_idx(0)
    master.save_voltage(args.out_file, append_file=False)
    # =============================================
    logger.info(mlg.string_end_script())
