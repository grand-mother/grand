#! /usr/bin/env python3

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
        "--no_rf_chain",
        action="store_false",
        default=True,
        help="don't add RF chain.",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        default="",
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
        default=-1, # -1 as a placeholder for None to maintain int type.
        help="Fix the random seed to reproduce same galactic noise, must be positive integer",
    )
    parser.add_argument(
        "--lst",
        type=check_float_day_hour,
        default=18.0,
        help="lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h for the EW arm.",
    )
    parser.add_argument(
        "--padding_factor",
        type=float,
        default=1.0,
        help="Increase size of signal with zero padding, with 1.2 the size is increased of 20%%. ",
    )
    # retrieve argument
    return parser.parse_args()


if __name__ == "__main__":
    import argparse
    from typing import Union
    import numpy as np

    import grand.manage_log as mlg
    from grand.sim.efield2voltage import Efield2Voltage

    # specific logger definition for script because __mane__ is "__main__" !
    logger = mlg.get_logger_for_script(__file__)

    logger.info("Computing voltage from the input electric field.")

    args = manage_args()
    # define a handler for logger : standard only
    mlg.create_output_for_logger(args.verbose, log_stdout=True)
    logger.info(mlg.string_begin_script())
    # =============================================
    seed = None if args.seed==-1 else args.seed
    logger.info(f"seed used for random number generator is {seed}.")

    master = Efield2Voltage(args.file.name, args.out_file, seed=seed, padding_factor=args.padding_factor)
    master.params["add_noise"]    = args.no_noise
    master.params["add_rf_chain"] = args.no_rf_chain
    master.params["lst"]          = args.lst

    #master.compute_voltage_event(0)
    #master.save_voltage(append_file=False)
    master.compute_voltage()    # saves automatically

    # =============================================
    logger.info(mlg.string_end_script())
