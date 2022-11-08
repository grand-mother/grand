#! /usr/bin/env python3

import argparse

import grand.manage_log as mlg
from grand.simu.master_simu import MasterSimuDetectorWithRootIo


# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)


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
        #type=argparse.FileType("w"),
    )
    parser.add_argument(
        "--verbose",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="logger verbosity.",
    )
    # retrieve argument
    return parser.parse_args()


if __name__ == "__main__":
    args = manage_args()
    # define a handler for logger : standard only
    mlg.create_output_for_logger(args.verbose, log_stdout=True)
    logger.info(mlg.string_begin_script())
    # =============================================
    master = MasterSimuDetectorWithRootIo(args.file.name)
    master.simu_du.set_flag_add_noise(args.no_noise)
    master.compute_event_idx(0)
    master.save_voltage(args.out_file, append_file=False)
    # =============================================
    logger.info(mlg.string_end_script())
