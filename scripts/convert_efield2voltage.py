#! /usr/bin/env python3
"""
Script to compute voltage from electric field.
Electric field traces are provided in a ROOT file.

To Run:
    python3 convert_efield2voltage.py <efield.root> -o <output.root> # # RF chain and noise added automatically.
    python3 convert_efield2voltage.py <efield.root> -o <output.root> --seed 0 --lst 10
    python3 convert_efield2voltage.py <efield.root> -o <output.root> --no_noise --no_rf_chain # Voc without noise and RF chain
    python3 convert_efield2voltage.py <efield.root> -o <output.root> --du_type GP300_nec (or GP300 or GP300_mat)
    python3 convert_efield2voltage.py <efield.root> -o <output.root> --no_rf_chain --rf_chain_nut #Calculates Voltage at the Nut (after LNA)
    python3 convert_efield2voltage.py <efield.root> -o <output.root> --no_rf_chain --rf_chain_gaa #RF chain in G@Auger

In this file:
    signal = Efield2Voltage(efield.root, out_voltage.root, seed=seed, padding_factor=args.padding_factor)
    signal.compute_voltage()    # saves automatically

    Options that can be given to compute_voltage() depending on what you want to compute.
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


Computing voltage with your own function to compute Voc.
    signal = Efield2Voltage(efield.root, out_voltage.root, seed=seed, padding_factor=args.padding_factor, du_type='GP300')
    my_volt = my_function(....)
    signal.voc = my_volt.voc
    signal.voc_f = my_volt.voc_f
    signal.vout = my_volt.vout
    signal.vout_f = my_volt.vout_f
    Note: make sure that the frequency bins in my_function() is equal to signal.freqs_mhz.
    signal.add(noise)          # make sure shape of noise broadcasts with the shape of signal.voc_f.
    signal.multiply(rf_chain)  # make sure shape of rf_chain broadcasts with the shape of signal.vout_f.

June 2023, JM and RK.
Feb 2024 modified by SN to add antenna model selection.
"""
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
        "directory",
        help="Simulation output data directory in GRANDROOT format.",
        # type=argparse.FileType("r"),
    )
    # parser.add_argument(
    #     "file",
    #     help="Efield input data file in GRANDROOT format.",
    #     type=argparse.FileType("r"),
    # )
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
        "--rf_chain_nut",
        action="store_true",
        default=False,
        help="add RF chain in antenna nut",
    )

    parser.add_argument(
        "--rf_chain_gaa",
        action="store_true",
        default=False,
        help="add RF chain for G@Auger setup",
    )

    parser.add_argument(
        "-o",
        "--out_file",
        default=None,
        help="output file in GRANDROOT format. If the file exists it is overwritten.",
        # required=True,
        # PB with option ???
        # type=argparse.FileType("w"),
    )
    parser.add_argument(
        "-od",
        "--out_directory",
        default=None,
        help="output directory in GRANDROOT format. If not given, is it the same as input directory",
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
    parser.add_argument(
        "--du_type",
        type=str,
        default='GP300',
        help="Choose between 4 different antenna models, GP300 -using hfss simulations, GP300_nec -using nec simulations, GP300_mat -using matlab simulations, Horizon",
    )
    parser.add_argument(
        "--target_duration_us",
        type=float,
        default=0,
        help="Adujust (and override) padding factor in order to get a signal of the given duration, in us",
    )
    parser.add_argument(
        "--target_sampling_rate_mhz",
        type=float,
        default=0,
        help="Target sampling rate of the data in Mhz",
    )
    parser.add_argument(
        "--add_jitter_ns",
        type=float,
        default=0,
        help="level of gaussian jitter (ns) to add to the trigger times",
    )
    parser.add_argument(
        "--calibration_smearing_sigma",
        type=float,
        default=0,
        help="Smear the stations amplitude calibrations with a gaussian centered in 1 and this input sigma",
    )
    # retrieve argument
    return parser.parse_args()


if __name__ == "__main__":
    import argparse
    from typing import Union
    import numpy as np

    import grand.manage_log as mlg
    #from grand import Efield2Voltage
    from grand.sim.efield2voltage import Efield2Voltage

    # specific logger definition for script because __mane__ is "__main__" !
    logger = mlg.get_logger_for_script(__file__)

    logger.info("Computing voltage from the input electric field.")

    args = manage_args()

    # If no output directory given, define it as input directory
    if args.out_directory is None:
        args.out_directory = args.directory

    # define a handler for logger : standard only
    mlg.create_output_for_logger(args.verbose, log_stdout=True)
    logger.info(mlg.string_begin_script())
    # =============================================
    seed = None if args.seed==-1 else args.seed
    logger.info(f"seed used for random number generator is {seed}.")

    # signal = Efield2Voltage(args.file.name, args.out_file, seed=seed, padding_factor=args.padding_factor)
    signal = Efield2Voltage(args.directory, args.out_file, output_directory=args.out_directory, seed=seed, padding_factor=args.padding_factor, du_type=args.du_type)
    signal.params["add_noise"]    = args.no_noise
    signal.params["add_rf_chain"] = args.no_rf_chain
    signal.params["lst"]          = args.lst
    signal.params["add_rf_chain_nut"] = args.rf_chain_nut
    signal.params["add_rf_chain_gaa"] = args.rf_chain_gaa
    signal.params["resample_to_mhz"]=args.target_sampling_rate_mhz
    signal.params["extend_to_us"]=args.target_duration_us
    signal.params["calibration_smearing_sigma"]=args.calibration_smearing_sigma
    signal.params["add_jitter_ns"]=args.add_jitter_ns
    #signal.compute_voltage_event(0)
    #signal.save_voltage(append_file=False)
    signal.compute_voltage()    # saves automatically

    # =============================================
    logger.info(mlg.string_end_script())
