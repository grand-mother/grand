#! /usr/bin/env python3

import argparse
from grand.dataio.root_files import get_file_event, _get_ttree_in_file
from grand.basis.traces_event import Handling3dTraces
import grand.manage_log as mlg
import matplotlib.pylab as plt


# plt.ion()

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("error", log_stdout=True)


def manage_args():
    parser = argparse.ArgumentParser(description="Information and plot event/traces for ROOT file")
    parser.add_argument(
        "file", help="path and name of ROOT file GRAND", type=argparse.FileType("r")
    )
    parser.add_argument(
        "-f",
        "--footprint",
        help="interactive plot (double click) of footprint, max value for each DU",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--time_val",
        help="interactive plot, value of each DU at time t defined by a slider",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--trace",
        type=int,
        help="plot trace x,y,z and power spectrum of detector unit (DU)",
        default=-100,
    )
    parser.add_argument(
        "--idx_evt",
        type=int,
        help="Select event with index <idx_evt>, given by -i option, idx_evt is always > 0 or = 0",
        default=-100,
    )
    parser.add_argument(
        "--trace_image",
        action="store_true",
        required=False,
        help="interactive image plot (double click) of norm of traces",
    )
    parser.add_argument(
        "--list_du",
        action="store_true",
        required=False,
        help="list of identifier of DU",
    )
    parser.add_argument(
        "--list_ttree",
        action="store_true",
        required=False,
        help="list of TTree present in file",
    )
    parser.add_argument(
        "--dump",
        type=int,
        default=-100,
        help="dump trace of DU",
    )  # retrieve argument
    parser.add_argument(
        "-i",
        "--info",
        action="store_true",
        required=False,
        help="some information about the contents of the file",
    )  # retrieve argument
    return parser.parse_args()


def main():
    #
    logger.info("Example script to deal with 3D traces.")

    args = manage_args()
    d_event = get_file_event(args.file.name)
    if args.idx_evt != -100:
        if args.id_evt < 0:
            logger.error("index events must >= 0")
            return
        if args.id_evt >= d_event.get_nb_events():
            logger.error(f"index events must < {d_event.get_nb_events()}")
            return
        d_event.load_event_idx(args.idx_evt)
    o_tevent = d_event.get_obj_handling3dtraces()
    if args.info:
        print(f"Nb DU         : {d_event.get_du_count()}")
        print(f"Size trace    : {d_event.get_size_trace()}")
        print(f"Nb events     : {d_event.get_nb_events()}")
        print("Idx\trun\tevent")
        for idx, evt in enumerate(d_event.l_events):
            print(f"{idx}\t{evt[1]}\t{evt[0]}")
    assert isinstance(o_tevent, Handling3dTraces)
    if args.list_du:
        print(f"Identifier DU : {o_tevent.d_idxdu.keys()}")
    if args.trace_image:
        o_tevent.plot_all_traces_as_image()
    if args.footprint:
        o_tevent.plot_footprint_4d_max()
        o_tevent.plot_footprint_val_max()
    if args.time_val:
        o_tevent.plot_footprint_time_slider()
    if args.trace != -100:
        if not args.trace in o_tevent.idt2idx.keys():
            logger.error(f"ERROR: unknown DU identifer")
            return
        o_tevent.plot_trace_du(args.trace)
        o_tevent.plot_psd_trace_du(args.trace)
    if args.list_ttree:
        print(_get_ttree_in_file(args.file.name))
    if args.dump != -100:
        if not args.dump in o_tevent.idt2idx.keys():
            logger.error(f"ERROR: unknown DU identifer")
            return
        idx_du = o_tevent.idt2idx[args.dump]
        tr_du = o_tevent.traces[idx_du]
        t_tr = o_tevent.t_samples[idx_du]
        for idx in range(o_tevent.get_size_trace()):
            print(f"{t_tr[idx]} {tr_du[0,idx]} {tr_du[1,idx]} {tr_du[2,idx]}")


if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # =============================================
    main()
    # =============================================
    plt.show()
    logger.info(mlg.string_end_script())
