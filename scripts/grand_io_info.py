#! /usr/bin/env python3

import argparse
from grand.io.root_file import FileSimuEfield
from grand.basis.traces_event import HandlingTracesOfEvent
import grand.manage_log as mlg
import matplotlib.pylab as plt


#plt.ion()

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=False)


def main():
    parser = argparse.ArgumentParser(
        description='Information and plot from ROOT file for GRAND ')
    parser.add_argument('file',
                        help='path and name of ROOT file GRAND',
                        type=argparse.FileType('r'))
    parser.add_argument('-n', '--net',
                        help='plot detector unit network', action="store_true", required=False)
    parser.add_argument('-t', '--trace', type=int,
                        help='plot trace x,y,z of detector unit (DU) at index TRACE (0<= TRACE < Nb DU)',
                        default=-100)
    parser.add_argument('--trace_norm', action="store_true", required=False,
                        help='plot norm of all traces as image ')
    # retrieve argument
    args = parser.parse_args()
    # 
    logger.info(mlg.chrono_start())
    d_efield = FileSimuEfield(args.file.name)
    logger.info(mlg.chrono_string_duration())
    print(f"Nb events  : {d_efield.get_nb_events()}")
    print(f"Nb DU      : {d_efield.get_nb_du()}")
    print(f"Size trace : {d_efield.get_size_trace()}")
    o_tevent = d_efield.get_obj_handlingtracesofevent()
    assert isinstance(o_tevent, HandlingTracesOfEvent)
    if args.trace_norm:
        o_tevent.plot_traces_norm()
    if args.net:
        #o_tevent.network.plot_du_pos()
        o_tevent.define_t_samples()
        #o_tevent.network.plot_values(o_tevent.get_max_abs(),"Max |Efield_i|")
        o_tevent.network.plot_values(o_tevent.get_max_norm(),"Max ||Efield||")        
        o_tevent.plot_histo_t_start()
        if True:
            # work in progress
            a_time, a_values = o_tevent.get_common_time_trace()
            o_tevent.network.plot_trace_time(a_time, a_values, "test")
    if args.trace != -100:
        if (0 > args.trace) or  (args.trace >= d_efield.get_nb_du()):
            print(f"ERROR: index of the trace must be >= 0 and <= {d_efield.get_nb_du()-1}")
            return
        o_tevent.plot_trace_idx(args.trace)
    
if __name__ == '__main__':
    logger.info(mlg.string_begin_script())
    #=============================================
    main()    
    #=============================================
    plt.show()   
    logger.info(mlg.string_end_script())
    