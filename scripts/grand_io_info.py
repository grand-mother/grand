#####! /usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from grand.io.root_file import FileSimuEfield
from grand.basis.traces_event import HandlingTracesOfEvent




def main():
    parser = argparse.ArgumentParser(
        description='Information and plot from ROOT file for GRAND ')
    parser.add_argument('file',
                        help='path and name of ROOT file GRAND',
                        type=argparse.FileType('r'))
    parser.add_argument('-n', '--net',
                        help='plot detector unit network', action="store_true", required=False)
    parser.add_argument('-t', '--trace', type=int,
                        help='plot trace x,y,z of detector unit (DU) at index TRACE (0<= TRACE < Nb DU)', required=False, default=-100)
    # retrieve argument
    args = parser.parse_args()
    # 
    d_efield = FileSimuEfield(args.file.name)
    print(f"Nb events  : {d_efield.get_nb_events()}")
    print(f"Nb DU      : {d_efield.get_nb_du()}")
    print(f"Size trace : {d_efield.get_size_trace()}")
    o_tevent = d_efield.get_obj_handlingtracesofevent()
    assert isinstance(o_tevent, HandlingTracesOfEvent)
    if args.net:
        #o_tevent.network.plot_du_pos()
        #o_tevent.network.plot_value(o_tevent.get_max_abs(),"Max abs |Efield|")
        #o_tevent.network.plot_value(o_tevent.get_max_norm(),"Max norm ||Efield||")        
        #o_tevent.plot_histo_t_start()
        o_tevent.define_t_samples()
        a_time = o_tevent.t_samples[0]
        a_values = o_tevent.get_norm()+1
        o_tevent.network.plot_trace_time(a_time, a_values, "demo with but NOK!!")
    if args.trace != -100:
        if (0 > args.trace) or  (args.trace >= d_efield.get_nb_du()):
            print(f"ERROR: index of the trace must be >= 0 and <= {d_efield.get_nb_du()-1}")
            return
        o_tevent.plot_trace_idx(args.trace)
    
if __name__ == '__main__': 
    main()
    plt.show()
    