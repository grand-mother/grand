#! /usr/bin/env python3

from grand.io.root.file.simu_efield_gerb import FileSimuEfield
import matplotlib.pyplot as plt

plt.ioff()
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='get information and plot from file root GRAND ')
    parser.add_argument('file',
                        help='path and name file GRAND',
                        type=argparse.FileType('r'))
    parser.add_argument('-n', '--net',
                        help='plot detector unit network',action="store_true", required=False)
    parser.add_argument('-t', '--trace',type=int,
                        help='plot trace x,y,z of unit network with index TRACE (>=0)', required=False,default=-1)
    # retrieve argument
    args = parser.parse_args()
    
    # 
    d_efield= FileSimuEfield(args.file.name)
    print(f"Nb events: {d_efield.get_nb_events()}")
    print(f"Nb du : {d_efield.get_nb_du()}")
    print(f"Size trace : {d_efield.get_size_trace()}")
    if args.net:
        plt.figure()
        for du_idx in range(d_efield.nb_du):
            #print(d_efield.du_pos[du_idx, 0], d_efield.du_pos[du_idx, 1])
            plt.plot(d_efield.du_pos[du_idx, 0], d_efield.du_pos[du_idx, 1], "*")
        plt.show()
    if args.trace >=0:
        if args.trace > d_efield.get_nb_du():
            print(f"idx trace must be between 0 and {d_efield.get_nb_du()-1}!")
            return
        idx = args.trace
        t_trace=d_efield.get_time_trace_ns()
        plt.figure()
        plt.title(f"Trace at index {idx}, identifier DU associted is {d_efield.tt_efield.du_id[idx]} ")
        plt.plot(t_trace[idx], d_efield.traces[idx,0],label='x')
        plt.plot(t_trace[idx], d_efield.traces[idx,1],label='y')
        plt.plot(t_trace[idx], d_efield.traces[idx,2],label='z')
        plt.xlabel("[ns]")
        plt.grid()
        plt.legend()
        plt.show()
    

if __name__ == '__main__':
    plt.show()
    main()