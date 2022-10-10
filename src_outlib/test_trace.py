'''

'''

from grand.store.trace import *
import matplotlib.pyplot as plt


G_file_efield = "/home/dc1/Coarse1.root"


def show_trace():
    d_efield= EfieldDC1()
    d_efield.read_efield_event(G_file_efield)
    plt.plot(d_efield.a3_trace[10,1])
    
    
def show_pos_det():
    d_efield= EfieldDC1()
    d_efield.read_efield_event(G_file_efield)
    plt.figure()
    for idx_du in range(d_efield.nb_du):
        print(d_efield.a3_pos[idx_du, 0], d_efield.a3_pos[idx_du, 1])
        plt.plot(d_efield.a3_pos[idx_du, 0], d_efield.a3_pos[idx_du, 1], "*")
    

if __name__ == '__main__':
    #show_trace()
    show_pos_det()
    plt.show()
    