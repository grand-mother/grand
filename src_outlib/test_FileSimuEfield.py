'''

'''

from grand.io.root.file.simu_efield_gerb import FileSimuEfield
import matplotlib.pyplot as plt



G_file_efield = "/home/dc1/Coarse1.root"


def show_trace(idx):
    d_efield= FileSimuEfield(G_file_efield)
    t_trace=d_efield.get_time_trace_ns()
    plt.figure()
    plt.plot(t_trace[idx], d_efield.traces[idx,1])
    
    
def show_pos_det():
    d_efield= FileSimuEfield(G_file_efield)
    plt.figure()
    for idx_du in range(d_efield.nb_du):
        print(d_efield.pos_du[idx_du, 0], d_efield.pos_du[idx_du, 1])
        plt.plot(d_efield.pos_du[idx_du, 0], d_efield.pos_du[idx_du, 1], "*")

def show_time_det():
    d_efield= FileSimuEfield(G_file_efield)
    t_trace=d_efield.get_time_trace_ns()
    print(t_trace.shape)
    print(t_trace[0,:10])
    print(t_trace[1,:10])
    print(t_trace.dtype)
    print(t_trace.shape)
    
def test_read_2_time():
    d_efield= FileSimuEfield(G_file_efield)
    d_efield.load_event_idx(0)
    
    
if __name__ == '__main__':
    test_read_2_time()
    show_trace(0)
    show_trace(1)
    show_pos_det()
    show_time_det()
    plt.show()
    pass