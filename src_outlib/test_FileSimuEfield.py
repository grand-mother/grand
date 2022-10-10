'''

'''

from grand.io.root_file import FileSimuEfield
import matplotlib.pyplot as plt
import grand.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

G_file_efield = "/home/dc1/Coarse6.root"


def show_trace(idx):
    d_efield= FileSimuEfield(G_file_efield)
    t_trace=d_efield.get_time_trace_ns()
    plt.figure()
    plt.plot(t_trace[idx], d_efield.traces[idx,0])
    plt.plot(t_trace[idx], d_efield.traces[idx,1])
    plt.plot(t_trace[idx], d_efield.traces[idx,2])
    
def show_pos_det():
    d_efield= FileSimuEfield(G_file_efield)
    plt.figure()
    for du_idx in range(d_efield.nb_du):
        print(d_efield.du_pos[du_idx, 0], d_efield.du_pos[du_idx, 1])
        plt.plot(d_efield.du_pos[du_idx, 0], d_efield.du_pos[du_idx, 1], "*")

def show_time_det():
    d_efield= FileSimuEfield(G_file_efield)
    t_trace=d_efield.get_time_trace_ns()
    print(t_trace.shape)
    print(t_trace[0,:10])
    print(t_trace[1,:10])
    print(t_trace.dtype)
    print(t_trace.shape)

def test_read():
    FileSimuEfield(G_file_efield)
    logger.info("End of creating")
     
def test_read_2_time():
    d_efield= FileSimuEfield(G_file_efield)
    d_efield.load_event_idx(0)
    
def test_histo_t_start():
    d_efield= FileSimuEfield(G_file_efield)
    o_trevt = d_efield.get_obj_handlingtracesofevent()
    o_trevt.plot_histo_t_start()
    
if __name__ == '__main__':
    logger.info(mlg.string_begin_script())
    #=============================================
    test_read()
    #test_read_2_time()
    #show_trace(38)
    # show_trace(1)
    #show_pos_det()
    #show_time_det()
    #test_histo_t_start()
    #=============================================    
    logger.info(mlg.string_end_script())
        
    plt.show()