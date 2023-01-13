'''
TEMPLATE WITH LOGGER INITIALISED
'''

import matplotlib.pyplot as plt

import grand.manage_log as mlg
from grand.io.root_trees import EfieldEventTree

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

G_file_efield = "/home/dc1/Coarse1.root"

def check_same_size_trace(f_root):
    logger.info(f'\nCheck {f_root}')
    d_ef = EfieldEventTree(f_root)
    ev_n, run_n = d_ef.get_list_of_events()[0]
    logger.info(f'load event={ev_n} run={run_n}')
    d_ef.get_event(ev_n, run_n)
    l_size_traces = []
    size_trace = len(d_ef.trace_x[0])
    logger.info(f'Size first trace: {size_trace}')
    plot_histo = False
    for idx_du in range(1, d_ef.du_count):
        size = len(d_ef.trace_x[idx_du])
        if size != size_trace:
            plot_histo = True
        l_size_traces.append(size)
    if plot_histo:
        plt.figure()
        plt.title(f'{f_root}')
        plt.hist(l_size_traces)
    else:
        logger.info(f'Same size.')

def check_all_dc1():
    check_same_size_trace("/home/dc1/Coarse1.root")
    check_same_size_trace("/home/dc1/Coarse2.root")
    check_same_size_trace("/home/dc1/Coarse3.root")
    check_same_size_trace("/home/dc1/Coarse4.root")
    check_same_size_trace("/home/dc1/Coarse5.root")
    check_same_size_trace("/home/dc1/Coarse6.root")
    

if __name__ == '__main__':
    logger.info(mlg.string_begin_script())
    #=============================================
    check_all_dc1()
    #=============================================    
    logger.info(mlg.string_end_script())
    plt.show()
