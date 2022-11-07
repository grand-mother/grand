'''

'''
import os 
import time

from grand.simu.master_simu import MasterSimuDetectorWithRootIo
import matplotlib.pyplot as plt
import grand.manage_log as mlg
from grand.io.root_trees import VoltageEventTree
import numpy as np 

np.random.seed(11)

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("info", log_stdout=True, log_file=None)

G_file_efield = "/home/dc1/Coarse1.root"
G_file_efield = "/home/dc1/Coarse2_xmax_add.root"

def test_VoltageTTree():
    f_root = 'test_volt.root'
    try:
        os.remove(f_root)
    except:
        pass
    #time.sleep(1)
    ovol = VoltageEventTree()
    ovol.run_number = 0
    ovol.event_number = 0
    nb_du = 10
    ovol.du_count = nb_du
    for idx in range(nb_du):
        trace = np.arange(idx, idx + 5999, dtype=np.float64)
        ovol.trace_x.append((trace).tolist())
        ovol.trace_y.append((trace + 10).tolist())
        ovol.trace_z.append((trace + 100).tolist())
    ovol.fill()
    ovol.write(f_root)

    
def test_Voc_du(idx):
    m_ios = MasterSimuDetectorWithRootIo(G_file_efield)
    m_ios.compute_event_du_idx(0, idx)
    v_oc_we = m_ios.simu_du.voc[idx][1]
    plt.figure()
    t_trace = m_ios.simu_du.du_time_efield[idx]
    plt.title("Voltage 0-padding: size*1.2")
    plt.plot(t_trace, m_ios.simu_du.voc[idx][0], label="V sn")
    plt.plot(t_trace, m_ios.simu_du.voc[idx][1], label="V we")
    #plt.plot(t_trace, m_ios.simu_du.voc[idx][2], label="V _z")
    plt.grid()
    plt.legend()
    plt.figure()
    plt.title("V_LNA")
    plt.plot(t_trace, m_ios.simu_du.v_out[idx][0], label="V sn")
    plt.plot(t_trace, m_ios.simu_du.v_out[idx][1], label="V we")
    #plt.plot(t_trace, m_ios.simu_du.v_out[idx][2], label="V _z")
    plt.grid()
    plt.legend()
    plt.figure()
    plt.title("E field")
    efield = m_ios.simu_du.du_efield[idx]
    plt.plot(t_trace, efield[0], label="x")
    plt.plot(t_trace, efield[1], label="y")
    plt.plot(t_trace, efield[2], label="z")   
    plt.grid()
    plt.legend()


def test_V_out_event():
    m_ios = MasterSimuDetectorWithRootIo(G_file_efield)
    m_ios.compute_event_idx(0)
    m_ios.save_voltage("volt_c2.root")


def test_Voc_event_many():
    m_ios = MasterSimuDetectorWithRootIo(G_file_efield)
    m_ios.compute_event_idx(0)
    m_ios.compute_event_idx(0)
    m_ios.compute_event_idx(0)


if __name__ == '__main__':
    logger = mlg.get_logger_for_script(__file__)
    logger.info(mlg.string_begin_script())
    # ================
    #test_VoltageTTree()
    #test_Voc_du(26)
    test_V_out_event()
    # test_Voc_event_many()
    # ================
    logger.info(mlg.string_end_script())
    plt.show()
