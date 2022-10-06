'''

'''

from grand.simu.master_simu import MasterSimuDetectorWithRootIo
import matplotlib.pyplot as plt
import grand.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_stdout=True, log_file="simu_with_rootio.txt")

G_file_efield = "/home/dc1/Coarse1.root"


def test_Voc_du(idx):
    m_ios = MasterSimuDetectorWithRootIo(G_file_efield)
    m_ios.compute_event_du_idx(0, idx)
    v_oc_we = m_ios.simu_du.voc_ew
    plt.figure()
    t_trace = m_ios.simu_du.du_time_efield[idx]
    plt.title("Voltage")
    #plt.plot(t_trace[:-1], v_oc[0].V, label="V sn")
    plt.plot(t_trace[:-1], v_oc_we.V, label="V we")
    #plt.plot(t_trace[:-1], v_oc[2].V, label="V _z")
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

def test_Voc_event():
    m_ios = MasterSimuDetectorWithRootIo(G_file_efield)
    m_ios.compute_event_idx(0)
    m_ios.save_voltage("out2.root")

def test_Voc_event_many():
    m_ios = MasterSimuDetectorWithRootIo(G_file_efield)
    m_ios.compute_event_idx(0)
    m_ios.compute_event_idx(0)
    m_ios.compute_event_idx(0)

if __name__ == '__main__':
    logger = mlg.get_logger_for_script(__file__)
    logger.info(mlg.string_begin_script())
    # ================
    #test_Voc_du(26)
    test_Voc_event()
    #test_Voc_event_many()
    # ================
    logger.info(mlg.string_end_script())
    plt.show()
