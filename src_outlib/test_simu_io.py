'''

'''

from grand.simu.all_du import MasterSimuDetectorWithRootIo
import matplotlib.pyplot as plt
import grand.manage_log as mlg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_stdout=True)

G_file_efield = "/home/dc1/Coarse1.root"


def test_Voc(idx):
    m_ios = MasterSimuDetectorWithRootIo(G_file_efield)
    v_oc = m_ios.compute_du_in_event(idx, 0)
    plt.figure()
    t_trace = m_ios.simu_du.du_time_efield[idx]
    print(t_trace.shape)
    print(v_oc[0].V.shape)
    plt.title("Voltage")
    plt.plot(t_trace[:-1], v_oc[0].V, label="V sn")
    plt.plot(t_trace[:-1], v_oc[1].V, label="V we")
    plt.plot(t_trace[:-1], v_oc[2].V, label="V _z")
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


if __name__ == '__main__':
    logger = mlg.get_logger_for_script(__file__)
    logger.info(mlg.string_begin_script())
    # ================
    test_Voc(26)
    # ================
    logger.info(mlg.string_end_script())
    plt.show()
