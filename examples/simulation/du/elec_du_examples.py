'''

'''

import matplotlib.pyplot as plt
import grand.manage_log as mlg

import grand.simu.du.elec_du as edu
 

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)


def StandingWaveRatioGP300_show_s11():
    o_s11 = edu.StandingWaveRatioGP300()
    o_s11.compute_s11()
    o_s11.plot_vswr()
    


def LowNoiseAmplificatorGP300_show_s21():
    o_s11 = edu.StandingWaveRatioGP300()
    o_s11.compute_s11()
    o_lna  = edu.LowNoiseAmplificatorGP300(2048)
    o_lna.update_with_s11(o_s11.s11)
    o_lna.plot_z()
    o_lna.plot_lna()
    

if __name__ == '__main__':
    logger.info(mlg.string_begin_script())
    #=============================================
    #StandingWaveRatioGP300_show_s11()
    LowNoiseAmplificatorGP300_show_s21()
    #=============================================    
    logger.info(mlg.string_end_script())
    plt.show()
