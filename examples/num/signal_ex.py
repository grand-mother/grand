"""
TEMPLATE WITH LOGGER INITIALISED
"""

import matplotlib.pyplot as plt
import numpy as np

import grand.manage_log as mlg
import grand.num.signal as gsig

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

G_freq_step = 0.5


def f2idx(freq, step=G_freq_step):
    return int(freq / step + 0.5)


def idx2f(idx, step):
    return idx * step


def plot_value(freq, value, s_tle=""):
    plt.figure()
    plt.title(s_tle)
    plt.plot(freq, value, "*")
    plt.xlabel("MHz")
    plt.grid()



if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # =============================================
    # to do
    # =============================================
    logger.info(mlg.string_end_script())
    plt.show()
