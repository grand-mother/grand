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


def complex_expansion_use():
    size_out = 2000
    f_step = G_freq_step
    f_start = 30
    f_cut = 240
    f_end = 250
    size_in = f2idx(f_end - f_start) + 1
    logger.info(f"{size_in}")
    data = np.zeros(size_in, dtype=np.complex64)
    freq_in = np.linspace(f_start, f_end, size_in)
    assert freq_in.size == size_in
    data[:] = 1.0 - 1j
    plot_value(freq_in, np.abs(data))
    freq, d_xpn = gsig.complex_expansion(size_out, f_step, f_start, f_cut, data)
    plot_value(freq, np.abs(d_xpn))


if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # =============================================
    complex_expansion_use()
    # =============================================
    logger.info(mlg.string_end_script())
    plt.show()
