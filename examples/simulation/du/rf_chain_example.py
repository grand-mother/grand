"""

"""

import matplotlib.pyplot as plt
import scipy.fft as sf
import numpy as np

import grand.manage_log as mlg
import grand.simu.du.rf_chain as edu


# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)


def plot_csv_file(
    f_name,
    col_x,
    col_y,
    x_label="MHz",
    y_label="TBD",
):  # pragma: no cover
    plt.figure()
    plt.title(f"{f_name}, x:{col_x}, y:{col_y}")
    data = np.loadtxt(f_name)
    plt.plot(data[:, col_x], data[:, col_y])
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid()


def plot_csv_file_abs(
    f_name,
    col_x,
    col_y1,
    col_y2,
    x_label="MHz",
    y_label="TBD",
):  # pragma: no cover
    plt.figure()
    plt.title(f"{f_name}, x:{col_x}, ")
    data = np.loadtxt(f_name)
    plt.plot(data[:, col_x], np.abs(data[:, col_y1] + 1j * data[:, col_y2]))
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.grid()


def StandingWaveRatioGP300_show_s11():
    o_s11 = edu.StandingWaveRatioGP300()
    out_f = sf.rfftfreq(1024, 0.5e-9) / 1e6
    o_s11.set_out_freq_mhz(out_f)
    o_s11.compute_s11()
    o_s11.plot_vswr()


def LowNoiseAmplificatorGP300_show_s21():
    dt = 0.5e-9
    out_f = sf.rfftfreq(2048, 0.5e-9) / 1e6
    o_lna = edu.LowNoiseAmplificatorGP300()
    o_lna.compute_for_freqs(out_f)
    o_lna._vswr.plot_vswr()
    o_lna.plot_z()
    o_lna.plot_gama()
    o_lna.plot_lna()


def CableGP300_plot():
    dt = 0.5e-9
    out_f = sf.rfftfreq(2048, 0.5e-9) / 1e6
    o_cbl = edu.CableGP300()
    o_cbl.compute_for_freqs(out_f)
    o_cbl.plot_cable()


def FilterGP300_plot():
    dt = 0.5e-9
    out_f = sf.rfftfreq(2048, 0.5e-9) / 1e6
    o_cbl = edu.VgaFilterBalunGP300()
    o_cbl.compute_for_freqs(out_f)
    o_cbl.plot_filter()


def plot_rho_kernel():
    out_f = sf.rfftfreq(1000, 0.5e-9) / 1e6
    print(out_f)
    o_lna = edu.LowNoiseAmplificatorGP300()
    o_lna.compute_for_freqs(out_f)
    o_lna.get_fft_rho_3d()
    o_lna.plot_rho_kernel()


def plot_total_kernel():
    out_f = sf.rfftfreq(1000, 0.5e-9) / 1e6
    print(out_f)
    rfchain = edu.RfChainGP300()
    rfchain.compute_for_freqs(out_f)
    rfchain.plot_kernel()
    rfchain.plot_tf()


if __name__ == "__main__":
    logger.info(mlg.string_begin_script())
    # =============================================
    StandingWaveRatioGP300_show_s11()
    LowNoiseAmplificatorGP300_show_s21()
    plot_rho_kernel()
    CableGP300_plot()
    FilterGP300_plot()
    plot_total_kernel()
    # =============================================
    logger.info(mlg.string_end_script())
    plt.show()
