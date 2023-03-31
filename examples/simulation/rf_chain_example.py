"""

"""
import numpy as np
import h5py
import scipy.fft as sf

import grand.simu.noise.rf_chain as grfc
from grand import grand_add_path_data
import grand.manage_log as mlg
import grand.simu.du.rf_chain as edu

import matplotlib.pyplot as plt
params = {
    "legend.fontsize": 14,
    "axes.labelsize": 22,
    "axes.titlesize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "figure.figsize": (10, 8),
    "axes.grid": False,
}
plt.rcParams.update(params)

freq_MHz = np.arange(30, 251, 1)

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standard only
mlg.create_output_for_logger("debug", log_stdout=True)

# To Run:
#   python3 plot_noise.py galactic_noise
#   options: [galactic, vswr, lna, vga, cable, rf_chain]

def plot(args="galactic", **kwargs):

    if args=="galactic":
        if 'lst' in kwargs.keys():
            lst = kwargs['lst']
        else:
            raise Exception("Provide LST info like plot('galactic_noise', lst=..)")

        lst = int(lst)

        gala_file = grand_add_path_data("sky/30_250galactic.mat")
        gala_show = h5py.File(gala_file, "r")
        gala_psd_dbm   = np.transpose(gala_show["psd_narrow_huatu"])
        # SL, dbm per MHz, P=mean(V*V)/imp with imp=100 ohms
        gala_power_dbm = np.transpose(gala_show["p_narrow_huatu"])
        # SL, microV per MHz, seems to be Vmax=sqrt(2*mean(V*V)), not std(V)=sqrt(mean(V*V)) 
        gala_voltage = np.transpose(gala_show["v_amplitude"])  
        # gala_power_mag = np.transpose(gala_show["p_narrow"])
        gala_freq = gala_show["freq_all"]

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        for l_g in range(3):
            plt.plot(gala_freq, gala_psd_dbm[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("PSD(dBm/Hz)", fontsize=15)
        plt.title("Galactic Noise PSD", fontsize=15)
        plt.grid(ls='--', alpha=0.3)

        plt.subplot(1, 3, 2)
        for l_g in range(3):
            plt.plot(gala_freq, gala_power_dbm[:, l_g, lst])
        # SL: gala_power_dbm = 1e6 * np.sqrt(2 * 100 * pow(10, gala_power_dbm/10) * 1e-3)
        plt.legend(["port X", "port Y", "port Z"], loc='upper right')
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Power(dBm)", fontsize=15)
        plt.title("Galactic Noise Power", fontsize=15)
        plt.grid(ls='--', alpha=0.3)

        plt.subplot(1, 3, 3)
        for l_g in range(3):
            plt.plot(gala_freq, gala_voltage[:, l_g, lst])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel("Voltage(uV)", fontsize=15)
        plt.title("Galactic Noise Voltage", fontsize=15)
        plt.tight_layout()
        plt.grid(ls='--', alpha=0.3)
        plt.subplots_adjust(top=0.85)
        plt.show()

    if args=='vswr':

        vswr = grfc.StandingWaveRatioGP300()
        vswr.compute_s11()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"VSWR, s11 parameter")
        ax1.set_title("db")
        ax1.plot(vswr._f_db_s11, vswr._db_s11[0], "k", label="0")
        ax1.plot(vswr._f_db_s11, vswr._db_s11[1], "y", label="1")
        ax1.plot(vswr._f_db_s11, vswr._db_s11[2], "b", label="2")
        ax1.set_ylabel(f"db")
        ax1.set_xlabel(f"[MHz]")
        ax1.grid()
        ax1.legend()
        ax2.set_title("abs(s_11)")
        ax2.plot(vswr.freqs_out, np.abs(vswr.s11[:, 0]), "k", label="0")
        ax2.plot(vswr.freqs_out, np.abs(vswr.s11[:, 1]), "y", label="1")
        ax2.plot(vswr.freqs_out, np.abs(vswr.s11[:, 2]), "b", label="2")
        ax2.set_ylabel(f"s11")
        ax2.set_xlabel(f"[MHz]")
        plt.grid(ls='--', alpha=0.3)
        ax2.legend()
        plt.show()

    if args=='lna':

        lna = grfc.LowNoiseAmplificatorGP300()
        lna.compute_for_freqs(freq_MHz)

        """
        plot of LNA S11
        """
        plt.figure()
        plt.title("LNA")
        plt.plot(lna.freqs_out, np.abs(lna.lna_gama[:, 0]), "k", label=r"0")
        plt.plot(lna.freqs_out, np.abs(lna.lna_gama[:, 1]), "y", label=r"1")
        plt.plot(lna.freqs_out, np.abs(lna.lna_gama[:, 2]), "b", label=r"2")
        plt.ylabel(r"abs(S$_{11}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()
        #plt.savefig("lna_s11.png", bbox_inches="tight")

        """
        plot of LNA S21
        """
        plt.figure()
        plt.title("LNA")
        plt.plot(lna.freqs_out, np.abs(lna.lna_s21[:, 0]), "k", label=r"0")
        plt.plot(lna.freqs_out, np.abs(lna.lna_s21[:, 1]), "y", label=r"1")
        plt.plot(lna.freqs_out, np.abs(lna.lna_s21[:, 2]), "b", label=r"2")
        plt.ylabel(r"abs(S$_{21}$)")
        plt.xlabel(f"Frequency [MHz]")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()
        #plt.savefig("lna_s21.png", bbox_inches="tight")

        """
        plot of intermediate calculation gamma
        """
        plt.figure()
        plt.title(r"S$_{11}^{\rm vswr}$")
        plt.plot(lna.freqs_out, np.abs(lna.antenna_gama[:, 0]), "k", label=r"0")
        plt.plot(lna.freqs_out, np.abs(lna.antenna_gama[:, 1]), "y", label=r"1")
        plt.plot(lna.freqs_out, np.abs(lna.antenna_gama[:, 2]), "b", label=r"2")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()

        """
        plot of intermediate calculation z
        """
        plt.figure()
        plt.title("")
        plt.plot(lna.freqs_out, np.abs(lna.z_ant[:, 0]), label=r"$\mathregular{Z_{A}}$")
        plt.plot(lna.freqs_out, np.abs(lna.z_in_lna[:, 0]), label=r"$\mathregular{Z^{in}_{LNA}}$")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()

        """
        plot of FFT LNA transfer function rho=rho_1*rho_2*rho_3
        """
        l_col = ["k", "y", "b"]
        
        plt.figure()
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        for port in range(3):
            plt.plot(lna.freqs_in, lna._dbs21_a[port], l_col[port])
        plt.ylim(20, 25)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}/dB} $", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right", fontsize=15)
        plt.title(r"$\mathregular{S_{21}}$" + " of LNA test", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        
        plt.figure(figsize=(9, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 3, 1)
        for port in range(3):
            plt.plot(lna.freqs_out, np.abs(lna._rho1[:, port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_1)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_1}$")
        plt.grid(ls='--', alpha=0.3)
        plt.subplot(1, 3, 2)
        for port in range(3):
            plt.plot(lna.freqs_out, np.abs(lna._rho2[:, port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_2)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_2}$")
        plt.grid(ls='--', alpha=0.3)
        plt.subplot(1, 3, 3)
        for port in range(3):
            plt.plot(lna.freqs_out, np.abs(lna._rho3[:, port]), l_col[port])
        plt.legend(["port X", "port Y", "port Z"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{mag(\rho_3)}$", fontsize=15)
        plt.xlim(30, 250)
        plt.title("the contribution of " + r"$\mathregular{ \rho_3}$")
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig("lna_rho1_2_3.png", bbox_inches="tight")

        """
        plot of LNA transfer function in time space
        """
        plt.figure()
        for port in range(3):
            plt.plot(lna.freqs_out, np.abs(lna.rho123[:, port]), color=l_col[port])
        plt.legend(["0", "1", "2"], loc="upper right")
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"abs($\rho$)", fontsize=15)
        plt.xlim(30, 250)
        plt.title(r"$\rho=\rho_1*\rho_2*\rho_3$")
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig("lna_rho.png", bbox_inches="tight")


        """
        plot of LNA transfer function in time space
        """
        plt.figure()
        plt.title("Rho kernel")
        kernel_rho = sf.fftshift(sf.irfft(lna.get_fft_rho_3d()), axes=1)
        # kernel_rho = sf.irfft(self.get_fft_rho_3d())
        print(kernel_rho.shape)
        # TODO: self.size_sig//2 or self.size_sig//2 -1 ?
        v_time = np.arange(lna.size_sig, dtype=np.float64) - lna.size_sig // 2
        dt_ns = 1e9 / (lna.freqs_out[1] * lna.size_sig * 1e6)
        v_time_ns = dt_ns * v_time
        plt.plot(v_time_ns, kernel_rho[0], "k", label="0")
        plt.plot(v_time_ns, kernel_rho[1], "y", label="1")
        plt.plot(v_time_ns, kernel_rho[2], "b", label="2")
        plt.xlabel("ns")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()
        plt.show()

    if args=="vga":

        vga = grfc.VgaFilterBalunGP300()
        vga.compute_for_freqs(freq_MHz)

        plt.figure(figsize=(6, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        # S11 and S21 DB
        plt.subplot(1, 2, 1)
        plt.plot(vga.freqs_in, vga.dbs11)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(vga.freqs_in, vga.dbs21)
        plt.grid(ls='--', alpha=0.3)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of Filter", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        # s21_add_vga
        plt.figure()
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.plot(vga.freqs_in, vga.dbs21_add_vga)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.title("S parameters of Filter add VGA", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.figure()
        plt.title("FFTS parameters of Filter add VGA", fontsize=15)
        plt.plot(vga.freqs_out, np.abs(vga.fft_vgafilbal))
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.show()

    if args=='cable':
        cable  = grfc.CableGP300()
        cable.compute_for_freqs(freq_MHz)

        plt.figure(figsize=(6, 3))
        # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
        plt.subplot(1, 2, 1)
        plt.plot(cable.freqs_in, cable.dbs11)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{11}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.grid(ls='--', alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(cable.freqs_in, cable.dbs21)
        plt.ylim(-10, 0)
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.ylabel(r"$\mathregular{S_{21}}$" + " mag/dB", fontsize=15)
        plt.legend(["port X", "port Y", "port Z"], loc="lower right")
        plt.suptitle("S parameters of cable", fontsize=15)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.grid(ls='--', alpha=0.3)

        plt.figure()
        plt.title("FFT cable")
        plt.plot(cable.freqs_out, np.abs(cable.fft_cable))
        plt.xlabel("Frequency(MHz)", fontsize=15)
        plt.grid(ls='--', alpha=0.3)
        plt.show()

    if args=='rf_chain':

        rfchain= grfc.RfChainGP300()
        rfchain.compute_for_freqs(freq_MHz)

        plt.figure()
        plt.title("Kernels associated to total transfer function of RF chain")
        kernel_0 = sf.fftshift(sf.irfft(rfchain.get_tf_3d()[0, :]))
        kernel_1 = sf.fftshift(sf.irfft(rfchain.get_tf_3d()[1, :]))
        kernel_2 = sf.fftshift(sf.irfft(rfchain.get_tf_3d()[2, :]))
        # kernel = sf.irfft(self.get_fft_rho_3d())
        # TODO: self.size_sig//2 or self.size_sig//2 -1 ?
        v_time = np.arange(rfchain.lna.size_sig, dtype=np.float64) - rfchain.lna.size_sig // 2
        dt_ns = 1e9 / (rfchain.lna.freqs_out[1] * rfchain.lna.size_sig * 1e6)
        v_time_ns = dt_ns * v_time
        plt.plot(v_time_ns, kernel_0, "k", label="port 1")
        plt.plot(v_time_ns, kernel_1, "y", label="port 2")
        plt.plot(v_time_ns, kernel_2, "b", label="port 3")
        plt.xlabel("ns")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()

        freqs = rfchain.lna.freqs_out
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.xlim(40, 260)
        plt.title("Amplitude total transfer function")
        plt.plot(freqs, np.abs(rfchain._total_tf[0]), "k", label="port 1")
        plt.plot(freqs, np.abs(rfchain._total_tf[1]), "y", label="port 2")
        plt.plot(freqs, np.abs(rfchain._total_tf[2]), "b", label="port 3")
        plt.xlabel("MHz")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.xlim(40, 260)
        plt.title("Phase total transfer function")
        plt.plot(freqs, np.angle(rfchain._total_tf[0], deg=True), "k", label="port 1")
        plt.plot(freqs, np.angle(rfchain._total_tf[1], deg=True), "y", label="port 2")
        plt.plot(freqs, np.angle(rfchain._total_tf[2], deg=True), "b", label="port 3")
        plt.xlabel("MHz")
        plt.ylabel("Deg")
        plt.grid(ls='--', alpha=0.3)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parser to select which noise quantity to plot. \
        To Run: ./plot_noise.py <plot_option>. \
        <plot_option>: galactic, vswr, lna, vga, cable, rf_chain. \
        Add --lst <int> for galactic noise. i.e ./plot_noise.py galactic --lst 18."
        )
    parser.add_argument(
        "plot_option",
        default="galactic",
        help="Option to select which noise quantity to plot.",
        )
    parser.add_argument(
        "--lst",
        default=18.0,
        type=float,
        help="lst for Local Sideral Time, galactic noise is variable with LST and maximal for 18h.",
        )

    args = parser.parse_args()

    options_list = ["vswr", "lna", "vga", "cable", "rf_chain"]

    if args.plot_option=="galactic":
        plot(args.plot_option, lst=args.lst)
    elif args.plot_option in options_list:
        plot(args.plot_option)
    else:
        raise Exception("Please provide a proper option for plotting noise. Options: galactic, vswr, lna, vga, cable, rf_chain.")

