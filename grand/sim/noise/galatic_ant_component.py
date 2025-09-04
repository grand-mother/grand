"""
Created on 26 août 2025

Refactoring of module grand.sim.noise.galaxy by Colley Jean-Marc

Goal of refactoring:
* for each call of galactic_noise(), the function read files model on disk to do the same thing
* clearify FFT normalization
* clearify method of noise generation

So 
* Separate ASD computing (galactic_ant_asd.py) and noise generation (galactic_ant_component.py) 
* Add class to perform the computing of ASD only one time during simulation

AND also
* Replace cubic interpolation by linear, more safe
* Simply check between what content of model galactic noise files and what we used finally 
* Add plot function in same module
"""

import numpy as np
import scipy.fft as sf
import scipy.signal as ss
import matplotlib.pyplot as plt

from grand.sim.noise.galatic_ant_asd import get_asd_galactic_ant_model
from grand.basis.signal import interpol_at_new_x


class GalacticAntComponent:
    """Component of the galactic signal through antenna

    1) define model with set_model_name() or set_model_file()
    2) define LST, freq sampling, size trace with set_lst_freq_size_out()
    3) create a random galactic signal with get_rfft_gal_ant() or get_traces_gal_ant()

    """

    def __init__(self):
        self.f_mod = None
        self.asd_mod = None
        self.asd = None

    def _interpolate_asd(self):
        # define LST
        asd_mod = self.asd_mod[:, :, self.lst]
        nb_freq = len(self.freqs_mhz)
        asd = np.zeros((nb_freq, 3))
        asd[:, 0] = interpol_at_new_x(self.f_mod[:, 0], asd_mod[:, 0], self.freqs_mhz, "linear")
        asd[:, 1] = interpol_at_new_x(self.f_mod[:, 0], asd_mod[:, 1], self.freqs_mhz, "linear")
        asd[:, 2] = interpol_at_new_x(self.f_mod[:, 0], asd_mod[:, 2], self.freqs_mhz, "linear")
        self.asd = asd

    def set_model_name(self, m_name):
        freq, asd = get_asd_galactic_ant_model(m_name)
        self.f_mod = freq
        self.asd_mod = asd

    def set_model_file(self, pn_model):
        m_asd = np.load(pn_model)
        self.f_mod = m_asd["fq"]
        self.asd_mod = m_asd["asd"]

    def set_lst_freq_size_out(self, lst, freqs_mhz, size_out):
        """Define LST, out frequency, size of trace"""
        self.lst = int(lst)
        assert self.lst >= 0 and self.lst < 24
        self.freqs_mhz = freqs_mhz
        self.nb_freq = len(freqs_mhz)
        self.size_out = size_out
        self._interpolate_asd()
        # rfft "backward" normalization
        # see documentation
        # https://github.com/luckyjim/mogwai/blob/main/doc/fft_and_white_noise.ipynb
        self.normalization = np.sqrt(self.size_out)
        # ASD normalisation
        # see documentation
        # https://github.com/luckyjim/mogwai/blob/main/doc/spectrum_with_Welch_method.ipynb
        f_nyquist_hz = freqs_mhz[-1] * 1e6
        self.fs_hz = 2 * f_nyquist_hz
        self.normalization *= np.sqrt(f_nyquist_hz)
        self.asd_nor = self.asd.T * self.normalization

    def get_rfft_gal_ant(self, nb_ant):
        # noise generation from PSD
        # see documentation
        # https://github.com/luckyjim/mogwai/blob/main/doc/noise_wf.ipynb
        angle = np.random.uniform(0, 2 * np.pi, size=(nb_ant, 3, self.nb_freq))
        # Mode 0 and Nyquist are real
        angle[:, :, 0], angle[:, :, -1] = 0.0, 0.0
        return np.exp(1j * angle) * self.asd_nor

    def get_traces_gal_ant(self, nb_ant):
        rfft_gal = self.get_rfft_gal_ant(nb_ant)
        return sf.irfft(rfft_gal)

    def plot_psd_inter(self):
        plt.figure()
        plt.title(f"Interpolate PSD galactic component at LST {self.lst}")
        plt.semilogy(self.freqs_mhz[1:-2], self.asd[1:-2, 0] ** 2, label="axis 0")
        plt.semilogy(self.freqs_mhz[1:-2], self.asd[1:-2, 1] ** 2, label="axis 1")
        plt.semilogy(self.freqs_mhz[1:-2], self.asd[1:-2, 2] ** 2, label="axis 2")
        plt.xlim([20, 260])
        plt.xlabel("Frequency [MHz]")
        plt.ylabel(r"PSD: [$\mu V^2/Hz$]")
        plt.grid()
        plt.legend()

    def plot_psd_model(self, lst):
        plt.figure()
        plt.title(f"Model PSD galactic component at LST {self.lst}")
        plt.semilogy(self.f_mod[1:-2], self.asd_mod[1:-2, 0, lst] ** 2, label="axis 0")
        plt.semilogy(self.f_mod[1:-2], self.asd_mod[1:-2, 1, lst] ** 2, label="axis 1")
        plt.semilogy(self.f_mod[1:-2], self.asd_mod[1:-2, 2, lst] ** 2, label="axis 2")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel(r"PSD: [$\mu V^2/Hz$]")
        plt.grid()
        plt.legend()

    def plot_psd_trace(self):
        nperseg = 1024
        trace = self.get_traces_gal_ant(1)[0]
        freq, psd0 = ss.welch(
            trace[0], window="hann", fs=self.fs_hz, scaling="density", nperseg=nperseg
        )
        freq, psd1 = ss.welch(
            trace[1], window="hann", fs=self.fs_hz, scaling="density", nperseg=nperseg
        )
        freq, psd2 = ss.welch(
            trace[2], window="hann", fs=self.fs_hz, scaling="density", nperseg=nperseg
        )
        freq *= 1e-6
        plt.figure()
        plt.title(f"PSD trace at LST {self.lst}")
        # remove mode 0 and Nyquist
        plt.semilogy(freq[1:-2], psd0[1:-2], label="axis 0")
        plt.semilogy(freq[1:-2], psd1[1:-2], label="axis 1")
        plt.semilogy(freq[1:-2], psd2[1:-2], label="axis 2")
        plt.xlim([20, 260])
        plt.ylim([1e-1, psd2.max()])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"PSD: [$\mu V^2/Hz$]")
        plt.grid()

    def plot_check_trace(self, axis=0):
        nperseg = 1024
        trace = self.get_traces_gal_ant(1)[0]
        freq, psd = ss.welch(
            trace[axis], window="hann", fs=self.fs_hz, scaling="density", nperseg=nperseg
        )
        freq *= 1e-6
        plt.figure()
        plt.title(f"PSD trace at LST {self.lst} and model for axis {axis}")
        # remove mode 0 and Nyquist
        plt.semilogy(freq[1:-2], psd[1:-2], label="Welch PSD trace")
        psd_mod = self.asd_mod[1:-2, axis, self.lst] ** 2
        plt.semilogy(self.f_mod[1:-2], psd_mod, label="model")
        plt.xlim([20, 260])
        plt.ylim([psd_mod.min(), psd_mod.max()])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"PSD: [$\mu V^2/Hz$]")
        plt.legend()
        plt.grid()

    def plot_trace_gal(self):
        trace = self.get_traces_gal_ant(1)[0]
        plt.figure()
        plt.title("Galactic component")
        plt.plot(trace[0], label="axis 0")
        plt.plot(trace[2], label="axis 2")
        plt.grid()
        plt.legend()


if __name__ == "__main__":
    gen_gal = GalacticAntComponent("GP300")
    size_out = 4096 * 2
    fs_hz = 2_000_000_000
    freqs_mhz = sf.rfftfreq(size_out, 1 / fs_hz) * 1e-6
    print(freqs_mhz[-1])
    gen_gal.set_lst_freq_size_out(2, freqs_mhz, size_out)
    gen_gal.plot_psd_model(2)
    # gen_gal.plot_psd_inter()
    # gen_gal.plot_psd_trace()
    # gen_gal.plot_check_trace(0)
    # gen_gal.plot_check_trace(1)
    # gen_gal.plot_check_trace(2)
    # gen_gal.plot_trace_gal()
    #
    traces = gen_gal.get_traces_gal_ant(10)
    from grand.basis.traces_event import Handling3dTraces

    evt = Handling3dTraces("Simulation galactic component")
    evt.init_traces(traces, f_samp_mhz=fs_hz * 1e-6)
    evt.set_unit_axis("$\mu V$", "dir", "galactic")
    evt.plot_trace_idx(5)
    evt.plot_psd_trace_idx(5)

    plt.show()
