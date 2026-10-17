import numpy as np
from grand.sim.detector.adc import ADC
from grand.sim.noise.galaxy import galactic_noise


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


freqs = np.fft.rfftfreq(2048, 1 / 2000.0)
noise = galactic_noise(18.0, 2048, freqs, 4, seed=1)          # LST 18 h, 4 antennas, same seed
attempt("1. galactic noise at LST 18 h, RMS", lambda: float("%.4g" % np.sqrt(np.mean(np.abs(noise) ** 2))))
attempt("2. galactic_noise at LST 24.0 h", lambda: galactic_noise(24.0, 2048, freqs, 1, seed=1).shape)
adc = ADC()
attempt("3. ADC.downsample of 1003 samples, 2000 to 500 MHz", lambda: adc.downsample(np.ones((1, 3, 1003)), 2000.0).shape)
