"""
Simulation of Galactic radio noise.
"""

import numpy as np

from grand.basis import validate as _validate
from grand.basis import data_model as _data_model

from grand import grand_add_path_data


def interpol_at_new_x(a_x, a_y, new_x):
    """
    Interpolate a discrete function and return zero outside its definition range.

    Parameters
    ----------
    a_x : ndarray
        Sample positions.
    a_y : ndarray
        Sample values.
    new_x : ndarray
        Positions to interpolate onto.

    Returns
    -------
    ndarray
        Cubic interpolation, zero outside the range of `a_x`.
    """
    from scipy import interpolate

    if a_x.shape[0] == 0:   # an assert, gone under python -O (#259)
        raise ValueError(_validate.message("galaxy.interpol_at_new_x", "no sample positions to interpolate from"))
    func_interpol = interpolate.interp1d(
        a_x,
        a_y,
        kind="cubic",
        bounds_error=False,
        fill_value=(0.0, 0.0),
    )
    return func_interpol(new_x)


def galactic_noise(f_lst, size_out, freqs_mhz, nb_ant, seed=None, du_type="GP300"):
    """
    Generate Galactic-noise voltage spectra for detector units.

    The precomputed Galactic-noise tables contain available-power spectral
    density ``P_L`` in W/Hz for 30--250 MHz, 72 local-sidereal-time bins
    (20-minute spacing), and the three antenna ports X, Y, Z. The requested
    LST is mapped to the nearest available 20-minute bin.

    For each port, the open-circuit RMS voltage in a 1 MHz native table bin is
    reconstructed from

    ``V_oc,RMS**2 = 4 * P_L * Re(Z_ant)``.

    The RMS voltage is then scaled to the requested uniform FFT-bin width,
    interpolated onto ``freqs_mhz``, and converted to complex rFFT
    coefficients using the normalization appropriate for SciPy/NumPy's
    backward-normalized ``irfft``.

    .. Authors:
       PengFei and Xidian group
       Modified by SN to support different antenna effective-length models.

    Parameters
    ----------
    f_lst : float
        Local sidereal time in hours. Must satisfy ``0 <= f_lst < 24``.
    size_out : int
        Length, in samples, of the time trace the spectrum belongs to.  It
        sets the normalization, so it must be the length the spectrum will be
        inverted to, whatever part of the frequency axis `freqs_mhz` covers.
    freqs_mhz : ndarray, shape (nb_freq,)
        Uniformly spaced output-frequency grid, in MHz: the full rFFT axis
        of `size_out` samples (``size_out // 2 + 1`` points), or a band of it.
        It is not checked against `size_out`.
    nb_ant : int
        Number of detector units for which independent noise is generated.
    seed : int or None, optional
        Random-number-generator seed. If None, a non-reproducible realization
        is generated; an integer seed gives reproducible output.
    du_type : str, optional
        Antenna model used for the Galactic-noise table: ``"GP300"`` (HFSS),
        ``"GP300_nec"`` (NEC), or ``"GP300_mat"`` (MATLAB).

    Returns
    -------
    ndarray, complex, shape (nb_ant, 3, nb_freq)
        Complex rFFT coefficients of the Galactic noise, for all detector
        units and antenna arms, in microvolts.

    Raises
    ------
    ValueError
        If `f_lst` is not finite or lies outside ``[0, 24)``, if `du_type` is
        not one of the three tabulated models, or if `freqs_mhz` is not
        uniformly spaced and increasing.

    Examples
    --------
    One realization of the noise spectrum for two units at 18 h local sidereal
    time, on a 1 MHz grid from 30 to 250 MHz:

    .. jupyter-execute::

        import numpy as np
        from grand.sim.noise.galaxy import galactic_noise

        freqs_mhz = np.arange(30.0, 251.0)
        spectrum = galactic_noise(18.0, 2048, freqs_mhz, nb_ant=2, seed=1)
        print(spectrum.shape)                                # (units, arms, frequencies)
        print("|V| at 30, 100 and 250 MHz on unit 0, arm X:",
              np.round(np.abs(spectrum[0, 0, [0, 70, 220]]), 1), "µV")
    """
    _validate.plausible(freqs_mhz, "freqs_mhz", "galactic_noise", "frequency_mhz")   # Hz gave all zeros (#266)
    # The Galactic-noise tables sample LST every 20 minutes (72 bins/24 h).
    # Select the nearest available bin. Integer-hour values map exactly, e.g.
    # f_lst=18.0 -> bin 54 -> LST 18:00.
    where = "galactic_noise"
    f_lst = _validate.as_real(f_lst, "f_lst", where)
    if not 0.0 <= f_lst < 24.0:
        raise ValueError(_validate.message(
            where, "'f_lst' (local sidereal time) must satisfy 0 <= f_lst < 24 hours, got %s" % f_lst))
    size_out = _validate.as_integer(size_out, "size_out", where, minimum=1)
    nb_ant = _validate.as_integer(nb_ant, "nb_ant", where, minimum=0)
    freqs_mhz = _validate.as_array(freqs_mhz, "freqs_mhz", where, ndim=1, min_length=2, finite=True)
    _validate.non_negative(freqs_mhz, "freqs_mhz", where, "MHz")
    lst_bin = int(np.floor(3.0 * f_lst + 0.5)) % 72

    # Available-power spectral-density tables, P_L [W/Hz].
    # Shape: (221 frequencies, 72 LST bins, 3 ports).
    gala_files = {
        "GP300": "noise/galactic_PL_per_Hz_gp13_GP300.npy",
        "GP300_nec": "noise/galactic_PL_per_Hz_gp13_GP300_nec.npy",
        "GP300_mat": "noise/galactic_PL_per_Hz_gp13_GP300_mat.npy",
    }

    _validate.one_of(du_type, tuple(gala_files), "du_type", where)

    gala_file = grand_add_path_data(gala_files[du_type])
    zant_file = grand_add_path_data("detector/RFchain_v2/Z_ant_3.2m.csv")

    gala_power = np.load(_data_model.check(gala_file, "galactic_noise"))  # (#279)
    if gala_power.shape != (221, 72, 3):
        raise ValueError(
            f"Unexpected Galactic-noise table shape {gala_power.shape} "
            f"for du_type='{du_type}'; expected (221, 72, 3)."
        )
    if not np.all(np.isfinite(gala_power)) or np.any(gala_power < 0.0):
        raise ValueError(
            f"Invalid Galactic-noise power values for du_type='{du_type}'."
        )

    # Select P_L at the requested LST: W/Hz, shape (221, 3).
    poc_per_hz = gala_power[:, lst_bin, :]

    # Convert W/Hz to available power in the native 1 MHz table bins.
    poc_1mhz = 1e6 * poc_per_hz

    # Use the same antenna-resistance table used to construct the P_L tables.
    zant = np.loadtxt(_data_model.check(zant_file, "galactic_noise"), delimiter=",", skiprows=1)
    zant_complex = np.column_stack(
        [
            zant[:, 1] + 1j * zant[:, 2],  # Z(1,1)
            zant[:, 3] + 1j * zant[:, 4],  # Z(2,2)
            zant[:, 5] + 1j * zant[:, 6],  # Z(3,3)
        ]
    )
    rant = np.real(zant_complex)

    if rant.shape != (221, 3):
        raise ValueError(
            f"Unexpected antenna-resistance shape {rant.shape}; "
            "expected (221, 3)."
        )
    if not np.all(np.isfinite(rant)) or np.any(rant <= 0.0):
        raise ValueError("Antenna resistance must be finite and positive.")

    # Available power and open-circuit RMS voltage are related by
    # P_L = V_oc,RMS^2 / (4 R_ant).
    voc2_1mhz = 4.0 * poc_1mhz * rant
    voc_rms_1mhz_uv = 1e6 * np.sqrt(voc2_1mhz)

    gala_freq_mhz = np.arange(30.0, 251.0)

    # Validate the requested FFT-frequency grid. The per-bin RMS voltage scales
    # as sqrt(bandwidth), so a uniform bin width is required here.
    freqs_mhz = np.asarray(freqs_mhz, dtype=float)
    if freqs_mhz.ndim != 1 or freqs_mhz.size < 2:
        raise ValueError(
            "freqs_mhz must be a one-dimensional array with at least two bins."
        )
    if not np.all(np.isfinite(freqs_mhz)):
        raise ValueError("freqs_mhz must contain only finite values.")

    df_mhz = np.diff(freqs_mhz)
    if np.any(df_mhz <= 0.0):
        raise ValueError("freqs_mhz must be strictly increasing.")
    if not np.allclose(df_mhz, df_mhz[0], rtol=1e-10, atol=1e-12):
        raise ValueError(
            "galactic_noise currently requires a uniformly spaced frequency grid."
        )

    nb_freq = freqs_mhz.size
    freq_res_mhz = df_mhz[0]

    # Scale the 1 MHz RMS voltage to the requested FFT-bin bandwidth and
    # interpolate each antenna port onto the requested frequency grid.
    voc_rms_bin_uv = voc_rms_1mhz_uv * np.sqrt(freq_res_mhz)
    v_amplitude = np.empty((nb_freq, 3), dtype=float)
    for port in range(3):
        v_amplitude[:, port] = interpol_at_new_x(
            gala_freq_mhz,
            voc_rms_bin_uv[:, port],
            freqs_mhz,
        )

    if not np.all(np.isfinite(v_amplitude)):
        raise ValueError("Interpolated Galactic-noise RMS amplitudes are not finite.")
    if np.any(v_amplitude < 0.0):
        raise ValueError("Interpolated Galactic-noise RMS amplitudes must be non-negative.")

    # Arrange as (port, frequency), then draw one Gaussian RMS amplitude and
    # one independent random phase for each DU, port, and frequency bin.
    v_amplitude = v_amplitude.T
    rng = np.random.default_rng(seed)
    amp = rng.normal(
        loc=0.0,
        scale=v_amplitude[np.newaxis, ...],
        size=(nb_ant, 3, nb_freq),
    )
    phase = 2.0 * np.pi * rng.random(size=(nb_ant, 3, nb_freq))

    # Interior positive-frequency rFFT bins. For a backward-normalized irFFT,
    # a conjugate pair contributes RMS sqrt(2)*|X_k|/N in the time domain.
    # Therefore a requested per-bin RMS sigma requires |X_k| = N*sigma/sqrt(2).
    v_complex = (
        np.abs(amp * size_out / np.sqrt(2.0))
        * np.exp(1j * phase)
    )

    # On a complete rFFT grid (the Efield2Voltage production use case), DC and
    # Nyquist are self-conjugate real coefficients and do not have a conjugate
    # partner. Their coefficient normalization is therefore N*sigma. Galactic
    # noise is zero at DC for the present 30--250 MHz tables, but handle it
    # generically. For odd N there is no Nyquist bin.
    if nb_freq == size_out // 2 + 1:
        v_complex[:, :, 0] = amp[:, :, 0] * size_out
        if size_out % 2 == 0:
            v_complex[:, :, -1] = amp[:, :, -1] * size_out

    return v_complex
