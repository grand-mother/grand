"""
Simulation of galaxy emission in radio frequency
"""

import h5py
import numpy as np
from grand import grand_add_path_data

def interpol_at_new_x(a_x, a_y, new_x):
    """
    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    :return: F(new_x) (float, (M)): interpolation of F at new_x
    """
    from scipy import interpolate
    assert a_x.shape[0] > 0
    func_interpol = interpolate.interp1d(
        a_x, a_y, "cubic", bounds_error=False, fill_value=(0.0, 0.0)
    )
    return func_interpol(new_x)

def galactic_noise(f_lst, size_out, freqs_mhz, nb_ant, seed=None, du_type='GP300'):
    r"""Returns the Fourier-domain voltage induced in each antenna arm by Galactic noise.

    The Galactic radio background is computed from LFMap sky brightness
    temperatures, folded through the response of a GRAND HorizonAntenna and
    tabulated over frequency (30-250 MHz) and local sidereal time (0-24 h).
    This routine reads that table and returns one realisation of the induced
    voltage spectrum for each antenna and each of the three arms.

    Spatial coherence between neighbouring detection units is **not**
    modelled — it is expected to be small given the sparsity of the array —
    so each unit receives an independent realisation of the sky-averaged
    noise.

    Parameters
    ----------
    f_lst : float
        Local sidereal time, in hours.  Truncated to an integer index into
        the tabulated LST axis; see the note below.
    size_out : int
        Length of the padded time trace the spectrum will be transformed
        back into.  It enters the normalisation of the returned amplitude.
    freqs_mhz : ndarray, shape (n_freq,)
        Output frequency axis, in MHz.  The tabulated model is interpolated
        onto it.
    nb_ant : int
        Number of detection units to generate realisations for.
    seed : int or None, optional
        Seed for the random generator.  ``None`` (the default) gives an
        independent realisation on every call; a fixed value makes the
        output reproducible, which is what the tests rely on.
    du_type : {'GP300', 'GP300_nec', 'GP300_mat'}, optional
        Which antenna-response simulation to use for the effective length:
        HFSS by default, or the NEC or MATLAB variants.

    Returns
    -------
    ndarray, shape (nb_ant, 3, n_freq)
        Complex voltage spectrum per antenna and arm, ordered X, Y, Z.

    Notes
    -----
    **The implementation and the published description differ.**  Section
    8.2 of `arXiv:2408.10926 <https://arxiv.org/abs/2408.10926>`_ states
    that the module "mimics the variability of the Galactic noise by
    multiplying, for each DU, the sky-averaged value of the noise by a
    different random value of its *phase* for each antenna arm and at each
    frequency" — that is, a fixed modulus with a randomised phase.  The code
    below also randomises the modulus, drawing it from a normal distribution
    scaled by the tabulated amplitude and taking the absolute value.  The
    two agree in mean power and differ in their fluctuations.  Which is
    intended has not been settled; it is tracked as part of the
    galactic-noise decision in the repository overhaul.

    ``f_lst`` is truncated rather than interpolated, so an LST of 18.9 h is
    treated as 18 h.  The ``TODO`` in the body marks the same point.

    Examples
    --------
    .. jupyter-execute::

        import numpy as np
        from grand.sim.noise.galaxy import galactic_noise

        freqs = np.linspace(30.0, 250.0, 221)
        v = galactic_noise(18.0, 1024, freqs, nb_ant=4, seed=0)

        print("shape (n_ant, 3 arms, n_freq):", v.shape)
        print("median |V| per arm (uV/MHz):",
              np.round(np.median(np.abs(v), axis=(0, 2)), 3))

    The Z arm is the one with vertical orientation and sees less of the sky
    than the horizontal X and Y arms; Fig. 7 of the paper shows the
    resulting difference across LST.

    ..Authors:
      PengFei and Xidian group.
      Modified by SN to support several antenna models for the effective length.
    """
    # TODO: why lst is an integer ?
    lst = int(f_lst)
    
    if du_type == 'GP300':
        lst = int(f_lst)
        gala_file = grand_add_path_data("noise/PG_ALL_jifen.mat")
        Zant_file = grand_add_path_data("detector/RFchain_v2/Z_ant_3.2m.csv")
        gala_show = h5py.File(gala_file, "r")
        gala_power = np.array(gala_show["PG_ALL_jifen"])
        gala_power = np.transpose(gala_power, (2, 0, 1)) #Watt/Hz
        Poc2X = 1e6*gala_power[:,:,0] #W
        Poc2Y = 1e6*gala_power[:,:,1] #W
        Poc2Z = 1e6*gala_power[:,:,2] #W
    
        zant = np.loadtxt(Zant_file, delimiter=",", skiprows=1)  # Skip header row if it exists
        # Extract real and imaginary parts and construct complex numbers
        zant_complex = np.column_stack([
            zant[:, 1] + 1j * zant[:, 2],  # Z(1,1)
            zant[:, 3] + 1j * zant[:, 4],  # Z(2,2)
            zant[:, 5] + 1j * zant[:, 6]   # Z(3,3)
        ])
        R = np.real(zant_complex)
        R_reshaped = R.T
        RantX = R_reshaped[0, :]
        RantY = R_reshaped[1, :]
        RantZ = R_reshaped[2, :]
        Voc2X = 4*Poc2X*RantX[:, np.newaxis]
        Voc2Y = 4*Poc2Y*RantY[:, np.newaxis]
        Voc2Z = 4*Poc2Z*RantZ[:, np.newaxis]
        VocX = 1e6*np.sqrt(Voc2X) # in uV
        VocY = 1e6*np.sqrt(Voc2Y) # in uV
        VocZ = 1e6*np.sqrt(Voc2Z) # in uV
        gala_voltage = np.stack((VocX, VocY, VocZ), axis=1)
        #gala_psd_dbm = np.transpose(gala_show["psd_narrow_huatu"])
        #gala_power_dbm = np.transpose(
        #    gala_show["p_narrow_huatu"]
        #)  # SL, dbm per MHz, P=mean(V*V)/imp with imp=100 ohms
        #gala_voltage = np.transpose(
        #    gala_show["v_amplitude"]
        #)  # SL, microV per MHz, seems to be Vmax=sqrt(2*mean(V*V)), not std(V)=sqrt(mean(V*V))
        ## gala_power_mag = np.transpose(gala_show["p_narrow"])
        gala_freq1 = np.arange(30.,251.)
        gala_freq = gala_freq1.reshape(221, 1)

        """f_start = 30
        f_end = 250
        # TODO: 221 is the number of frequency ? why ? and comment to explain
        nb_freq = 221
        v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
        galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
        galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
        galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)"""
        v_amplitude_infile = gala_voltage[:, :, lst - 1]
    
    elif du_type == 'GP300_nec':
        gala_file = grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_nec.npy")
        gala_file1 = grand_add_path_data("noise/Pocmax_30-250_Watt_per_MHz_nec.npy")
        gala_file2 = grand_add_path_data("noise/Pocmax_30-250_dBm_per_MHz_nec.npy")
        gala_voltage = np.load(gala_file)
        gala_voltage = np.transpose(gala_voltage, (0, 2, 1)) #micro Volts per MHz (max)
        gala_power_watt = np.load(gala_file1) 
        gala_power_watt = np.transpose(gala_power_watt, (0, 2, 1)) #watt per MHz
        gala_power_dbm = np.load(gala_file2)
        gala_power_dbm = np.transpose(gala_power_dbm, (0, 2, 1)) # dBm per MHz
        gala_freq1 = np.arange(30.,251.)
        gala_freq = gala_freq1.reshape(221, 1)
        """f_start = 30
        f_end = 250
        # TODO: 221 is the number of frequency ? why ? and comment to explain
        nb_freq = 221
        v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
        galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
        galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
        galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)"""
        v_amplitude_infile = gala_voltage[:, :, lst - 1]
        
    elif du_type == 'GP300_mat':
        gala_file = grand_add_path_data("noise/Vocmax_30-250MHz_uVperMHz_mat.npy")
        gala_file1 = grand_add_path_data("noise/Pocmax_30-250_Watt_per_MHz_mat.npy")
        gala_file2 = grand_add_path_data("noise/Pocmax_30-250_dBm_per_MHz_mat.npy")
        gala_voltage = np.load(gala_file)
        gala_voltage = np.transpose(gala_voltage, (0, 2, 1)) #micro Volts per MHz (max)
        gala_power_watt = np.load(gala_file1) 
        gala_power_watt = np.transpose(gala_power_watt, (0, 2, 1)) #watt per MHz
        gala_power_dbm = np.load(gala_file2)
        gala_power_dbm = np.transpose(gala_power_dbm, (0, 2, 1)) # dBm per MHz
        gala_freq1 = np.arange(30.,251.)
        gala_freq = gala_freq1.reshape(221, 1)
        """f_start = 30
        f_end = 250
        # TODO: 221 is the number of frequency ? why ? and comment to explain
        nb_freq = 221
        v_complex_double = np.zeros((nb_ant, size_out, 3), dtype=complex)
        galactic_v_time = np.zeros((nb_ant, size_out, 3), dtype=float)
        galactic_v_m_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)
        galactic_v_p_single = np.zeros((nb_ant, int(size_out / 2) + 1, 3), dtype=float)"""
        v_amplitude_infile = gala_voltage[:, :, lst - 1]

    # SL
    nb_freq = len(freqs_mhz)
    freq_res = freqs_mhz[1] - freqs_mhz[0]
    v_amplitude_infile = v_amplitude_infile * np.sqrt(freq_res)
    v_amplitude = np.zeros((nb_freq, 3))
    v_amplitude[:, 0] = interpol_at_new_x(gala_freq[:, 0], v_amplitude_infile[:, 0], freqs_mhz)
    v_amplitude[:, 1] = interpol_at_new_x(gala_freq[:, 0], v_amplitude_infile[:, 1], freqs_mhz)
    v_amplitude[:, 2] = interpol_at_new_x(gala_freq[:, 0], v_amplitude_infile[:, 2], freqs_mhz)

    '''
    a_nor = np.zeros((nb_ant, nb_freq, 3), dtype=float)
    phase = np.zeros((nb_ant, nb_freq, 3), dtype=float)
    v_complex = np.zeros((nb_ant, 3, nb_freq), dtype=complex)
    for l_ant in range(nb_ant):
        for l_fq in range(nb_freq):
            for l_axis in range(3):
                # Generates a normal distribution with 0 as the mean and
                # v_amplitude[l_fq, l_axis] as the standard deviation
                a_nor[l_ant, l_fq, l_axis] = np.random.normal(
                    loc=0, scale=v_amplitude[l_fq, l_axis]
                )
                # phase of random Gauss noise
                phase[l_ant, l_fq, l_axis] = 2 * np.pi * np.random.random_sample()
                # SL *size_out is because default scipy fft is normalised backward, *1/2 is because mean(cos(x)*cos(x)))
                v_complex[l_ant, l_axis, l_fq] = abs(a_nor[l_ant, l_fq, l_axis] * size_out / 2)
                v_complex[l_ant, l_axis, l_fq] *= np.exp(1j * phase[l_ant, l_fq, l_axis])
    '''

    # RK: above loop is replaced by lines below. Also np.random.default_rng(seed) is used instead of np.random.seed().
    #     if seed is a fixed number, same set of randomly generated number is produced. This is useful for testing.
    v_amplitude = v_amplitude.T
    rng   = np.random.default_rng(seed)     
    amp   = rng.normal(loc=0, scale=v_amplitude[np.newaxis,...], size=(nb_ant, 3, nb_freq))
    phase = 2 * np.pi * rng.random(size=(nb_ant, 3, nb_freq))
    v_complex = np.abs(amp * size_out / 2) * np.exp(1j * phase)

    return v_complex




