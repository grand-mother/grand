"""Electromagnetic-energy proxy from the fitted radio amplitude."""

import numpy as np

from grand.basis import validate as _validate


def recons_energy_from_voltage(amplitude, sin_alpha, a=1.96e7, b=7.90e6):
    """Reconstruct the electromagnetic energy from the fitted radio amplitude.

    The amplitude is in ADC counts. Just a first proxy.

    Parameters
    ----------
    amplitude : float or array-like
        Scaling factor from the best-fit ADF (in ADC units).
    sin_alpha : float or array-like
        Sine of the geomagnetic angle between the shower axis and the geomagnetic field.
    a : float
        Slope parameter obtained from the fit on simulations (default: 1.96e7).
    b : float
        Offset parameter obtained from the fit on simulations (default: 7.90e6).
    See arXiv:2507.04324 for details on how the values of a and b are determined.

    Returns
    -------
    energy : float or array-like
        Reconstructed electromagnetic energy in eV.
        Events yielding negative reconstructed energies are set to zero.
        Such cases may correspond to genuine cosmic-ray events for which the
        voltage-based energy proxy fails, as this reconstruction is not very robust
        and should only be interpreted as a first-order estimator.
    """
    where = "recons_energy_from_voltage"
    scalar = np.ndim(amplitude) == 0 and np.ndim(sin_alpha) == 0
    amplitude = _validate.as_array(amplitude, "amplitude", where, finite=True)
    sin_alpha = _validate.as_array(sin_alpha, "sin_alpha", where, finite=True)
    if np.any(sin_alpha == 0):
        raise ValueError(_validate.message(
            where, "'sin_alpha' is 0: the shower axis is parallel to the geomagnetic field, "
            "so the geomagnetic emission vanishes and the energy cannot be reconstructed"))
    energy = (amplitude / sin_alpha - b) / a
    # np.maximum, not the builtin max, so that arrays work as documented
    energy = np.maximum(energy, 0.0) * 1e18
    return float(energy) if scalar else energy