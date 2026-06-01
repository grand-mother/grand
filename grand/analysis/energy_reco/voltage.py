
import numpy as np 

def recons_energy_from_voltage(amplitude, sin_alpha, a=1.96e7, b=7.90e6):
    """
    Reconstructs the electromagnetic energy from the measured radio signal amplitude in ADC counts.
    Just a first proxy

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
    
    energy = (amplitude / sin_alpha - b) / a   
    energy =max(energy, 0.0) * 1e18 
    return energy