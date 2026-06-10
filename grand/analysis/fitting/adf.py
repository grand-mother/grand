import numpy as np 
import grand.analysis.constants as cons
import grand.analysis.physics as che
import grand.analysis.coords.array_shower as co
import grand.analysis.geom.angles as an
import sys
#print(sys.path)
from iminuit import minimize

def ADF_parameters(theta, phi, delta_omega, amplitude, Xants, Xsource, groundAltitude=cons.groundAltitude, Bvec=cons.Bvec):
    """
    Compute all geometric parameters for the ADF function.
    
    Inputs:
        theta, phi          : shower direction angles (rad) (from ADF_recons, best fit values)
        delta_omega         : ADF shape parameter (output from ADF_recons, best fit values)
        amplitude           : ADF amplitude (output from ADF_recons, best fit values)
        Xants  : (N,3) positions of antennas
        Xsource   : (3,) position of Xsource (from SWF)
        Bvec   : (3,) magnetic field
        groundAltitude : altitude of ground
    
    Returns:
        eta       : (N,) azimuthal angle in shower plane
        omega     : (N,) angle wrt shower axis
        omega_cr  : (N,) Cherenkov angle for each antenna
        l_ant     : (N,) distance from Xsource to antenna
        adf       : (N,) ADF amplitude for each antenna
    """

    K = co.shower_direction_vector(theta, phi)
   
    asym_coeff = -0.003*np.rad2deg(theta)+0.220
    asym = asym_coeff/np.sqrt(1. - np.dot(K,Bvec)**2)
    

    l_ant = an.distance_source_antenna(Xants, Xsource)
    eta = an.eta(theta, phi, Bvec, Xants, Xsource)
    omega = an.omega(theta, phi, Xants, Xsource)
    
    omega_cr = np.array([che.compute_Cerenkov(Xants[i,:], K, (groundAltitude-Xsource[2])/K[2], Xsource, 2e3)
                         for i in range(Xants.shape[0])])
    
    # limit for small theta
    if theta <70*np.pi/180: 
            omega_cr = np.minimum(omega_cr, 0.6*np.pi/180)
    
    adf = amplitude/l_ant / (1.+4.*( ((np.tan(omega)/np.tan(omega_cr))**2 - 1. )/delta_omega)**2)
    adf *= 1. + asym*np.cos(eta) # 
    
    return eta, omega, omega_cr, l_ant, adf

def ADF_loss(params, Aants, Xants, Xsource, uncertainty=0.075):
    """
    Compute chi² for the ADF function.
    
    Inputs:
        params  : either a list/array [theta, phi, delta_omega, amplitude]
                  or four separate scalars (theta, phi, delta_omega, amplitude)
        Aants   : measured peak amplitudes (N,)
        Xants   : antenna positions (N,3)
        Xsource : shower source position (3,) (from SWF)
        uncertainty : relative uncertainty on amplitudes (default: 7.5%)
    
    Returns:
        chi² value
    """

    theta, phi, delta_omega, amplitude = params

    # Compute model
    _, _, _, _, amplitude_model = ADF_parameters(theta, phi, delta_omega, amplitude, Xants, Xsource, groundAltitude=cons.groundAltitude, Bvec=cons.Bvec)
    
    chi2 = np.sum((Aants - amplitude_model)**2 / (uncertainty*Aants)**2)
    return chi2


def recons_ADF(theta_pwf, phi_pwf, Aants, Xants, Xsource):
    """
    Fit the ADF parameters to measured antenna peak amplitudes.
    
    This function performs a chi² minimization using the ADF_loss function to 
    reconstruct the best-fit shower parameters: direction (theta, phi), ADF width (delta_omega), 
    and amplitude.
    
    Inputs:
        theta_pwf : initial guess for shower zenith angle (rad)
        phi_pwf   : initial guess for shower azimuth angle (rad)
        Aants     : measured peak amplitudes at antennas (N,)
        Xants     : antenna positions (N,3)
        Xsource   : Xsource position (3,)
    
    Returns:
        theta_adf     : reconstructed zenith angle (rad)
        phi_adf       : reconstructed azimuth angle (rad)
        delta_omega   : best-fit ADF shape parameter
        amplitude     : best-fit ADF amplitude
    """

    # Define bounds for each parameter
    bounds = [
        [theta_pwf - 2*np.pi/180, theta_pwf + 2*np.pi/180],
        [phi_pwf   - 1*np.pi/180, phi_pwf   + 1*np.pi/180],
        [1.25, 3.0],
        [1e6, 1e10]
    ]

    params_in = np.array(bounds).mean(axis=1)

    # Adjust initial amplitude guess using measured max amplitude and mean antenna distance
    l_ant = an.distance_source_antenna(Xants, Xsource)
    params_in[3] = Aants.max() * l_ant.mean()

    # Run minimization using iminuit
    result = minimize(
        ADF_loss,
        params_in,
        args=(Aants, Xants, Xsource),
        method="migrad",
        bounds=bounds
    )

    # Extract best-fit parameters from minimization result
    theta_adf, phi_adf, delta_omega, amplitude = result.x
    return theta_adf, phi_adf, delta_omega, amplitude