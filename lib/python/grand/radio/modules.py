from . import config
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import astropy.units as u

import logging
logger = logging.getLogger(__name__)

from ..config import load
load("/home/martineau/GRAND/soft/grand/tests/radio/config.py")
phigeo = (config.magnet.declination / u.deg).value
thetageo = 90+(config.magnet.inclination / u.deg).value
# XXX / TODO: instead of np.deg2rad(az) use astropy functionality
# XXX / TODO: Handling of astropy units missing


# ============================================================================
def _get_shower_angles(ush):
    '''
    Parameters
    ----------
    ush : propagation vector(s) of the shower in the GRAND referential. Dimension = nb of vectors x [u_x,u_y,u_z]

    Returns
    -------
    zen :   array
        zenith of the showers in GRAND, in deg
    az : array
        azimuth of the showers in GRAND, in deg
    '''

    if np.size(ush) == 3:
        ush = np.array([ush])  # python is so amazingly shity with vector manipulation... This is needed to force definition of line vector
    if np.shape(ush)[1] != 3:
       print("Nb of columns of shower vectors must be 3: [u_x,u_y,u_z]")
       return([0,0])
    # First normalize shower vectors
    n = np.linalg.norm(ush,axis=1)
    ush = ush/n[:,None]
    zen=np.rad2deg(np.arccos(ush[:,2])) * u.deg # Zenith in antenna frame
    zen[zen<0] = zen[zen<0] +360*u.deg
    az=np.rad2deg(np.arctan2(ush[:,1],ush[:,0])) * u.deg
    az[az<0] = az[az<0] + 360*u.deg
    if np.size(az) == 1:
      az = az[0]
      zen = zen[0]
    return zen, az

def _get_shower_vector(zen,az):
    '''
    Parameters
    ----------
    zen :   array
        zenith of the showers in GRAND, in deg
    az : array
        azimuth of the shower in GRAND, in deg

    Returns
    -------
    ush : propagation vectors of the shower in the GRAND referential. Shape = (nshowers*[ush_x,ush_y,ush_z])

    '''
    caz = np.cos(np.deg2rad(az))
    saz = np.sin(np.deg2rad(az))
    czen = np.cos(np.deg2rad(zen))
    szen = np.sin(np.deg2rad(zen))
    ush = np.array([np.multiply(caz,szen),np.multiply(saz,szen),czen])  # Using GRAND conventions here
    ush = ush.T
    return ush

def _geomagnetic_angle(zen, az):
    '''
    Parameters
    ----------
    zen : float
        zenith of the shower in GRAND, in deg
    az : float
        azimuth of the shower in GRAND, in deg

    Returns
    -------
    geo_angle: float
        geomagnetic angle in deg
    '''

    # Direction of shower
    az_rad = np.deg2rad(az)
    zen_rad = np.deg2rad(zen)
    v = np.array([np.cos(az_rad) * np.sin(zen_rad),
                  np.sin(az_rad) * np.sin(zen_rad),
                  np.cos(zen_rad)])

    # Direction where Earth mag field points to at Ulastai
    # North + Magnetic North, pointing towards East in this case
    az_B = np.deg2rad(phigeo)
    # Direction where we point to . Inclination=63.05deg
    zen_B = np.deg2rad(thetageo)
    B = np.array([  np.cos(az_B)* np.sin(zen_B),
                    np.sin(az_B)* np.sin(zen_B),
                    np.cos(zen_B)])

    geo_angle = np.arccos(np.matmul(v, B))

    return np.rad2deg(geo_angle)

# ============================================================================


def _calculateXmax(primarytype, energy):
    ''' Xmax value in g/cm2
    Arguments:
    ----------
    primarytype: str
        nature of primary: electron, pion, proton, iron
    energy: float
        primary energy in eV

    Returns:
    ----------
    Xmax: float
        Xmax value in g/cm2

    Note: factor approximated from https://pos.sissa.it/301/1100/pdf
    '''

    #    current version for tau decays
    if primarytype == 'electron':  # aprroximated by gamma shower
        a = 82.5  # g/cm2
        c = 342.5  # g/cm2
    if primarytype == 'pion':  # pion, kaon .... aprroximated by proton
        a = 62.5  # g/cm2
        c = 357.5  # g/cm2
    if primarytype == 'proton'or primarytype == 'Proton':  # pion, kaon .... approximated by proton
        a = 62.5  # g/cm2
        c = 357.5  # g/cm2
        #print("ATTENTION: zenith to be corrected for CR in following use")
    if primarytype == 'iron' or primarytype == 'Iron':  # aprroximated by proton
        a = 60  # g/cm2 # just approximated
        c = 177.5  # g/cm2
        #print("ATTENTION: zenith to be corrected for CR in following use")

    Xmax = a*np.log10(energy*10**-12.)+c  # eV* 10**-12. to be in TeV

    # /abs(np.cos(np.pi-zen2)) # XXX / TODO: how to correct for slanted shower
    return Xmax

# ============================================================================


def _dist_decay_Xmax(zen_inj, injh, xmax):
    ''' Calculate the height of Xmax and the distance from injection point to Xmax along the shower axis

    Arguments:
    ----------
    zen: float
        GRAND zenith in deg, for CR shower use _get_CRzenith()
    injh: float
        injectionheight above sealevel in m
    xmax: float
        Xmax in g/cm2

    Returns:
    --------
    h: float
        vertical Xmax_height in m
    ai: float
        Xmax_distance injection to Xmax along shower axis in m
    '''
    # TODO: not checked for upward going trajs

    from astropy.constants import R_earth
    Re = R_earth.value  # m, Earth radius

    # % Using isothermal Model, numbers from Zhaires
    rho_0 = 1.225  # ; % kg/m3
    M = 0.028966  # ;  %kg/mol - 1000g/mol
    g = 9.81  # ; %ms-2
    T = 288.  # ; % K
    R = 8.32  # ; J/K/mol , J=kg m2/s2

    zen_inj = np.deg2rad((zen_inj/u.deg).value)

    hD = (injh/u.m).value
    step = 10  # m
    if hD > 10000:
        step = 100  # m
    xmax = xmax * 10. # g/cm2 to kg/m2: 1g/cm2 = 10kg/m2
    gamma = np.pi-zen_inj

    X = 0.
    i = 0.
    h = hD
    ai = 0
    while X < xmax:
        i = i+1
        ai = i*step  #m
        hi = np.sqrt((Re+hD)**2. + ai**2. - 2 * ai * (Re+hD) * np.cos(gamma))- Re # cos(gamma)= + to - at 90dg
        # Xmax in g/cm2, slanted = Xmax, vertical/ cos(theta);
        # density in g/cm3, h: m->100cm, np.pi-zen2 since it is defined as where the showers comes from, abs(cosine) so correct for minus values
        X = X + rho_0*np.exp(-g*M*hi/(R*T)) * step # (deltah*100) *abs(1./np.cos(np.pi-zen2))

    hi = hi * u.m
    ai = ai * u.m
    logger.debug("Xmax = " + str(xmax) + "corresponds to  height = " + str(h) + ", at distance = " + str(ai) + "from injection point.")
    print("Xmax height = ",hi)
    return hi, ai  # Xmax_height in m, Xmax_distance in m

# ============================================================================


def _get_CRXmaxPosition(zen, azim, xmax, injh=0, corePos=[0, 0, 0]):
    ''' Calculates vector to Xmax position for CRs

    Arguments:
    ----------
    zen: float
        GRAND zenith in deg, for CR shower use _get_CRzenith()
    azim: float
        GRAND azimuth in deg
    xmax: float
        Xmax in g/cm2
    injh: float (default set), for CR = 100000m (100km)
        injection height wrt sealevel in m
    core: numpy array (default set)
        shower core position

    Returns:
    ---------
    new: numpy array
        position of Xmax in m

    Note: *For neutrinos: the algorithm builds on the injection at (0,0,injh),
            accepts electron and pion primaries
          *For CR: it calculates the distance between origin at athomphere
            top along shower axis to get the Xmax position
            -- origin defined at (0,0,0)m at the moment
           *accepts only proton and iron primaries currently
    XXX / TODO: *add the option for a shower_core != (0,0,0)
                *cross check the output with matias simulations

    '''
    # Zenith angle at injection point, distance from injection to core
    zen_inj, a = _get_zenith_inj(zen, injh, corePos[2])
    a = (a / u.m).value

    # height of xmax, distance from decay to xmax
    h, ai = _dist_decay_Xmax(zen_inj, injh, xmax)
    ai = (ai / u.m).value
    ax = a - ai # DIstance from Xmax to ground

    # Now compute Xmax position in referential
    zenr = np.deg2rad((zen/u.deg).value)
    azimr = np.deg2rad((azim/u.deg).value)
    u_sh = np.array([np.cos(azimr) * np.sin(zenr), np.sin(azimr) * np.sin(zenr), np.cos(zenr)])  # TODO: switch to new angular conventions
    corePos = (corePos / u.m).value
    new = -abs(a-ai)*u_sh + corePos  # TODO: extend treatment to neutrinos
    new = new * u.m

    try:
        return new
    except:
        logger.error("Xmax position not calculated")

# ============================================================================


def _get_zenith_inj(zen, injh, GdAlt):
    ''' Compute zenith angle at injection point

    Arguments:
    ----------
    zen: float
        GRAND zenith in deg
    injh: float
        injection height wrt to sealevel in m
    GdAlt: float
        ground altitude of array/observer in m (should be substituted)

    Returns:
    --------
    zen_inj: float
        GRAND zenith computed at shower core position in deg

    NOTE: To be included in other functions
    '''

    from astropy.constants import R_earth
    Re = R_earth.value  # m, Earth radius
    injh = (injh / u.m).value
    GdAlt = (GdAlt / u.m).value
    zen = np.deg2rad((zen / u.deg).value)
    a = np.sqrt((Re + injh)**2. - (Re+GdAlt)**2 * np.sin(np.pi - zen)**2) - (Re+GdAlt)*np.cos(np.pi-zen) # Total track length
    zen_inj = np.rad2deg(np.pi-np.arccos((a**2 + (Re+injh)**2 - (Re+GdAlt)**2)/(2*a*(Re+injh))))  * u.deg
    print("Zenith angle at injection point:",zen_inj)
    return zen_inj, a * u.m

# ============================================================================


def _project_onShoweraxis(p, v, core=np.array([0., 0., 0.])):
    ''' calculates the orthogonal projection of a position onto the shower axis

    Arguments:
    ----------
    p: numpy array
        position
    v: numpy array
        shower direction
    core: numpy array (default set)
        core position

    Returns:
    --------
    numpy array
        projected position on shower axis
    '''

    v = v/np.linalg.norm(v)
    # tarnslate to local frame
    local_p = p - core
    # project on v == shower axis
    proj = np.dot(local_p, v)  # /np.dot(v,v)

    return core + proj*v
# ============================================================================


def _distance_toShoweraxis(p, v, core=np.array([0., 0., 0.])):
    '''calculates orthogonal distance of position to the shower axis,
    calls _project_onShoweraxis()

    Arguments:
    ----------
    p: numpy array
        position
    v: numpy array
        shower direction
    core: numpy array (default set)
        core position

    Returns:
    --------
    float
        distance to shower axis
    '''

    return np.linalg.norm(p - _project_onShoweraxis(p, v, core))


# ============================================================================

def get_LDF(trace, p, v, core=np.array([0., 0., 0.])):
    '''
    XXX / TODO : to be finished
    '''

    #from signal_treatment import p2p
    # loop over all antenna position and traces ...
    #p2p(trace) or hilbert

    # return distance to axis, Amplitude # in shower plane
    return 0
# ============================================================================


def correction():
    ''' returns correction factor for early late effect

    XXX / TODO : to be finished
    '''
    return 0

# ============================================================================


def correct_EarlyLate(trace):
    '''
    XXX / TODO : to be finished

    NOTE Rio guys have already a methods here...

    correct for early late effect,
    following the approch given arxiv:1808.00729 in for the energy fluence

    energy fluence [corrected in shower plane] = energy fluence [ sim. at ground] *factor**2
    -> for amplitude only: amp [corrected] = amp. [sim.] * factor ...
    '''

    # simply assume trace = np.array([t, x, y, z])

    # Needed input:
    # * trace to scale
    # * position p
    # * shower direction v
    # * shower core
    # * maybe Xmax position

    # Xmax position: calculated or simulated as input ...
    # x: distance along axis from projected antenna position to shower core
    # R: distance along axis from Xmax position to shower core

    #R0 = R + x
    #factor = R/R0

    return 0


# ============================================================================

def correct_chargeexcess():
    '''
    XXX / TODO : to be finished

    # get the strength of the charge excess wrt to geomagnetic
    # charge excess fraction is a sin(alpha) |Ec|/|Eg|
    # alpha=angle between B and v, Ec and Eg field according askaryan and geomagnetic effect
    # correction done in time and for each position
    '''
    return 0

# ============================================================================


def get_polarisation_vector(trace, zen, azim, phigeo, thetageo):
    '''
    XXX / TODO : to be finished
    '''
    # rotate electric field trace in shower coordinates, use frame.py for it
    # get polarisation:
    # get EvxB as horizontal component along propagtion direction: trace, max()
    # and EvxvxB as vertical component along propagtion direction: trace, max()
    # how to handle curvature of wavefront... work with line of sight...

    #B = np.array([np.cos(phigeo)*np.sin(inc), np.sin(phigeo)*np.sin(inc),np.cos(inc)])
    #B=B/np.linalg.norm(B)

    #v = np.array([np.cos(az)*np.sin(zen),np.sin(az)*np.sin(zen),np.cos(zen)])
    #v=v/np.linalg.norm(v)
    #print(v)

    #vxB = np.cross(v,B)
    #vxB = vxB/np.linalg.norm(vxB)
    #vxvxB = np.cross(v,vxB)
    #vxvxB = vxvxB/np.linalg.norm(vxvxB)

    # rotation to showeframe
    #Eshower= np.zeros([len(txt1.T[1]),3])
    #Eshower.T[0]= txt1.T[1]* v[0] +txt1.T[2]*v[1]+ txt1.T[3]*v[2]
    #Eshower.T[1]= txt1.T[1]* vxB[0] +txt1.T[2]*vxB[1]+ txt1.T[3]*vxB[2]
    #Eshower.T[2]= txt1.T[1]* vxvxB[0] +txt1.T[2]*vxvxB[1]+ txt1.T[3]*vxvxB[2]

    from frame import get_rotation
    # rotation in shower coordinates
    R = get_rotation(zen, azim, phigeo, thetageo)
    #Eshower=(Ev, EvxB, EvxvxB)
    Eshower = np.dot(trace[:, 1:], R)

    return 0

# -------------------------------------------------------------------


def main():
    if (len(sys.argv) > 1):
        print("""

            Usage: python3 modules.py
            Example: python3 modules.py

        """)
        sys.exit(0)

    primary = "proton"
    energy = 63e15
    zen = 180.-70.50
    azim = 180.-0.

    p = np.array([1, 1, 0])
    v = np.array([1, 2, 0])
    core = np.array([0., 0., 0.])

    #pos= _get_XmaxPosition(primary, energy, zen, azim, injh=100000)
    pos = _project_onShoweraxis(p, v, core)
    print(np.linalg.norm(p)**2.-np.linalg.norm(pos)**2.)
    print(pos, _distance_toShoweraxis(p, v, core)**2.)


if __name__ == "__main__":
    main()
