from pathlib import Path
import numpy as np
import grand.analysis.constants as cons
import math
from numba import njit

kwd = {"fastmath": {"reassoc", "contract", "arcp"}}

@njit(**kwd)
def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+cons.R_earth)**2 + R2 ) - cons.R_earth)/1e3 # Altitude in km
    rh = cons.ns*np.exp(cons.kr*h)
    n = 1.+1e-6*rh
    
    return (n)

@njit(**kwd)
def ZHSEffectiveRefractionIndex(X0,Xa):

    R02 = X0[0]**2 + X0[1]**2
    
    # Altitude of emission in km
    h0 = (math.sqrt( (X0[2]+cons.R_earth)**2 + R02 ) - cons.R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)
    
    # Refractivity at emission 
    # rh0 = cons.ns*math.exp(cons.kr*h0)

    modr = math.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        #nint = math.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth  = (math.sqrt( (Next[2]+cons.R_earth)**2 + nextR2 ) - cons.R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (math.exp(cons.kr*nexth)-math.exp(cons.kr*currh))/(cons.kr*(nexth-currh))
            else:
                s += math.exp(cons.kr*currh)

            Curr = Next
            currh = nexth
            # print (currh)

        avn = cons.ns*s/nint
        # print(avn)
        n_eff = 1. + 1e-6*avn # Effective (average) index

    else:

        # without numerical integration
        hd = Xa[2]/1e3 # Antenna altitude
        #if (np.abs(hd-h0) > 1e-10):
        avn = (cons.ns/(cons.kr*(hd-h0)))*(math.exp(cons.kr*hd)-math.exp(cons.kr*h0))
        #else:
        #    avn = ns*math.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)


# --------------------------------------
# For regularisation of the SWF fit
# --------------------------------------
@njit
def DensityAtAltitude(h):
    """
    Atmospheric density at altitude h (m).

    Returns
    -------
    rho : float
        Density in g/cm³.
    """

    if h >= cons.h_top_atm:
        return 0.

    layer = 0

    for i in range(1, 5):
        if h >= cons.linsley_h[i]:
            layer = i

    b = cons.linsley_b[layer]
    c = cons.linsley_c[layer]

    if layer == 4:
        return b / c

    return b / c * np.exp(-h * 100. / c)


# --------------------------------------
# Tabulated atmospheres (alternative to the default Linsley model)
# --------------------------------------
# File name of each tabulated model, next to this module.
# Columns: altitude a.s.l. [m], density [g/cm3], vertical depth [g/cm2].
ATMOSPHERE_TABLES = {
    "DunhuangFromChao": "atmosphere_DunhuangFromChao.txt",  # ZHAireS GP300 (DC2)
}
_loaded_atmospheres = {}


def load_atmosphere(name="Linsley"):
    """
    Atmosphere used by SlantDepth: None for the default Linsley model, or the
    (2, N) array [altitude (m), log(density in g/cm3)] of a tabulated model.
    """
    if name == "Linsley":
        return None

    if name not in ATMOSPHERE_TABLES:
        raise ValueError(
            f"Unknown atmosphere model {name!r}; "
            f"available: Linsley, {', '.join(ATMOSPHERE_TABLES)}"
        )

    if name not in _loaded_atmospheres:
        path = Path(__file__).parent / ATMOSPHERE_TABLES[name]
        alt, rho, _ = np.loadtxt(path, unpack=True)
        _loaded_atmospheres[name] = np.ascontiguousarray([alt, np.log(rho)])

    return _loaded_atmospheres[name]


def atmosphere_top(atm=None):
    """Top of the atmosphere (m a.s.l.): density is 0 above it."""
    return cons.h_top_atm if atm is None else float(atm[0, -1])


@njit
def DensityFromTable(h, atm):
    """Density (g/cm3) at altitude h (m) from a tabulated atmosphere
    (log-linear interpolation, 0 above the table top)."""
    if h > atm[0, -1]:
        return 0.
    return np.exp(np.interp(h, atm[0], atm[1]))


@njit
def _density(h, atm):
    if atm is None:
        return DensityAtAltitude(h)
    return DensityFromTable(h, atm)


@njit
def SlantDepth(X0, u, nsteps=200, atm=None):
    """
    Atmospheric depth from X0 to the top of the atmosphere.

    Parameters
    ----------
    X0 : array (3,)
        Starting position in m.
    u : array (3,)
        Unit direction towards the top of the atmosphere.
    nsteps : int
        Number of integration steps.
    atm : None or array (2, N)
        Atmosphere from load_atmosphere(): None = Linsley (default).

    Returns
    -------
    X : float
        Atmospheric depth in g/cm².
    """

    R = cons.earth_radius_m
    if atm is None:
        h_top = cons.h_top_atm
    else:
        h_top = atm[0, -1]

    # Position relative to Earth's center
    qx = X0[0]
    qy = X0[1]
    qz = X0[2] + R

    r0 = np.sqrt(qx**2 + qy**2 + qz**2)
    h0 = r0 - R

    if h0 >= h_top:
        return 0.

    if h0 < 0.:
        return np.nan

    # Intersection with the top of the atmosphere
    b = qx*u[0] + qy*u[1] + qz*u[2]
    c = r0**2 - (R + h_top)**2

    if b < 0.:
        return np.nan

    s_top = -b + np.sqrt(b**2 - c)

    # Trapezoidal integration
    X = 0.
    s_prev = 0.
    rho_prev = _density(h0, atm)

    for i in range(1, nsteps + 1):

        s = s_top * (i / nsteps)**2

        x = X0[0] + s*u[0]
        y = X0[1] + s*u[1]
        z = X0[2] + s*u[2]

        h = np.sqrt(x**2 + y**2 + (z + R)**2) - R

        rho = _density(h, atm)

        X += 0.5 * (rho_prev + rho) * (s - s_prev) * 100.

        s_prev = s
        rho_prev = rho

    return X