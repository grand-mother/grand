import numpy as np
import grand.analysis.constants as cons

def RefractionIndexAtPosition(X):

    R2 = X[0]*X[0] + X[1]*X[1]
    h = (np.sqrt( (X[2]+cons.R_earth)**2 + R2 ) - cons.R_earth)/1e3 # Altitude in km
    rh = cons.ns*np.exp(cons.kr*h)
    n = 1.+1e-6*rh
    
    return (n)

def ZHSEffectiveRefractionIndex(X0,Xa):

    R02 = X0[0]**2 + X0[1]**2
    
    # Altitude of emission in km
    h0 = (np.sqrt( (X0[2]+cons.R_earth)**2 + R02 ) - cons.R_earth)/1e3
    # print('Altitude of emission in km = ',h0)
    # print(h0)
    
    # Refractivity at emission 
    rh0 = cons.ns*np.exp(cons.kr*h0)

    modr = np.sqrt(R02)
    # print(modr)

    if (modr > 1e3):

        # Vector between antenna and emission point
        U = Xa-X0
        # Divide into pieces shorter than 10km
        #nint = np.int(modr/2e4)+1
        nint = int(modr/2e4)+1
        K = U/nint

        # Current point coordinates and altitude
        Curr  = X0
        currh = h0
        s = 0.

        for i in np.arange(nint):
            Next = Curr + K # Next point
            nextR2 = Next[0]*Next[0] + Next[1]*Next[1]
            nexth  = (np.sqrt( (Next[2]+cons.R_earth)**2 + nextR2 ) - cons.R_earth)/1e3
            if (np.abs(nexth-currh) > 1e-10):
                s += (np.exp(cons.kr*nexth)-np.exp(cons.kr*currh))/(cons.kr*(nexth-currh))
            else:
                s += np.exp(cons.kr*currh)

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
        avn = (cons.ns/(cons.kr*(hd-h0)))*(np.exp(cons.kr*hd)-np.exp(cons.kr*h0))
        #else:
        #    avn = ns*np.exp(kr*h0)

        n_eff = 1. + 1e-6*avn # Effective (average) index

    return (n_eff)