'''
    Translation of initial computeVoltage version (https://github.com/grand-mother/simulations/blob/master/computeVoltage_massProd.py)
    being adapted to GRAND software

    TODO:
        - problem with shape of A and B in computevoltage for time traces with odd number of lines
        - handle upward going showers
        - referential change (TopoToAntenna)
        - IMPORTANT: how to handle astropy units for electric field and voltage numpy arrays...

    ATTENTION:
    ----- computevoltage : stacking changed,  voltage time now in ns
'''


#!/usr/bin/env python
import os
from os.path import  join
import sys
import math
import numpy as np
import astropy.units as u

import logging
logger = logging.getLogger("ComputeVoltage")

import pylab as plt
import glob
from astropy import constants as const

#from . import signal_processing
from . import config
from . import io_utils
from . import shower
from . import detector
from . import modules

#from . import computevoltage_orig

#import _table_voltage,_load_to_array, _load_eventinfo_fromhdf, _load_path

import linecache
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.interpolate import interp1d
#import astropy.units as u

from ..config import load
load("/home/martineau/GRAND/soft/grand/tests/radio/config.py")

PRINT_ON=True
DISPLAY = 1

## Earth radius in m
from astropy import constants as const
#EARTH_RADIUS=6370949. #m

## step in azimuth in npy file in deg
azstep=5
## Multiplication factor: freq*2 if h/2 and sizeant/2
freqscale=1

##if antenna is loaded or not in npy file --- NOTE: Not needed
#loaded=1

#============================================================================
def compute_ZL(freq, DISPLAY = False, R = 300, C = 6.5e-12, L = 1e-6):
#============================================================================

  ''' Function to compute impedance from GRAND load

  Arguments:
  ----------
  freq: float
    frequency in Hz
  DISPLAY: True/False
    optional, plotting function
  R, C, L: float
    SI UNits: Ohms, Farrads, Henry

  Returns:
  ----------
  RL, XL: float
    Impdedance
  '''

  w = 2*np.pi*freq # Pulsation
  w2 = w*w
  # ZC = 1./(i*C*w)
  # ZL = i*L*w
  # ZR = R
  # Analytic formulas for R//C//L
  deno = (L*L*w2+R*R*(1-L*C*w2)*(1-L*C*w2))
  RL = R*L*L*w2/deno  # Computed formula
  XL = R*R*L*w*(1-L*C*w2)/deno  # Computed formula
  if DISPLAY:
    plt.figure(1)
    plt.plot(freq/1e6,RL,label="R$_L$")
    plt.plot(freq/1e6,XL,label="X$_L$")
    plt.legend(loc="best")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Load impedance ($\Omega$)")
    plt.xlim([min(freq/1e6), max(freq/1e6)])
    plt.show()

  return RL, XL
#============================================================================


def load_leff(**kwargs):
  ''' Function to load antenna response model (npy file)

  Arguments:
  ----------
  Path to antenna model files


  '''

  # Compute load impendance
  fr=np.arange(20,301,5)
  RLp, XLp = compute_ZL(fr*1e6)

  try:
      leffx, leffy,leffz = kwargs["leff"]  #TODO fix :)
  except KeyError:
      leff_x = config.antenna.leff.x
      leff_y = config.antenna.leff.y
      leff_z = config.antenna.leff.z


  #logger.debug("Loading antenna effective length file",leff_x)
  print('Loading',leff_x)
  global freq1,realimp1,reactance1,theta1,phi1,lefftheta1,leffphi1,phasetheta1,phasephi1,RL1,XL1
  freq1,realimp1,reactance1,theta1,phi1,lefftheta1,leffphi1,phasetheta1,phasephi1=np.load(leff_x) ### this line cost 6-7s
  print('Done.')
  RL1=interp1d(fr, RLp, bounds_error=False, fill_value=0.0)(freq1[:,0])
  XL1=interp1d(fr, XLp, bounds_error=False, fill_value=0.0)(freq1[:,0])

  print('Loading',leff_y)
  global freq2,realimp2,reactance2,theta2,phi2,lefftheta2,leffphi2,phasetheta2,phasephi2,RL2,XL2
  freq2,realimp2,reactance2,theta2,phi2,lefftheta2,leffphi2,phasetheta2,phasephi2=np.load(leff_y) ### this line cost 6-7s
  print('Done')
  RL2=interp1d(fr, RLp, bounds_error=False, fill_value=0.0)(freq2[:,0])
  XL2=interp1d(fr, XLp, bounds_error=False, fill_value=0.0)(freq2[:,0])

  print('Loading',leff_z)
  global freq3,realimp3,reactance3,theta3,phi3,lefftheta3,leffphi3,phasetheta3,phasephi3,RL3,XL3
  freq3,realimp3,reactance3,theta3,phi3,lefftheta3,leffphi3,phasetheta3,phasephi3=np.load(leff_z) ### this line cost 6-7s
  print('Done')
  RL3=interp1d(fr, RLp, bounds_error=False, fill_value=0.0)(freq3[:,0])
  XL3=interp1d(fr, XLp, bounds_error=False, fill_value=0.0)(freq3[:,0])



#============================================================================
def TopoToAntenna(u,alpha,beta):
#============================================================================

    '''from coordinates in the topography frame to coordinates in the antenna

    Arguments:
    ----------
    u: numpy array
        shower vector
    alpha: float
        surface angle alpha in deg
    beta: float
        surface beta alpha in deg

    Returns:
    ----------
    numpy array:
        shower vector in coordinates of antenna

    '''
    # TODO: update using GRAND referential classes. Move to modules.
    alpha=alpha*np.pi/180 #around y
    beta=beta*np.pi/180 #around z
    cb = np.cos(beta)
    sb = np.sin(beta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    roty = np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
    roty = np.linalg.inv(roty)  # Since we rotate referential, inverse transformation should be applied
    rotz = np.array([[cb,-sb,0],[sb,cb,0],[0,0,1]])
    rotz = np.linalg.inv(rotz) # Since we rotate referential, inverse transformation should be applied
    rotyz=roty.dot(rotz)  # beta and then alpha rotation. This induces a EW component for x arm

    # Now rotate along zp so that we are back with x along NS
    xarm = [1,0,0]  #Xarm
    xarmp = rotyz.dot(xarm)  # Rotate Xarm along slope
    # Compute antrot, angle of NS direction in antenna ref = angle to turn Xarm back to North
    antrot = math.atan2(xarmp[1],xarmp[0])*180/np.pi
    cz = np.cos(antrot*np.pi/180)
    sz = np.sin(antrot*np.pi/180)
    rotzant = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    rotzant = np.linalg.inv(rotzant)
    rottot = rotzant.dot(rotyz)

    [xp,yp,zp] = rottot.dot(u)
    return np.array([xp,yp,zp])

#===========================================================================================================
def get_voltage(t, Ex, Ey, Ez, zen, az,alpha=0, beta=0, typ="X"):
#===========================================================================================================
    ''' Applies the antenna response to the Efield

    Arguments:
    ----------
    time1: numpy array
        time in s (ATTENTION)
    Ex: numpy array
        x component in muV/m
    Ey: numpy array
        y component in muV/m
    Ez: numpy array
        z component in muV/m
    zen: float
        Effective zenith (ie source as seen at antenna position) in GRAND ref in deg
    azim: float
        Effective azimuth (ie source as seen at antenna position) in GRAND ref in deg
    alpha: float
        surface angle alpha in deg
    beta: float
        surface angle beta in deg
    typ: str
        hand over arm (X,Y,Z)

    Returns:
    --------
    voltage: numpy array
        voltage trace in one arm
    time: numpy array
        time trace
    '''

    # Load proper antenna response matrix
    if typ=="X":
       fileleff = config.antenna.leff.x
       freq=freq1
       realimp=realimp1
       reactance=reactance1
       theta=theta1
       phi=phi1
       lefftheta=lefftheta1
       leffphi=leffphi1
       phasetheta=phasetheta1
       phasephi=phasephi1
       RL=RL1
       XL=XL1
    if typ=="Y":
       fileleff = config.antenna.leff.y
       freq=freq2
       realimp=realimp2
       reactance=reactance2
       theta=theta2
       phi=phi2
       lefftheta=lefftheta2
       leffphi=leffphi2
       phasetheta=phasetheta2
       phasephi=phasephi2
       RL=RL2
       XL=XL2
    if typ=="Z":
       fileleff = config.antenna.leff.z
       freq=freq3
       realimp=realimp3
       reactance=reactance3
       theta=theta3
       phi=phi3
       lefftheta=lefftheta3
       leffphi=leffphi3
       phasetheta=phasetheta3
       phasephi=phasephi3
       RL=RL3
       XL=XL3

    # Compute effective theta, phi in antenna tilted frame (taking slope into account, with x=SN)  # TODO: cleaner way with GRAND frame classes?
    ush = -modules._get_shower_vector(zen,az)  # Vector pointing towards Xmax
    ushp = TopoToAntenna(ush,alpha,beta)  # Now  in antenna frame
    zenp, azp = modules._get_shower_angles(ushp)
    if azp>360 * u.deg:
        azp = azp-360 * u.deg
    elif azp<0 * u.deg:
        azp = azp+360 * u.deg

    if typ=='X':
        if PRINT_ON:
            print('Alpha and beta of surface slope:',alpha, beta)
            print('Zenith & azimuth in GRAND framework:',modules._get_shower_angles(ush))
            print('Zenith & azimuth in antenna framework:',zenp, azp)

    if zenp>90 * u.deg:
        print('Signal originates below antenna horizon! No antenna response computed. Abort.')
        logger.info('Signal originates below antenna horizon! No antenna response computed. Abort.')
        return([],[])

    # Now take care of Efield signals
    delt = np.mean(np.diff(t));
    Fs = round(1/delt)
    timeoff=t[0] # time offset, to get absolute time
    t = t-timeoff #reset to zero

    # Rotate Efield to antenna frame (x along actual arm)
    Etot=np.array([Ex,Ey,Ez])
    [Exp,Eyp,Ezp] = TopoToAntenna(Etot,alpha,beta)  #TODO: cleaner way?
    szen = np.sin(zenp);  # numpy takes care about angle units (deg), must not apply pi/180 factor here
    czen = np.cos(zenp);
    saz = np.sin(azp);
    caz = np.cos(azp);
    print("Zen & Az = ",zenp,azp)
    #print(czen,szen,caz,saz)

    amplituder = szen*(caz*Exp+saz*Eyp)+czen*Ezp
    amplitudet = czen*(caz*Exp+saz*Eyp)-szen*Ezp
    amplitudep = -saz*Exp+caz*Eyp

    ###plots
    if DISPLAY==2:
        import pylab as pl
        import matplotlib.pyplot as plt

        plt.figure(1,  facecolor='w', edgecolor='k')
        plt.title(typ)
        plt.subplot(211)
        plt.plot(t*1e9,Eyp, label="Ey = EW")
        plt.plot(t*1e9,Exp, label="Ex = NS")
        plt.plot(t*1e9,Ezp, label="Ez = UP")

        plt.xlabel('Time (nsec)')
        plt.ylabel('Electric field (muV/m)')
        plt.legend(loc='best')
        plt.subplot(212)
        plt.plot(t*1e9,amplituder, label="E$_{ r}$")
        plt.plot(t*1e9,amplitudet, label="E$_{ \Theta}$")
        plt.plot(t*1e9,amplitudep, label="E$_{ \Phi}$")
        plt.xlabel('Time (nsec)')
        plt.ylabel('Electric field (muV/m)')
        plt.legend(loc='best')
        plt.show()


    ##################################
    ### all the settings for the 3 different antenna arms:

    nfreq=len(freq[:,0])
    f=np.zeros(nfreq)
    RA=np.zeros(nfreq)
    XA=np.zeros(nfreq)
    ltr1=np.zeros(nfreq)
    lta1=np.zeros(nfreq)
    lpr1=np.zeros(nfreq)
    lpa1=np.zeros(nfreq)
    ltr2=np.zeros(nfreq)
    lta2=np.zeros(nfreq)
    lpr2=np.zeros(nfreq)
    lpa2=np.zeros(nfreq)

    if azstep==5:
        roundazimuth=round(azp.value/10)*10+round((azp.value-10*round(azp.value/10))/5)*5
    elif azstep==1:
        roundazimuth=round(azp.value)
    else:
        print('Error on azimuth step!')
        logger.error('Error on azimuth step!')
        return(0)
    if roundazimuth>=91 and roundazimuth<=180:
        roundazimuth=180-roundazimuth
    if roundazimuth>=181 and roundazimuth<=270:
        roundazimuth=roundazimuth-180
    if roundazimuth>=271 and roundazimuth<=360:
        roundazimuth=360-roundazimuth

    for i in range(nfreq):   # Using interpolation for every angle
        f[i]=freq[i,0]*freqscale
        indtheta=np.nonzero(theta[i,:]==int(zenp.value))[0]
        indphi=np.nonzero(phi[i,:]==roundazimuth)[0]
        indcom=np.intersect1d(indtheta,indphi)
        ltr1[i]=lefftheta[i,indcom]
        lta1[i]=np.deg2rad(phasetheta[i,indcom]) #*np.pi/180
        lpr1[i]=leffphi[i,indcom]
        lpa1[i]=np.deg2rad(phasephi[i,indcom]) #*np.pi/180
        indtheta=np.nonzero(theta[i,:]==int(zenp.value)+1)[0]
        indphi=np.nonzero(phi[i,:]==roundazimuth)[0]
        indcom=np.intersect1d(indtheta,indphi)
        ltr2[i]=lefftheta[i,indcom]
        lta2[i]=np.deg2rad(phasetheta[i,indcom]) #*np.pi/180
        lpr2[i]=leffphi[i,indcom]
        lpa2[i]=np.deg2rad(phasephi[i,indcom]) #*np.pi/180

        ltr=interp1d([int(zenp.value),int(zenp.value)+1],np.transpose([ltr1,ltr2]))(zenp.value)
        lta=interp1d([int(zenp.value),int(zenp.value)+1],np.transpose([lta1,lta2]))(zenp.value)
        lpr=interp1d([int(zenp.value),int(zenp.value)+1],np.transpose([lpr1,lpr2]))(zenp.value)
        lpa=interp1d([int(zenp.value),int(zenp.value)+1],np.transpose([lpa1,lpa2]))(zenp.value)

    ###############################
    # Now go for the real thing

    fmin=f[0]
    fmax=f[-1]
    f=f*1e6
    #nf  = int(2**np.floor(np.log(len(amplitudet))/np.log(2))) # changed in July2019 - shortens the trace
    nf = len(amplitudet) # TODO: check that this is still OK?
    while Fs/nf > fmin*1e6:   # <== Make sure that the DFT resolution is at least fmin.
        nf *= 2
    F = rfftfreq(nf)*Fs

    modulust = interp1d(f, ltr, bounds_error=False, fill_value=0.0)(F)
    phaset   = interp1d(f, lta, bounds_error=False, fill_value=0.0)(F)
    modulusp = interp1d(f, lpr, bounds_error=False, fill_value=0.0)(F)
    phasep   = interp1d(f, lpa, bounds_error=False, fill_value=0.0)(F)

    phaset -= phaset[0] # Switch the phase origin to be consistent with a real signal.
    phasep -= phasep[0] # Switch the phase origin to be consistent with a real signal.

    # Switch to frequency domain
    #B and D are V in freq domain, they are complex
    A = rfft(amplitudet, nf)
    ct = np.cos(phaset)
    st = np.sin(phaset)
    B = np.zeros(A.shape)
    B[1:-1:2] = modulust[1:-1:2]*(A[1:-1:2]*ct[1:-1:2]-A[2:-1:2]*st[2:-1:2])
    B[2:-1:2] = modulust[2:-1:2]*(A[1:-1:2]*st[1:-1:2]+A[2:-1:2]*ct[2:-1:2])
    B[0]  = A[0]*modulust[0]
    B[-1] = A[-1]*modulust[-1]

    C = rfft(amplitudep, nf)
    cp = np.cos(phasep)
    sp = np.sin(phasep)
    D = np.zeros(C.shape)
    D[1:-1:2] = modulusp[1:-1:2]*(C[1:-1:2]*cp[1:-1:2]-C[2:-1:2]*sp[2:-1:2])
    D[2:-1:2] = modulusp[2:-1:2]*(C[1:-1:2]*sp[1:-1:2]+C[2:-1:2]*cp[2:-1:2])
    D[0]  = C[0]*modulusp[0]
    D[-1] = C[-1]*modulusp[-1]

    vt=irfft(B)
    vp=irfft(D)
    voltage = vp + vt
    timet     = np.arange(0, len(vt))/Fs
    timep     = np.arange(0, len(vp))/Fs

    return(voltage, timet+timeoff)

#===========================================================================================================
# Compute the time dependent voltage
#===========================================================================================================
if __name__ == '__main__':
    print("computeVoltageNew()")

    # Load simulated event HDF5 file
    exfile = "/home/martineau/GRAND/GRANDproto300/data/test/coreas/event_000001.hdf5"

    #from ..config import load
    #load("/home/martineau/GRAND/soft/grand/tests/radio/config.py")
    load_leff()

    # First retrieve shower infos
    #1st method: using dicts
    sh, ant_ID, positions, slopes = io_utils._load_eventinfo_fromhdf(exfile)
    print(sh.keys())
    print("Zenith=",sh["zenith"])
    print("Azimuth=",sh["azimuth"])
    print("Core=",sh["core"])
    print("Energy=",sh["energy"])
    print("Xmax=",sh["xmax"])
    print("Injection height=",sh["injection_height"])
    try:
         xmax = sh["xmax"]
    except KeyError:
         # No Xmax value available, compute it from (average) model.
         xmax = modules._calculateXmax(sh["primary"],sh["energy"].value)

    if sh["primary"] == "proton" or sh["primary"] == "iron":
        xmaxpos = modules._get_CRXmaxPosition( sh["zenith"], sh["azimuth"],sh["xmax"], sh["injection_height"],sh["core"])
    else:
        print("No computation yet!")  # TODO
    print("Xmax position in GRAND ref=",xmaxpos)
    print("Distance to ground=",np.linalg.norm(xmaxpos),"m")

    # #2nd method: using class
    # sh = shower.SimulatedShower()
    # from astropy.table import Table
    # g = Table.read(exfile, path="/event")
    # shower.loadInfo_toShower(sh, g.meta)
    # print(sh.energy)

    # Now get antenna positions from hdf5 file, using dicts
    arrayfile = "/home/martineau/GRAND/soft/grand/lib/python/grand/example/det.txt"
    det = detector.Detector()
    det.create_from_file(arrayfile)      #create detector=antenna array from file defined in config file
    array = det.position  # get all antennas positions
    IDs = det.ID  # get all antennas IDs  #TODO: change IDs to int instead of float

    # loop over existing single antenna files as raw output from simulations
    logger.info("Looping over antennas ...")
    # Coordinates in a local frame (ENU by default)
    from grand import ECEF, LTP
    obstime = "2019-12-25"
    origin = ECEF(latitude=38.85 * u.deg, longitude=92.5 * u.deg, height=0 * u.m,representation_type="geodetic")  # TODO: include this in the config file
    #print("Center of GRAND array:",origin)
    for i in range(np.size(IDs)):  # Would be more logical to loop on all existing traces rather than units in the detector
        # First fetch traces
        efield = io_utils._load_efield_fromhdf(exfile,ant = "/"+str(int(IDs[i])))
        if np.size(efield) == 1:  # No trace for that antenna
            print("No signal on antenna",int(IDs[i]),", skipping it.")
            continue

        # Now  build effective angles to Xmax (source)
        print("Now working on antenna",int(IDs[i]),".")
        ant = array[i]
        ush = (ant-xmaxpos)/np.linalg.norm(xmaxpos-ant)
        ush = ush / u.m
        #ush = LTP(ush,location=origin, magnetic=True, orientation=("N", "W", "U"), obstime=obstime) # How can we get coordinates values???
        print("Source radiating in direction:", ush)
        print("towards position",ant)
        zen_eff,az_eff = modules._get_shower_angles(ush)  # Effective zenith & az angles (ie Xmax pos seen from the antenna)
        print("Corresponding effective angles:", zen_eff, az_eff)

        alpha = 0 # TODO: fetch antenna slope
        beta = 0
        # Now go to work
        voltage_NS, timeNS = get_voltage(efield.T[0]*1e-9, efield.T[1], efield.T[2], efield.T[3], zen_eff, az_eff, alpha, beta, typ="X")
        voltage_EW, timeEW = get_voltage(efield.T[0]*1e-9, efield.T[1], efield.T[2], efield.T[3], zen_eff, az_eff, alpha, beta, typ="Y")
        voltage_vert, timevert = get_voltage(efield.T[0]*1e-9, efield.T[1], efield.T[2], efield.T[3], zen_eff, az_eff, alpha, beta, typ="Z")

        ###plots
        if DISPLAY>0:
            import pylab as pl
            import matplotlib.pyplot as plt

            plt.figure(17,  facecolor='w', edgecolor='k')
            plt.subplot(311)
            plt.plot(efield.T[0],efield.T[2], label="Ey = EW")
            plt.plot(efield.T[0],efield.T[1], label="Ex = NS")
            plt.plot(efield.T[0],efield.T[3], label="Ez = UP")
            plt.xlabel('Time (nsec)')
            plt.ylabel('Electric field (muV/m)')
            plt.legend(loc='best')
            plt.subplot(312)
            plt.plot(timeEW*1e9,voltage_EW, label="EW")
            plt.plot(timeNS*1e9,voltage_NS, label="NS")
            plt.plot(timevert*1e9,voltage_vert, label="Vertical")
            plt.xlabel('Time (nsec)')
            plt.ylabel('Voltage (muV)')
            plt.legend(loc='best')

            # Now compare to old computationss
            # voltage_NS, timeNS  = computevoltage_orig.get_voltage(efield.T[0]*1e-9, efield.T[1], efield.T[2], efield.T[3], -ush, alpha, beta, typ="X")
            # voltage_EW, timeEW  = computevoltage_orig.get_voltage(efield.T[0]*1e-9, efield.T[1], efield.T[2], efield.T[3], -ush, alpha, beta, typ="Y")
            # voltage_vert, timevert  = computevoltage_orig.get_voltage(efield.T[0]*1e-9, efield.T[1], efield.T[2], efield.T[3], -ush, alpha, beta, typ="Z")
            # plt.subplot(313)
            # plt.plot(timeEW*1e9,voltage_EW, label="EW")
            # plt.plot(timeNS*1e9,voltage_NS, label="NS")
            # plt.plot(timevert*1e9,voltage_vert, label="Vertical")
            # plt.xlabel('Time (nsec)')
            # plt.ylabel('Voltage (muV)')
            # plt.legend(loc='best')

            plt.show()


        # ATTENTION EW AND NS WERE SWITCHED
        # ATTENTION voltage time now in ns
        #return np.vstack((timeNS*1e9,voltage_NS,voltage_EW,voltage_vert)) # switched to be consistent to efield treatment
        #return np.stack([timeNS*1e9,voltage_NS,voltage_EW,voltage_vert], axis=-1)
