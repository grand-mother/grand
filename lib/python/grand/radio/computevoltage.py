'''
    original from HandsOn session 19/04/2019
    --- several times modified
    
    TODO:
        - problem with shape of A and B in computevoltage for time traces with odd number of lines
        - handle neutrinos and in general upward going showers
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

import logging
logger = logging.getLogger("ComuteVoltage")

import pylab as plt
import glob

from radio_simus.signal_processing import filters
from .__init__ import antx, anty, antz

import linecache
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.interpolate import interp1d
#import astropy.units as u


PRINT_ON=False
DISPLAY = 0

## Earth radius in m
EARTH_RADIUS=6370949. #m
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



# Compute load impendance
#impRLC R = 300;C = 6.5e-12;L = 1e-6; 20 300 MHz
fr=np.arange(20,301,5)
RLp, XLp = compute_ZL(fr*1e6)

print('--- ATTENTION: current version of computevoltage only valid for Cosmic Rays') 

#wkdir = '/home/laval1NS/zilles/radio-simus/lib/python/radio_simus/GRAND_antenna/'
# Load antenna response files
freespace = 0
#if freespace==1:
  #fileleff_x=wkdir+'butthalftripleX4p5mfreespace_leff.npy' # 
  #fileleff_y=wkdir+'butthalftripleY4p5mfreespace_leff.npy' # 'HorizonAntenna_leff_notloaded.npy' if loaded=0, EW component
  #fileleff_z=wkdir+'butthalftripleZ4p5mfreespace_leff.npy'
#else:
  #fileleff_x=wkdir+'HorizonAntenna_SNarm_leff_loaded.npy' # 'HorizonAntenna_leff_notloaded.npy' if loaded=0, NS component
  #fileleff_y=wkdir+'HorizonAntenna_EWarm_leff_loaded.npy' # 'HorizonAntenna_leff_notloaded.npy' if loaded=0, EW component
  #fileleff_z=wkdir+'HorizonAntenna_Zarm_leff_loaded.npy' # 'HorizonAntenna_leff_notloaded.npy' if loaded=0, Vert component
  
fileleff_x = antx
fileleff_y = anty
fileleff_z = antz

if PRINT_ON:
    print('Loading',fileleff_x,'...')  
logger.debug("Loading antenna files " + fileleff_x +"....")
freq1,realimp1,reactance1,theta1,phi1,lefftheta1,leffphi1,phasetheta1,phasephi1=np.load(fileleff_x) ### this line cost 6-7s
RL1=interp1d(fr, RLp, bounds_error=False, fill_value=0.0)(freq1[:,0])
XL1=interp1d(fr, XLp, bounds_error=False, fill_value=0.0)(freq1[:,0])
freq2,realimp2,reactance2,theta2,phi2,lefftheta2,leffphi2,phasetheta2,phasephi2=np.load(fileleff_y) ### this line cost 6-7s
RL2=interp1d(fr, RLp, bounds_error=False, fill_value=0.0)(freq2[:,0])
XL2=interp1d(fr, XLp, bounds_error=False, fill_value=0.0)(freq2[:,0])
freq3,realimp3,reactance3,theta3,phi3,lefftheta3,leffphi3,phasetheta3,phasephi3=np.load(fileleff_z) ### this line cost 6-7s
RL3=interp1d(fr, RLp, bounds_error=False, fill_value=0.0)(freq3[:,0])
XL3=interp1d(fr, XLp, bounds_error=False, fill_value=0.0)(freq3[:,0])
if PRINT_ON:
    print('Done.')




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
def get_voltage(time1, Ex, Ey, Ez, zenith_sim, azimuth_sim,alpha=0, beta=0, typ="X"):
#===========================================================================================================
    ''' Applies the antenna response
    
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
    zenith_sim: float
        GRAND zenith in deg
    azimuth_sim: float
        GRAND azimuth in deg
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
       fileleff = fileleff_x
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
       fileleff = fileleff_y
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
       fileleff = fileleff_z
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

    # Compute effective theta, phi in antenna tilted frame (taking slope into account, with x=SN)
    caz = np.cos(np.deg2rad(azimuth_sim))
    saz = np.sin(np.deg2rad(azimuth_sim))
    czen = np.cos(np.deg2rad(zenith_sim))
    szen = np.sin(np.deg2rad(zenith_sim))
    ush = -np.array([caz*szen, saz*szen,czen])  # Vector pointing towards source
    ushp = TopoToAntenna(ush,alpha,beta)  # Xmax vector in antenna frame
    zen=np.arccos(ushp[2])*180/np.pi  # Zenith in antenna frame
    azim=math.atan2(ushp[1],ushp[0])*180/np.pi
    if azim>360:
        azim = azim-360
    elif azim<0:
        azim = azim+360
    if typ=='X':
        if PRINT_ON:
            print('Alpha and beta of surface slope:',alpha, beta)
            print('Zenith & azimuth in GRAND framework:',zenith_sim, azimuth_sim)
            print('Zenith & azimuth in antenna framework:',zen, azim)
    
    if (freespace==0) and (zen>90):
        print('Signal originates below antenna horizon! No antenna response computed. Abort.')
        logger.info('Signal originates below antenna horizon! No antenna response computed. Abort.')
        return([],[])
    
    # Now take care of Efield signals
    delt = np.mean(np.diff(time1));
    Fs = round(1/delt)
    timeoff=time1[0] # time offset, to get absolute time
    time1 = (time1-time1[0]) #reset to zero
    # Rotate Efield to antenna frame (x along actual arm)
    Etot=np.array([Ex,Ey,Ez])
    [Exp,Eyp,Ezp] = TopoToAntenna(Etot,alpha,beta)
    szen = np.sin(zen*np.pi/180);
    czen = np.cos(zen*np.pi/180);
    saz = np.sin(azim*np.pi/180);
    caz = np.cos(azim*np.pi/180);
    #amplituder = szen*(caz*Exp+saz*Eyp)+czen*Ezp
    amplitudet = czen*(caz*Exp+saz*Eyp)-szen*Ezp
    amplitudep = -saz*Exp+caz*Eyp
    # if typ == "Z":
    #     plt.figure(12)
    #     plt.plot(Exp)
    #     plt.plot(Eyp)
    #     plt.plot(Ezp)

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
        roundazimuth=round(azim/10)*10+round((azim-10*round(azim/10))/5)*5
    elif azstep==1:
        roundazimuth=round(azim)
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
        indtheta=np.nonzero(theta[i,:]==int(zen))[0]
        indphi=np.nonzero(phi[i,:]==roundazimuth)[0]
        indcom=np.intersect1d(indtheta,indphi)
        ltr1[i]=lefftheta[i,indcom]
        lta1[i]=np.deg2rad(phasetheta[i,indcom]) #*np.pi/180
        lpr1[i]=leffphi[i,indcom]
        lpa1[i]=np.deg2rad(phasephi[i,indcom]) #*np.pi/180
        indtheta=np.nonzero(theta[i,:]==int(zen)+1)[0]
        indphi=np.nonzero(phi[i,:]==roundazimuth)[0]
        indcom=np.intersect1d(indtheta,indphi)
        ltr2[i]=lefftheta[i,indcom]
        lta2[i]=np.deg2rad(phasetheta[i,indcom]) #*np.pi/180
        lpr2[i]=leffphi[i,indcom]
        lpa2[i]=np.deg2rad(phasephi[i,indcom]) #*np.pi/180

        ltr=interp1d([int(zen),int(zen)+1],np.transpose([ltr1,ltr2]))(zen)
        lta=interp1d([int(zen),int(zen)+1],np.transpose([lta1,lta2]))(zen)
        lpr=interp1d([int(zen),int(zen)+1],np.transpose([lpr1,lpr2]))(zen)
        lpa=interp1d([int(zen),int(zen)+1],np.transpose([lpa1,lpa2]))(zen)

    ###############################
    # Now go for the real thing
    
    fmin=f[0]
    fmax=f[-1]
    f=f*1e6
    #nf  = int(2**np.floor(np.log(len(amplitudet))/np.log(2))) # changed in July2019 - shortens the trace
    nf = len(amplitudet)
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
def compute_antennaresponse(signal, zenith_sim, azimuth_sim, alpha=0., beta=0.):
#===========================================================================================================
    ''' 
    calls compute voltage for each component and stacks the results
    
    Parameters
    -----------
    signal: numpy array
        electric field trace time/ns, Ex, Ey,Ez in muV/m
    zenith_sim: float
        zenith of shower in deg (GRAND)
    azimuth_sim: float
        azimuth of shower in deg (GRAND)
    alpha, beta: float
        antenna angles, optional, in deg
        
    Returns
    ---------
    numpy array
        voltage traces, Time in ns, Vx,Vy,Vz in muV
    '''
    
    voltage_NS, timeNS = get_voltage(signal.T[0]*1e-9, signal.T[1], signal.T[2], signal.T[3], zenith_sim, azimuth_sim, alpha, beta, typ="X")
    voltage_EW, timeEW = get_voltage(signal.T[0]*1e-9, signal.T[1], signal.T[2], signal.T[3], zenith_sim, azimuth_sim, alpha, beta, typ="Y")
    voltage_vert, timevert = get_voltage(signal.T[0]*1e-9, signal.T[1], signal.T[2], signal.T[3],zenith_sim, azimuth_sim, alpha, beta, typ="Z")
        
    # ATTENTION EW AND NS WERE SWITCHED 
    # ATTENTION voltage time now in ns 
    #return np.vstack((timeNS*1e9,voltage_NS,voltage_EW,voltage_vert)) # switched to be consistent to efield treatment
    return np.stack([timeNS*1e9,voltage_NS,voltage_EW,voltage_vert], axis=-1)


##===========================================================================================================
#def inputfromtxt(input_file_path):
##===========================================================================================================
    #''' ATTENTION Should be deleted and substituted in modules
    #'''

    #particule = ['eta','pi+','pi-','pi0','Proton','p','proton','gamma','Gamma','electron','Electron','e-','K+','K-','K0L','K0S','K*+'
    #,'muon+','muon-','Muon+','Muon-','mu+','mu-','tau+','tau-','nu(t)','Positron','positron','e+']


    #showerID=str(input_file_path).split("/")[-2] # should be equivalent to folder name
    #datafile = input_file_path+'/inp/'+showerID+'.inp'
    ##datafile = glob.glob(input_file_path+'/inp/*.inp')[0]
    #if os.path.isfile(datafile) ==  False:  # File does not exist 
        #try:
            #datafile = input_file_path+'/'+showerID+'.inp' 
            #if os.path.isfile(datafile) ==  True:  # File does not exist 
                #print('\n Get shower parameters from ', datafile )
        #except IOError:
            #print('--- ATTENTION: Could not find ZHaireS input file in folder',datafile,'! Aborting.')
            #exit()
    #else: # File exists
      #print('\n Get shower parameters from ', datafile )
    ## ToDo: implement line-by-line reading 
    #data = open(datafile, 'r') 
    #for line in data:
        #if 'PrimaryZenAngle' in line:
            #zen=float(line.split(' ',-1)[1])
            #zen = 180-zen  #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
        #if 'PrimaryAzimAngle' in line:
            #azim = float(line.split(' ',-1)[1])+180 #conversion to GRAND convention i.e. pointing towards antenna/propagtion direction
            #if azim>=360:
                #azim= azim-360
    #try:
        #zen
    #except NameError:
        #print('--- ATTENTION: zenith could not be read-in')
        #zen = 100. #Case of a cosmic for which no injection height is defined in the input file and is then set to 100 km by ZHAireS
    #try:
        #azim
    #except NameError:
        #print('--- ATTENTION: azimuth could not be read-in')
        #azim = 0
        
    #print('Shower direction from input file -- azmimuth/deg ='+str(azim)+' , zenith/deg = '+str(zen))

    #return zen,azim

#===========================================================================================================
def compute(opt_input,path, zenith_sim, azimuth_sim):
#===========================================================================================================
    ''' Reads in efield in ascii format (zhaires) and applies the antenna reponse t one antenna or whole array
        By default grep all antennas from the antenna file
        
        Arguments:
        ----------
        opt_input: str
            hand over mode txt or manual
        path: str
            path to event
        zenith_sim: float
            GRAND zenith in deg
        zenith_sim: float
            GRAND zenith in deg
        
        Returns:
        --------
            --
        Note: Possible plot of traces if desired
    '''
    
    
    posfile = path + '/antpos.dat'
    positions=np.genfromtxt(posfile)
    start=0
    end=len(positions)

    alpha_sim=0
    beta_sim=0
    if opt_input=='txt' and len(sys.argv)==4: # just one specific antenna handed over
            start=int(sys.argv[-1]) # antenna ID
            end=start+1
    elif opt_input=='manual':
        if len(sys.argv)>=7: # Slope paras
            alpha_sim = float(sys.argv[5]) # antenna slope
            beta_sim = float(sys.argv[6]) # antenna slope
            print('Antenna slope: (',alpha_sim,',',beta_sim,')')
        if len(sys.argv)==8: # just one specif antenna handed over
            start=int(sys.argv[-1]) # antenna ID
            end=start+1

    

    print('Now looping over',end-start,'antenna(s) in folder',path)
    for l in range(start,end):
         
        efieldtxt=path+'/a'+str(l)+'.trace'
        print('\n** Antenna',l,', Efield file:',efieldtxt)

        try:
	    # Loading traces
            efield = np.loadtxt(efieldtxt,usecols=(0,1,2,3),unpack=True)

        except IOError:
            print('IOError: file not found')

        
        
        ### APPLY ANTENNA RESPONSE
        trace = compute_antennaresponse(efield, zenith_sim, azimuth_sim, alpha=alpha_sim, beta=beta_sim )

        ### TODO: add here peak-to-peak computation, but how do we wann store it...
        
        
        ###plots
        if DISPLAY==1:
            import pylab as pl
            import matplotlib.pyplot as plt
            
            plt.figure(1,  facecolor='w', edgecolor='k')
            plt.subplot(211)
            plt.plot(efield[0],efield[2], label="Ey = EW")
            plt.plot(efield[0],efield[1], label="Ex = NS")
            plt.plot(efield[0],efield[3], label="Ez = UP")
            plt.xlabel('Time (nsec)')
            plt.ylabel('Electric field (muV/m)')
            plt.legend(loc='best')
            plt.subplot(212)
            plt.plot(trace[0]*1e9,trace[1], label="EW")
            plt.plot(trace[0]*1e9,trace[2], label="NS")
            plt.plot(trace[0]*1e9,trace[3], label="Vertical")
            plt.xlabel('Time (nsec)')
            plt.ylabel('Voltage (muV)')
            plt.legend(loc='best')
	    
            plt.show()

############### end of loop over antennas


####################################################################################################################################
####################################################################################################################################
####################################################################################################################################

#===========================================================================================================
# Compute the time dependent voltage
#===========================================================================================================
if __name__ == '__main__':

    #print('Nb of paras=',len(sys.argv))
    if ((str(sys.argv[2])=="manual") & (len(sys.argv)<5)) or ((str(sys.argv[2])=="txt") & (len(sys.argv)<3)):
        print("""
	Wrong minimum number of arguments. All angles are to be expressed in degrees and in GRAND convention.
        Usage:
        if ZHAireS inp file input (Several antennas):
            python computevoltage.py [path to traces]  [input_option]
            example: python computevoltage.py ./ txt 
        if manual input (Single antenna):
            python computevoltage.py [path to traces] [input_option] [zenith] [azimuth] [opt: alpha,beta] [opt: antenna ID]]
            example: python computeVoltage.py ./  manual 85 205 10 5
            
        """)
	
        ## -> computes voltage traces for EW, NS and Vertical antenna component and saves the voltage traces in out_'.txt (same folder as a'.trace)
        ## -> produces a new json file with copying the original one, but saves as well additional informations as p2p-voltages, and peak times and values in *.voltage.json in the same folder as the original json file
        sys.exit(0)


    #folder containing the traces and where the output should go to
    path=sys.argv[1] 
    print("path to traces=",path)
    
    # Decide where to retrieve the shower parameters : txt for ZHAireS input file or manual to hand them over by hand
    opt_input = str(sys.argv[2])
    print("opt_input = ",opt_input)

    if opt_input=='txt':
        import in_out
        # Read the ZHAireS input (.inp) file to extract the primary type, the energy, the injection height and the direction
        inp_file = str(sys.argv[1])
        zenith_sim,azimuth_sim ,energy,injh,primarytype,core,task= in_out.inputfromtxt(inp_file)

    elif opt_input=='manual':
        zenith_sim = float(sys.argv[3]) #deg
        azimuth_sim = float(sys.argv[4]) #deg
        
    print("VOLTAGE COMPUTATION STARTED")
    compute(opt_input,path, zenith_sim, azimuth_sim)
    
    print("VOLTAGE COMPUTED")
