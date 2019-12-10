'''Script to perform an interpolation between to electric field traces at a desired position
TODO: use magnetic field values and shower core from config-file
'''


import numpy as np
from scipy import signal
from utils import getn
import operator

from os.path import split
import sys

from frame import get_rotation, UVWGetter
from io_utils import load_trace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from __init__ import phigeo, thetageo

#################################################################
# Not needed at the moment, removed later
# def rfftfreq(n, d=1.0, nyquist_domain=1):
# '''calcs frequencies for rfft, exactly as numpy.fft.rfftfreq, lacking that function in my old numpy version.
# Parameters:
# ---------
#n: int
# Number of points.
#d: float
# Sample spacing, default is set to 1.0 to return in units of sampling freq.

# Returns:
# -------
# f: array of floats
# frequencies of rfft, length is n/2 + 1
# '''
# if n % 2 == 0:
#f = numpy.array([n/2 - i for i in range(n/2,-1,-1)]) / (d*n)
# else:
#f = numpy.array([(n-1)/2 + 1 - i for i in range(n/2,-1,-1)]) / (d*n)
# if nyquist_domain is 1 you're done and return directly
# if nyquist_domain != 1:
# if nyquist_domain even, mirror frequencies
#if (nyquist_domain % 2) == 0: f = f[::-1]
#sampling_freq = 1./d
#fmax = 0.5*sampling_freq
#f += (nyquist_domain-1)*fmax
# return f
####################################

def unwrap(phi, ontrue=None):
    """Unwrap the phase to a strictly decreasing function.

    Parameters:
    ----------
        phi: numpy array, float
            phase of the signal trace
        ontrue: str
            printing option, default=None
    Returns:
    ----------
        phi_unwrapped: numpy array, float
            unwarpped phase of the signal trace
    """

    phi_unwrapped = np.zeros(phi.shape)
    p0 = phi_unwrapped[0] = phi[0]
    pi2 = 2. * np.pi
    l = 0
    for i0, p1 in enumerate(phi[1:]):
        i = i0 + 1
        if p1 >= p0:
            l += np.floor_divide(p1 - p0, pi2) + 1
        phi_unwrapped[i] = p1 - l * pi2
        p0 = p1
        if ontrue is not None:
            print(i, phi[i], phi[i-1], l, phi_unwrapped[i], abs(phi[i] - phi[i-1]),
                  abs(phi[i] - phi[i-1] + np.pi), abs(phi[i] - phi[i-1] - np.pi), l)
    return phi_unwrapped


def interpolate_trace(t1, trace1, x1, t2, trace2, x2, xdes, upsampling=None,  zeroadding=None, ontrue=None, flow=60.e6, fhigh=200.e6):
    """Interpolation of signal traces at the specific position in the frequency domain
    
    The interpolation of traces needs as input antenna position 1 and 2, their traces (filtered or not) 
    in one component, their time, and the desired antenna position and returns the trace ( in x,y,z coordinate system) and the time from the desired antenna position.
    Zeroadding and upsampling of the signal are optional functions.

    IMPORTANT NOTE:
    The interpolation of the phases includes the interpolation of the signal arrival time. A linear interpolation implies a plane radio
    emission wave front, which is a simplification as it is hyperbolic in shape. However, the wave front can be estimated as a plane between two simulated observer positions 
    for a sufficiently dense grid of observers, as then parts of the wave front are linear on small scales.

    This script bases on the diploma thesis of Ewa Holt (KIT, 2013) in the context of AERA/AUGER. It is based on the interpolation of the amplitude and the pahse in the frequency domain. 
    This can lead to misidentifiying of the correct phase. We are working on the interplementaion on a more robust interpolation of the signal time.
    Feel free to include it if you have some time to work on it. The script is completely modular so that single parts can be substitute easily.
        

    Parameters:
    ----------
            t1: numpy array, float
                time in ns of antenna 1
            trace1: numpy array, float
                single component of the electric field's amplitude of antenna 1
            x1: numpy array, float
                position of antenna 1
            t2: numpy array, float
                time in ns of antenna 2
            trace2: numpy array, float
                single component of the electric field's amplitude of antenna 2
            x2: numpy array, float
                position of antenna 2
            xdes: numpy arry, float
                antenna position for which trace is desired, in meters
            upsampling: str
                optional, True/False, performs upsampling of the signal, by a factor 8
            zeroadding: str
                optional, True/False, adds zeros at the end of the trace of needed
            ontrue: str
                optional, True/False, just a plotting command
            flow: float
                lower frequency - optional, define the frequency range for plotting, if desired (DISPLAY=True/False)
            fhigh: float
                higher frequency - optional, define the frequency range for plotting, if desired (DISPLAY=True/False)

    Returns:
    ----------
        xnew: numpy array, float
            time for signal at desired antenna position in ns
        tracedes: numpy array, float
            interpolated electric field component at desired antenna position
    """
    DISPLAY = False

    # hand over time traces of one efield component -t1=time, trace1=efield- and the position 
    # x1 of the first antenna, the same for the second antenna t2,trace2, x2.
    # xdes is the desired antenna position (m) where you would like to have the efield trace in time
    # if necessary you have to do an upsampling of the trace: upsampling=On
    # onTrue=On would give you printings to the terminal to check for correctness
    # flow= lower freq in Hz, fhigh=higher freq in Hz, not necessarily needed

    factor_upsampling = 1
    if upsampling is not None:
        factor_upsampling = 8
    c = 299792458.e-9  # m/ns

    # calculating weights: should be done with the xyz coordinates
    # since in star shape pattern it is mor a radial function connection the poistion of 
    # same signal as linear go for that solution.
    # if lines ar on a line, it will give the same result as before
    tmp1 = np.linalg.norm(x2 - xdes)
    tmp2 = np.linalg.norm(x1 - xdes)
    tmp = 1. / (tmp1 + tmp2)
    weight1 = tmp2 * tmp
    weight2 = tmp1 * tmp

    if np.isinf(weight1):
        print("weight = inf")
        print(x1, x2, xdes)
        weight1 = 1.
        weight2 = 0.
    if np.isnan(weight1):
        print('Attention: projected positions equivalent')
        weight1 = 1.
        weight2 = 0.
    epsilon = np.finfo(float).eps
    if (weight1 > 1. + epsilon) or (weight2 > 1 + epsilon):
        print("weight larger 1: ", weight1, weight2, x1, x2, xdes, np.linalg.norm(
            x2-x1), np.linalg.norm(x2-xdes), np.linalg.norm(xdes-x1))
    if weight1 + weight2 > 1 + epsilon:
        print("PulseShape_Interpolation.py: order in simulated positions. Check whether ring or ray structure formed first")
        print(weight1, weight2, weight1 + weight2)

    # get refractive indey at the antenna positions
    n1 = getn(x1[2])
    n2 = getn(x2[2])

    #################################################################################
    # linearly interpolation of the phases

    # first antenna
    # upsampling if necessary
    if upsampling is not None:
        trace1 = signal.resample(trace1, len(trace1)*factor_upsampling)
        t1 = np.linspace(t1[0], t1[-1], len(trace1)
                            * factor_upsampling, endpoint=False)

    if zeroadding is True:
        max_element = len(trace1)  # to shorten the array after zeroadding
        print(max_element)
        xnew = np.linspace(t1[0], 1.01*t1[-1],
                              int((1.01*t1[-1]-t1[0])/(t1[2]-t1[1])))
        print(len(xnew))
        xnew = xnew*1.e-9  # ns -> s
        zeros = np.zeros(len(xnew)-max_element)
        f = trace1
        f = np.hstack([f, zeros])
    if zeroadding is None:
        f = trace1
        xnew = t1*1.e-9

    fsample = 1./((xnew[1]-xnew[0]))  # Hz

    freq = np.fft.rfftfreq(len(xnew), 1./fsample)
    FFT_Ey = np.fft.rfft(f)

    Amp = np.abs(FFT_Ey)
    phi = np.angle(FFT_Ey)
    phi_unwrapped = unwrap(phi, ontrue)

    #############################

    # second antenna
    ## t in ns, Ex in muV/m, Ey, Ez
    # NOTE: Time binning always 1ns

    # upsampling if needed
    if upsampling is not None:
        trace = signal.resample(trace2, len(trace2)*factor_upsampling)
        trace2 = trace
        t2 = np.linspace(t2[0], t2[-1], len(trace2)
                            * factor_upsampling, endpoint=False)

    if zeroadding is True:
        # get the same length as xnew
        xnew2 = np.linspace(
            t2[0], t2[0] + (xnew[-1]-xnew[0])*1e9, len(xnew))
        xnew2 = xnew2*1.e-9
        f2 = trace2
        f2 = np.hstack([f2, zeros])
    if zeroadding is None:
        f2 = trace2
        xnew2 = t2*1e-9  # ns -> s
    fsample2 = 1./((xnew2[1]-xnew2[0]))  # *1.e-9 to get time in s

    freq2 = np.fft.rfftfreq(len(xnew2), 1./fsample2)
    FFT_Ey = np.fft.rfft(f2)

    Amp2 = np.abs(FFT_Ey)
    phi2 = np.angle(FFT_Ey)
    phi2_unwrapped = unwrap(phi2, ontrue)

    ### Get the pulsh sahpe at the desired antenna position

    # get the phase

    # getnp.zeros([len(phi2)]) the angle for the desired position
    phides = weight1 * phi_unwrapped + weight2 * phi2_unwrapped
    if ontrue is not None:
        print(phides)
    #if DISPLAY:
        #phides2 = phides.copy()

    # re-unwrap: get -pi to +pi range back and check whether phidesis inbetwwen
    phides = np.mod(phides + np.pi, 2. * np.pi) - np.pi

    #################################################################################
    ### linearly interpolation of the amplitude

    #Amp, Amp2
    # Since the amplitude shows a continuous unipolar shape, a linear interpolation is sufficient

    Ampdes = weight1 * Amp + weight2 * Amp2
    #if DISPLAY:
        #Ampdes2 = Ampdes.copy()

    # inverse FFT for the signal at the desired position
    Ampdes = Ampdes.astype(np.complex64)
    phides = phides.astype(np.complex64)
    #if DISPLAY:
        #phides2 = phides2.astype(np.complex64)
    Ampdes *= np.exp(1j * phides)

    tracedes = (np.fft.irfft(Ampdes))
    tracedes = tracedes.astype(float)

    # PLOTTING

    if DISPLAY:
        import matplotlib.pyplot as plt
        import pylab

        fig1 = plt.figure(1, dpi=120, facecolor='w', edgecolor='k')
        plt.subplot(311)
        plt.plot(freq, phi, 'ro-', label="first")
        plt.plot(freq2, phi2, 'bo-', label="second")
        plt.plot(freq2, phides, 'go--', label="interpolated")
        #plt.plot(freq2, phi_test, 'co--', label= "real")
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"phase (rad)", fontsize=16)
        plt.xlim(flow, fhigh)

        #pylab.legend(loc='upper left')

        plt.subplot(312)
        ax = fig1.add_subplot(3, 1, 2)
        plt.plot(freq, phi_unwrapped, 'r+')
        plt.plot(freq2, phi2_unwrapped, 'bx')
        plt.plot(freq2, phides2, 'g^')
        #plt.plot(freq2, phi_test_unwrapped, 'c^')
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"phase (rad)", fontsize=16)
        # plt.show()
        # plt.xlim([0,0.1e8])
        # plt.xlim([1e8,2e8])
        # plt.ylim([-10,10])
        # ax.set_xscale('log')
        plt.xlim(flow, fhigh)

        plt.subplot(313)
        ax = fig1.add_subplot(3, 1, 3)
        plt.plot(freq, Amp, 'r+')
        plt.plot(freq2, Amp2, 'bx')
        plt.plot(freq2, Ampdes2, 'g^')
        #plt.plot(freq2, Amp_test, 'c^')
        plt.xlabel(r"Frequency (Hz)", fontsize=16)
        plt.ylabel(r"Amplitude muV/m/Hz ", fontsize=16)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.ylim([1e1, 10e3])
        plt.xlim(flow, fhigh)

        plt.show()

    if zeroadding is True:
        # hand over time of first antenna since interpolation refers to that time
        return xnew[0:max_element]*1.e9, tracedes[0:max_element]

    if upsampling is not None:
        return xnew[0:-1:8]*1.e9, tracedes[0:-1:8]
    else:
        xnew = np.delete(xnew, -1)
        return xnew*1.e9, tracedes  # back to ns

################################## CONTROL  

        #if DISPLAY==1:
            ##### PLOTTING

            #import matplotlib.pyplot as plt
            #plt.plot(xnew_planey0, np.real(tracedes_planey0), 'g:', label= "plane 0")
            #plt.plot(xnew_planey1, np.real(tracedes_planey1), 'b:', label= "plane 1")
            #plt.plot(xnew_desiredy, np.real(tracedes_desiredy), 'r-', label= "desired")

            #plt.xlabel(r"time (s)", fontsize=16)
            #plt.ylabel(r"Amplitude muV/m ", fontsize=16)
            #plt.legend(loc='best')

            #plt.show()
            
            
#################################     


def _ProjectPointOnLine(a, b, p):
    ''' Helper function
    line defined by vector a and b, project othogonally vector p to it
    '''
    ap = p-a
    ab = b-a
    nrm = np.dot(ab,ab)
    if nrm <= 0.:
        print(a, b)
    point = a + np.dot(ap,ab) / nrm * ab
    return point


#################################     

        
def do_interpolation(desired, array, zenith, azimuth, phigeo=phigeo, thetageo=thetageo, shower_core=np.array([0,0,0]), DISPLAY=False):
    '''
    Reads in arrays, looks for neigbours, calls the interpolation and saves the traces
    
    Parameters:
    ----------
    desired: str
        path to list of desired antenna positions (x,y,y info)
    array: str
        path to antenna list, already simulated -- base
        The script accepts starshape as well as grid arrays
    zenith: float
        GRAND zenith in deg
    azimuth: float
        GRAND azimuth in deg
    phigeo, thetageo: float
        angles of magnetic field in deg
    shower_core: numpy array
        position of shower core for correction
    DISPLAY: True/False
        enables printouts and plots
    
    
    Returns:
    ----------
        --
    Saves traces via index infomation in same folder as desired antenna positions

    
    NOTE: The selection of the neigbours is sufficiently stable, but does not always pick the "best" neigbour, still looking for an idea
    TODO: Read-in and save only hdf5 files
    '''
    print(shower_core)

    # DESIRED
    # Hand over a list file including the antenna positions you would like to have.
    positions = np.loadtxt(desired) # positions[:,0]:x, positions[:,1]:y,positions[:,2]:z
    if DISPLAY:
        print('desired positions: ')
        print(positions, len(positions))
    if len(positions) <=1:
        print("Files of desired positions has to consist of at least two positions, Bug to be fixed")


    # SIMULATION
    # Read in simulated position list
    #positions_sims=np.loadtxt(array+"/antpos.dat")
    positions_sims=np.loadtxt(array)
    if DISPLAY:
        print('simulated positions: ')
        print(positions_sims, len(positions_sims))
    if len(positions_sims) <=1:
        print("Files of simulated positions has to consist of at least two positions, Bug to be fixed")
    
    
    # SELECTION: For interpolation only select the desired position which are "in" the plane of simulated antenna positions
    a = positions_sims[0]-positions_sims[10]
    a = a/np.linalg.norm(a)
    b = positions_sims[0]-positions_sims[-1]
    b = b/np.linalg.norm(b)
    if(a==b).all():
        a = positions_sims[0]-positions_sims[80]
        a = a/np.linalg.norm(a)
    n=np.cross(a,b)
    n = n/np.linalg.norm(n)

    # test wether desired points are in_plane, needed assumption for interpolation
    ind=[]
    for i in np.arange(0,len(positions[:,1])):
        if (np.dot(positions_sims[0]- positions[i], n) ==0. ):
            ind.append(i)
    #print("in-plane: ", ind)

    #------------------------
    if DISPLAY:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        #ax = fig.gca(projection='3d')

        #ax.scatter(positions_sims[:,0], positions_sims[:,1], positions_sims[:,2], label = "simulated")
        #ax.scatter(positions[:,0], positions[:,1], positions[:,2], label = "desired")
        #ax.scatter(shower_core[0], shower_core[1], shower_core[2], label = "shower core")
        for j in range(0,len(positions[:,1])):
                ax.annotate(str(j), ((positions[j,0], positions[j,1])))
        ax.scatter(positions_sims[:,0], positions_sims[:,1], label = "simulated")
        ax.scatter(positions[ind,0], positions[ind,1], label = "desired")
        ax.scatter(shower_core[0], shower_core[1],  label = "shower core")
        
        plt.title("XYZ coordinates")
        plt.legend(loc=2)
        plt.show()
    #------------------------

        
    ##--##--##--##--##--##--##--##--##-##--##--##-##--##--## START: WRAP UP AS FUNCTION (PROJECTION AND ROTATION)
    #### START: UNDO projection
    #define shower vector
    az_rad=np.deg2rad(180.+azimuth)#Note ZHAIRES units used
    zen_rad=np.deg2rad(180.-zenith)

    # shower vector  = direction of line for backprojection, TODO should be substituded bey line of sight Xmax - positions
    v = np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)])
    v = v/np.linalg.norm(v)
    
    # for back projection position vector line is projected position
    # for back projection normal vector of plane to intercsect == v
    n = v
        
    for i in np.arange(0,len(positions[:,1])):
        b=-np.dot(n,positions[i,:])/ np.dot(n, v)
        positions[i,:] = positions[i,:] + b*v - shower_core # correct by shower core position
    for i in np.arange(0,len(positions_sims[:,1])):
        b=-np.dot(n,positions_sims[i,:])/ np.dot(n, v)
        positions_sims[i,:] = positions_sims[i,:] + b*v - shower_core # correct by shower core position
    
    

    ### START: ROTATE INTO SHOWER COORDINATES, and core for offset by core position, alreadz corrected in projection
    #GetUVW = UVWGetter(shower_core[0], shower_core[1], shower_core[2], zenith, azimuth, phigeo, thetageo)
    GetUVW = UVWGetter(0., 0., 0., zenith, azimuth, phigeo, thetageo)



    # Rotate only "in"plane desired positions
    pos= []
    for i in np.arange(0,len(positions[:,1])):
        if i in ind:
            pos.append(GetUVW(positions[i,:], ))    
    pos=np.asarray(pos)
    
    # Rotate simulated positions
    pos_sims= np.zeros([len(positions_sims[:,1]),3])
    for i in np.arange(0,len(positions_sims[:,1])):
        pos_sims[i,:] = GetUVW(positions_sims[i,:], )
    ##--##--##--##--##--##--##--##--##-##--##--##-##--##--## END: WRAP UP AS FUNCTION (PROJECTION AND ROTATION)
    
    # ------------------
    if DISPLAY:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)
        #ax2 = fig2.gca(projection='3d')
        
        #ax2.scatter(pos_sims[:,0], pos_sims[:,1], pos_sims[:,2], label = "simulated")
        #ax2.scatter(pos[:,0], pos[:,1], pos[:,2], label = "desired")
        for j in range(0,len(pos[:,1])):
                ax2.annotate(str(j), ((pos[j,1], pos[j,2])))
        ## x component should be 0
        ax2.scatter(pos_sims[:,1], pos_sims[:,2], label = "simulated")
        ax2.scatter(pos[:,1], pos[:,2], label = "desired")
        ax2.scatter(0, 0, marker ="x",  label = "core")
        
        plt.title("shower coordinates")
        plt.legend(loc=2)
        plt.show()
    # ------------------

      

    # calculate radius and angle for simulated positions and store some in list
    points=[]
    for i in np.arange(0,len(pos_sims[:,1])):  # position should be within one plane yz plane, remove x=v component for simplicity
        #points.append([i, pos_sims[i,1], pos_sims[i,2] ])
        theta2 = np.arctan2(pos_sims[i,2], pos_sims[i,1])
        radius2 = np.sqrt( pos_sims[i,1]**2 + pos_sims[i,2]**2 )
        if round(theta2,4) == -3.1416:
            theta2*=-1
        points.append([i, theta2, radius2])
        

    #loop only over desired in-plane positions, acting as new reference 
    for i in np.arange(0,len(pos[:,1])):  # position should be within one plane yz plane, remove x=v component for simplicity

        theta = np.arctan2(pos[i,2], pos[i,1])
        radius = np.sqrt( pos[i,1]**2 + pos[i,2]**2 )
        #print("index of desired antenna ", ind[i], theta, radius, )
        
        
        # The 4 quadrants -- in allen 4 Ecken soll Liebe drin stecken
        points_I=[]
        points_II=[]
        points_III=[]
        points_IV=[]
        
        for m in np.arange(0,len(points)): # desired points as reference
            delta_phi = points[m][1]-theta
            if delta_phi > np.pi:
                delta_phi = delta_phi -2.*np.pi
            delta_r = points[m][2]-radius
            
            #distance = np.sqrt(delta_r**2 + (delta_r *delta_phi)**2 ) # weighting approach1
            #distance= np.sqrt((pos_sims[m,1]-pos[i,1])**2. +(pos_sims[m,2]-pos[i,2])**2.) # euclidean distance
            distance= np.sqrt(points[m][2]**2. +radius**2. -2.*points[m][2]*radius* np.cos(points[m][1]-theta) ) #polar coordinates
            
            if delta_phi >= 0. and  delta_r >= 0:
                points_I.append((m,abs(round(delta_phi,2)),abs(round(delta_r,2)), distance))
            if delta_phi > 0. and  delta_r < 0:
                points_II.append((m,abs(round(delta_phi,2)),abs(round(delta_r,2)), distance))
            if delta_phi <= 0. and  delta_r <= 0:
                points_III.append((m,abs(round(delta_phi,2)),abs(round(delta_r,2)), distance))
            if delta_phi < 0. and  delta_r > 0:
                points_IV.append((m,abs(round(delta_phi,2)),abs(round(delta_r,2)), distance))
        
    
        if not points_I:
            print("list - Quadrant 1 - empty --> no interpolation for ant", str(ind[i]))
            continue
        if not points_II:
            print("list - Quadrant 2 - empty --> no interpolation for desired ant", str(ind[i]))  
            continue
        if not points_III:
            print("list - Quadrant 3 - empty --> no interpolation for ant", str(ind[i]))   
            continue
        if not points_IV:
            print("list - Quadrant 4 - empty --> no interpolation for ant", str(ind[i]))
            continue

        points_I=np.array(points_I, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])
        points_II=np.array(points_II, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])
        points_III=np.array(points_III, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])
        points_IV=np.array(points_IV, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_r', 'f4'), ('distance', 'f4')])

        ## Sort points; not optimal (the best) solution for all, but brings stable/acceptable results
        points_I = np.sort(points_I, order=['distance', 'delta_phi', 'delta_r'])    
        points_II = np.sort(points_II, order=['distance', 'delta_phi', 'delta_r']) 
        points_III = np.sort(points_III, order=[ 'distance','delta_phi', 'delta_r']) 
        points_IV = np.sort(points_IV, order=['distance', 'delta_phi', 'delta_r']) 
        #indizes of 4 closest neigbours: points_I[0][0], points_II[0][0], points_III[0][0], points_IV[0][0]
        
        # try to combine the one with roughly the same radius first and then the ones in phi
        point_online1=_ProjectPointOnLine(pos_sims[points_I[0][0]], pos_sims[points_IV[0][0]], pos[i])# Project Point on line 1 - I-IV
        point_online2=_ProjectPointOnLine(pos_sims[points_II[0][0]], pos_sims[points_III[0][0]], pos[i])# Project Point on line 2 - II-III

        # ------------------
        if DISPLAY:
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(1,1,1)

            for j in range(0,len(pos_sims[:,1])):
                ax3.annotate(str(j), ((pos_sims[j,1], pos_sims[j,2])))
            
            ## x component should be 0
            ax3.scatter(pos_sims[:,1], pos_sims[:,2], label = "simulated")
            ax3.scatter(pos[i,1], pos[i,2], label = "desired")
            ax3.scatter(pos_sims[points_I[0][0],1], pos_sims[points_I[0][0],2], label = "1")
            ax3.scatter(pos_sims[points_II[0][0],1], pos_sims[points_II[0][0],2], label = "2")
            ax3.scatter(pos_sims[points_III[0][0],1], pos_sims[points_III[0][0],2], label = "3")
            ax3.scatter(pos_sims[points_IV[0][0],1], pos_sims[points_IV[0][0],2], label = "4")
            
            ax3.scatter(point_online1[1], point_online1[2], marker ="x")
            ax3.scatter(point_online2[1], point_online2[2], marker ="x")            
            ax3.scatter(0, 0, marker ="x",  label = "core")
            ax3.plot([0, pos[i,1]], [0, pos[i,2]])

            plt.legend(loc=2)
            plt.show()
        # ------------------

   
        
        
        
        ## the interpolation of the pulse shape is performed, in x, y and z component
        # TODO: read-in table instead textfile
        directory=split(array)[0]+"/"
        print("Read traces from ", directory)
        
        txt0 = load_trace(directory, points_I[0][0], suffix=".trace")
        txt1 = load_trace(directory, points_IV[0][0], suffix=".trace")
        xnew1, tracedes1x = interpolate_trace(txt0.T[0], txt0.T[1], positions_sims[points_I[0][0]] , txt1.T[0], txt1.T[1], positions_sims[points_IV[0][0]], point_online1 ,upsampling=None, zeroadding=None) 
        xnew1, tracedes1y = interpolate_trace(txt0.T[0], txt0.T[2], positions_sims[points_I[0][0]] , txt1.T[0], txt1.T[2], positions_sims[points_IV[0][0]], point_online1 ,upsampling=None, zeroadding=None) 
        xnew1, tracedes1z = interpolate_trace(txt0.T[0], txt0.T[3], positions_sims[points_I[0][0]] , txt1.T[0], txt1.T[3], positions_sims[points_IV[0][0]], point_online1 ,upsampling=None, zeroadding=None) 
        
        txt2 = load_trace(directory, points_II[0][0], suffix=".trace")
        txt3 = load_trace(directory, points_III[0][0], suffix=".trace")
        xnew2, tracedes2x = interpolate_trace(txt2.T[0], txt2.T[1], positions_sims[points_II[0][0]] , txt3.T[0], txt3.T[1], positions_sims[points_III[0][0]], point_online2 ,upsampling=None, zeroadding=None) 
        xnew2, tracedes2y = interpolate_trace(txt2.T[0], txt2.T[2], positions_sims[points_II[0][0]] , txt3.T[0], txt3.T[2], positions_sims[points_III[0][0]], point_online2 ,upsampling=None, zeroadding=None) 
        xnew2, tracedes2z = interpolate_trace(txt2.T[0], txt2.T[3], positions_sims[points_II[0][0]] , txt3.T[0], txt3.T[3], positions_sims[points_III[0][0]], point_online2 ,upsampling=None, zeroadding=None)         
        
        ###### Get the pulse shape of the desired position from projection on line1 and 2
        xnew_desiredx, tracedes_desiredx =interpolate_trace(xnew1, tracedes1x, point_online1, xnew2, tracedes2x, point_online2, positions[ind[i]], zeroadding=None)      
        xnew_desiredy, tracedes_desiredy =interpolate_trace(xnew1, tracedes1y, point_online1, xnew2, tracedes2y, point_online2, positions[ind[i]], zeroadding=None) 
        xnew_desiredz, tracedes_desiredz =interpolate_trace(xnew1, tracedes1z, point_online1, xnew2, tracedes2z, point_online2, positions[ind[i]], zeroadding=None) 





        # TODO Save as hdf5 file instead of textfile
        print("Interpolated trace stord as ",split(desired)[0]+ '/a'+str(ind[i])+'.trace')
        FILE = open(split(desired)[0]+ '/a'+str(ind[i])+'.trace', "w+" )
        for i in range( 0, len(xnew_desiredx) ):
                print("%3.2f %1.5e %1.5e %1.5e" % (
                    xnew_desiredx[i], tracedes_desiredx[i], tracedes_desiredy[i], tracedes_desiredz[i]), end='\n', file=FILE)
        FILE.close()
        
        #delete after iterate
        del points_I, points_II, points_III, points_IV
#-------------------------------------------------------------------



def main():
    if ( len(sys.argv)>1 ):
        print("""
            Example on how to do interpolate a signal
                -- read in list of desired poistion
                -- read in already simulated arrazs
                -- find neigbours and perform interpolation
                -- save interpolated trace
            
            Usage: python3 interpolate.py 
            Example: python3 interpolate.py 
            
            NOTE: Still hadrcoded pathes in file since it is only a library file
        """)
        sys.exit(0)
        
        
    path="/mnt/c/Users/Anne/work/CoREAS/"
    # path to list of desied antenna positions, traces will be stored in that corresponding folder
    #desired  = sys.argv[1]
    desired=path+"/Test_inter/new_antpos_rect.dat"
    # underlaying array with simulated positions
    array = path+"/Test_inter/Stshp_XmaxLibrary_0.1259_70.529_0_Proton_17/antpos.dat"
    # Shower directions in deg and GRAND convention    
    zenith = (180-70.5) # GRAND deg
    azimuth = 180+0 # GRAND deg
    
    
    # call the interpolation: Angles of magnetic field and shower core information needed, but set to default values
    do_interpolation(desired, array, zenith, azimuth, phigeo=phigeo, thetageo=thetageo, shower_core=np.array([0,0,2900]), DISPLAY=False)


if __name__== "__main__":
  main()




