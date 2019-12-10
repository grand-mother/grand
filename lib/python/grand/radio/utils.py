import numpy as np

import logging
logger = logging.getLogger("Utils")

#===========================================================================================================
#def load_trace(directory, index, suffix=".trace"):
    #"""Load data from a trace file (ascii file)
    
    #Arguments:
    #----------
    #directory: str
        #path to folder
    #index: int
        #antenna ID
    #suffix: str
        #file ending
        
    #Returns:
    #--------
    #numpy array
        #field trace
        
    #Raises:
    #--------
    #IOError:
        #adopt to ID naming
        
    #Note: currently only usable for Zhaires simulations    
    
    #TODO: readin- hdf5 files and return numpy array
    #"""
    #try:
        #path = "{:}/a{:}{:}".format(directory, index, suffix)
        #return np.loadtxt(path)
        ##with open(path, "r") as f:
            ##return np.array([*map(float, line.split()) for line in f])
    #except IOError:
        #path = "{0}/a{1:04d}{2}".format(directory, index+1, suffix)
        #with open(path, "r") as f:
            #return np.array([map(float, line.split()) for line in f])
        
#===========================================================================================================

def getn(h):
    """Get the refractive index
    
    Arguments:
    ----------
    h: float
        height in m wrt to sealevel
        
    Returns:
    --------
    float
        refractive index at height h
        
    Note: Reference: Zhaires (see email M. Tueros 25/11/2016)
    """
    # h in meters
    return 1. + 325E-06 * np.exp(-0.1218E-03 * h)

#===========================================================================================================

def getCerenkovAngle(h):
   """Get the Cerenkov angle
    
    Arguments:
    ----------
    h: float
        height in m wrt to sealevel
        
    Returns:
    --------
    float
        Cherenkov angle at height h in deg
   """
   return np.rad2deg(np.arccos(1. / getn(h)))

#===========================================================================================================
def get_integratedn(injh2, position):
    '''Calculates the integrated n from a specific height (0,0,height) to the observer position
        --- works currently only for neutrinos
        
    Arguments:
    ----------
    injh2: float
        injection height wrt sealevel in m (0.,0., injh2)
    position: numpy array
        observer position wrt sealevel in m 
    
    Returns:
    --------
    n : float
        integrated refractive index along a path from injection to observer
    
    Note: assumption coordinate system so that tau decay at (0.,0, injectionheight)
    Note: calculation of integrated n implemented similar as in Zhaires
    '''
    
    Re= 6370949 # m, Earth radius
    ########
    # line of sight
    ux= position[0] -0.
    uy= position[1] -0.
    uz= position[2] -injh2
    
    nint= 10000 # so many sub tracks alng line of sight
    # vector k along the line of sight
    kx=ux/nint
    ky=uy/nint
    kz=uz/nint
    
    #current positions, start with injh as first point of emission
    currpx=0.
    currpy=0.
    currpz=injh2
    currh=currpz # just in that case here, since particle injected directly induce a shower
    
    ns=325E-06
    kr=-0.1218E-03
    
    #print "inhh, antenna height ", injh2, position[2]
    summe=0.
    for i in range(0,nint):
        nextpx=currpx+kx
        nextpy=currpy+ky
        nextpz=currpz+kz
        
        nextR=np.sqrt( nextpx*nextpx +nextpy*nextpy )
        nexth= ( np.sqrt((( injh2 - nextpz  ) + Re) * (( injh2  - nextpz  ) + Re) + nextR*nextR) - Re) /1e3
        
        if (abs(currh-nexth)>1e-10 ):
            summe=summe+ (  np.exp(kr*nexth) -   np.exp(kr*currh)  )/ (kr*( nexth - currh) )
        else:
            summe=summe+ np.exp(kr*currh)
        
        currpx=nextpx
        currpy=nextpy
        currpz=nextpy
        currR=nextR
        currh=nexth
        
        
    avn= ns*summe/nint
    n= 1.+ avn
    
    return  n # integrated n

#===========================================================================================================
def mag(x):
    ''' Calculates the length of a vector or the absolute value
    '''
    return np.sqrt(x.dot(x))


#=========================================================================================================== 
def _getAngle(refpos=[0.,0.,1e6],theta=None,azim=None,ANTENNAS=None, core=[0.,0.,0.]): # theta and azim in Grand convention
    """ Get angle between antenna and shower axis (injection point or Xmax)
        Arguments:
        ----------
        refpos: numpy array
            sofar position of injection or Xmax
        theta: float
            GRAND zenith in deg
        azim: float
            GRAND azimuth in deg
        ANTENNAS: numpy array
            observer position in m
        core: numpy array
            optional, core position in m, not yet used
            
        Returns:
        --------
        float
            Angle in deg between shower axis and vector reference position to observer position
            
        Note: currently only working for neutrinos
        TODO make it working for CRs, currently only for neutrinos
    """

    zenr = np.radians(theta)
    azimr= np.radians(azim)
    ANTENNAS1 = np.copy(ANTENNAS)-core

    # Compute angle between shower and decay-point-to-antenna axes
    u_ant = ANTENNAS1-refpos
    u_ant = (u_ant/np.linalg.norm(u_ant))

    u_sh = [np.cos(azimr)*np.sin(zenr), np.sin(azimr)*np.sin(zenr), np.cos(zenr)]
    ant_angle = np.arccos(np.matmul(u_ant, u_sh))

    return np.rad2deg(ant_angle)

#===========================================================================================================

#def rfftfreq(n, d=1.0, nyquist_domain=1):
	#'''calcs frequencies for rfft, exactly as numpy.fft.rfftfreq, lacking that function in my old numpy version.
	#Arguments:
	#---------
		#n: int 
			#Number of points.
		#d: float
			#Sampler:spacing, default isr:set to 1.0 to return in units ofr:sampling freq. 
		
	#Returns:
	#-------
		#f: array of floats
			#frequencies of rfft, length is n/2 + 1
	#'''
	#if n % 2 == 0:
		#f = array([n/2 - i for i in range(n/2,-1,-1)]) / (d*n)
	#else:
		#f = array([(n-1)/2 + 1 - i for i in range(n/2,-1,-1)]) / (d*n)
	## if nyquist_domain is 1 you're done and return directly
	#if nyquist_domain != 1:
		## if nyquist_domain even, mirror frequencies
		#if (nyquist_domain % 2) == 0: f = f[::-1]
		#sampling_freq = 1./d
		#fmax = 0.5*sampling_freq 
		#f += (nyquist_domain-1)*fmax
	#return f

#===========================================================================================================

def time2freq(trace):
    '''
    Conversion time to frequency domain
    
    Goal: coherent normalization of the fft: np.sum(trace**2) * dt = np.sum(spectrum**2) * df
        forward FFT with corrected normalization to conserve power.
        -- additional sqrt(2) needed to account for omitted negative frequencies when using "real fft"
    
    Arguments:
    ----------
    trace: numpy array
        trace in time domain
        
    Returns:
        numpy array
        trace in frequency domain
        
    ToDO: check units
    
    '''
    return np.fft.rfft(trace, axis=-1) * 2 ** 0.5 

#===========================================================================================================

def freq2time(spectrum, n=None):
    """
    Conversion frequency to time domain
    
    Goal: coherent normalization of the fft: np.sum(trace**2) * dt = np.sum(spectrum**2) * df
        performs backward FFT with correct normalization that conserves the power
        -- addional division 1/sqrt(2) to account for omitted negative frequencies when using "real fft"
    
    Arguments
    ----------
    spec: complex np array
        the frequency spectrum
    n: int
        the number of sample in the time domain (relevant if time trace has an odd number of samples)
        
        
    """
    return np.fft.irfft(spectrum, axis=-1, n=n) / 2 ** 0.5

