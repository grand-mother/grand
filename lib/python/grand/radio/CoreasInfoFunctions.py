################################
#### by A. Zilles
################################

#!/usr/bin/env python

import os
from os.path import split, join, realpath
import numpy as np
import sys
import glob

import logging
logger = logging.getLogger("CoreasInfo")

from astropy import units as u



#===========================================================================================================
def inputfromtxt_coreas(input_file_path): # still ongoing work
#===========================================================================================================
    '''
    Get shower parameter from inp and reas file for coreas simulations
    
    Parameters:
        input_file_path: str
            path of inp file
        
    Returns:
        zen: float
            zenith, deg in GRAND conv.
        azim: float
            azimuth, deg in GRAND conv.
        energy: float
            primary energy in eV
        injh: not working
            Coreas can not yet accept an injection height
        primarytype: str
            'proton' or 'iron'
        core: numpy array
            core position in meters
        task: str
            ID of shower
    '''
    
    
    if os.path.isfile(input_file_path) ==  False:  # File does not exist 
        print('--- ATTENTION: inp-file does not exist')
        exit()
        
    #datafile = file(input_file_path) # why it is not working...
    datafile = open(input_file_path, 'r') 

    for line in datafile:
        # NOTE: CORSIKA/CoREAS angle = direction of propgation == GRAND conventions
        if 'THETAP' in line:
            zen=float(line.split('    ',-1)[1]) *u.deg# propagation direction
            zen=180*u.deg-zen # to GRAND
        if 'PHIP' in line:
            azim = float(line.split('    ',-1)[1])*u.deg # propagation direction, already in GRAND
        #if 'RASPASSHeight' in line:
            #injh = float(line.split(' ',-1)[2])*u.m
        if 'ERANGE' in line:
            energy = line.split('    ',-1)[1] # GeV by default
            if energy[-1]=='\n':
                energy=energy[0:-1]
            energy = float(energy) *1e9 *u.eV# GeV to eV
        if 'PRMPAR' in line:
            primarytype = str(line.split('    ',-1)[1])
            if primarytype[-1]=='\n':
                primarytype=primarytype[0:-1]
            if primarytype == str(14):
                primarytype ='proton'
            if primarytype == str(5626):
                primarytype ='iron'

    try:
        energy
    except NameError:
        print('No primary energy found in the ZHAireS input text file.')
        exit()
    try:
        primarytype
    except NameError:
        print('ATTENTION: No particle type found')
        primarytype = None
    try:
        injh
    except NameError:
        injh = 100000.e2*u.m #Case of a cosmic for which no injection height is defined in the input file and is then set to 100 km in cm
            
    # Get reas file
    path, reas = os.path.split(input_file_path)
    base = os.path.basename(reas)
    base1 = os.path.splitext(base)
    file_path= path[0:-4]+'/'+base1[0]+".info"
    #file_path= path[0:-4]+'/'+base1[0]+".reas"

    datafile = open(file_path, 'r') 
    for line in datafile:
        if 'TASK' in line:
            task = str(line.split('  ',-1)[1])
            if task[-1]=='\n':
                task=task[0:-1]
            if task[-1]=='\r':
                task=task[0:-1]
            
        if 'CORE' in line:
            #print(line)
            offset = line.split('  ',-1)
            offset[-1]=offset[-1].rstrip()
            core = list([float(offset[1]), float(offset[2]), float(offset[3])])*u.m # in cm to m 
    try:
        task
    except NameError:
        task = None
    try:
        core
    except NameError:
        core = None       
        
        
    if task:
        if core is not None:
            return zen,azim,energy,injh,primarytype,core,task
        else:
            return zen,azim,energy,injh,primarytype,task
    else:
        return zen,azim,energy,injh,primarytype
#===========================================================================================================

def _get_positions_coreas(path):
    '''
    read in antenna positions from Coreas simulations, wrt to sealevel
    
    Parameters:
    datafile: str
        path to folder of run
    
    Returns:
    positions: numpy array
        x,y,z component of antenna positions in meters
    ID_ant: list
        corresponding antenna ID for identification !- [0,1,2,....]
        
        
    NOTE: units assign to positions and slope assuming meters and degree 
    '''
    datafile = open(path, 'r') 
    x_pos1=[]
    y_pos1=[]
    z_pos1=[]
    ID_ant=[]
    #positions=[]
    
    alpha=[]
    beta=[]
    for line in datafile:
    # Coreas
        if 'AntennaPosition =' in line: #list file
            #print(line,line.split('    ',-1) )
            x_pos1.append(float(line.split('  ',-1)[1])/100.) #*u.cm) # cm to m
            y_pos1.append(float(line.split('  ',-1)[2])/100.) #*u.cm) # cm to m
            z_pos1.append(float(line.split('  ',-1)[3])/100.) #*u.cm) # cm to m
            ID_ant.append(str(line.split('  ',-1)[4]))
            alpha.append(0)
            beta.append(0)
        if 'ANTENNA' in line: #info file
            x_pos1.append(float(line.split('  ',-1)[2])) #*u.m) 
            y_pos1.append(float(line.split('  ',-1)[3])) #*u.m) 
            z_pos1.append(float(line.split('  ',-1)[4])) #*u.m) 
            ID_ant.append(str(line.split('  ',-1)[1]))
            alpha.append(float(line.split('  ',-1)[5]))
            beta.append(float(line.split('  ',-1)[6]))
            
            
    x_pos1=np.asarray(x_pos1)
    y_pos1=np.asarray(y_pos1)
    z_pos1=np.asarray(z_pos1)
    positions=np.stack((x_pos1,y_pos1, z_pos1), axis=-1 )*u.m
    slopes=np.stack((alpha, beta), axis=-1 )*u.deg
    #print(ID_ant)    
    
    return positions, ID_ant, slopes

#===========================================================================================================

def _get_Xmax_coreas(path):
    """ read Xmax value from simulations, inspred by Rio group
        --- work in progress
        
    Parameters:
        path: str
            path to folder of run
    
    Returns:
        Xmax: float 
            slant depth in g/cm^2
    """
    
    simInput_file = glob.glob(path+"/*.reas")[0]

    try:
        fp = open(simInput_file,'r')
        for line in fp:
            if 'DepthOfShowerMaximum' in line:
                Xmax = float(line.split()[2])
        fp.close()
    except:
        Xmax = None

    return Xmax

#===========================================================================================================

def _get_distanceXmax_coreas(path):
    '''
    Reads the distance to Xmax from reas file , function from Rio group
    
    Parameters:
        path: str
            path to folder of run
    
    Returns:
        d_Xmax: float 
            distance to Xmax in m
    '''

    simInput_file = glob.glob(path+"/*.reas")[0]
    
    try:
        fp = open(simInput_file,'r')
        for line in fp:
            if 'DistanceOfShowerMaximum' in line:
                d_Xmax = float(line.split()[2])/100 # from cm to m
        fp.close()
    except:
        d_Xmax = None
        
    return d_Xmax

#===========================================================================================================

def _get_showercore_coreas(path):
    '''
    Reads the distance to Xmax from reas file, inspired by Rio group
    
    Parameters:
        path: str
            path to folder of run
    
    Returns:
        numpy array 
            shower core in m
    '''

    simInput_file = glob.glob(path+"/*.reas")[0]

    
    try:
        fp = open(simInput_file,'r')
        for line in fp:
            if 'CoreCoordinateNorth' in line:
                CoreX = float(line.split()[2])/100
            if 'CoreCoordinateWest' in line:
                CoreY = float(line.split()[2])/100
            if 'CoreCoordinateVertical' in line:
                CoreZ = float(line.split()[2])/100
            fp.close()
            
        return np.array([CoreX,CoreY,CoreZ])
        
    except:
        logger.debug('Shower core not found')
        return None
        
    
    
    
        
