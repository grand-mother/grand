# Local frame transforms for pulse shape computations.
import numpy as np

from __init__ import phigeo, thetageo
#from .__init__ import phigeo, thetageo

import logging
logger = logging.getLogger("Frame")


        #showeraxis = -1 * hp.spherical_to_cartesian(zenith, azimuth)  # -1 is because shower is propagating towards us
        #if(magnetic_field_vector is None):
            #magnetic_field_vector = hp.get_magnetic_field_vector(site=site)
        #magnetic_field_normalized = magnetic_field_vector / linalg.norm(magnetic_field_vector)
        #vxB = np.cross(showeraxis, magnetic_field_normalized)
        #e1 = vxB
        #e2 = np.cross(showeraxis, vxB)
        #e3 = np.cross(e1, e2)
        #e1 /= linalg.norm(e1)
        #e2 /= linalg.norm(e2)
        #e3 /= linalg.norm(e3)
        #self.__transformation_matrix_vBvvB = copy.copy(np.matrix([e1, e2, e3]))
        #self.__inverse_transformation_matrix_vBvvB = np.linalg.inv(self.__transformation_matrix_vBvvB)





def get_rotation(zen, az, phigeo=phigeo, thetageo=thetageo):
    """Utility function for getting the rotation matrix between frames
    
    Arguments:
    ----------
    zen: float
        zenith of shower in deg (GRAND)
    az: float
        azimuth of shower in deg (GRAND)
    phigeo: float
        angle magnetic field in deg
    thetageo: float
        angle magnetic field in deg
        
    Returns:
    --------
    numpy array
        rotation matrix
    """
    zen = np.deg2rad(zen)
    az = np.deg2rad(az) 
    phigeo = np.deg2rad(phigeo) 
    thetageo = np.deg2rad(thetageo)     
    
    #magnetic field vector
    s = np.sin(thetageo)
    B = np.array([np.cos(phigeo) * s, np.sin(phigeo) * s,
                     np.cos(thetageo)])
    
    # shower vector   
    s = np.sin(zen)
    v = np.array([np.cos(az) * s, np.sin(az) * s, np.cos(zen)])


    vxB = np.cross(v, B)
    vxB /= np.linalg.norm(vxB)
    vxvxB = np.cross(v, vxB)
    vxvxB /= np.linalg.norm(vxvxB)
    
    return np.array((v, vxB, vxvxB))

# ---------------------------------------------------------

def UVWGetter(cx=0., cy=0., cz=0., zen=0., az=0., phigeo=phigeo, thetageo=thetageo):
    """Closure for getting coordinates in the shower frame.
    
    Arguments:
    ----------
    cx, cy, cz: float
        center eg. of antenna positions
    zen: float
        zenith of shower in deg (GRAND)
    az: float
        azimuth of shower in deg (GRAND)
    phigeo: float
        angle magnetic field in deg
    thetageo: float
        angle magnetic field in deg
        
    Returns:
    --------
    numpy array
        rotated vector in shower frame
    """
    R = get_rotation(zen, az, phigeo, thetageo)
    origin = np.array((cx, cy, cz))

    def GetUVW(pos):
       return np.dot(R, pos - origin)
    return GetUVW

# ---------------------------------------------------------

def XYZGetter(cx, cy, cz, zen, az, phigeo=phigeo, thetageo=thetageo):
    """Closure for getting back to the main frame
        
    Arguments:
    ----------
    cx, cy, cz: float
        center eg. of antenna positions
    zen: float
        zenith of shower in deg (GRAND)
    az: float
        azimuth of shower in deg (GRAND)
    phigeo: float
        angle magnetic field in deg
    thetageo: float
        angle magnetic field in deg
        
    Returns:
    --------
    numpy array
        rotated vector from shower frame in main frame
    """

    Rt = get_rotation(zen, az, phigeo, thetageo).T
    origin = np.array((cx, cy, cz))

    def GetXYZ(pos):
        return np.dot(Rt, pos) + origin
    return GetXYZ

##########################################################################################################

def _create_starshape(zen, az, phigeo=0.72, thetageo=147.43, gdalt = 2734, stepsize = 25 ): # Bfield lenghu
    ''' 
    produce a starshape and rotates it into XYZ coordinates
    
    Arguments:
    ----------
    zen: float
        zenith of shower in deg (GRAND)
    az: float
        azimuth of shower in deg (GRAND)
    phigeo: float
        angle magnetic field in deg
    thetageo: float
        angle magnetic field in deg 
    gdalt: float
        groundaltitude in meters
    stepsize: float
        distance between positions in star shape in meters
        
    Returns:
    --------
    numpy array
        positions in starshape patter in XYZ coordinates (not UVW), in meters
    
    
    NOTE: calculation done in CORSIKA coordinates
    '''
    
    zen= 180.-zen # GRAND to CORSIKA
    zen_rad = np.deg2rad(zen)
    az_rad = np.deg2rad(az)
    rot = get_rotation(zen_rad, az_rad, phigeo=phigeo, thetageo=thetageo)
    v = rot[0]
    vxB = rot[1]
    vxvxB = rot[2]
    
    # 160 Antennen
    ang_split= 8.
    #stepsize = 25. #m
    rings= 21

    pos=[]
    for i in np.arange(1,rings):
      for j in np.arange(ang_split):
          xyz = i*(stepsize)*(np.cos(j*(2./(ang_split))*np.pi)*vxB+np.sin(j *(2./(ang_split))*np.pi)*vxvxB)
          #c = xyz[2]/v[2]
          #pos.append([-(xyz[0]-c*v[0]), (xyz[1]-c*v[1]),  gdalt] )
          pos.append([(xyz[0]), xyz[1],  xyz[2]] )

    
    return np.array(pos) 

#########################################################################################################

def _project_starshape(azimuth, zenith, dist_fromxmax, n, core=np.array([0.,0.,0.]) ,max_ang=2.5, thetageo= 147.43, phigeo=0.72):
    ''' 
    This function calls  _create_starshape (produce a starshape and rotates it into XYZ coordinates), 
    and projects the positions on a given plane
    
    Arguments:
    ----------
    zen: float
        zenith of shower in deg (GRAND)
    az: float
        azimuth of shower in deg (GRAND)
    dist_fromxmax: float
        distance of shower core to Xmax, in meters
    n: numpy array
        normal vector of screen (plane)
    core: numpy array
        shower core position, in meters
    max_ang: float
        maximum opening angle of cone, in deg   
    phigeo: float
        angle magnetic field in deg
    thetageo: float
        angle magnetic field in deg 

    Returns:
    --------
    numpy array
        positions in starshape patter projected on screen, in meters
    
    
    NOTE: calculation done in ZHAIRES coordinates 
    TODO: projection should not be done along v but along line of sight Xmax - xyz0
    '''
    
    #define shower vector, normal vector plane , core 
    az_rad=np.deg2rad(180.+azimuth)#Note ZHAIRES units used
    zen_rad=np.deg2rad(180.-zenith)


    # shower vector 
    v = np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)])
    v = v/np.linalg.norm(v)
    
    ### setting starshape
    max_ang = np.deg2rad(max_ang) # Most distant antenans are 2degs from axis 
    d1 = dist_fromxmax*np.tan(max_ang)
    step = d1/20. 
    
    xyz0 = _create_starshape(zenith, azimuth, phigeo=0.72, thetageo=147.43, gdalt = 2734, stepsize = step )
    number= len(xyz0.T[0])
    #### star shape pattern in xyz, projected on plane
    xyz=np.zeros([number,3]) 
    rings=int(number/8)
    for i in np.arange(1,rings+1):  
        for j in np.arange(8):
            # line-plane intersection
            # projection of shower mountain plane, xyz0 antenna position as position vector of line of sight
            b=-np.dot(n,xyz0[(i-1)*8+j])/ np.dot(n, v)
            xyz[(i-1)*8+j]=xyz0[(i-1)*8+j] +b*v + core  #rojected
            
    return xyz
       
#########################################################################################################
       
def _project_onshowerplane(positions, azimuth, zenith, d = None, core=np.array([0.,0.,0.])):
    ''' 
    This function projects the positions on a given plane back onto shower plane (vertical to shower propagation), 
    using line-plane intersection, coordinates still in XYZ, not shower coordinates
    Use UVWGetter() to  calculate positions in shower coordinates
    
    Arguments:
    ----------
    positions: numpy array
        positions which got projected on a plane
    zen: float
        zenith of shower in deg (GRAND)
    az: float
        azimuth of shower in deg (GRAND)
    d: numpy array
        vector direction of projection
    core: numpy array
        shower core position, in meters


    Returns:
    --------
    numpy array
        positions projected on plane orthogonal to shower axis, in meters
    
    
    NOTE: calculation done in ZHAIRES coordinates 
    TODO: projection should not be done along v but along line of sight Xmax - xyz0
    '''
        
       
    #### UNDO projection
    
    #define shower vector
    az_rad=np.deg2rad(180.+azimuth)#Note ZHAIRES units used
    zen_rad=np.deg2rad(180.-zenith)

    if d is None:
        # shower vector  = direction of line for backprojection, TODO should be substituded bey line of sight Xmax - positions
        v = np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)])
        v = v/np.linalg.norm(v)
        d = v
    
    # for back projection position vector line is projected position
    # for back projection normal vector of plane to intercsect == v, normal vector of shower plane
    n = np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)])
    n = n/np.linalg.norm(n)
        
    pos= np.zeros([len(positions[:,1]),3])
    for i in np.arange(0,len(positions[:,1])):
        b=-np.dot(n,positions[i,:])/ np.dot(n, d)
        pos[i,:] = positions[i,:] + b*d - core # correct by shower core position

    return pos
