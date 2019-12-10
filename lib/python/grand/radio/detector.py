'''ONGOING WORK

Start to rewrite detectro class with containers, still ongoing
At the bottom you find some line giving examples how to use the function of the class

Use: python3.7 detectors.py

#see SLACK with Valentin 26Sept
#* Internal conversion from any coordinate szstem to ECEF, see example in grand/tools/geomagnet.py
#* input type of antenna coordinates could be a Union[grand.ECEF, grand.LTP], ie. local or global coordinates (--> grand pacakage)
#* use Final decorator (typing_extension)


#ToDo after adding grand 
#* add positions of a whole array, in ECEF and m
#* ...
#* antennas positions are stored in a single numpy.array, idealy a grand.coordinates.ECEF object.
## * numpy arrays: myarra.flags.writeable = False
##* implement an AntennaArray object with attributes: type:str, position:grand.coordinates.ECEF, orientation:grand.coordinates.Horizontal, etc.
# * add function if slope not given, calculate it


### NOTE: Anne tried to set up Antenna Class, but was not handy enough in her examples. are the information in Dtector not sufficient enough
'''



#from grand import *
import astropy.units as u
import collections

import numpy as np

from typing import Optional, List, Union


from . import config
site, origin = config.site, config.origin #, latitude, longitude # not yet needed



#from astropy import log
import logging
logger = logging.getLogger("Detector")
#logger.info("================   Detector set up at: " +site +"    ================")

## grand package does not want to work for me, sorry
#DEFAULT: add this info to confog file, != shower core
#origin_default = grand.ECEF(x = 0 * u.m, y = 0 * u.m, z = 0 * u.m)
try:
    print("Detector origin =", str(origin) )
    origin_default = origin
except:
    origin_default = np.array([ 0 * u.m, 0 * u.m, 0 * u.m])



class AlreadySet(Exception):
    """Raised when attempting to re-set an already set attribute"""
    logger.debug('Already set')
    pass

    
    
class Detector:
    ''' info on detector
    
        
        Immutable container with both static and runtime checks, The fields can be accessed as attributes/
        
        1. A static analyses can be done with mypy. It will ensure that the proper types are used as arguments when setting the shower attributes. Special unit  can not be checked (eg energy given in meters...)
        2. In addition using the @property class decorator we can perform runtime checks when an instance attribute is modified.
        
        ToDo:
            a long list -- see the comments above
            unit test missing
    
    '''

    _attributes = ("origin", "location", 
                   "position", "slope", "type")#, "antenna")

    
    def __init__(self, **kwargs):
        # list of instance attributes
        self.__origin: Optional[Union[list, str]] = origin_default
        self.__location: Optional[str] = site
        self.__ID: Optional[int] = []
        self.__position = []
        self.__slope = []
        self.__type = []
        
        for attr, value in kwargs.items():
            if attr not in self._attributes:
                raise ValueError(f"Invalid attribute {attr}")
            setattr(self, attr, value)
     
     
    # Do I need that 
    def __str__(self) -> str:
        attributes = ", ".join([attr + "=" + repr(getattr(self, attr))
                                for attr in self._attributes])
        #return f"Shower({attributes})"
        return f"{self.__class__.__name__}({attributes})"
        
    @staticmethod
    def _assert_not_set(attr):
        #if attr is not None:
        if attr not in (None, site, origin_default):
            raise AlreadySet()
    
    
    @property
    def origin(self) -> Union[list, str]:
        """origin of array, in m """
        #return self.__origin
        if hasattr(self.__origin, 'unit'):
            print("Unit of origin: ",self.__origin.unit)
            return self.__origin
        else:
            return self.__origin*u.m

    @origin.setter
    def origin(self, value: Union[list, str]):
        self._assert_not_set(self.__origin)
        self.__origin = value

    @origin.deleter
    def origin(self):
        self.__origin = None
        
        
    @property
    def location(self) -> Union[list, str]:
        """location/site of array"""
        return self.__location

    @location.setter
    def location(self, value: Union[list, str]):
        self._assert_not_set(self.__location)
        self.__location = value

    @location.deleter
    def location(self):
        self.__location = None


    @property
    def ID(self) -> Union[list, str, int]:
        """IDs of antennas in array, accepts only lists"""
        return np.asarray(self.__ID)[0]

    @ID.setter
    def ID(self, value: Union[int, str]):
        self.__ID.append(value)
  
  
    @property
    def position(self) -> Union[list, str]:
        """position of antennas array, in m, accepts only lists"""
        #return np.asarray(self.__position)[0]*u.m
        if hasattr(self.__position, 'unit'):
            print("Unit of antenna positions: ",self.__position.unit)
            return np.asarray(self.__position)[0]
        else:
            return np.asarray(self.__position)[0]*u.m

    @position.setter
    def position(self, value: Union[list, str]):
        self.__position.append(value)

       
    @property
    def slope(self) -> Union[list, str]:
        """local slopes (alpha,beta) of antennas in array, in deg, accepts only lists"""
        return np.asarray(self.__slope)*u.deg

    @slope.setter
    def slope(self, value: Union[list, str]):
        self.__slope.append(value)
        
        
    @property
    def type(self) -> Union[list, str, int]:
        """type of antennas in array, strings, accepts only lists"""
        return np.asarray(self.__type)[0]

    @type.setter
    def type(self, value: Union[int, str]):
        self.__type.append(value)



    def find_position(self, antID):
        ''' find antenna position per ID in detector array
        Arguments:
        ----------
        det: object
        ID: int
            desired antenna ID
            
        Returns:
        --------
        positions: array
            desired antenna position
        '''

        index = np.where(self.ID == int(antID))[0][0]
        return self.position[index]


    def find_slope(self, antID):
        ''' find antenna slope per ID in detector array
        Arguments:
        ----------
        det: object
        antID: int
            desired antenna ID
            
        Returns:
        --------
        positions: array
            desired antenna slope
        '''
        index = np.where(self.ID == int(antID))[0][0]
        return self.slope[index]


    def find_antenna(self, antID):
        ''' return antenna with ID as namedtuple
        Arguments:
        ----------
        antID: int
            desired antenna ID
            
        Returns:
        --------
            named tuple Antenna(ID, psoition, slope, type)
        '''
        index = np.where(self.ID == int(antID))[0][0]
        Antenna = collections.namedtuple('Antenna', 'ID position slope type')
        return Antenna(ID=antID, 
                       position=self.position[index], 
                       slope=self.slope[index],
                       type=self.type[index])


 
    def create_from_file(self, array_file): 
        ''' reading in whole antenna array as file: antID, positions in m
            and slope wrt horizontal in deg(sets default values or to be caluclated), antenna type (or default)
            
        Arguments:
        ----------
        self
        array_file: str
            filepath containing ID, positions and slope
        
        Returns:
        -------
        -  
        
        TODO: add calculation of slope from Turtle
        
        '''
        logger.info("Creating array from: " + str(array_file))
        ant_array = np.loadtxt(array_file,  comments="#")
        self.ID = ant_array[:, 0].tolist()
        self.position = ant_array[:, 1:4].tolist()
        for i in range(0,len(ant_array.T[0])):
            try:
                self.slope = ant_array[i, 4:6].tolist()
            except: #add exception
                logger.debug("slope needs to be caluclated")
                self.slope = (0,0)
            try:
                self.type = str(ant_array[i,6])
            except: #add exception
                logger.debug("Type needs to be defined")
                self.type = "HorizonAntenna" # default type
 
 
 
 
 
################# TESTING


#### namedtuples -- cannot use keys from tuples...        
#det = Detector(origin = np.array([ 0 *u.m,  0 *u.m ,  0* u.m]))
##det.origin = ( 0 *u.m,  0 *u.m ,  0* u.m)
#det.location = "Lenghu" #site
#det.create_from_file("/home/laval1NS/zilles/CoREAS/regular_array_slopes.txt")
#print(det.position)
#print(det._attributes, det.origin, det.location)


#print(det.position.T, det.ID)
#print(det.find_position(1), det.find_slope(1))
#print(det.find_antenna(1))
        
#ant=det.find_antenna(1)
#print(ant.position)
        
        



