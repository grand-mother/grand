import astropy.units as u
from typing import Optional, List, Union

import numpy as np

import logging
logger = logging.getLogger("Shower")

gcm2= u.gram /(u.cm**2)

''' NOTE:
* from typing_extension import Final, mypy would would raise bad usage
* numpy arrays: myarra.flags.writeable = False


* Usage of @property decorator: in case an attribute is derived from other attributes in the class -> the derived attribute will update whenever the source attributes is changed, 
attribute becomes a property by defining it as a function and add the @property decorator before the definition.
* Usage of setter method for the property: to update the source attributes whenever the property is set

'''

class AlreadySet(Exception):
    """Raised when attempting to re-set an already set attribute"""
    logger.debug('Already set')
    pass


class Shower():
    ''' info on shower 
    
        Immutable container with both static and runtime checks, The fields can be accessed as attributes/
        
        1. A static analyses can be done with mypy. It will ensure that the proper types are used as arguments when setting the shower attributes. Special unit  can not be checked (eg energy given in meters...)
        2. In addition using the @property class decorator we can perform runtime checks when an instance attribute is modified. (not yet done, do we need that...)
        
        NOTE: In principle all kind of information can be added, e.g. shower core info, injectionheight etc if available
              Do we want to have a calculated Xmax position, or shall we load it from a calculation or simulation if available...
        
        showerID: str
        primary: str
        energy: float in eV
        zenith: float in deg, GRAND
        azimuth: float in deg, GRAND
        direction: numpy vector
        injectionheight: float in m
        trigger:  list, int
        
        simulations: str
        Xmax: float in g/cm2
        # Xmax_position: possible further attribute in class, to be defined
                
        recoXmax: float in g/cm2
        recoenergy: float in eV
        recozenith: float in deg, GRAND
        recoazimuth: float in deg, GRAND
        
        
        TODO: 
        * How do I prevent that attributes can overwritten
        * missing impulse function: vector type, calculate impulse vector
               
    '''
    
    _attributes = ("showerID", "primary", "energy", "zenith",
                            "azimuth", "injectionheight", "trigger")
    
    def __init__(self, **kwargs):
        # list of instance attributes
        self.__showerID: Optional[str] = None
        self.__primary: Optional[str] = None
        self.__energy: Optional[u.Quantity] = None
        self.__zenith: Optional[u.Quantity] = None
        self.__azimuth: Optional[u.Quantity] = None
        self.__injectionheight: Optional[u.Quantity] = None
        
        self.__trigger: List[Union[int, str]] = None
        

        for attr, value in kwargs.items():
            if attr not in self._attributes:
                raise ValueError(f"Invalid attribute {attr}")
            setattr(self, attr, value)


    def __str__(self) -> str:
        attributes = ", ".join([attr + "=" + repr(getattr(self, attr))
                                for attr in self._attributes])
        #return f"Shower({attributes})"
        return f"{self.__class__.__name__}({attributes})"


    @staticmethod
    def _assert_not_set(attr):
        if attr is not None:
            raise AlreadySet()


    @property
    def showerID(self) -> str:
        """The very unique ID of a shower"""
        return self.__showerID

    @showerID.setter
    def showerID(self, value: str):
        self._assert_not_set(self.__showerID)
        self.__showerID = value
        
    @showerID.deleter
    def showerID(self):
        self.__showerID = None


    @property
    def primary(self) -> str:
        """The primary particle initiating the shower"""
        return self.__primary

    @primary.setter
    def primary(self, value: str):
        self._assert_not_set(self.__primary)
        self.__primary = value
        
    @primary.deleter
    def primary(self):
        self.primary = None


    @property
    def energy(self) -> u.Quantity:
        """The total energy contained in the shower"""
        return self.__energy

    @energy.setter
    def energy(self, value: u.Quantity):
        self._assert_not_set(self.__energy)
        self.__energy = value.to(u.eV)
        
    @energy.deleter
    def energy(self):
        self.__energy = None


    @property
    def zenith(self) -> u.Quantity:
        """The zenith angle of the shower axis"""
        return self.__zenith

    @zenith.setter
    def zenith(self, value: u.Quantity):
        self._assert_not_set(self.__zenith)
        self.__zenith = value.to(u.deg)
        
    @zenith.deleter
    def zenith(self):
        self.__zenith = None


    @property
    def azimuth(self) -> u.Quantity:
        """The azimuth angle of the shower axis"""
        return self.__azimuth

    @azimuth.setter
    def azimuth(self, value: u.Quantity):
        self._assert_not_set(self.__azimuth)
        self.__azimuth = value.to(u.deg)
        
    @azimuth.deleter
    def azimuth(self):
        self.__azimuth = None


    @property
    def injectionheight(self) -> u.Quantity:
        """The injectionheight of the shower"""
        return self.__injectionheight

    @injectionheight.setter
    def injectionheight(self, value: u.Quantity):
        self._assert_not_set(self.__injectionheight)
        self.__injectionheight = value.to(u.m)
        
    @injectionheight.deleter
    def injectionheight(self):
        self.__injectionheight = None


    @property
    def trigger(self) -> List[Union[str,int]]:
        """The trigger of the shower
           could be a value or a list of values or strings"""
        return self.__trigger

    @trigger.setter
    def trigger(self, value: List[Union[str,int]] ):
        self._assert_not_set(self.__trigger)
        self.__trigger = value
        
    @trigger.deleter
    def trigger(self):
        self.__trigger = None
        
    
    def direction(self): 
        """The shower direction (numpy array) -- cross-check definition: np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)]) """
        try:
            return np.array([np.cos(self.azimuth.to(u.rad))*np.sin(self.zenith.to(u.rad)),
                             np.sin(self.azimuth.to(u.rad))*np.sin(self.zenith.to(u.rad)),
                             np.cos(self.zenith.to(u.rad))])
        except:
            logger.error("Direction cannot be calculated")

    
    
#=============================================  

### simulated shower 
class SimulatedShower(Shower):  
    ''' info on simulations; Additional attributes: simulation, Xmax'''
        
    _attributes = Shower._attributes + ("simulation", "Xmax")
    
    def __init__(self, **kwargs):
        self.__simulation: Optional[str] = None
        self.__Xmax: Optional[u.Quantity] = None
        
        super().__init__(**kwargs)
    
    @property
    def simulation(self) -> u.Quantity:
        """name of simulation program"""
        return self.__simulation

    @simulation.setter
    def simulation(self, value: u.Quantity):
        self._assert_not_set(self.__simulation)
        self.__simulation = value
        
        
    @simulation.deleter
    def simulation(self):
        self.__simulation = None
    
    @property
    def Xmax(self) -> u.Quantity:
        """The simulated/ calculated Xmax value"""
        return self.__Xmax

    @Xmax.setter
    def Xmax(self, value: u.Quantity):
        self._assert_not_set(self.__Xmax)
        self.__Xmax = value.to(gcm2)

    @Xmax.deleter
    def Xmax(self):
        self.__Xmax = None

    #def Xmax_position(self): 
        #""" calculated/simulated Xmax_position"""
        
#=================================================        

### reconstructed event
class ReconstructedShower(Shower):
    ''' info on reconstructed values; 
    Additional attributes: recoenergy, recoXmax, recozenith, recoazimuth
    
    TODO:
        *missing reco injectionheight       
    '''
    
    _attributes = Shower._attributes + ("recoenergy", "recoXmax", "recozenith", "recoazimuth",)
    
    def __init__(self, **kwargs):
        self.__recoenergy: Optional[u.Quantity] = None
        self.__recozenith: Optional[u.Quantity] = None
        self.__recoazimuth: Optional[u.Quantity] = None
        self.__recoXmax: Optional[u.Quantity] = None
        
        super().__init__(**kwargs)
    
    @property
    def recoenergy(self) -> u.Quantity:
        """value of reconstructed energy"""
        return self.__recoenergy

    @recoenergy.setter
    def recoenergy(self, value: u.Quantity):
        self._assert_not_set(self.__recoenergy)
        self.__recoenergy = value.to(u.eV)

    @recoenergy.deleter
    def recoenergy(self):
        self.__recoenergy = None


    @property
    def recozenith(self) -> u.Quantity:
        """value of reconstructed zenith"""
        return self.__recozenith

    @recozenith.setter
    def recozenith(self, value: u.Quantity):
        self._assert_not_set(self.__recozenith)
        self.__recozenith = value.to(u.deg)

    @recozenith.deleter
    def recozenith(self):
        self.__recozenith = None
    
    
    @property
    def recoazimuth(self) -> u.Quantity:
        """value of reconstructed azimuth"""
        return self.__recoazimuth

    @recoazimuth.setter
    def recoazimuth(self, value: u.Quantity):
        self._assert_not_set(self.__recoazimuth)
        self.__recoazimuth = value.to(u.deg)

    @recoazimuth.deleter
    def recoazimuth(self):
        self.__recoazimuth = None
        
        
    @property
    def recoXmax(self) -> u.Quantity:
        """value of reconstructed Xmax"""
        return self.__recoXmax

    @recoXmax.setter
    def recoXmax(self, value: u.Quantity):
        self._assert_not_set(self.__recoXmax)
        self.__recoXmax = value.to(gcm2) 
    
    @recoXmax.deleter
    def recoXmax(self):
        self.__recoXmax = None
    

    def recodirection(self): 
        """The reconstructed shower direction (numpy array) -- cross-check definition: np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)]) """
        try:
            return np.array([np.cos(self.recoazimuth.to(u.rad))*np.sin(self.recozenith.to(u.rad)),
                             np.sin(self.recoazimuth.to(u.rad))*np.sin(self.recozenith.to(u.rad)),
                             np.cos(self.recozenith.to(u.rad))])
        except:
            logger.error("Direction cannot be reconstructed")
    
    
#=================================================        


def loadInfo_toShower(name, info=None):
    '''load meta info from hdf5 file to object attributes, object initiated beforehand
    
    example:        testshower = SimulatedShower()
                    loadInfo_toShower(testshower, f.meta)
    
    TODO: 
        *add missing attributes if desired 
    '''
    try:
        showerID = info["ID"]
        name.showerID = str(showerID)
    except IOError:
        showerID = None
    
    try:
        primary = info["primary"]
        name.primary = str(primary)
    except IOError:
        primary = None
        
    try:
        energy = info["energy"]
        if hasattr(energy, 'unit'):
            name.energy = energy
        else:
            name.energy = float(energy)* u.eV
    except IOError:
        energy = None
    
    try:
        zenith = info["zenith"]
        if hasattr(zenith, 'unit'):
            name.zenith = zenith
        else:
            name.zenith = float(zenith)* u.deg
    except IOError:
        zenith = None
    
    try:
        azimuth = info["azimuth"]
        if hasattr(azimuth, 'unit'):
            name.azimuth = azimuth
        else:
            name.azimuth = float(azimuth)* u.deg
    except IOError:
        azimuth = None
    
    try:
        injectionheight = info["injection_height"]
        if hasattr(injectionheight, 'unit'):
            name.injectionheight =  injectionheight
        else:
            name.injectionheight =  float(injectionheight)* u.m
    except IOError:
        injectionheight = None


    ## for simulation input only
    try:
        simulation = info["simulation"]
        name.simulation = str(simulation)
        #try: ## TODO it does not raise an error if attribute does not exist
            #name.simulation = str(simulation)
        #except: # what kind of exception
            #logger.debug("for "+str(name)+" info on simulation could not be set")
    except (IOError, KeyError):
        logger.error("No simulation info provided")
        simulation = None
        
    try:
        Xmax = info["Xmax"]
        if hasattr(Xmax, 'unit'):
            name.Xmax = Xmax
        else:
            name.Xmax = float(Xmax)*gcm2
        #try: ## TODO it does not raise an error if attribute does not exist -- logger
    except (IOError, KeyError):
        logger.error("No XMax value provided")
        Xmax = None
    

        
