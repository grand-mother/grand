"""
Read/Write python interface to GRAND data (real and simulated) stored in Cern ROOT TTrees.

This is the interface for accessing GRAND ROOT TTrees that do not require the user (reader/writer of the TTrees) to have any knowledge of ROOT. It also hides the internals from the data generator, so that the changes in the format are not concerning the user.
"""

from logging import getLogger
import sys
import datetime
import os

import ROOT
import numpy as np
import glob

from collections import defaultdict

# This import changes in Python 3.10
if sys.version_info.major >= 3 and sys.version_info.minor < 10:
    from collections import MutableSequence
else:
    from collections.abc import MutableSequence
        
from dataclasses import dataclass, field    

thismodule = sys.modules[__name__]

#from grand.io.root_trees import * # this is home/grand/grand (at least in docker) or ../../grand
#sys.path.append("../../grand/dataio/")  #matias: i need this to make it work on my system. got to figure it out 
#from root_trees import *
from grand.dataio.root_trees import *


###########################################################################################################################################################################################################
#
# RawShowerTree
#
##########################################################################################################################################################################################################


@dataclass
## The class for storing a shower simulation-only data for each event
class RawShowerTree(MotherEventTree):
    """The class for storing a shower simulation-only data for each event"""

    _type: str = "rawshower"

    _tree_name: str = "trawshower"
    
    ### Name and version of the shower simulator
    _sim_name: StdString = StdString("")

    ###X Event name (the task name, can be usefull to track the original simulation) 
    _event_name: StdString = StdString("")

    ### Event Date  (used to define the atmosphere and/or the magnetic field)
    _event_date: StdString = StdString("")
    
    ### Unix time of this event date
    _unix_date: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))
   
    ### Random seed
    _rnd_seed: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))
    
    ### Energy in neutrinos generated in the shower (GeV). Useful for invisible energy computation
    _energy_in_neutrinos: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    
    ###X Primary energy (GeV) 
    energy_primary: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    
    ### Shower azimuth (deg, CR convention)
    _azimuth: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))

    ### Shower zenith  (deg, CR convention)
    _zenith: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    
    ### Primary particle type (PDG)
    primary_type: StdVectorListDesc = field(default=StdVectorListDesc("string"))

    # Primary injection point [m] in Shower coordinates
    primary_inj_point_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))

    ### Primary injection altitude [m] in Shower Coordinates
    primary_inj_alt_shc: StdVectorListDesc = field(default=StdVectorListDesc("float"))

    # primary injection direction in Shower Coordinates
    primary_inj_dir_shc: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))

    ### Atmospheric model name TODO:standardize
    _atmos_model: StdString = StdString("")

    # Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters
    _atmos_model_param: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    
    # Table of air density [g/cm3] and vertical depth [g/cm2] versus altitude [m]
    atmos_altitude: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    atmos_density: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))
    atmos_depth: StdVectorListDesc = field(default=StdVectorListDesc("vector<float>"))

        
    ### Magnetic field parameters: Inclination, Declination, Fmodulus.: In shower coordinates. Declination
    #The Earth’s magnetic field, B, is described by its strength, Fmodulus = ∥B∥; its inclination, I, defined
    # as the angle between the local horizontal plane and the field vector; and its declination, D, defined
    # as the angle between the horizontal component of B, H, and the geographical North (direction of
    # the local meridian). The angle I is positive when B points downwards and D is positive when H is 
    # inclined towards the East.    
    _magnetic_field: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))

    ### Shower Xmax depth  (g/cm2 along the shower axis)
    _xmax_grams: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    
    ### Shower Xmax position in shower coordinates [m]
    _xmax_pos_shc: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float64))
    
    ### Distance of Xmax  [m] to the ground
    _xmax_distance: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))
    
    ### Altitude of Xmax  [m]. Its important for the computation of the index of refraction at maximum, and of the cherenkov cone
    _xmax_alt: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    ### high energy hadronic model (and version) used TODO: standarize
    _hadronic_model: StdString = StdString("")
    
    ### low energy model (and version) used TODO: standarize
    _low_energy_model: StdString = StdString("")
    
    ### Time it took for the simulation of the cascade (s). In the case shower and radio are simulated together, use TotalTime/(nant-1) as an approximation
    _cpu_time: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))


    #### ZHAireS/Coreas
    # * THINNING *
    # Thinning energy, relative to primary energy
    # this is EFRCTHN in Coreas (the 0th THIN value)
    _rel_thin: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    # this is the maximum weight, computed in zhaires as PrimaryEnergy*RelativeThinning*WeightFactor/14.0 (see aires manual section 3.3.6 and 2.3.2) to make it mean the same as Corsika Wmax
    # this is WMAX in Coreas (the 1st THIN value) - Weight limit for thinning
    _maximum_weight: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))
    
    # this is the ratio of energy at wich thining starts in hadrons and electromagnetic particles. In Aires is always 1
    # this is THINRAT in Coreas (the 0th THINH value) - hadrons
    _hadronic_thinning: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    # this is the ratio of electromagnetic to hadronic maximum weights.
    # this is WEIRAT in Coreas (the 1st THINH value)
    _hadronic_thinning_weight: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    # Maximum radius (in cm) at observation level within which all particles are subject to inner radius thinning. In corsika particles are sampled following a r^(-4) distribution
    # Aires has a similar feature, but the definition is much more complex...so this will be left empty for now.
    # this is RMAX in Coreas (the 2nd THIN value)
    _rmax: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    # * CUTS *
    #gamma energy cut (GeV)
    _lowe_cut_gamma: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    #electron/positron energy cut (GeV)
    _lowe_cut_e: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    #muons energy cut (GeV)
    _lowe_cut_mu: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    #mesons energy cut (GeV)
    _lowe_cut_meson: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))

    #nucleons energy cut (GeV)
    _lowe_cut_nucleon: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float64))


    ###META ZHAireS/Coreas

    ### Core position with respect to the antenna array (undefined for neutrinos)
    _shower_core_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    
 
    ### Longitudinal Pofiles (those compatible between Coreas/ZHAires)
    
    ## Longitudinal Profile of vertical depth (g/cm2) (we remove this becouse corsika seems to output always in slant
    #long_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of slant depth (g/cm2)
    #long_slantdepth: StdVectorListDesc = field(default=StdVectorListDesc("float")) (we renamed this to long_pd_dpepth, these coments are to be removed once we agree)
    long_pd_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of Gammas      
    long_pd_gammas: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of e+
    long_pd_eplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of e-
    long_pd_eminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of mu+
    long_pd_muplus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of mu-
    long_pd_muminus: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of All charged particles
    long_pd_allch: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of Nuclei
    long_pd_nuclei: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of Number of Hadrons
    long_pd_hadr: StdVectorListDesc = field(default=StdVectorListDesc("float"))

    ## Longitudinal Profile of Energy of created neutrinos (GeV)
    long_ed_neutrino: StdVectorListDesc = field(default=StdVectorListDesc("float"))


    ## Longitudinal Profile of low energy gammas (GeV)
    long_ed_gamma_cut: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    ## Longitudinal Profile of low energy e+/e- (GeV)
    long_ed_e_cut: StdVectorListDesc = field(default=StdVectorListDesc("float"))           
    ## Longitudinal Profile of low energy mu+/mu- (GeV)
    long_ed_mu_cut: StdVectorListDesc = field(default=StdVectorListDesc("float"))           
    ## Longitudinal Profile of low energy hadrons (GeV)
    long_ed_hadr_cut: StdVectorListDesc = field(default=StdVectorListDesc("float"))
    
    ## Longitudinal Profile of energy deposit by gammas (GeV)
    long_ed_gamma_ioniz: StdVectorListDesc = field(default=StdVectorListDesc("float"))                          
    ## Longitudinal Profile of energy deposit by e+/e-  (GeV)
    long_ed_e_ioniz: StdVectorListDesc = field(default=StdVectorListDesc("float"))           
    ## Longitudinal Profile of energy deposit by muons  (GeV)
    long_ed_mu_ioniz: StdVectorListDesc = field(default=StdVectorListDesc("float"))           
    ## Longitudinal Profile of energy deposit by hadrons (GeV)
    long_ed_hadr_ioniz: StdVectorListDesc = field(default=StdVectorListDesc("float"))     
 
    # extra values
    long_ed_depth: StdVectorListDesc = field(default=StdVectorListDesc("float"))

    _first_interaction: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))

    @property
    def sim_name(self):
         return str(self._sim_name)
    
    @sim_name.setter
    def sim_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")
    
        self._sim_name.string.assign(value)

    @property
    def rel_thin(self):
        """Thinning energy, relative to primary energy"""
        return self._rel_thin[0]

    @rel_thin.setter
    def rel_thin(self, value: np.float64) -> None:
        self._rel_thin[0] = value 
 


    @property
    def maximum_weight(self):
        """Weight limit for thinning"""
        return self._maximum_weight[0]

    @maximum_weight.setter
    def maximum_weight(self, value: np.float64) -> None:
        self._maximum_weight[0] = value
 


    @property
    def hadronic_thinning(self):
        """hadronic thinning ratio"""
        return self._hadronic_thinning[0]

    @hadronic_thinning.setter
    def hadronic_thinning(self, value: np.float64) -> None:
        self._hadronic_thinning[0] = value



    @property
    def hadronic_thinning_weight(self):
        """hadronic thinning weight ratio"""
        return self._hadronic_thinning_weight[0]

    @hadronic_thinning_weight.setter
    def hadronic_thinning_weight(self, value: np.float64) -> None:
        self._hadronic_thinning_weight[0] = value

    

    @property
    def rmax(self):
        """Maximum radius (in cm) at observation level within which all particles are subject to inner radius thinning"""
        return self._rmax[0]

    @rmax.setter
    def rmax(self, value: np.float64) -> None:
        self._rmax[0] = value



    @property
    def lowe_cut_gamma(self):
        """gamma energy cut (GeV)"""
        return self._lowe_cut_gamma[0]

    @lowe_cut_gamma.setter
    def lowe_cut_gamma(self, value: np.float64) -> None:
        self._lowe_cut_gamma[0] = value  
      


    @property
    def lowe_cut_e(self):
        """electron energy cut (GeV)"""
        return self._lowe_cut_e[0]

    @lowe_cut_e.setter
    def lowe_cut_e(self, value: np.float64) -> None:
        self._lowe_cut_e[0] = value 



    @property
    def lowe_cut_mu(self):
        """muon energy cut (GeV)"""
        return self._lowe_cut_mu[0]

    @lowe_cut_mu.setter
    def lowe_cut_mu(self, value: np.float64) -> None:
        self._lowe_cut_mu[0] = value 



    @property
    def lowe_cut_meson(self):
        """meson energy cut (GeV)"""
        return self._lowe_cut_meson[0]

    @lowe_cut_meson.setter
    def lowe_cut_meson(self, value: np.float64) -> None:
        self._lowe_cut_meson[0] = value 



    @property
    def lowe_cut_nucleon(self):
        """nucleon energy cut (GeV)"""
        return self._lowe_cut_nucleon[0]

    @lowe_cut_nucleon.setter
    def lowe_cut_nucleon(self, value: np.float64) -> None:
        self._lowe_cut_nucleon[0] = value 



    @property
    def event_name(self):
        """Event name"""
        return str(self._event_name)

    @event_name.setter
    def event_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for event_name {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._event_name.string.assign(value)



    @property
    def event_date(self):
        """Event Date"""
        return str(self._event_date)

    @event_date.setter
    def event_date(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for date {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._event_date.string.assign(value)



    @property
    def rnd_seed(self):
        """Random seed"""
        return self._rnd_seed[0]

    @rnd_seed.setter
    def rnd_seed(self, value):
        self._rnd_seed[0] = value



    @property
    def energy_in_neutrinos(self):
        """Energy in neutrinos generated in the shower (GeV). Usefull for invisible energy"""
        return self._energy_in_neutrinos[0]

    @energy_in_neutrinos.setter
    def energy_in_neutrinos(self, value):
        self._energy_in_neutrinos[0] = value


    @property
    def azimuth(self):
        """Shower azimuth TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth. Also, geoid vs sphere problem"""
        return self._azimuth[0]

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth[0] = value



    @property
    def zenith(self):
        """Shower zenith TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth"""
        return self._zenith[0]

    @zenith.setter
    def zenith(self, value):
        self._zenith[0] = value



    @property
    def atmos_model(self):
        """Atmospheric model name TODO:standarize"""
        return str(self._atmos_model)

    @atmos_model.setter
    def atmos_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._atmos_model.string.assign(value)



    @property
    def atmos_model_param(self):
        """Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters"""
        return np.array(self._atmos_model_param)

    @atmos_model_param.setter
    def atmos_model_param(self, value):
        self._atmos_model_param = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("atmos_model_param", self._atmos_model_param)



    @property
    def magnetic_field(self):
        """Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense."""
        return np.array(self._magnetic_field)

    @magnetic_field.setter
    def magnetic_field(self, value):
        self._magnetic_field = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("magnetic_field", self._magnetic_field)



    @property
    def xmax_grams(self):
        """Shower Xmax depth (g/cm2 along the shower axis)"""
        return self._xmax_grams[0]

    @xmax_grams.setter
    def xmax_grams(self, value):
        self._xmax_grams[0] = value



    @property
    def xmax_pos_shc(self):
        """Shower Xmax position in shower coordinates"""
        return np.array(self._xmax_pos_shc)

    @xmax_pos_shc.setter
    def xmax_pos_shc(self, value):
        self._xmax_pos_shc = np.array(value).astype(np.float64)
        self._tree.SetBranchAddress("xmax_pos_shc", self._xmax_pos_shc)



    @property
    def xmax_distance(self):
        """Distance of Xmax [m]"""
        return self._xmax_distance[0]

    @xmax_distance.setter
    def xmax_distance(self, value):
        self._xmax_distance[0] = value



    @property
    def xmax_alt(self):
        """Altitude of Xmax (m, in the shower simulation earth. Its important for the index of refraction )"""
        return self._xmax_alt[0]

    @xmax_alt.setter
    def xmax_alt(self, value):
        self._xmax_alt[0] = value



    @property
    def hadronic_model(self):
        """High energy hadronic model (and version) used TODO: standarize"""
        return str(self._hadronic_model)

    @hadronic_model.setter
    def hadronic_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._hadronic_model.string.assign(value)



    @property
    def low_energy_model(self):
        """High energy model (and version) used TODO: standarize"""
        return str(self._low_energy_model)

    @low_energy_model.setter
    def low_energy_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._low_energy_model.string.assign(value)



    @property
    def cpu_time(self):
        """Time it took for the shower + efield simulation."""
        return np.array(self._cpu_time)

    @cpu_time.setter
    def cpu_time(self, value):
        self._cpu_time = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("cpu_time", self._cpu_time)
        


    @property
    def shower_core_pos(self):
        """Shower core position"""
        return np.array(self._shower_core_pos)

    @shower_core_pos.setter
    def shower_core_pos(self, value):
        self._shower_core_pos = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("shower_core_pos", self._shower_core_pos)        
        


    @property
    def unix_date(self):
        """The date of the event in seconds since epoch"""
        return self._unix_date[0]

    @unix_date.setter
    def unix_date(self, val: np.uint32) -> None:
        self._unix_date[0] = val



    @property
    def first_interaction(self):
        """Height of the first interaction"""
        return self._first_interaction[0]

    @first_interaction.setter
    def first_interaction(self, value):
        self._first_interaction[0] = value
    
    
#####################################################################################################################################################################################################
#
# RawEfieldTree
# 
#####################################################################################################################################################################################################
        
@dataclass
## The class for storing Efield simulation-only data common for each event
class RawEfieldTree(MotherEventTree):
    """The class for storing Efield simulation-only data common for each event"""

    _type: str = "rawefield"

    _tree_name: str = "trawefield"

    #Per Event Things
    ## Name and version of the electric field simulator
    _efield_sim: StdString = StdString("")

    ## Name of the atmospheric index of refraction model
    _refractivity_model: StdString = StdString("")
    _refractivity_model_parameters: StdVectorList = field(default_factory=lambda: StdVectorList("double"))    
    _atmos_refractivity: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    
    
    ## The antenna time window is defined around a t0 that changes with the antenna, starts on t0+t_pre (thus t_pre is usually negative) and ends on t0+post
    _t_pre: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    _t_post: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    _t_bin_size: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))    
    
    #Per antenna things
    _du_id: StdVectorList = field(default_factory=lambda: StdVectorList("int"))  # Detector ID
    _du_name: StdVectorList = field(default_factory=lambda: StdVectorList("string"))  # Detector Name
    ## Number of detector units in the event - basically the antennas count
    _du_count: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))


        
    _t_0: StdVectorList = field(default_factory=lambda: StdVectorList("float"))  # Time window t0
    _p2p: StdVectorList = field(default_factory=lambda: StdVectorList("float"))  # peak 2 peak amplitudes (x,y,z,modulus)

    ## X position in shower referential
    _du_x: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## Y position in shower referential
    _du_y: StdVectorList = field(default_factory=lambda: StdVectorList("float"))
    ## Z position in shower referential
    _du_z: StdVectorList = field(default_factory=lambda: StdVectorList("float"))    
    

    ## Efield traces 
    ## Efield trace in X direction
    _trace_x: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Efield trace in Y direction
    _trace_y: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))
    ## Efield trace in Z direction
    _trace_z: StdVectorList = field(default_factory=lambda: StdVectorList("vector<float>"))



    @property
    def du_count(self):
        """Number of detector units in the event - basically the antennas count"""
        return self._du_count[0]

    @du_count.setter
    def du_count(self, value: np.uint32) -> None:
        self._du_count[0] = value

    @property
    def efield_sim(self):
         return str(self._efield_sim)
    
    @efield_sim.setter
    def efield_sim(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required.")
    
        self._efield_sim.string.assign(value)

    @property
    def refractivity_model(self):
        """Name of the atmospheric index of refraction model"""
        return str(self._refractivity_model)

    @refractivity_model.setter
    def refractivity_model(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for site {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._refractivity_model.string.assign(value)

    @property
    def refractivity_model_parameters(self):
        """Refractivity model parameters"""
        return self._refractivity_model_parameters

    @refractivity_model_parameters.setter
    def refractivity_model_parameters(self, value) -> None:
        # A list of strings was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._refractivity_model_parameters.clear()
            self._refractivity_model_parameters += value
        # A vector was given
        elif isinstance(value, ROOT.vector("double")):
            self._refractivity_model_parameters._vector = value
        else:
            raise ValueError(
                f"Incorrect type for refractivity_model_parameters {type(value)}. Either a list, an array or a ROOT.vector of unsigned shorts required."
            )


    @property
    def atmos_refractivity(self):
        """refractivity for each altitude at atmos_altiude table"""
        return self._atmos_refractivity


    @atmos_refractivity.setter
    def atmos_refractivity(self, value):
        # A list was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._atmos_refractivity.clear()
            self._atmos_refractivity += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._atmos_refractivity._vector = value
        else:
            raise ValueError(
                f"Incorrect type for atmos_refractivity {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )            
        

    @property
    def t_pre(self):
        """Starting time of antenna data collection time window. The window starts at t0+t_pre, thus t_pre is usually negative."""
        return self._t_pre[0]

    @t_pre.setter
    def t_pre(self, value):
        self._t_pre[0] = value

    @property
    def t_post(self):
        """Finishing time of antenna data collection time window. The window ends at t0+t_post."""
        return self._t_post[0]

    @t_post.setter
    def t_post(self, value):
        self._t_post[0] = value

    @property
    def t_bin_size(self):
        """Time bin size"""
        return self._t_bin_size[0]

    @t_bin_size.setter
    def t_bin_size(self, value):
        self._t_bin_size[0] = value




    @property
    def du_id(self):
        """Detector ID"""
        return self._du_id

    @du_id.setter
    def du_id(self, value):
        # A list was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_id.clear()
            self._du_id += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("int")):
            self._du_id._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_id {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )

    @property
    def du_name(self):
        """Detector Name"""
        return self._du_name    


    @du_name.setter
    def du_name(self, value):
        # A list of strings was given
        if isinstance(value, list):
            # Clear the vector before setting
            self._du_name.clear()
            self._du_name += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("string")):
            self._du_name._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_name {type(value)}. Either a list or a ROOT.vector of strings required."
            )



    @property
    def t_0(self):
        """Time window t0"""
        return self._t_0

    @t_0.setter
    def t_0(self, value):
        # A list was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._t_0.clear()
            self._t_0 += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._t_0._vector = value
        else:
            raise ValueError(
                f"Incorrect type for t_0 {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )

    @property
    def p2p(self):
        """Peak 2 peak amplitudes (x,y,z,modulus)"""
        return self._p2p

    @p2p.setter
    def p2p(self, value):
        # A list was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._p2p.clear()
            self._p2p += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("float")):
            self._p2p._vector = value
        else:
            raise ValueError(
                f"Incorrect type for p2p {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )        

                

    @property
    def du_x(self):
        """X position in site's referential"""
        return self._du_x

    @du_x.setter
    def du_x(self, value) -> None:
        # A list of strings was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_x.clear()
            self._du_x += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._du_x._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_x {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def du_y(self):
        """Y position in site's referential"""
        return self._du_y

    @du_y.setter
    def du_y(self, value) -> None:
        # A list of strings was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_y.clear()
            self._du_y += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._du_y._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_y {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )

    @property
    def du_z(self):
        """Z position in site's referential"""
        return self._du_z

    @du_z.setter
    def du_z(self, value) -> None:
        # A list of strings was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._du_z.clear()
            self._du_z += value
        # A vector was given
        elif isinstance(value, ROOT.vector("float")):
            self._du_z._vector = value
        else:
            raise ValueError(
                f"Incorrect type for du_z {type(value)}. Either a list, an array or a ROOT.vector of floats required."
            )



    @property
    def trace_x(self):
        """Efield trace in X direction"""
        return self._trace_x

    @trace_x.setter
    def trace_x(self, value):
        # A list was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trace_x.clear()
            self._trace_x += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_x._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trace_x {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )


    @property
    def trace_y(self):
        """Efield trace in Y direction"""
        return self._trace_y

    @trace_y.setter
    def trace_y(self, value):
        # A list was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trace_y.clear()
            self._trace_y += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_y._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trace_y {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )
        


    @property
    def trace_z(self):
        """Efield trace in Z direction"""
        return self._trace_z

    @trace_z.setter
    def trace_z(self, value):
        # A list was given
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._trace_z.clear()
            self._trace_z += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._trace_z._vector = value
        else:
            raise ValueError(
                f"Incorrect type for trace_z {type(value)}. Either a list, an array or a ROOT.vector of float required."
            )
        

#############################################################################################################################################################################################################################
#
#   RawMetaTree
#
#############################################################################################################################################################################################################################

@dataclass
## The class for storing meta-data for each event, that is meta to the shower and efield simulation (like, coreposition, array name, antenna selection, etc)
class RawMetaTree(MotherEventTree):
    """The class for storing data about the event generation that is meta to the shower/efield simulation itself"""


    _type: str = "rawmeta"

    _tree_name: str = "trawmeta"
    
    ### Array over wich the event was simulated (use "starshape" for...starshapes)
    _array_name: StdString = StdString("")
    
    #In the simulation, the coordinates are in "shower coordinates" whose origin is at the core position. So core position is always 0,0,0. The core position this represents in your array is meta the simulator 
    ### Core position with respect to the antenna array (undefined for neutrinos)
    _shower_core_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32)) 
        
    #In the simulation, the origin of time is when the shower hits the ground.
    #There is a date (no time of the day, just the date) that can be used to get the magnetic field from the magnetic field model, but nothing else.
    #If the event you are simulating represents an event that happened at a specific time, you set here its second and nanosecond.     
    ### Unix second of the shower t0 (when the core traveling at c arrives at the ground?)
    _unix_second: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))    
    
    ### Unix nanosecond of the shower t0 (when the core traveling at c arrives at the ground?)
    _unix_nanosecond: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))    
    
    #The event you are simulating might be the result of several trials at input generation until you found one that has some chance of triggering. You need to store this info for efficiency studies.
    ### statistical weight given to the event
    _event_weight: np.ndarray = field(default_factory=lambda: np.zeros(1, np.uint32))    
    ### tested core positions
    _tested_cores: StdVectorList = field(default_factory=lambda:StdVectorList("vector<float>"))    

    @property
    def array_name(self):
        """array name"""
        return str(self._array_name)

    @array_name.setter
    def array_name(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for array_name {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._array_name.string.assign(value)

    @property
    def shower_core_pos(self):
        """Shower core position"""
        return np.array(self._shower_core_pos)

    @shower_core_pos.setter
    def shower_core_pos(self, value):
        self._shower_core_pos = np.array(value).astype(np.float32)
        self._tree.SetBranchAddress("shower_core_pos", self._shower_core_pos)         

    @property
    def unix_second(self):
        """Unix second of the shower t0 (when the core traveling at c arrives at the ground?)"""
        return self._unix_second[0]

    @unix_second.setter
    def unix_second(self, val: np.uint32) -> None:
        self._unix_second[0] = val


    @property
    def unix_nanosecond(self):
        """Unix nanosecond of the shower t0 (when the core traveling at c arrives at the ground?)"""
        return self._unix_nanosecond[0]

    @unix_nanosecond.setter
    def unix_nanosecond(self, val: np.uint32) -> None:
        self._unix_nanosecond[0] = val
        
        
    @property
    def event_weight(self):
        """The event statistical weight"""
        return self._event_weight[0]

    @event_weight.setter
    def event_weight(self, val: np.uint32) -> None:
        self._event_weight[0] = val
        
    @property
    def tested_cores(self):
        """tested cores"""
        return self._tested_cores
        
    @tested_cores.setter
    def tested_cores(self, value):
        # A list was given  
        if (
            isinstance(value, list)
            or isinstance(value, np.ndarray)
            or isinstance(value, StdVectorList)
        ):
            # Clear the vector before setting
            self._tested_cores.clear()
            self._tested_cores += value
        # A vector of strings was given
        elif isinstance(value, ROOT.vector("vector<float>")):
            self._tested_cores._vector = value
        else:
            raise ValueError(
                f"Incorrect type for _tested_cores {type(value)}. Either a list, an array or a ROOT.vector of vector<float> required."
            )                

#############################################################################################################################################################################################################################
#
#   RawZHAiresTree
#
#############################################################################################################################################################################################################################
       
@dataclass
## The class for storing shower data for each event specific to ZHAireS only
class RawZHAireSTree(MotherEventTree):
    """The class for storing shower data for each event specific to ZHAireS only"""

    _type: str = "eventshowerzhaires"

    _tree_name: str = "teventshowerzhaires"

    # ToDo: we need explanations of these parameters
      

    _other_parameters: StdString = StdString("")



    @property
    def other_parameters(self):
        """Other parameters"""
        return str(self._other_parameters)

    @other_parameters.setter
    def other_parameters(self, value):
        # Not a string was given
        if not (isinstance(value, str) or isinstance(value, ROOT.std.string)):
            raise ValueError(
                f"Incorrect type for other_parameters {type(value)}. Either a string or a ROOT.std.string is required."
            )

        self._other_parameters.string.assign(value)        

#############################################################################################################################################################################################################################
#
#   RawCoreasTree
#
#############################################################################################################################################################################################################################
     
@dataclass
## The class for storing shower data for each event specific to Coreas only
class RawCoreasTree(MotherEventTree):
    """The class for storing shower data for each event specific to Coreas only"""

    _type: str = "eventshowercoreas"

    _tree_name: str = "teventshowercoreas"

    _AutomaticTimeBoundaries: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _ResolutionReductionScale: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _GroundLevelRefractiveIndex: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _GPSSecs: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _GPSNanoSecs: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _DepthOfShowerMaximum: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _DistanceOfShowerMaximum: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _GeomagneticAngle: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _nshow: StdVectorList = field(default_factory=lambda: np.zeros(1, np.float32))
    _ectmap: StdString = field(default_factory=lambda: np.zeros(1, np.float32))
    _maxprt: StdString = field(default_factory=lambda: np.zeros(1, np.float32))
    _radnkg: StdString = field(default_factory=lambda: np.zeros(1, np.float32))
    _parallel_ectcut: StdString = field(default_factory=lambda: np.zeros(1, np.float32))
    _parallel_ectmax: StdString = field(default_factory=lambda: np.zeros(1, np.float32))



    @property
    def AutomaticTimeBoundaries(self):
        """Automatic time boundaries"""
        return self._AutomaticTimeBoundaries
    
    @AutomaticTimeBoundaries.setter
    def AutomaticTimeBoundaries(self, value):
        self._AutomaticTimeBoundaries[0] = value 
    


    @property
    def ResolutionReductionScale(self):
        """Resolution reduction scale"""
        return self._ResolutionReductionScale
    
    @ResolutionReductionScale.setter
    def ResolutionReductionScale(self, value):
        self._ResolutionReductionScale[0] = value 



    @property
    def GroundLevelRefractiveIndex(self):
        """Ground level refractive index"""
        return self._GroundLevelRefractiveIndex
    
    @GroundLevelRefractiveIndex.setter
    def GroundLevelRefractiveIndex(self, value):
        self._GroundLevelRefractiveIndex[0] = value 



    @property
    def GPSSecs(self):
        """GPS seconds"""
        return self._GPSSecs
    
    @GPSSecs.setter
    def GPSSecs(self, value):
        self._GPSSecs[0] = value  



    @property
    def GPSNanoSecs(self):
        """GPS nanoseconds"""
        return self._GPSNanoSecs
    
    @GPSNanoSecs.setter
    def GPSNanoSecs(self, value):
        self._GPSNanoSecs[0] = value  
        


    @property
    def DepthOfShowerMaximum(self):
        """Depth of shower maximum"""
        return self._DepthOfShowerMaximum
    
    @DepthOfShowerMaximum.setter
    def DepthOfShowerMaximum(self, value):
        self._DepthOfShowerMaximum[0] = value  
        


    @property
    def DistanceOfShowerMaximum(self):
        """Distance of shower maximum"""
        return self._DistanceOfShowerMaximum
    
    @DistanceOfShowerMaximum.setter
    def DistanceOfShowerMaximum(self, value):
        self._DistanceOfShowerMaximum[0] = value 



    @property
    def GeomagneticAngle(self):
        """Geomagnetic angle"""
        return self._GeomagneticAngle
    
    @GeomagneticAngle.setter
    def GeomagneticAngle(self, value):
        self._GeomagneticAngle[0] = value  



    @property
    def nshow(self):
        """nshow"""
        return self._nshow
    
    @nshow.setter
    def nshow(self, value):
        self._nshow[0] = value 



    @property
    def ectmap(self):
        """ectmap"""
        return self._ectmap
    
    @ectmap.setter
    def ectmap(self, value):
        self._ectmap[0] = value 



    @property
    def maxprt(self):
        """maxprt"""
        return self._maxprt
    
    @maxprt.setter
    def maxprt(self, value):
        self._maxprt[0] = value 



    @property
    def radnkg(self):
        """radnkg"""
        return self._radnkg
    
    @radnkg.setter
    def radnkg(self, value):
        self._radnkg[0] = value 
    


    @property
    def parallel_ectcut(self):
        """first value of PARALLEL option in Coreas input"""
        return self._parallel_ectcut
    
    @parallel_ectcut.setter
    def parallel_ectcut(self, value):
        self._parallel_ectcut[0] = value

    

    @property
    def parallel_ectmax(self):
        """first value of PARALLEL option in Coreas input"""
        return self._parallel_ectmax
    
    @parallel_ectmax.setter
    def parallel_ectmax(self, value):
        self._parallel_ectmax[0] = value
