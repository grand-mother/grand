# Table of Contents

* [EventParametersGenerator](#EventParametersGenerator)
  * [GenerateEventParametersFile](#EventParametersGenerator.GenerateEventParametersFile)
  * [GetCorePositionFromParametersFile](#EventParametersGenerator.GetCorePositionFromParametersFile)
  * [GetTestedPositionsFromParametersFile](#EventParametersGenerator.GetTestedPositionsFromParametersFile)
  * [GetEventWeightFromParametersFile](#EventParametersGenerator.GetEventWeightFromParametersFile)
  * [GetEventUnixTimeFromParametersFile](#EventParametersGenerator.GetEventUnixTimeFromParametersFile)
* [raw\_root\_trees](#raw_root_trees)
  * [RawShowerTree](#raw_root_trees.RawShowerTree)
    * [long\_depth](#raw_root_trees.RawShowerTree.long_depth)
    * [long\_slantdepth](#raw_root_trees.RawShowerTree.long_slantdepth)
    * [long\_gammas](#raw_root_trees.RawShowerTree.long_pd_gamma)
    * [long\_eplus](#raw_root_trees.RawShowerTree.long_pd_eplus)
    * [long\_eminus](#raw_root_trees.RawShowerTree.long_pd_eminus)
    * [long\_muplus](#raw_root_trees.RawShowerTree.long_pd_muplus)
    * [long\_muminus](#raw_root_trees.RawShowerTree.long_pd_muminus)
    * [long\_allch](#raw_root_trees.RawShowerTree.long_pd_allch)
    * [long\_nuclei](#raw_root_trees.RawShowerTree.long_pd_nuclei)
    * [long\_hadr](#raw_root_trees.RawShowerTree.long_pd_hadr)
    * [long\_neutrino](#raw_root_trees.RawShowerTree.long_ed_neutrino)
    * [long\_gamma\_cut](#raw_root_trees.RawShowerTree.long_ed_gamma_cut)
    * [long\_gamma\_ioniz](#raw_root_trees.RawShowerTree.long_ed_gamma_ioniz)
    * [long\_e\_cut](#raw_root_trees.RawShowerTree.long_ed_e_cut)
    * [long\_e\_ioniz](#raw_root_trees.RawShowerTree.long_ed_e_ioniz)
    * [long\_mu\_cut](#raw_root_trees.RawShowerTree.long_ed_mu_cut)
    * [long\_mu\_ioniz](#raw_root_trees.RawShowerTree.long_ed_mu_ioniz)
    * [long\_hadr\_cut](#raw_root_trees.RawShowerTree.long_ed_hadr_cut)
    * [long\_hadr\_ioniz](#raw_root_trees.RawShowerTree.long_ed_hadr_ioniz)
    * [relative\_thinning](#raw_root_trees.RawShowerTree.rel_thin)
    * [weight\_factor](#raw_root_trees.RawShowerTree.maximum_weight)
    * [gamma\_energy\_cut](#raw_root_trees.RawShowerTree.lowe_cut_gamma)
    * [electron\_energy\_cut](#raw_root_trees.RawShowerTree.lowe_cut_e)
    * [muon\_energy\_cut](#raw_root_trees.RawShowerTree.lowe_cut_mu)
    * [meson\_energy\_cut](#raw_root_trees.RawShowerTree.lowe_cut_meson)
    * [nucleon\_energy\_cut](#raw_root_trees.RawShowerTree.lowe_cut_nucleon)
    * [event\_name](#raw_root_trees.RawShowerTree.event_name)
    * [event\_date](#raw_root_trees.RawShowerTree.event_date)
    * [rnd\_seed](#raw_root_trees.RawShowerTree.rnd_seed)
    * [energy\_in\_neutrinos](#raw_root_trees.RawShowerTree.energy_in_neutrinos)
    * [prim\_energy](#raw_root_trees.RawShowerTree.energy_primary)
    * [shower\_azimuth](#raw_root_trees.RawShowerTree.azimuth)
    * [shower\_zenith](#raw_root_trees.RawShowerTree.zenith)
    * [prim\_type](#raw_root_trees.RawShowerTree.primary_type)
    * [prim\_injpoint\_shc](#raw_root_trees.RawShowerTree.primary_inj_point_shc)
    * [prim\_inj\_alt\_shc](#raw_root_trees.RawShowerTree.primary_inj_alt_shc)
    * [prim\_inj\_dir\_shc](#raw_root_trees.RawShowerTree.primary_inj_dir_shc)
    * [site](#raw_root_trees.RawShowerTree.site)
    * [site_alt](#raw_root_trees.RawShowerTree.site_alt)
    * [site_lon](#raw_root_trees.RawShowerTree.site_lon)
    * [site_lat](#raw_root_trees.RawShowerTree.site_lat)
    * [atmos\_model](#raw_root_trees.RawShowerTree.atmos_model)
    * [atmos\_model\_param](#raw_root_trees.RawShowerTree.atmos_model_param)
    * [atmos\_altitude](#raw_root_trees.RawShowerTree.atmos_altitude)
    * [atmos\_density](#raw_root_trees.RawShowerTree.atmos_density)
    * [atmos\_depth](#raw_root_trees.RawShowerTree.atmos_depth)
    * [magnetic\_field](#raw_root_trees.RawShowerTree.magnetic_field)
    * [xmax\_grams](#raw_root_trees.RawShowerTree.xmax_grams)
    * [xmax\_pos\_shc](#raw_root_trees.RawShowerTree.xmax_pos_shc)
    * [xmax\_distance](#raw_root_trees.RawShowerTree.xmax_distance)
    * [xmax\_alt](#raw_root_trees.RawShowerTree.xmax_alt)
    * [hadronic\_model](#raw_root_trees.RawShowerTree.hadronic_model)
    * [low\_energy\_model](#raw_root_trees.RawShowerTree.low_energy_model)
    * [cpu\_time](#raw_root_trees.RawShowerTree.cpu_time)
    * [shower\_core\_pos](#raw_root_trees.RawShowerTree.shower_core_pos)
    * [unix\_date](#raw_root_trees.RawShowerTree.unix_date)
  * [RawEfieldTree](#raw_root_trees.RawEfieldTree)
    * [du\_count](#raw_root_trees.RawEfieldTree.du_count)
    * [refractivity\_model](#raw_root_trees.RawEfieldTree.refractivity_model)
    * [refractivity\_model\_parameters](#raw_root_trees.RawEfieldTree.refractivity_model_parameters)
    * [atmos\_refractivity](#raw_root_trees.RawEfieldTree.atmos_refractivity)
    * [t\_pre](#raw_root_trees.RawEfieldTree.t_pre)
    * [t\_post](#raw_root_trees.RawEfieldTree.t_post)
    * [t\_bin\_size](#raw_root_trees.RawEfieldTree.t_bin_size)
    * [du\_id](#raw_root_trees.RawEfieldTree.du_id)
    * [du\_name](#raw_root_trees.RawEfieldTree.du_name)
    * [t\_0](#raw_root_trees.RawEfieldTree.t_0)
    * [p2p](#raw_root_trees.RawEfieldTree.p2p)
    * [trace\_x](#raw_root_trees.RawEfieldTree.trace_x)
    * [trace\_y](#raw_root_trees.RawEfieldTree.trace_y)
    * [trace\_z](#raw_root_trees.RawEfieldTree.trace_z)
    * [pos\_x](#raw_root_trees.RawEfieldTree.du_x)
    * [pos\_y](#raw_root_trees.RawEfieldTree.du_y)
    * [pos\_z](#raw_root_trees.RawEfieldTree.du_z)
  * [RawZHAireSTree](#raw_root_trees.RawZHAireSTree)
    * [relative\_thining](#raw_root_trees.RawZHAireSTree.relative_thining)
    * [weight\_factor](#raw_root_trees.RawZHAireSTree.maximum_weight)
    * [gamma\_energy\_cut](#raw_root_trees.RawZHAireSTree.lowe_cut_gamma)
    * [electron\_energy\_cut](#raw_root_trees.RawZHAireSTree.lowe_cut_e)
    * [muon\_energy\_cut](#raw_root_trees.RawZHAireSTree.lowe_cut_mu)
    * [meson\_energy\_cut](#raw_root_trees.RawZHAireSTree.lowe_cut_meson)
    * [nucleon\_energy\_cut](#raw_root_trees.RawZHAireSTree.lowe_cut_nucleon)
    * [other\_parameters](#raw_root_trees.RawZHAireSTree.other_parameters)

<a id="EventParametersGenerator"></a>

# EventParametersGenerator

<a id="EventParametersGenerator.GenerateEventParametersFile"></a>

#### GenerateEventParametersFile

```python
def GenerateEventParametersFile(EventName,
                                Primary,
                                Energy,
                                zenith,
                                azimuth,
                                CorePosition,
                                ArrayName,
                                EventWeight=1,
                                EventUnixTime=0,
                                EventUnixNanosecond=0,
                                OutMode="a",
                                TestedPositions="None")
```

The function generates an event parameters file in a specific format for use in simulation programs. 
The file includes information such as the event name, primary particle, energy, zenith and azimuth angles, core position, 
and array name. 
Optionally, the Event Weight and Unix Time and Unix Nanosecond can be specified
Additionally, the file can include a list of tested positions that were tried before generating the event. 

The idea behind having this file is to make it friendly for other simulation programs, and to avoid putting extra things in the Aires/ZHAireS/CoREAS .inp file
For now, the only really needed parameters are ArrayName and CorePosition, but if we implement an antenna selection this would be the place to put that information.
Also is the place for other information external to Aires/ZHAireS regarding event generation (i.e. parameters of the core position generation or the antenna selection)

Inputs:
- EventName (str): The name of the event.
- Primary (str): The type of primary particle.
- Energy (float): The energy of the primary particle in GeV.
- Zenith (float): The zenith angle of the event in degrees.
- Azimuth (float): The azimuth angle of the event in degrees.
- CorePosition (tuple): The (x,y,z) coordinates of the core position in meters.
- ArrayName (str): The name of the array used for the simulation.
- EventWeight (float,optional): The statistical weight of the event. Default is 1
- EventUnixTime (unisgned int, optional): The event time in seconds since EPOCH. Default is 0
- EventUnixTime (unisgned int, optional): The nanoseconds in the event second since EPOCH. Default is 0    
- OutMode (str, optional): The mode for opening the output file. Default is "a" for append mode.
- TestedPositions (list of tuples, optional): A list of (x,y,z) tuples representing tested core positions.

Output:
- None

<a id="EventParametersGenerator.GetCorePositionFromParametersFile"></a>

#### GetCorePositionFromParametersFile

```python
def GetCorePositionFromParametersFile(filename)
```

This function reads the core position from a file generated by the function GenerateEventParametersFile.

Input:
filename (string): The name of the file to read from.

Output:
A tuple containing the x, y, and z coordinates of the core position.
If the file is not found, or if the core position is not found or the format is incorrect, 
the function returns (0.0, 0.0, 0.0).

<a id="EventParametersGenerator.GetTestedPositionsFromParametersFile"></a>

#### GetTestedPositionsFromParametersFile

```python
def GetTestedPositionsFromParametersFile(file_path)
```

Reads the core positions tested before generating the event from the given EventParameters file.

**Arguments**:

- `file_path` _str_ - The path to the EventParameters file.
  

**Returns**:

  A list of tuples representing the core positions tested before generating the event. Each tuple has three
  elements (x, y, z) representing the position of the core.
  

**Raises**:

- `FileNotFoundError` - If the EventParameters file is not found.
- `ValueError` - If the core positions tested section is found but the expected data cannot be read due to formatting
  errors.

<a id="EventParametersGenerator.GetEventWeightFromParametersFile"></a>

#### GetEventWeightFromParametersFile

```python
def GetEventWeightFromParametersFile(filename)
```

This function reads the EventWeight from a given Event Parameters file.

**Arguments**:

- `filename` _str_ - The name of the Event Parameters file.
  

**Returns**:

- `float` - The value of the EventWeight parameter.
  

**Raises**:

- `FileNotFoundError` - If the file cannot be found.

<a id="EventParametersGenerator.GetEventUnixTimeFromParametersFile"></a>

#### GetEventUnixTimeFromParametersFile

```python
def GetEventUnixTimeFromParametersFile(filename)
```

This function reads the EventUnixTime and EventUnixNanosecond from a given Event Parameters file.

**Arguments**:

- `filename` _str_ - The name of the Event Parameters file.
  

**Returns**:

- `tuple` - A tuple containing the Unix time in seconds and nanoseconds.
  

**Raises**:

- `FileNotFoundError` - If the file cannot be found.

<a id="raw_root_trees"></a>

# raw\_root\_trees

Read/Write python interface to GRAND data (real and simulated) stored in Cern ROOT TTrees.

This is the interface for accessing GRAND ROOT TTrees that do not require the user (reader/writer of the TTrees) to have any knowledge of ROOT. It also hides the internals from the data generator, so that the changes in the format are not concerning the user.

<a id="raw_root_trees.RawShowerTree"></a>

## RawShowerTree Objects

```python
@dataclass
class RawShowerTree(MotherEventTree)
```

The class for storing a shower simulation-only data for each event

<a id="raw_root_trees.RawShowerTree.long_depth"></a>

#### long\_depth

```python
@property
def long_depth()
```

Longitudinal profile depth (g/cm2)

<a id="raw_root_trees.RawShowerTree.long_slantdepth"></a>

#### long\_slantdepth

```python
@property
def long_slantdepth()
```

Longitudinal profile of slant depth (g/cm2)

<a id="raw_root_trees.RawShowerTree.long_pd_gamma"></a>

#### long\_gammas

```python
@property
def long_pd_gamma()
```

Longitudinal profile of gammas

<a id="raw_root_trees.RawShowerTree.long_pd_eplus"></a>

#### long\_eplus

```python
@property
def long_pd_eplus()
```

Longitudinal profile of positrons

<a id="raw_root_trees.RawShowerTree.long_pd_eminus"></a>

#### long\_eminus

```python
@property
def long_pd_eminus()
```

Longitudinal profile of electrons

<a id="raw_root_trees.RawShowerTree.long_pd_muplus"></a>

#### long\_muplus

```python
@property
def long_pd_muplus()
```

Longitudinal profile of positrons

<a id="raw_root_trees.RawShowerTree.long_pd_muminus"></a>

#### long\_muminus

```python
@property
def long_pd_muminus()
```

Longitudinal profile of electrons

<a id="raw_root_trees.RawShowerTree.long_pd_allch"></a>

#### long\_allch

```python
@property
def long_pd_allch()
```

Longitudinal profile of all charged particles

<a id="raw_root_trees.RawShowerTree.long_pd_nuclei"></a>

#### long\_nuclei

```python
@property
def long_pd_nuclei()
```

Longitudinal profile of nuclei

<a id="raw_root_trees.RawShowerTree.long_pd_hadr"></a>

#### long\_hadr

```python
@property
def long_pd_hadr()
```

Longitudinal profile of hadrons

<a id="raw_root_trees.RawShowerTree.long_ed_neutrino"></a>

#### long\_neutrino

```python
@property
def long_ed_neutrino()
```

Longitudinal profile of created neutrinos

<a id="raw_root_trees.RawShowerTree.long_ed_gamma_cut"></a>

#### long\_gamma\_cut

```python
@property
def long_ed_gamma_cut()
```

Longitudinal profile of low energy gammas

<a id="raw_root_trees.RawShowerTree.long_ed_gamma_ioniz"></a>

#### long\_gamma\_ioniz

```python
@property
def long_ed_gamma_ioniz()
```

Longitudinal profile of gamma energy deposit

<a id="raw_root_trees.RawShowerTree.long_ed_e_cut"></a>

#### long\_e\_cut

```python
@property
def long_ed_e_cut()
```

Longitudinal profile of low energy e+/e-

<a id="raw_root_trees.RawShowerTree.long_ed_e_ioniz"></a>

#### long\_e\_ioniz

```python
@property
def long_ed_e_ioniz()
```

Longitudinal profile of energy deposit by e+/e-

<a id="raw_root_trees.RawShowerTree.long_ed_mu_cut"></a>

#### long\_mu\_cut

```python
@property
def long_ed_mu_cut()
```

Longitudinal profile of low energy muons

<a id="raw_root_trees.RawShowerTree.long_ed_mu_ioniz"></a>

#### long\_mu\_ioniz

```python
@property
def long_ed_mu_ioniz()
```

Longitudinal profile of muon energy deposit

<a id="raw_root_trees.RawShowerTree.long_ed_hadr_cut"></a>

#### long\_hadr\_cut

```python
@property
def long_ed_hadr_cut()
```

Longitudinal profile of low energy hadrons

<a id="raw_root_trees.RawShowerTree.long_ed_hadr_ioniz"></a>

#### long\_hadr\_ioniz

```python
@property
def long_ed_hadr_ioniz()
```

Longitudinal profile of hadrons energy deposit

<a id="raw_root_trees.RawShowerTree.rel_thin"></a>

#### relative\_thinning

```python
@property
def rel_thin()
```

Thinning energy, relative to primary energy

<a id="raw_root_trees.RawShowerTree.maximum_weight"></a>

#### weight\_factor

```python
@property
def maximum_weight()
```

Weight factor

<a id="raw_root_trees.RawShowerTree.lowe_cut_gamma"></a>

#### gamma\_energy\_cut

```python
@property
def lowe_cut_gamma()
```

gamma energy cut (GeV)

<a id="raw_root_trees.RawShowerTree.lowe_cut_e"></a>

#### electron\_energy\_cut

```python
@property
def lowe_cut_e()
```

electron energy cut (GeV)

<a id="raw_root_trees.RawShowerTree.lowe_cut_mu"></a>

#### muon\_energy\_cut

```python
@property
def lowe_cut_mu()
```

muon energy cut (GeV)

<a id="raw_root_trees.RawShowerTree.lowe_cut_meson"></a>

#### meson\_energy\_cut

```python
@property
def lowe_cut_meson()
```

meson energy cut (GeV)

<a id="raw_root_trees.RawShowerTree.lowe_cut_nucleon"></a>

#### nucleon\_energy\_cut

```python
@property
def lowe_cut_nucleon()
```

nucleon energy cut (GeV)

<a id="raw_root_trees.RawShowerTree.event_name"></a>

#### event\_name

```python
@property
def event_name()
```

Event name

<a id="raw_root_trees.RawShowerTree.event_date"></a>

#### event\_date

```python
@property
def event_date()
```

Event Date

<a id="raw_root_trees.RawShowerTree.rnd_seed"></a>

#### rnd\_seed

```python
@property
def rnd_seed()
```

Random seed

<a id="raw_root_trees.RawShowerTree.energy_in_neutrinos"></a>

#### energy\_in\_neutrinos

```python
@property
def energy_in_neutrinos()
```

Energy in neutrinos generated in the shower (GeV). Usefull for invisible energy

<a id="raw_root_trees.RawShowerTree.energy_primary"></a>

#### prim\_energy

```python
@property
def energy_primary()
```

Primary energy (GeV) TODO: Check unit conventions. # LWP: Multiple primaries? I guess, variable count. Thus variable size array or a std::vector

<a id="raw_root_trees.RawShowerTree.azimuth"></a>

#### shower\_azimuth

```python
@property
def azimuth()
```

Shower azimuth TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth. Also, geoid vs sphere problem

<a id="raw_root_trees.RawShowerTree.zenith"></a>

#### shower\_zenith

```python
@property
def zenith()
```

Shower zenith TODO: Discuss coordinates Cosmic ray convention is bad for neutrinos, but neurtino convention is problematic for round earth

<a id="raw_root_trees.RawShowerTree.primary_type"></a>

#### prim\_type

```python
@property
def primary_type()
```

Primary particle type TODO: standarize (PDG?)

<a id="raw_root_trees.RawShowerTree.primary_inj_point_shc"></a>

#### prim\_injpoint\_shc

```python
@property
def primary_inj_point_shc()
```

Primary injection point in Shower coordinates

<a id="raw_root_trees.RawShowerTree.primary_inj_alt_shc"></a>

#### prim\_inj\_alt\_shc

```python
@property
def primary_inj_alt_shc()
```

Primary injection altitude in Shower Coordinates

<a id="raw_root_trees.RawShowerTree.primary_inj_dir_shc"></a>

#### prim\_inj\_dir\_shc

```python
@property
def primary_inj_dir_shc()
```

primary injection direction in Shower Coordinates

<a id="raw_root_trees.RawShowerTree.atmos_model"></a>

#### atmos\_model

```python
@property
def atmos_model()
```

Atmospheric model name TODO:standarize

<a id="raw_root_trees.RawShowerTree.atmos_model_param"></a>

#### atmos\_model\_param

```python
@property
def atmos_model_param()
```

Atmospheric model parameters: TODO: Think about this. Different models and softwares can have different parameters

<a id="raw_root_trees.RawShowerTree.atmos_altitude"></a>

#### atmos\_altitude

```python
@property
def atmos_altitude()
```

height above sea level in meters, for the atmos_density and atmos_depth table

<a id="raw_root_trees.RawShowerTree.atmos_density"></a>

#### atmos\_density

```python
@property
def atmos_density()
```

Table of air density [g/cm3]

<a id="raw_root_trees.RawShowerTree.atmos_depth"></a>

#### atmos\_depth

```python
@property
def atmos_depth()
```

Table of vertical depth [g/cm2]

<a id="raw_root_trees.RawShowerTree.magnetic_field"></a>

#### magnetic\_field

```python
@property
def magnetic_field()
```

Magnetic field parameters: Inclination, Declination, modulus. TODO: Standarize. Check units. Think about coordinates. Shower coordinates make sense.

<a id="raw_root_trees.RawShowerTree.xmax_grams"></a>

#### xmax\_grams

```python
@property
def xmax_grams()
```

Shower Xmax depth (g/cm2 along the shower axis)

<a id="raw_root_trees.RawShowerTree.xmax_pos_shc"></a>

#### xmax\_pos\_shc

```python
@property
def xmax_pos_shc()
```

Shower Xmax position in shower coordinates

<a id="raw_root_trees.RawShowerTree.xmax_distance"></a>

#### xmax\_distance

```python
@property
def xmax_distance()
```

Distance of Xmax [m]

<a id="raw_root_trees.RawShowerTree.xmax_alt"></a>

#### xmax\_alt

```python
@property
def xmax_alt()
```

Altitude of Xmax (m, in the shower simulation earth. Its important for the index of refraction )

<a id="raw_root_trees.RawShowerTree.hadronic_model"></a>

#### hadronic\_model

```python
@property
def hadronic_model()
```

High energy hadronic model (and version) used TODO: standarize

<a id="raw_root_trees.RawShowerTree.low_energy_model"></a>

#### low\_energy\_model

```python
@property
def low_energy_model()
```

High energy model (and version) used TODO: standarize

<a id="raw_root_trees.RawShowerTree.cpu_time"></a>

#### cpu\_time

```python
@property
def cpu_time()
```

Time it took for the shower + efield simulation.

<a id="raw_root_trees.RawShowerTree.shower_core_pos"></a>

#### shower\_core\_pos

```python
@property
def shower_core_pos()
```

Shower core position

<a id="raw_root_trees.RawShowerTree.unix_date"></a>

#### unix\_date

```python
@property
def unix_date()
```

The date of the event in seconds since epoch

<a id="raw_root_trees.RawEfieldTree"></a>

## RawEfieldTree Objects

```python
@dataclass
class RawEfieldTree(MotherEventTree)
```

The class for storing Efield simulation-only data common for each event

<a id="raw_root_trees.RawEfieldTree.du_count"></a>

#### du\_count

```python
@property
def du_count()
```

Number of detector units in the event - basically the antennas count

<a id="raw_root_trees.RawEfieldTree.refractivity_model"></a>

#### refractivity\_model

```python
@property
def refractivity_model()
```

Name of the atmospheric index of refraction model

<a id="raw_root_trees.RawEfieldTree.refractivity_model_parameters"></a>

#### refractivity\_model\_parameters

```python
@property
def refractivity_model_parameters()
```

Refractivity model parameters

<a id="raw_root_trees.RawEfieldTree.atmos_refractivity"></a>

#### atmos\_refractivity

```python
@property
def atmos_refractivity()
```

refractivity for each altitude at atmos_altiude table

<a id="raw_root_trees.RawEfieldTree.t_pre"></a>

#### t\_pre

```python
@property
def t_pre()
```

Starting time of antenna data collection time window. The window starts at t0-t_pre, thus t_pre is usually positive.

<a id="raw_root_trees.RawEfieldTree.t_post"></a>

#### t\_post

```python
@property
def t_post()
```

Finishing time of antenna data collection time window. The window ends at t0+t_post.

<a id="raw_root_trees.RawEfieldTree.t_bin_size"></a>

#### t\_bin\_size

```python
@property
def t_bin_size()
```

Time bin size

<a id="raw_root_trees.RawEfieldTree.du_id"></a>

#### du\_id

```python
@property
def du_id()
```

Detector ID

<a id="raw_root_trees.RawEfieldTree.du_name"></a>

#### du\_name

```python
@property
def du_name()
```

Detector Name

<a id="raw_root_trees.RawEfieldTree.t_0"></a>

#### t\_0

```python
@property
def t_0()
```

Time window t0

<a id="raw_root_trees.RawEfieldTree.p2p"></a>

#### p2p

```python
@property
def p2p()
```

Peak 2 peak amplitudes (x,y,z,modulus)

<a id="raw_root_trees.RawEfieldTree.trace_x"></a>

#### trace\_x

```python
@property
def trace_x()
```

Efield trace in X direction

<a id="raw_root_trees.RawEfieldTree.trace_y"></a>

#### trace\_y

```python
@property
def trace_y()
```

Efield trace in Y direction

<a id="raw_root_trees.RawEfieldTree.trace_z"></a>

#### trace\_z

```python
@property
def trace_z()
```

Efield trace in Z direction

<a id="raw_root_trees.RawEfieldTree.du_x"></a>

#### pos\_x

```python
@property
def du_x()
```

X position in site's referential

<a id="raw_root_trees.RawEfieldTree.du_y"></a>

#### pos\_y

```python
@property
def du_y()
```

Y position in site's referential

<a id="raw_root_trees.RawEfieldTree.du_z"></a>

#### pos\_z

```python
@property
def du_z()
```

Z position in site's referential

<a id="raw_root_trees.RawZHAireSTree"></a>

## RawZHAireSTree Objects

```python
@dataclass
class RawZHAireSTree(MotherEventTree)
```

The class for storing shower data for each event specific to ZHAireS only

<a id="raw_root_trees.RawZHAireSTree.relative_thining"></a>

#### relative\_thining

```python
@property
def relative_thining()
```

Relative thinning energy

<a id="raw_root_trees.RawZHAireSTree.maximum_weight"></a>

#### weight\_factor

```python
@property
def maximum_weight()
```

Weight factor

<a id="raw_root_trees.RawZHAireSTree.lowe_cut_gamma"></a>

#### gamma\_energy\_cut

```python
@property
def lowe_cut_gamma()
```

Low energy cut for gammas(GeV)

<a id="raw_root_trees.RawZHAireSTree.lowe_cut_e"></a>

#### electron\_energy\_cut

```python
@property
def lowe_cut_e()
```

Low energy cut for electrons (GeV)

<a id="raw_root_trees.RawZHAireSTree.lowe_cut_mu"></a>

#### muon\_energy\_cut

```python
@property
def lowe_cut_mu()
```

Low energy cut for muons (GeV)

<a id="raw_root_trees.RawZHAireSTree.lowe_cut_meson"></a>

#### meson\_energy\_cut

```python
@property
def lowe_cut_meson()
```

Low energy cut for mesons (GeV)

<a id="raw_root_trees.RawZHAireSTree.lowe_cut_nucleon"></a>

#### nucleon\_energy\_cut

```python
@property
def lowe_cut_nucleon()
```

Low energy cut for nucleons (GeV)

<a id="raw_root_trees.RawZHAireSTree.other_parameters"></a>

#### other\_parameters

```python
@property
def other_parameters()
```

Other parameters

