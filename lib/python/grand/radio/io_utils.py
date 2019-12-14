################################
# by A. Zilles
################################

#!/usr/bin/env python

from astropy import units as u
import os
from os.path import split, join, realpath
import numpy as np
import sys
import glob

import logging
logger = logging.getLogger(__name__)


# ============================================================================

def load_trace(directory, index, suffix=".trace"):
    """Load data from a trace file

   Parameters
   ---------
        directory: str
            path to file
        index: ind
            index number of antenna
        suffix: str
            optional, suffix of file

   Returns
   ---------
        numpy array

    XXX / TODO: read-in hdf5 files and return numpy array
    """

    path = "{:}/a{:}{:}".format(directory, index, suffix)
    with open(path, "r") as f:
        return np.array([list(map(float, line.split())) for line in f])


# ============================================================================


def _table_efield(efield, pos=None, slopes=None, info={}, save=None, ant="/"):
    '''
    Load electric field trace in table with header info
    (numpy array to astropy table)

    Parameters
    ---------
    efield: numpy array
        electric field trace
    pos: numpy array
        position of antenna
    info: dict
        contains shower info

    Returns
    ---------
    efield_ant: astropy table

    '''
    from astropy.table import Table, Column

    info.update({'position': pos*u.m, 'slopes': slopes*u.deg})

    a = Column(data=efield.T[0], unit=u.ns, name='Time')
    b = Column(data=efield.T[1], unit=u.u*u.V/u.meter, name='Ex')
    c = Column(data=efield.T[2], unit=u.u*u.V/u.meter, name='Ey')
    d = Column(data=efield.T[3], unit=u.u*u.V/u.meter, name='Ez')
    efield_ant = Table(data=(a, b, c, d,), meta=info)

    if save:
        efield_ant.write(save, path=ant+'efield', format="hdf5",
                         append=True, compression=True, serialize_meta=True)

    return efield_ant

# ============================================================================


def table_voltage(voltage, pos=None, slopes=None, info={}, save=None, ant="/"):
    '''
    Load voltage trace in table with header info
    (numpy array to astropy table)

    Parameters
    ---------
    voltage: numpy array
        voltage trace
    pos: numpy array
        position of antenna
    info: dict
        contains shower info

    Returns
    ---------
    voltage_ant: astropy table

    '''
    from astropy.table import Table, Column

    info.update({'position': pos*u.m, 'slopes': slopes*u.deg})

    a = Column(data=voltage.T[0], unit=u.ns, name='Time')
    b = Column(data=voltage.T[1], unit=u.u*u.V, name='Vx')
    c = Column(data=voltage.T[2], unit=u.u*u.V, name='Vy')
    d = Column(data=voltage.T[3], unit=u.u*u.V, name='Vz')
    voltage_ant = Table(data=(a, b, c, d,), meta=info)

    processing_info = {'voltage': (
        'antennaresponse', 'noise', 'filter', 'digitise')}
    if save is not None:
        if 'antennaresponse' in info['voltage']:
            path_tmp = ant+'voltages'
        if 'noise' in info['voltage']:
            path_tmp = ant+'voltages_noise'
        if 'filter' in info['voltage']:
            path_tmp = ant+'filtered'
        if 'digitise' in info['voltage']:
            path_tmp = ant+'voltages_digitise'

        voltage_ant.write(save, path=path_tmp, format="hdf5",
                          append=True, compression=True, serialize_meta=True)

    return voltage_ant

# ============================================================================



def load_trace_to_table(path_raw,  pos=None, slopes=None, info={},
                        content="e", simus="zhaires", save=None, ant="/"):
    """Load data from an electric field trace file to astropy table

   Parameters
   ---------
        path_raw: str
            path to file -- electric field (.trace) or voltage trace (.dat)
        pos: numpy array, floats
            optional, position of antenna
        info: str
            optional. shower infos
        content: str
            e/efield or v/voltages
        sim: str
            coreas/zhaires, pick the simulation
        save: str
            optional,path to save a hdf5 file


   Returns
   ---------
        astropy table
    """

    if content == "efield" or content == "e":
        efield = np.loadtxt(path_raw)
        #zhaires: time in ns and efield in muV/m
        if simus == "coreas":
            efield.T[0] *= 1e9  # s to ns
            # coreas cgs to SI, V/m to muV/m
            efield.T[1] *= 2.99792458e4 * 1.e6
            efield.T[2] *= 2.99792458e4 * 1.e6
            efield.T[3] *= 2.99792458e4 * 1.e6

        efield_ant = _table_efield(
            efield, pos=pos, slopes=slopes, info=info, save=save, ant=ant)
    if content == "voltages" or content == "v":
        voltage = np.loadtxt(path_raw)
        efield_ant = table_voltage(
            voltage, pos, slopes=slopes, info=info, save=save, ant=ant)

    return efield_ant


# ============================================================================


def _load_ID_fromhdf(path_hdf5):
    """ Load ID from hdf5 file for identification of shower event

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        ID: str, float
            shower ID, number of simulation
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")

    # Get shower infomation
    try:
        ID = g.meta['ID'],               # shower ID, number of simulation
    except:
        ID = None

    return ID[0]

# ============================================================================


def _load_primary_fromhdf(path_hdf5):
    """ Load primary from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        primary: str, float
            type of primary
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")

    # Get shower infomation
    try:
        primary = g.meta['primary'],        # primary (electron, pion)
    except:
        primary = None

    return primary[0]

# ============================================================================


def _load_xmax_fromhdf(path_hdf5):
    """ Load Xmax from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        xmax: str, float
            grammage in g/cm2
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")

    # Get shower infomation
    try:
        xmax = g.meta['Xmax'],        # primary (electron, pion)
    except:
        xmax = None

    return xmax[0]

    # ============================================================================


def _load_energy_fromhdf(path_hdf5):
    """ Load primary's energy from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        energy: float
            primary's energy in EeV
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")

    try:
        energy = g.meta['energy'],               # EeV
    except:
        energy = None

    return energy[0]

# ============================================================================


def _load_energy_fromhdf(path_hdf5):
    """ Load primary's energy from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        energy: float
            primary's energy in EeV
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")

    try:
        energy = g.meta['energy'],               # EeV
    except:
        energy = None

    return energy[0]

# ============================================================================


def _load_zenith_fromhdf(path_hdf5):
    """ Load primary's zenith from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        zenith: float
            primary's zenith in deg (GRAND)
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")
    try:
        zenith = g.meta['zenith'],               # deg (GRAND frame)
    except:
        zenith = None

    return zenith[0]

# ============================================================================


def _load_azimuth_fromhdf(path_hdf5):
    """ Load primary's azimuth from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        azimuth: float
            primary's azimuth in deg (GRAND)
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")
    try:
        azimuth = g.meta['azimuth'],                # deg (GRAND frame)
    except:
        azimuth = None

    return azimuth[0]

# ============================================================================


def _load_injectionheight_fromhdf(path_hdf5):
    """ Load primary's injectionheight from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        injectionheight: float
            primary's injectionheight in m wrt sealevel
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")
    try:
        # m (injection height in the local coordinate system)
        injection_height = g.meta['injection_height'],
    except:
        injection_height = None

    return injection_height[0]

# ============================================================================


def _load_task_fromhdf(path_hdf5):
    """ Load task from hdf5 file, for identifaction

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        task: str
            e.g. DANTON event number
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")
    try:
        task = g.meta['task'],    # Identification
    except:
        task = None

    return task[0]

# ============================================================================


def _load_core_fromhdf(path_hdf5):
    """ Load shower core from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        core: numpy array
            shower core (x,y,z) in m
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")
    try:
        core = g.meta['core'],    # m, numpy array, core position
    except:
        core = None

    return core[0]

# ============================================================================


def _load_simulation_fromhdf(path_hdf5):
    """ Load shower core from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        simulation: str
            simulation program/version identifier like coreas or zhaires
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path="/event")
    try:
        simulation = g.meta['simulation']  # coreas or zhaires
    except:
        simulation = None

    return simulation

# ============================================================================


def _load_showerinfo_fromhdf(path_hdf5):
    """Load data from hdf5 file and restore shower info, as a list

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        shower: list
            shower parameters etc
    """

    ID = _load_ID_fromhdf(path_hdf5)
    primary = _load_primary_fromhdf(path_hdf5)
    xmax = _load_xmax_fromhdf(path_hdf5)
    energy = _load_energy_fromhdf(path_hdf5)
    zenith = _load_zenith_fromhdf(path_hdf5)
    azimuth = _load_azimuth_fromhdf(path_hdf5)
    injection_height = _load_injectionheight_fromhdf(path_hdf5).value/100 * u.m # Was initialy in cm?  # TODO: check
    task = _load_task_fromhdf(path_hdf5)
    core = _load_core_fromhdf(path_hdf5)
    simulation = _load_simulation_fromhdf(path_hdf5)

    shower = {
        "ID": ID,               # shower ID, number of simulation
        "primary": primary,        # primary (electron, pion)
        "xmax": xmax,               # g/cm2
        "energy": energy,               # EeV
        "zenith": zenith,               # deg (GRAND frame)
        "azimuth": azimuth,                # deg (GRAND frame)
        # m (injection height in the local coordinate system)
        "injection_height": injection_height,
        "task": task,    # Identification
        "core": core,    # m, numpy array, core position
        "simulation": simulation  # coreas or zhaires
    }

    return shower

# ============================================================================


def _load_antID_fromhdf(path_hdf5, path1="/event"):
    """ Load antIDs from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        antID: list
            list of antenna ID contained in event
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path=path1)
    try:
        ant_ID = g["ant_ID"]
    except:
        ant_ID = None

    return ant_ID

# ============================================================================


def _load_positions_fromhdf(path_hdf5, path1="/event"):
    """ Load antenna positions from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        positions: numpy array
            list of antenna positions (x,y,z) contained in event

    NOTE: The hdf5 file
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path=path1)
    try:
        positions = np.array(
            [g["pos_x"], g["pos_y"], g["pos_z"]]).T*g["pos_x"].unit
    except:
        positions = None

    return positions


# ============================================================================

def _load_slopes_fromhdf(path_hdf5, path1="/event"):
    """ Load antenna slopes from hdf5 file

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        slopes: numpy array
            list of antenna slopes (alpha,beta) contained in event
    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path=path1)
    try:
        slopes = np.array([g["alpha"], g["beta"]]).T*g["alpha"].unit
    except:
        slopes = None

    return slopes

# ============================================================================


def _load_eventinfo_fromhdf(path_hdf5, path1="/event"):
    """Load data from hdf5 file to numpy array and restore shower info

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file

   Returns
   ---------
        shower: list
            shower parameters etc
        position: numpy array in m (cross-check)
            positions of all antennas in array (x,y,z)
        slopes: numpy array in deg
            slopes of all antennas in array (alpha, beta)

    """

    from astropy.table import Table

    g = Table.read(path_hdf5, path=path1)

    # Get shower infomation
    shower = _load_showerinfo_fromhdf(path_hdf5)

    # Get array infomation
    ant_ID = _load_antID_fromhdf(path_hdf5)
    positions = _load_positions_fromhdf(path_hdf5)
    slopes = _load_slopes_fromhdf(path_hdf5)

    return shower, ant_ID, positions, slopes

# ============================================================================


def _load_path(path_hdf5, path="/analysis"):
    """Load any data from hdf5 file and restores infomation on analysis

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file
        path: str (default set)
            can be any keyword


   Returns
   ---------
        astropy table
            returns content of selected path
        info
            returns table infomation
        meta: dict

    """

    from astropy.table import Table

    f = None
    info = None
    meta = None
    try:
        f = Table.read(path_hdf5, path=path)
        try:
            info = f.info
        except:
            logger.info(f"Info on {path} is not availble")
        try:
            meta = f.meta
        except:
            logger.info(f"Meta on {path} is not available")
    except:
        logger.warning(f"{path} could not be loaded from {path_hdf5}")

    return f, info, meta


# ============================================================================

def _load_efield_fromhdf(path_hdf5, ant):
    """Load electric field data from hdf5 file for a single antenna as numpy array

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file
        ant: int
            ant ID

    Returns
    ---------
        numpy array/None
            electric field trace: time in ns, Ex, Ey, Ez in muV/s
    """

    from astropy.table import Table

    try:
        efield = Table.read(path_hdf5, path=ant+"/efield")
        return np.array([efield['Time'],
                         efield['Ex'],
                         efield['Ey'],
                         efield['Ez']]).T
    except:
        return None

# ============================================================================


def _load_voltage_fromhdf(path_hdf5, ant="/"):
    """Load voltage data from hdf5 file for a single antenna as numpy array

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file
        ant: int
            ant ID

    Returns
    ---------
        numpy array/None
            voltagetrace: time in ns, Vx, Vy, Vz in muV

    TODO: How to we handle renaming of path or several voltages pathes ...
    """

    from astropy.table import Table

    try:
        voltage = Table.read(path_hdf5, path=ant+"/voltages")
        return np.array([voltage['Time'],
                         voltage['Vx'],
                         voltage['Vy'],
                         voltage['Vz']]).T
    except:
        return None

# ===========================================================================================================


def _load_to_array(path_hdf5, content="efield", ant="/"):
    """Load trace data from hdf5 file for a single antenna
    as numpy array and restore infomation on antenna

   Parameters
   ---------
        path_hdf5: str
            path to hdf5 file
        content: str
            grep efield or voltages traces

   Returns
   ---------
        efield1 or voltage1: numpy array
            containing the electric field or voltage trace trace:
            time, Ex, Ey, Ez or time, Vx, Vy, Vz
        efield['Time'] or voltage['Time'].unit: str
            unit of time column
        efield['Ex'] or voltage['Vx'].unit: str
            unit of efield or voltage field
        position: numpy array in m (cross-check)
            position of the antenna
        slopes: numpy array in deg
            slopes of the antenna
    """

    from astropy.table import Table

    if content == "efield" or content == "e":
        efield1 = _load_efield_fromhdf(path_hdf5, ant=ant)

        try:
            # position=efield.meta['position']
            position = _load_slopes_fromhdf(path_hdf5, path1=ant+"/efield")
        except IOError:
            position = None
        try:
            # slopes=efield.meta['slopes']
            slopes = _load_slopes_fromhdf(path_hdf5, path1=ant+"/efield")
        except IOError:
            slopes = None

        # TODO do we one to return atsropy units...
        return efield1, efield['Time'].unit, efield['Ex'].unit, position, slopes

    if content == "voltages" or content == "v":

        try:
            # position=voltage.meta['position']
            position = _load_slopes_fromhdf(path_hdf5, path1=ant+"/voltages")
        except IOError:
            position = None
        try:
            # slopes=voltage.meta['slopes']
            slopes = _load_slopes_fromhdf(path_hdf5, path1=ant+"/voltages")
        except IOError:
            slopes = None

        # TODO do we one to return atsropy units...
        return voltage1, voltage['Time'].unit, voltage['Vx'].unit,position, slopes


# ============================================================================


def load_eventinfo_tohdf(path, showerID, simus, name_all=None):
    """Load data from simulation to hdf5 file,
       directly savin as hdf5 file option

   Parameters
   ---------
        path: str
            path simulated event set
        showerID: str
            identifaction string for single event/shower
        simus: str
            coreas/zhaires
        name_all: str (optional)
            path to store the hdf5 file

   Returns
   ---------
        shower: dict
            contains shower parameters and other infos
        ID_ant: numpy array
            antenna ID of whole array
        ID_ant: numpy array
            antenna positions of whole array [x,y,z]
        ID_ant: numpy array
            slopes of whole array [alphha,beta]

        saves hdf5 file with event table and meta data if name_all !=None

   """

    if simus == 'zhaires':

        # taken from Matias scripts -- slighty modified functions,  to be tested
        from . import AiresInfoFunctions as AiresInfo
        sryfile = glob.glob(path+"/*.sry")
        tmp = AiresInfo.ReadAiresSry(str(sryfile[0]), "GRAND")
        zen, azim, energy, primary, xmax, distance, taskname = tmp
        posfile = path + '/antpos.dat'
        positions, ID_ant, slopes = AiresInfo._get_positions_zhaires(posfile)

        from .AiresInfoFunctions import GetSlantXmaxFromSry
        try:
            Xmax = GetSlantXmaxFromSry(sryfile, outmode="GRAND")
            core = np.array([0, 0, 0])  # infomation not yet accessible
        except:
            Xmax = None
        # correction of shower core
        try:
            positions = positions + np.array([core[0], core[1], 0.*u.m])
        except:
            logger.debug("No core position information availble")

    if simus == 'coreas':
        from . import CoreasInfoFunctions as CoreasInfo
        # *list contains not alpha and beta
        # posfile = path +'SIM'+str(showerID)+'.list'
        # produced file *info contains
        # original ant ID , positions , alpha and beta
        posfile = path + 'SIM'+str(showerID)+'.info'
        positions, ID_ant, slopes = CoreasInfo._get_positions_coreas(posfile)

        inputfile = path+'/inp/SIM'+showerID+'.inp'
        tmp = CoreasInfo.inputfromtxt_coreas(inputfile)
        zen, azim, energy, injh, primarytype, core, task = tmp

        try:
            Xmax = CoreasInfo.get_Xmax_coreas(path)
        except ValueError:
            Xmax = None

        # correction of shower core
        try:
            positions = positions + np.array([core[0], core[1], 0.*u.m])
        except:
            logger.debug("No core position information availble")
        # ---------------------------------------------------------------------

    # load shower info from inp file via dictionary
    shower = {
        "ID": showerID,               # shower ID, number of simulation
        "primary": primarytype,        # primary (electron, pion)
        "energy": energy,               # eV
        "zenith": zen,               # deg (GRAND frame)
        "azimuth": azim,                # deg (GRAND frame)
        # m (injection height in the local coordinate system)
        "injection_height": injh,
        "task": task,    # Identification
        "core": core,    # m, numpy array, core position
        "Xmax": Xmax,    # depth of shower maximum in g/cm2
        "simulation": simus  # coreas or zhaires
    }
    ####################################
    logger.info(f"Shower summary: {shower}")

    if name_all is not None:
        from astropy.table import Table, Column
        a1 = Column(data=np.array(ID_ant), name='ant_ID')
        b1 = Column(data=positions.T[0], unit=u.m, name='pos_x')
        c1 = Column(data=positions.T[1], unit=u.m, name='pos_y')
        d1 = Column(data=positions.T[2], unit=u.m, name='pos_z')
        e1 = Column(data=slopes.T[0], unit=u.deg, name='alpha')
        f1 = Column(data=slopes.T[1], unit=u.deg, name='beta')
        event_info = Table(data=(a1, b1, c1, d1, e1, f1,), meta=shower)
        event_info.write(name_all, path='event', format="hdf5",
                         append=True,  compression=True, serialize_meta=True)
        logger.info("Event info has been saved to " + name_all)

    return shower, ID_ant, positions, slopes
