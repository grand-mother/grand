#! /usr/bin/env python
import astropy.units as u
from grand import ECEF, LTP
from grand.simulation import Antenna, ShowerEvent, TabulatedAntennaModel

import os
grand_astropy = True
try:
    if os.environ['GRAND_ASTROPY']=="0":
        grand_astropy=False
except:
    pass
    
# Note: This examples requires some test data to be downloaded localy. You can
# get these data from the grand store as:
#
# wget https://github.com/grand-mother/store/releases/download/101/HorizonAntenna_EWarm_leff_loaded.npy
#
# mkdir -p tests/simulation/data/zhaires
# cd tests/simulation/data/zhaires
# wget https://github.com/grand-mother/store/releases/download/101/zhaires-test.tar.gz
# tar -xzf zhaires-test.tar.gz
# cd -


# Load the radio shower simulation data
shower = ShowerEvent.load('tests/simulation/data/zhaires')
if shower.frame is None:
    shower.localize(39.5 * u.deg, 90.5 * u.deg) # Coreas showers have no
                                                # localization info. This must
                                                # be set manually

# Define an antenna model
#
# A tabulated model of the Butterfly antenna is used. Note that a single EW
# arm is assumed here for the sake of simplicity

antenna_model = TabulatedAntennaModel.load(
    'HorizonAntenna_EWarm_leff_loaded.npy')

# Loop over electric fields and compute the corresponding voltages
for antenna_index, field in shower.fields.items():
    # Compute the antenna local frame
    #
    # The antenna is placed within the shower frame. It is oriented along the
    # local magnetic North by using an ENU/LTP frame (x: East, y: North, z:
    # Upward)
    
    if grand_astropy:
        print("field", field.electric.r, shower.frame)

        # LWP: joins field.electric.r and shower.frame
        antenna_location = shower.frame.realize_frame(field.electric.r)
        print(antenna_location)
        antenna_frame = LTP(location=antenna_location, orientation='ENU',
                        magnetic=True, obstime=shower.frame.obstime)
        # LWP: joins antenna_model and antenna_frame
        antenna = Antenna(model=antenna_model, frame=antenna_frame)
        print("antenna_frame", antenna_frame)
        #print("antenna_model", antenna_model)
        #print("antenna", antenna)
    else:
        # LWP: Move shower frame by field.electric.r, then somehow add this to Antenna without astropy
        # This need deep moditication of the Antenna class, so for now do the astropy stuff
        print("field", field.electric.r, shower.frame)

        # LWP: joins field.electric.r and shower.frame
        antenna_location = shower.frame.realize_frame(field.electric.r)
        print(antenna_location)
        antenna_frame = LTP(location=antenna_location, orientation='ENU',
                        magnetic=True, obstime=shower.frame.obstime)
        # LWP: joins antenna_model and antenna_frame
        antenna = Antenna(model=antenna_model, frame=antenna_frame)
        print("antenna_frame", antenna_frame)
        #print("antenna_model", antenna_model)
        #print("antenna", antenna)      

    # Compute the voltage on the antenna
    #
    # The electric field is assumed to be a plane-wave originating from the
    # shower axis at the depth of maximum development. Note that the direction
    # of observation and the electric field components are provided in the
    # shower frame. This is indicated by the `frame` named argument.

    direction = shower.maximum - field.electric.r
    print("computing voltage")
    field.voltage = antenna.compute_voltage(direction, field.electric,
                                            frame=shower.frame)
    print("computed voltage", field.voltage)
    exit()
