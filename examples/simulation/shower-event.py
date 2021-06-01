#! /usr/bin/env python
import astropy.units as u
from grand import ECEF, LTP
from grand.simulation import Antenna, ShowerEvent, TabulatedAntennaModel
#from grand.simulation import ShowerEvent, TabulatedAntennaModel
#from grand.simulation.antenna .generic import compute_voltage
import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import BaseRepresentation, CartesianRepresentation

# Load the radio shower simulation data
showerdir = '../../tests/simulation/data/zhaires'
shower = ShowerEvent.load(showerdir)
if shower.frame is None:
    shower.localize(39.5 * u.deg, 90.5 * u.deg) # Coreas showers have no
                                                # localization info. This must
                                                # be set manually

print("Shower frame=",shower.frame)
print("Zenith (Zhaires?!) =",shower.zenith)
print("Azimuth (Zhaires?!) =",shower.azimuth)
print("Xmax=",shower.maximum)
#shower.core=CartesianRepresentation(0, 0, 2900, unit='m')
print("Core=",shower.core)


# Define an antenna model
#
# A tabulated model of the Butterfly antenna is used. Note that a single EW
# arm is assumed here for the sake of simplicity

antenna_model = TabulatedAntennaModel.load('./HorizonAntenna_EWarm_leff_loaded.npy')

# Loop over electric fields and compute the corresponding voltages
for antenna_index, field in shower.fields.items():
    # Compute the antenna local frame
    #
    # The antenna is placed within the shower frame. It is oriented along the
    # local magnetic North by using an ENU/LTP frame (x: East, y: North, z: Upward)
    antenna_location = shower.frame.realize_frame(field.electric.r)
    print(antenna_index,"Antenna pos=",antenna_location)

    antenna_frame = LTP(location=antenna_location, orientation='NWU',magnetic=True, obstime=shower.frame.obstime)
    antenna = Antenna(model=antenna_model, frame=antenna_frame)


    # Compute the voltage on the antenna
    #
    # The electric field is assumed to be a plane-wave originating from the
    # shower axis at the depth of maximum development. Note that the direction
    # of observation and the electric field components are provided in the
    # shower frame. This is indicated by the `frame` named argument.
    direction = shower.maximum - field.electric.r
    print("Direction to Xmax = ",direction)
    #print(antenna_frame.realize_frame(direction))
    Exyz = field.electric.E.represent_as(CartesianRepresentation)

    field.voltage = antenna.compute_voltage(direction, field.electric,frame=shower.frame)

    plt.figure()
    plt.subplot(211)
    plt.plot(field.electric.t,Exyz.x,label='Ex')
    plt.plot(field.electric.t,Exyz.y,label='Ey')
    plt.plot(field.electric.t,Exyz.z,label='Ez')
    plt.xlabel('Time ('+str(field.electric.t.unit)+')')
    plt.ylabel('Efield ('+str(Exyz.x.unit)+')')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(field.voltage.t,field.voltage.V,label='V$_{EW}$')
    plt.xlabel('Time ('+str(field.voltage.t.unit)+')')
    plt.ylabel('Voltage ('+str(field.voltage.V.unit)+')')
    plt.legend(loc='best')
    plt.show()
