#! /usr/bin/env python

import os.path as osp

import numpy as np
from matplotlib import pyplot as plt

from grand.simu.du.process_ant import AntennaProcessing
from grand.simu.shower.gen_shower import ShowerEvent
from grand.io.file_leff import TabulatedAntennaModel
import grand.manage_log as mlg
from grand import ECEF, Geodetic, LTP, GRANDCS
from grand import SphericalRepresentation
from grand import grand_add_path_data_model, grand_get_path_root_pkg

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_stdout=True)

logger.info(mlg.string_begin_script())

# Load the radio shower simulation data
showerdir = osp.join(grand_get_path_root_pkg(), "tests/simulation/data/zhaires")
shower = ShowerEvent.load(showerdir)

logger.info(shower.frame)
logger.info(shower.maximum)

if shower.frame is None:
    shower.localize(39.5, 90.5)  # Coreas showers have no
    # localization info. This must
    # be set manually

logger.info(shower.frame)
logger.info(shower.maximum)
logger.info("---------------------------------")
logger.info(f"Zenith (Zhaires?!) {shower.zenith}")
logger.info(f"Azimuth (Zhaires?!) {shower.azimuth}")
logger.info(f"Xmax={shower.maximum.flatten()}")
logger.info(f"Core={shower.core.flatten()}")
logger.info(f"obstime={shower.frame.obstime}")
logger.info(f"{vars(shower.frame)} Shower frame")
logger.info("---------------------------------")

# Define an antenna model
#
# A tabulated model of the Butterfly antenna is used. Note that a single EW
# arm is assumed here for the sake of simplicity
# path_ant = grand_add_path_data("detector/HorizonAntenna_EWarm_leff_loaded.npy")
path_ant = grand_add_path_data_model("detector/GP300Antenna_EWarm_leff.npy")
antenna_model = TabulatedAntennaModel.load(path_ant)
# logger.debug(antenna_model)
for i_ite in range(2000):
    logger.info(f"ite {i_ite}")
    counter = 0
    # Loop over electric fields and compute the corresponding voltages
    for antenna_index, field in shower.fields.items():
        counter += 1
        logger.debug(f"antenna_index={antenna_index}")
        # logger.debug(f'field={field}')
        if counter == 2:
            # TODO: debug when counter == 3
            #           File "/home/jcolley/projet/grand_wk/binder/grand/grand/libs/gull.py", line 67, in __init__
            #     raise LibraryError(r)
            # grand.libs.gull.LibraryError: A GULL library error occurred: b'{ gull_snapshot_create [#2], src/gull.c:357 } invalid syntax [/home/jcolley/projet/grand_wk/binder/grand/grand/libs/data/gull/IGRF13.COF:1]'

            break

        # Compute the antenna local frame
        #
        # The antenna is placed within the shower frame. It is oriented along the
        # local magnetic North by using an ENU/LTP frame (x: East, y: North, z: Upward)
        antpos_wrt_shower = field.electric.pos_xyz
        # RK: if antenna location was saved in LTP frame in zhaires.py, next step would not required.
        antenna_location = LTP(
            x=antpos_wrt_shower.x,
            y=antpos_wrt_shower.y,
            z=antpos_wrt_shower.z,
            frame=shower.frame,
        )
        logger.info(shower.frame)
        antenna_frame = LTP(
            location=antenna_location,
            orientation="NWU",
            magnetic=True,
            obstime=shower.frame.obstime,
        )
        antenna = AntennaProcessing(model_leff=antenna_model, frame=antenna_frame)

        logger.info(f"{antenna_index} Antenna pos in shower frame {antpos_wrt_shower.flatten()}")
        logger.info(
            f"{vars(antenna_location)} {antenna_location.flatten()} antenna pos LTP in shower frame"
        )
        logger.info("---------------------------------")
        logger.info(f"{vars(antenna_frame)} antenna frame")
        logger.info("---------------------------------")

        # Compute the voltage on the antenna
        #
        # The electric field is assumed to be a plane-wave originating from the
        # shower axis at the depth of maximum development. Note that the direction
        # of observation and the electric field components are provided in the
        # shower frame. This is indicated by the `frame` named argument.
        Exyz = field.electric.e_xyz
        logger.info(mlg.chrono_start())
        # Xmax, Efield, and input frame are all in shower frame.
        logger.debug("compute_voltage")
        logger.info(shower.frame)
        field.voltage = antenna.compute_voltage(shower.maximum, field.electric, frame=shower.frame)
        logger.info(mlg.chrono_string_duration())
        logger.info(f"\nVpp= {max(field.voltage.V) - min(field.voltage.V)}")
plt.figure()
plt.subplot(211)
plt.title("example/simulation/shower-event.py")
plt.plot(field.electric.a_time, Exyz.x, label="Ex")
plt.plot(field.electric.a_time, Exyz.y, label="Ey")
plt.plot(field.electric.a_time, Exyz.z, label="Ez")
plt.xlabel("Time (ns)")
plt.ylabel(r"Efield ($\mu$V/m)")
plt.grid()
plt.legend(loc="best")
plt.subplot(212)
plt.plot(field.voltage.t, field.voltage.V, label="V$_{EW}$")
plt.xlabel("Time (ns)")
plt.ylabel(r"Voltage ($\mu$V)")
plt.legend(loc="best")
plt.grid()

logger.info(mlg.string_end_script())
plt.show()
