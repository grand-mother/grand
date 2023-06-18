#! /usr/bin/env python

import os.path as osp
import numpy as np
from matplotlib import pyplot as plt

import grand.manage_log as mlg
from grand import LTP, CartesianRepresentation
from grand import grand_add_path_data
from grand.io.root_files import File
from grand.basis.type_trace import ElectricField
from grand.simu.du.antenna_model import AntennaModel
from grand.simu.du.process_ant import AntennaProcessing
from grand.simu.shower.gen_shower import ShowerEvent

# specific logger definition for script because __mane__ is "__main__" !
logger = mlg.get_logger_for_script(__file__)

# define a handler for logger : standart output and file log.txt
mlg.create_output_for_logger("debug", log_stdout=True)

logger.info(mlg.string_begin_script())

# Load the radio shower simulation data
shower = ShowerEvent()
e_path = grand_add_path_data("test_efield.root")
tefield= File(e_path)
shower.origin_geoid  = tefield.run.origin_geoid  # [lat, lon, height]
shower.load_root(tefield.shower)                 # calculates grand_ref_frame, shower_frame, Xmax in LTP etc

logger.info("---------------------------------")
logger.info(f"frame: {shower.frame}")
logger.info(f"obstime: {shower.frame.obstime}")
logger.info(f"Zenith: {shower.zenith}")
logger.info(f"Azimuth: {shower.azimuth}")
logger.info(f"Xmax (shc): {shower.maximum}")
logger.info(f"Core={shower.core.flatten()}")
logger.info(f"{vars(shower.frame)} Shower frame")
logger.info("---------------------------------")

# Define an antenna model
#
# A tabulated model of the GP300 antenna is used. Note that all three arms of an
# can be accessed as antenna_model.leff_sn (leff_ew, leff_z)
antenna_model  = AntennaModel()

# Loop over electric fields of each antenna and compute the corresponding voltages
#for antenna_index in range(tefield.du_count): # for all antennas.
for antenna_index in range(2):                 # only for first two antennas.
    logger.debug(f"antenna_index={antenna_index}")

    # Compute the antenna local frame
    #
    # The antenna is placed within the shower frame. It is oriented along the
    # local magnetic North by using an ENU/LTP frame (x: East, y: North, z: Upward)
    # RK: if antenna location is saved in LTP frame, next step is not required.
    antenna_location = LTP(
        x=tefield.du_xyz[antenna_index, 0],
        y=tefield.du_xyz[antenna_index, 1],
        z=tefield.du_xyz[antenna_index, 2],
        frame=shower.frame,
    )
    logger.info(shower.frame)
    antenna_frame = LTP(
        location=antenna_location,
        orientation="NWU",
        magnetic=True,
        obstime=shower.frame.obstime,
    )
    antenna = AntennaProcessing(model_leff=antenna_model.leff_sn, pos=antenna_frame)

    logger.info(f"{antenna_index} Antenna pos in shower frame {tefield.du_xyz[antenna_index]}")
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
    e_trace = CartesianRepresentation(
        x=tefield.traces[antenna_index, 0],
        y=tefield.traces[antenna_index, 1],
        z=tefield.traces[antenna_index, 2],
    )
    efield_idx = ElectricField(tefield.traces_time[antenna_index] * 1e-9, e_trace)

    logger.info(mlg.chrono_start())
    logger.debug("compute_voltage")
    logger.info(shower.frame)
    # Xmax, Efield, and input frame are all in shower frame.
    voltage = antenna.compute_voltage(shower.maximum, efield_idx, shower.frame)
    logger.info(mlg.chrono_string_duration())
    logger.info(f"\nVpp= {max(voltage.V) - min(voltage.V)}")

    plt.figure()
    plt.subplot(211)
    plt.title("example/simu/shower_event.py")
    plt.plot(efield_idx.a_time, efield_idx.e_xyz.x, label="Ex", color='b')
    plt.plot(efield_idx.a_time, efield_idx.e_xyz.y, label="Ey", color='y')
    plt.plot(efield_idx.a_time, efield_idx.e_xyz.z, label="Ez", color='k')
    plt.xlabel("Time (ns)")
    plt.ylabel(r"Efield ($\mu$V/m)")
    plt.grid()
    plt.legend(loc="best")
    plt.subplot(212)
    plt.plot(voltage.t, voltage.V, label="V$_{SN}$", color='b')
    plt.xlabel("Time (ns)")
    plt.ylabel(r"Voltage ($\mu$V)")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f'test_efield_voltage_ant{antenna_index}.png')

logger.info(mlg.string_end_script())
plt.show()
