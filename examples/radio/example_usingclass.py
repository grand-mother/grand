#!/usr/bin/env python
"""
This is an example of using grand classes for analysing a simulated event.  The
analysis loops over events in a folded. For each event, the shower parameters
and the triggered antennas are extracted. Finally, a summary plot is generated.

authors:
  A. Zilles, V. Niess
"""

import argparse
from pathlib import Path

from astropy.table import Table # XXX use an event loader instead
import astropy.units as u

import matplotlib.pyplot as plt
import numpy as np

from grand import logger
from grand.radio.detector import Detector
from grand.radio.shower import SimulatedShower, loadInfo_toShower


def analyse(eventfolder, arrayfile):
    """
    Simple trigger analysis illustrating the usage of the Detector and
    SimulatedShower classes
    """
    # Create an empty detector
    det = Detector() # XXX Load from file or config by default

    # Fill the detector array from a file
    det.create_from_file(arrayfile) # XXX read directly from config if not
                                    # specified
    # XXX Log the detector statistics

    # Loop over all HDF5 event files in a folder and get the showers
    logger.info("Looping over events in ...")
    event = []

    eventfolder = Path(eventfolder)
    for path in sorted(eventfolder.glob("*.hdf5")):
        logger.info(f"Reading Event from {path}")

        # Create the shower object and set its attributes
        shower = SimulatedShower() # XXX add a class metod for loading from
        loadInfo_toShower(shower, path)

        logger.info("Event info:")
        logger.info(f"- showerID          = {shower.showerID}")
        logger.info(f"- primary           = {shower.primary}")
        logger.info(f"- energy/eV         = {shower.energy}")
        logger.info(f"- zenith/deg        = {shower.zenith}")
        logger.info(f"- azimuth/deg       = {shower.azimuth}")
        logger.info(f"- injectionheight/m = {shower.injectionheight}")

        event.append(shower)

        # Example of trigger analysis
        analysis = Table.read(path, path="/analysis") # XXX: Get this from I/O
        if (np.sum(analysis["trigger_aggr_xy"]) > 5 or
            np.sum(analysis["trigger_aggr_any"]) > 5):
            nxy = np.sum(analysis["trigger_aggr_xy"])
            nany = np.sum(analysis["trigger_aggr_any"])
            logger.info(
                f"=> shower triggers (aggr): any = {nany}, xy = {nxy}")

            # Add the trigger info to the class
            shower.trigger = 1
        else:
            shower.trigger = 0

    ### ANALYSIS ###############################################################
    logger.info("Analysing event ...")

    # Calculate the fraction of detected events
    trigger = np.array([e.trigger for e in event])
    ntrig, ntot = sum(trigger), len(trigger)
    fraction = int(100 * float(ntrig) / ntot)
    logger.info(f"{ntrig} out of {ntot} ({fraction}%) events detected")

    # Selection of triggered events
    T = trigger == 1

    # Vectorize the shower parameters for the plots
    energy  = np.array([e.energy  / u.eV  for e in event])
    zenith  = np.array([e.zenith  / u.deg for e in event])
    azimuth = np.array([e.azimuth / u.deg for e in event])
    primary = np.array([e.primary         for e in event])

    ### PLOT ###################################################################
    plt.rcParams.update({"figure.figsize": (12, 5)})
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    opts1 = {"color": "orange", "bins": 50, "alpha": 0.5}
    opts2 = {"bins": 50}

    ax1.hist(energy, label=f"all = {ntot}", **opts1)
    ax1.hist(energy[T], label=f"trigger (aggr.) = {ntot}", **opts2)
    ax1.legend()
    ax1.set_title("energy (eV)")

    ax2.hist(zenith, **opts1)
    ax2.hist(zenith[T], **opts2)
    ax2.set_title("zenith (deg)")

    ax3.hist(azimuth, **opts1)
    ax3.hist(azimuth[T], **opts2)
    ax3.set_title("azimuth (deg)")

    ax4.hist(primary, **opts1)
    ax4.hist(primary[T], **opts2)
    ax4.set_title("primary")

    fig.tight_layout()

    outfile = eventfolder / "trigger_stats.png"
    plt.savefig(outfile)
    logger.info(f"PNG saved to {outfile}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", default="data/coreas",
        help="set the simulated events folder to EVENTS")
    parser.add_argument("--array",  default="data/array.txt",
        help="set the array description file to ARRAY")
    parser.add_argument("--logger", default="INFO",
        help="set the logger verbosity level to LOGGER")
    args = parser.parse_args()

    # Configure the logger
    logger.setLevel(args.logger)

    # Run the analysis
    analyse(args.events, args.array)
