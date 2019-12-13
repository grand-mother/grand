import astropy.units as u
from grand.radio import config as radio
from numpy import array as Array
from pathlib import Path


radio.site.name      = "Lenghu"
radio.site.longitude = 92.334037             * u.deg
radio.site.latitude  = 38.870398             * u.deg
radio.site.obsheight = 2734.0                * u.m
radio.site.origin    = Array([0, 0, 0]) * u.m

# Antenna file
radio.array = "../../CoREAS/regular_array_slopes.txt"

# Magnetic field (optionnal, to force using a given magnetic field)
radio.magnet.magnitude   = 54.021 * u.uT
radio.magnet.inclination = 57.43  * u.deg
radio.magnet.declination = 0.72   * u.deg

# Definition sigma in muV (50-200MHz)
radio.processing.vrms1     = 15 * u.uV # After filtering
radio.processing.vrms2     = 28 * u.uV # Before filtering
radio.processing.tsampling = 2  * u.ns # For digitisation

# Antenna responses
prefix = Path("../lib/python/radio_simus/GRAND_antenna")
radio.antenna.leff.x = prefix / "HorizonAntenna_SNarm_leff_loaded.npy"
radio.antenna.leff.y = prefix / "HorizonAntenna_EWarm_leff_loaded.npy"
radio.antenna.leff.z = prefix / "HorizonAntenna_Zarm_leff_loaded.npy"
