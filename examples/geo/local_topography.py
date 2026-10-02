#! /usr/bin/env python
"""Plots the ground elevation around the GP13 centre station, in a local frame.

It needs the SRTM topography tiles around the site: about two 26 MB files,
downloaded into data/topography/ and cached there.  It did that without
asking (#218); it now downloads only with --download, and otherwise uses the
tiles already there (missing ones read as NaN, with a warning).

    python local_topography.py --download
"""
import argparse

from grand import ECEF, Geodetic, LTP, topography
import matplotlib.pyplot as pl
import numpy as np

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--download", action="store_true",
                    help="download the topography tiles of the area (about 50 MB) if not cached")
args = parser.parse_args()


# Set the local frame origin. Center station of GP13.
origin = Geodetic(
    latitude=40.98,
    longitude=93.93,
    height=0,
)

# Get the corresponding topography data. Note that does are dowloaded from the
# web and cached which might take some time. Reducing the area results in less
# data to be downloaded, i.e. speeding up this step
radius = 2000  # m
if args.download:
    print("Downloading the topography tiles around the site into data/topography/ (about 50 MB)")
    topography.update_data(origin, radius=radius)


# Generate a grid of local coordinates using numpy.meshgrid
x = np.linspace(-1 * radius, radius, 1001)
y = np.linspace(-1 * radius, radius, 1001)
X, Y = np.meshgrid(x, y)
coordinates = LTP(
    x=X.flatten(),
    y=Y.flatten(),
    z=np.zeros(X.size),
    location=origin,
    orientation="ENU",
    magnetic=False,
)

# Get the local ground elevation. Note that local coordinates naturally account
# for the Earth curvature.
zg = topography.elevation(coordinates, reference="local")
zg = zg.reshape(X.shape)

# Plot the result using contour levels. The Earth curvature is clearly visible
# at large distances from the origin.
pl.figure()
pl.contourf(x / 1000, y / 1000, zg / 1000, 40, cmap="terrain")
pl.colorbar(label="Local Altitude (km)")
pl.xlabel("Easting (km)")
pl.ylabel("Northing (km)")
pl.title("Elevation wrt LTP")
pl.show()
