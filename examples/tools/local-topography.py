#! /usr/bin/env python
from grand import ECEF, Geodetic, LTP, topography
import matplotlib.pyplot as pl
import numpy as np


# Set the local frame origin
origin = Geodetic(
    latitude=42.923516,
    longitude=86.716069,
    height=0,
)

# Get the corresponding topography data. Note that does are dowloaded from the
# web and cached which might take some time. Reducing the area results in less
# data to be downloaded, i.e. speeding up this step
radius = 200000 # m
topography.update_data(origin, radius=radius)


# Generate a grid of local coordinates using numpy.meshgrid
x = np.linspace(-1*radius, radius, 1001)
y = np.linspace(-1*radius, radius, 1001)
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
zg = topography.elevation(coordinates, reference='local')
zg = zg.reshape(X.shape)

# Plot the result using contour levels. The Earth curvature is clearly visible
# at large distances from the origin.
pl.figure()
pl.contourf(x/1000, y/1000, zg/1000, 40, cmap='terrain')
pl.colorbar(label="Local Altitude (km)")
pl.xlabel("Easting (km)")
pl.ylabel("Northing (km)")
pl.title("Elevation wrt LTP")
pl.show()
