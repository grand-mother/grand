#! /usr/bin/env python
import astropy.units as u
from grand import ECEF, LTP, topography
import matplotlib.pyplot as pl
import numpy as np


# Set the local frame origin
origin = ECEF(latitude=42.923516 * u.deg, longitude=86.716069 * u.deg,
              representation_type='geodetic')

# Get the corresponding topography data. Note that does are dowloaded from the
# web and cached which might take some time. Reducing the area results in less
# data to be downloaded, i.e. speeding up this step
radius = 200 * u.km
topography.update_data(origin, radius=radius)


# Generate a grid of local coordinates using numpy.meshgrid
x = np.linspace(-radius, radius, 1001)
y = np.linspace(-radius, radius, 1001)
X, Y = np.meshgrid(x, y)
coordinates = LTP(X.flatten(), Y.flatten(), np.zeros(X.size) << u.km,
                  location=origin, orientation='ENU', magnetic=False)

# Get the local ground elevation. Note that local coordinates naturally account
# for the Earth curvature.
zg = topography.elevation(coordinates)
zg = zg.reshape(X.shape)

# Plot the result using contour levels. The Earth curvature is clearly visible
# at large distances from the origin.
pl.figure()
pl.contourf(x.to_value('km'), y.to_value('km'), zg.to_value('km'), 40)
pl.colorbar()
pl.xlabel('easting (km)')
pl.ylabel('northing (km)')
pl.title('local altitude (km)')
pl.show()
