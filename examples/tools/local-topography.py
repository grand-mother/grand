#! /usr/bin/env python
import astropy.units as u
from grand import ECEF, LTP, topography
import matplotlib.pyplot as plot
import numpy


# Set the local frame origin
origin = ECEF(latitude=42.923516 * u.deg, longitude=86.716069 * u.deg,
              representation_type="geodetic")

# Get the corresponding topography data. Note that does are dowloaded from the
# web and cached which might take some time. Reducing the area results in less
# data to be downloaded, i.e. speeding up this step
radius = 200 * u.km
topography.update_data(origin, radius=radius)


# Generate a grid of local coordinates using numpy.meshgrid
x = numpy.linspace(-radius, radius, 1001)
y = numpy.linspace(-radius, radius, 1001)
X, Y = numpy.meshgrid(x, y)
coordinates = LTP(X.flatten(), Y.flatten(), numpy.zeros(X.size) << u.km,
                  location=origin, orientation="ENU", magnetic=False)

# Get the local ground elevation. Note that local coordinates naturally account
# for the Earth curvature.
zg = topography.elevation(coordinates)
zg = zg.reshape(X.shape)

# Plot the result using contour levels. The Earth curvature is clearly visible
# at large distances from the origin.
plot.figure()
plot.contourf(x.to_value("km"), y.to_value("km"), zg.to_value("km"), 40)
plot.colorbar()
plot.xlabel("easting (km)")
plot.ylabel("northing (km)")
plot.title("local altitude (km)")
plot.show()
