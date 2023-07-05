#! /usr/bin/env python
from grand import (
    Coordinates,
    CartesianRepresentation,
    SphericalRepresentation,
    GeodeticRepresentation,
    topography,
)
from grand import ECEF, Geodetic, GRANDCS, LTP
from grand import Geomagnet
from grids import (
    create_grid_univ,
)  # GRAND_tools rep https://github.com/kumikokotera/GRAND_tools. Add GRAND_tools/grid_shape rep to PYTHONPATH
import matplotlib.pyplot as pl
import numpy as np


# Set the local frame origin
subeiD = Geodetic(latitude=41.2699858577, longitude=96.5302861717, height=0)
dunhuang = Geodetic(latitude=40.902317, longitude=94.054550, height=0)

## Topography
# Get the corresponding topography data. Note that does are dowloaded from the
# web and cached which might take some time. Reducing the area results in less
# data to be downloaded, i.e. speeding up this step
radius = 1000  # m
topography.update_data(dunhuang, radius=radius)

# Generate a grid of local coordinates using numpy.meshgrid
x = np.linspace(-1 * radius, radius, 1001)
y = np.linspace(-1 * radius, radius, 1001)
X, Y = np.meshgrid(x, y)
coordinates = GRANDCS(
    x=X.flatten(), y=Y.flatten(), z=np.zeros(X.size), location=dunhuang
)  # CS: x = North, y = West
# Get the local ground elevation. Note that local coordinates naturally account for the Earth curvature.
zg = topography.elevation(
    coordinates, reference="local"
)  # Altitude measured wrt cart ref centered on origin "location"
zg = zg.reshape(X.shape)

# Now go to Geodetic coordinates
coord_geo = Geodetic(coordinates)
# Now reformat output
lat = coord_geo.latitude.reshape(X.shape)  # Format as matrix
lat = lat[0]  # Grab first line: list of latitudes
lon = coord_geo.longitude.reshape(X.shape)  # Format as matrix
lon = lon[:, 0]  # Grab first col: list of longitudes

## Detector
# Generate grid
pos_hex, offset2 = create_grid_univ(
    GridShape="hexhex",
    radius=816,
    angle=0,
    do_offset=False,
    Nrand=None,
    randeff=None,
    DISPLAY=False,
    directory=None,
    do_prune=False,
    input_n_ring=3,
)
det_cs = GRANDCS(
    x=pos_hex[0, :].flatten(),
    y=pos_hex[1, :].flatten(),
    z=np.zeros(pos_hex[0, :].size),
    location=dunhuang,
)
print("Nb of units:", len(det_cs.x))
det_cs.z = topography.elevation(det_cs, reference="local")
# Now go to geodetic
det_geo = Geodetic(det_cs)


# Write to file
antpos = open("layout_GP100.dat", "w")
for i, _ in enumerate(det_cs.x):
    tofile = (
        " A"
        + str(i)
        + " "
        + "{:.1f}".format(det_cs.x[i])
        + " "
        + "{:.1f}".format(det_cs.y[i])
        + " "
        + "{:.6f}".format(det_geo.latitude[i])
        + " "
        + "{:.6f}".format(det_geo.longitude[i])
        + " "
        + "{:.1f}".format(det_cs.z[i])
    )
    print(tofile)
    antpos.write(tofile + "\n")
antpos.close()


# Plot the result using contour levels. The Earth curvature is clearly visible
# at large distances from the origin.
pl.figure()
pl.contourf(x / 1000, y / 1000, zg, 40, cmap="terrain")
pl.plot(det_cs.x / 1000, det_cs.y / 1000, "+k")
pl.plot(0, 0, "or")
pl.colorbar(label="Altitude (m)")
pl.xlabel("Northing (km)")
pl.ylabel("Westing (km)")
pl.title("Elevation wrt LTP")

pl.figure()
pl.contourf(lon, lat, zg.T, 40, cmap="terrain")
pl.plot(det_geo.longitude, det_geo.latitude, "+k")
pl.plot(dunhuang.longitude, dunhuang.latitude, "or")

pl.colorbar(label="Altitude (m)")
pl.xlabel("Latitude (deg)")
pl.ylabel("Longitude (deg)")
pl.title("Elevation wrt LTP")

fig = pl.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(det_cs.x / 1000, det_cs.y / 1000, det_cs.z)
pl.xlabel("Northing (km)")
pl.ylabel("Westing (km)")
pl.title("Elevation wrt LTP (m)")

pl.show()
