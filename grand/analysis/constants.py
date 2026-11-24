from grand import ECEF, Geodetic, Geomagnet, GRANDCS, LTP
from grand import topography, Reference, geoid_undulation
import numpy as np

c_light = 2.997924580e8
R_earth = 6371007.0
ns = 325
kr = -0.1218
n_atm = 1.000136

groundAltitude = 1235.351 

coord_daq   = Geodetic(latitude=40.99746387, longitude=93.94868871, height=0) 
B = Geomagnet(location=coord_daq)  # default model and obstime is used. Watch out X = EW!
Bvec = np.array([B.field.y[0], -B.field.x[0], B.field.z[0]]) # Bvec_y < 0 because B_g pointing west by 0.6°
Bvec = np.array([B.field.y[0], 0, B.field.z[0]]) # x = magnetic North!
Bn = Bvec/np.linalg.norm(Bvec)


# -----------------------------------------
# Change for adding prior on SWF
# -----------------------------------------
# Linsley atmosphere (CORSIKA US Standard)
# Altitudes in m, grammage in g/cm², c in cm

linsley_h = np.array([0., 4e3, 10e3, 40e3, 100e3])

linsley_a = np.array([
    -186.555305,
    -94.919,
    0.61289,
    0.,
    0.01128292,
])

linsley_b = np.array([
    1222.6562,
    1144.9069,
    1305.5948,
    540.1778,
    1.,
])

linsley_c = np.array([
    994186.38,
    878153.55,
    636143.04,
    772170.16,
    1e9,
])

h_top_atm = 112829.2  # m
earth_radius_m = 6371e3