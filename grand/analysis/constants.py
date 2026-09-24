from grand import ECEF, Geodetic, Geomagnet, GRANDCS, LTP
from grand import topography, Reference, geoid_undulation
import numpy as np

c_light = 2.997924580e8
R_earth = 6371007.0
ns = 325
kr = -0.1218
n_atm = 1.000136

groundAltitude = 1231

coord_daq   = Geodetic(latitude=40.99746387, longitude=93.94868871, height=0) 
B = Geomagnet(location=coord_daq)  # default model and obstime is used. Watch out X = EW!
Bvec = [B.field.y[0], -B.field.x[0], B.field.z[0]] # Bvec_y < 0 because B_g pointing west by 0.6°
Bvec = [B.field.y[0], 0, B.field.z[0]] # x = magnetic North!
Bn = Bvec/np.linalg.norm(Bvec)

