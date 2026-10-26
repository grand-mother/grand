from typing import Optional, Union

import math # used in 'schmidt_legendre_poly' function.
import numpy as np
import datetime
from numbers import Number
from .coordinates import Geodetic, ECEF, LTP, GRANDCS, GeodeticRepresentation, horizontal_to_ecef
from grand import path_grandlib_data, grandcs_origin
from numba import jit

pi  = 3.14159265358979323846
d2r = pi/180           # factor to convert angle from degree to radian
r2d = 180/pi           # factor to convert angle from radian to degree

# The default observation time if none is specified
_default_obstime: Union[str, datetime.date] = datetime.date(2026, 12, 31)

#DATADIR: Final = grand_get_path_root_pkg() + "/data/geomagnet"
_datadir = path_grandlib_data("geomagnet")

# The default geo-magnetic model, i.e. IGRF13.
# Reference: https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
_default_model: str = "WMMHR"


def geodetic_to_geocentric(
    longitude, # geodetic longitude in degrees
    latitude, # geodetic latitude in degrees
    height=0 # geodetic height above ellipsoid
    ):
    """
    Equation 7 and 8 of the WMM2025 report.
    Convert geodetic coordinate in "ellipsoid" reference into spherical geocentic coordinates.
    Note that 'longitude' is same for both coordiante systems

    use 'math' for floats as it is faster than 'numpy'. Use 'numpy' for arrays.

    longitude = geodetic longitude [deg] # note that throughout GRANDlib, lon, lat is in degree.
    latitude  = geodetic latitude [deg]
    height    = altitude above ellipsoid Earth model [m]

    WGS84 ellipsoid
    A   = semi-major axis
    e   = eccentricity

    Returns:
        longitude: geocentric longitude. same as geodetic longitude. [degree]
        lat_     : geocentric latitude [degree]. 0 at equator and 90 at the north pole
        r        : geocentric radius [meters]
    """
    e  = 0.081819190842622 # 1/f = 298.257223563; e2 =f(2−f); e = sqrt(e2)
    e2 = e*e               # eccentricity squared

    A    = 6378137 # semi-major axis of the ellipsoid
    slat = np.sin(latitude * d2r)
    clat = np.cos(latitude * d2r)
    Rc   = A/np.sqrt(1 - (e2 * slat**2))

    p = (Rc + height) * clat
    z = (Rc * (1 - e2) + height) * slat
    r = np.sqrt(p**2 + z**2)
    # Latitude is replaced with colatitude angle as it is commonly used in this calculation.
    lat_ = pi/2 - np.arcsin(z/r)    # "pi/2 - " is added to Eq 7 of WMM2025 report to convert angle to colatitude.

    return longitude, lat_*r2d, r

#def compute_enu(latitude, longitude):
#    """
#    Compute East, North, Up (ENU) unit vectors in ECEF frame given latitude and longitude.
#
#    Parameters:
#    latitude (float): latitude in degrees
#    longitude (float): longitude in degrees
#
#    Returns:
#    tuple: (e, n, u) where each is a list of 3 components
#    """
#    # Convert degrees to radians
#    lat = latitude * d2r   # latitude
#    lon = longitude * d2r  # longitude
#
#    slat = np.sin(lat)
#    clat = np.cos(lat)
#    slon = np.sin(lon)
#    clon = np.cos(lon)
#
#    # East vector
#    e = [-slon, clon, 0.0]
#
#    # North vector
#    n = [-clon * slat, -slon * slat, clat]
#
#    # Up vector
#    u = [clon * clat, slon * clat, slat]
#
#    # return basis vector of ENU orientation
#    #return np.vstack((e, n, u)).T
#    return e, n, u

def basis_vector(name, latitude, longitude, azimuth):
    tag = name[0].upper()
    if tag == "E":
        return horizontal_to_ecef(latitude, longitude, 90 + azimuth, 0)
    elif tag == "W":
        return horizontal_to_ecef(latitude, longitude, 270 + azimuth, 0)
    elif tag == "N":
        return horizontal_to_ecef(latitude, longitude, azimuth, 0)
    elif tag == "S":
        return horizontal_to_ecef(latitude, longitude, 180 + azimuth, 0)
    elif tag == "U":
        return horizontal_to_ecef(latitude, longitude, 0, 90)
    elif tag == "D":
        return horizontal_to_ecef(latitude, longitude, 0, -90)
    else:
        raise ValueError(f"Invalid frame orientation `{name}`")

def get_filename(model, epoch_year, latest_epoch):
    """
    Get filename based on input observation time and the geomagnetic model.
    """
    #a          = 6371200 # geomagnetic reference radius. This 'a' is different than the global 'a'.
    #obstime    = date_decimal(obstime)
    #epoch_year = int(obstime - (obstime%5)) # obstime is in decimal year. Coefficients are updated every 5 years.
    #if epoch_year%5 != 0:
    #    raise ValueError("Epoch year must be a multiple of 5. Given epoch year is %s."%epoch_year)

    # Select coeffients file based on the observation time and geomagnetic model.
    model       = model.upper()
    #interpolate = False # interpolate coefficients for IGRF model with epoch year before 2025.
    # WMM high resolution model
    if model=="WMMHR":
        if epoch_year==latest_epoch:
            #coeff_file = _datadir+"geomagnet/WMMHR%s.COF"%latest_epoch
            coeff_file = _datadir / f"WMMHR{latest_epoch}.COF"
        else:
            raise ValueError("Epoch year must be 2025 to use WMMHR. Given epoch year is %s."%epoch_year)
    # WMM model
    elif model=="WMM":
        if epoch_year>=2020:
            #coeff_file = _datadir+"geomagnet/WMM%s.COF"%epoch_year
            coeff_file = _datadir / f"WMM{epoch_year}.COF"
        else:
            raise ValueError("Epoch year must be after 2020 to use WMM. Given epoch year is %s. \
                Use IGRF model instead to compute geomagnetic field before 2020."%epoch_year)
    # IGRF model
    elif model=="IGRF":
        # Ignore data before 1945, assuming GRAND collaboration does not need it for any of its scientific projects.
        if epoch_year>=1945 and epoch_year<(latest_epoch-5): 
            #coeff_file  = _datadir+"geomagnet/DGRF%s.COF"%epoch_year 
            #coeff_file1 = _datadir+"geomagnet/DGRF%s.COF"%(epoch_year+5) 
            coeff_file  = _datadir / f"DGRF{epoch_year}.COF"
            coeff_file1 = _datadir / f"DGRF{epoch_year+5}.COF" 
            #interpolate = True 
        elif epoch_year==latest_epoch-5: 
            #coeff_file  = _datadir+"geomagnet/DGRF%s.COF"%epoch_year 
            #coeff_file1 = _datadir+"geomagnet/IGRF%s.COF"%latest_epoch 
            coeff_file  = _datadir / f"DGRF{epoch_year}.COF"
            coeff_file1 = _datadir / f"IGRF{latest_epoch}.COF" 
            #interpolate = True 
        elif epoch_year==latest_epoch: 
            #coeff_file = _datadir+"geomagnet/IGRF%s.COF"%epoch_year 
            coeff_file = _datadir / f"IGRF{epoch_year}.COF" 
        else: 
            raise Exception("Input observation year (%s) is outside the range."%epoch_year) 
    else: 
        raise Exception("Invalid model or epoch year: %s %s."%(model, epoch_year))  

    return coeff_file   
    
def load_coefficients(filepath):
    """
    Load geomagnetic model cofficients based on the input file.
    Coeffients are arranged in similar manner for IGRF and WMM models.

    n, m, gnm, hnm, dgnm, dhnm
    """
    dtype = [('n', int),
             ('m', int),
             ('gnm', float),
             ('hnm', float),
             ('dgnm', float),
             ('dhnm', float)]

    # load file with coefficients and stored data as named tuple.
    coeff_file = np.loadtxt(filepath, comments=['9999', '#'], skiprows=1, dtype=dtype) 
    n_array    = coeff_file['n']
    m_array    = coeff_file['m']
    n_max      = int(n_array.max())

    Gnm  = np.zeros((n_max+1, n_max+1))
    Hnm  = np.zeros((n_max+1, n_max+1))
    dGnm = np.zeros((n_max+1, n_max+1))
    dHnm = np.zeros((n_max+1, n_max+1))

    # Vectorized scatter is faster than a Python loop.
    Gnm[n_array, m_array]  = coeff_file['gnm']
    Hnm[n_array, m_array]  = coeff_file['hnm']
    dGnm[n_array, m_array] = coeff_file['dgnm']
    dHnm[n_array, m_array] = coeff_file['dhnm']

    return n_max, Gnm, Hnm, dGnm, dHnm

def date_decimal(
    obstime: Union[str, datetime.date] = None # Example: '2020-01-19' or datetime.date(2020, 1, 19).
    ) -> Union[float, Optional[int]]:
    """
    Compute decimal year.
    """
    if isinstance(obstime, str):
        split_datetime = obstime.split('-')
        year  = int(split_datetime[0])
        month = int(split_datetime[1])
        day   = int(split_datetime[2])
    elif isinstance(obstime, datetime.date):
        year  = obstime.year
        month = obstime.month
        day   = obstime.day

    # A tuple avoids constructing a NumPy array on every call.
    start_day = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
    leap_year = ((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)))
    days_in_month = start_day[month] - start_day[month-1] + (month == 2) * leap_year
    
    if month < 1 or month > 12:
        raise Exception("Month must be from 1 to 12. Provided month is %s."%month)   
    if day < 1 or day > days_in_month:
        raise Exception("Day must be from 1 to %s for month %s. Provided day is %s."%(days_in_month, month, day)) 
    
    day_in_year = start_day[month-1] + day + (leap_year if month > 2 else 0)
    decimal_year = year + (day_in_year / (365.0 + leap_year))
    
    return decimal_year

def time_adjusted_gauss_coefficients(coeff_array, dcoeff_array, t, t0=2025):
    """
    Converts coefficients provided at a reference time to a given time
    """
    coeff_array_t = coeff_array + (t - t0)*dcoeff_array  # secular variation coefficients
    
    return coeff_array_t

@jit(nopython=True, cache=True, fastmath=True) # 'jit' decorator is used to speed up 'for' loops
def get_S(n_max):
    # Recursively compute the normalization factor S. Compute 'S' only once for all input thetas.
    S = np.zeros((n_max+1, n_max+1))
    S[0,0] = 1    # Eq. 5.26a
    for n in range(1, n_max+1):
        for m in range(n+1):
            if m == 0:
                S[n,0] = S[n - 1, 0] * (2.*n - 1)/n  # Eq 5.26b. Note: typo, P instead of S in notation.
            else:
                S[n,m] = S[n,m-1]*math.sqrt(((n-m+1)*(int(m==1)+1))/(n+m)) # Eq 5.26c

    return S

@jit(nopython=True, cache=True, fastmath=True) # 'jit' decorator is used to speed up 'for' loops 
def get_P(n_max, theta): 
    """Recursively compute Gaussian normalized associated Lagendre polynomials for each 'theta'. 
    """
    #if (P is None) and (dP is None): # reassigning the same matrix is faster than generating a new matrix.
    P  = np.zeros((n_max+1, n_max+1)) 
    dP = np.zeros((n_max+1, n_max+1)) 

    P[0,0] = 1 # Eq. 5.24a 
    dP[0,0]= 0 # Eq. 5.25a 
    costh = np.cos(theta*d2r)   # x=costh in legendre function P(x) 
    sinth = np.sin(theta*d2r) 

    for n in range(1, n_max+1): 
        for m in range(n+1): 
            if m == n: 
                P[n,n]  = sinth * P[n-1, n-1] 
                dP[n,n] = sinth * dP[n-1, n-1] + costh * P[n-1, n-1]            # Eq 5.25b 
            else: 
                K = 0 if n==1 else ((n-1)**2 - m**2) / ((2*n-1) * (2*n-3))      # Eq 5.24d, 5.24e 
                P[n,m]  = costh * P[n-1,m] - K * P[n-2,m]                       # Eq 5.24 
                dP[n,m] = costh * dP[n-1,m] - sinth * P[n-1,m] - K * dP[n-2,m]  # Eq 5.25c 

    return P, dP


@jit(nopython=True, cache=True, fastmath=True)
def _field_kernel(n_max, phi_arr, theta_arr, r_arr, lat_arr, Gnmt, Hnmt, S):
    """
    Fast geomagnetic-field kernel.

    This replaces the Python-level location loop in Geomagnet.field(). It also avoids
    allocating M, N, cos(M*phi), sin(M*phi), and temporary row-sum arrays for every
    location.
    """
    nloc = len(theta_arr)
    Bx = np.zeros(nloc)
    By = np.zeros(nloc)
    Bz = np.zeros(nloc)
    a = 6371200.0

    for i in range(nloc):
        theta_deg = theta_arr[i]    # theta angle in degree
        phi       = phi_arr[i]*d2r  # degree to radian
        theta     = theta_deg *d2r  # degree to radian
        r         = r_arr[i]        # m

        # Compute associated Lagendre polynomials and its derivatives. 
        P, dP = get_P(n_max, theta_deg) # compute P, dP matrices for each 'theta'

        ar = a / r
        ar_pow = ar * ar * ar  # n = 1 gives (a/r)^(n+2) = (a/r)^3

        bx_geo = 0.0
        by_geo = 0.0
        bz_geo = 0.0

        for n in range(1, n_max + 1):
            msum_x = 0.0
            msum_y = 0.0
            msum_z = 0.0

            for m in range(n + 1):
                angle = m * phi
                c = math.cos(angle)
                s = math.sin(angle)

                g = Gnmt[n, m]
                h = Hnmt[n, m]
                p = P[n, m] * S[n, m]
                dp = -dP[n, m] * S[n, m]

                gh = g * c + h * s
                msum_x += gh * dp
                msum_y += m * (g * s - h * c) * p
                msum_z += gh * p

            bx_geo -= ar_pow * msum_x   # X-component (equation 10 of WMM2025 report)
            by_geo += ar_pow * msum_y   # Y-component (equation 11 of WMM2025 report). Note that cos(latitude) becomes sin(colatitude))
            bz_geo -= (n + 1.0) * ar_pow * msum_z # Z-component (equation 12 of WMM2025 report)
            ar_pow *= ar       # increase a/r ratio as n increases.

        by_geo /= math.sin(theta)

        # Rotate from geocentric coordinates to ellipsoidal reference frame.
        # Equation 17 of WMM2025 report
        # pi/2 - theta is latitude angle in geocentric coordinate system. theta is colatitude angle.
        dlat = (pi / 2.0 - theta) - lat_arr[i] * d2r
        cd = math.cos(dlat)
        sd = math.sin(dlat)
        Bx[i] = bx_geo * cd - bz_geo * sd
        By[i] = by_geo
        Bz[i] = bx_geo * sd + bz_geo * cd

    return Bx, By, Bz

class BfieldRepresentation:
    """
    Generic container for the geomagnetic field
    """
    def __new__(
        cls,
        x: Union[Number, np.ndarray] = None,           # X-component of magnetic field
        y: Union[Number, np.ndarray] = None,           # Y-component of magnetic field
        z: Union[Number, np.ndarray] = None,           # Z-component of magnetic field
        declination: Union[Number, np.ndarray] = None, # declination angle of magnetic field
        inclination: Union[Number, np.ndarray] = None, # inclination angle of magnetic field
        location: Geodetic = None,  # location where magnetic field has been computed
        orientation: str   = 'NED', # orientation of the coordinate frame. 'NED' is the default orientation of WMMHR.
        magnetic: bool     = False, # shift orientation by magnetic declination?
        parent = None,  # store Geomagnet instance to retrieve model and obstime info required for 'to_grandcs' method.
    ):
        if all(isinstance(i, Number) for i in [x,y,z]):
            pass
        elif all(isinstance(i, np.ndarray) for i in [x,y,z]):
            n = len(x)
            assert n == len(y), (
                "Length of x and y array must be the same. \
				x: %i, y: %i"
                % (n, len(y))
            )
            assert n == len(z), (
                "Length of x and z array must be the same. \
							   x: %i, z: %i"
                % (n, len(z))
            )
        else:
            raise TypeError(type(x))

        # create a new instance of the class.
        obj = super().__new__(cls)

        # Attach an attribute during creation
        obj._x = x
        obj._y = y
        obj._z = z
        obj._declination = declination
        obj._inclination = inclination
        obj._location    = location
        obj._orientation = orientation
        obj._magnetic    = magnetic
        obj._parent      = parent

        if obj is None:
            raise RuntimeError("__new__ returned None!")

        return obj

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, v):
        self._x = v

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, v):
        self._y = v

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, v):
        self._z = v

    @property
    def declination(self):
        return self._declination

    @declination.setter
    def declination(self, v):
        self._declination = v

    @property 
    def inclination(self):
        return self._inclination

    @inclination.setter
    def inclination(self, v):
        self._inclination = v

    @property 
    def location(self):
        return self._location

    @location.setter
    def location(self, v):
        self._location = v

    @property 
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, v):
        self._orientation = v

    @property 
    def magnetic(self):
        return self._magnetic

    @magnetic.setter
    def magnetic(self, v):
        self._magnetic = v

    @property 
    def parent(self):
        return self._parent

    def norm(self):
        return np.sqrt(self._x*self._x + self._y*self._y + self._z*self._z)

    def _to_ecef(self):
        """
        Vnew_frame = np.matmul(R, Vold_frame)
        R = columns with old basis expressed in new frame.

        Compute basis vectors of LTP frame with 'NED' (default is NED, it can be any other) orientation in ECEF frame.
        """
        # ux.shape = (3, nlocation)
        ux = basis_vector(              # N
            self._orientation[0], 
            self._location.latitude,
            self._location.longitude,
            0) # azimuth=0
        uy = basis_vector(               # E
            self._orientation[1], 
            self._location.latitude,
            self._location.longitude,
            0) # azimuth=0
        uz = basis_vector(               # D
            self._orientation[2], 
            self._location.latitude,
            self._location.longitude,
            0) # azimuth=0

        # Place-holder for rotation matrices.
        # Rotation matrix to transform vectors from LTP with 'NED' orientation to ECEF frame.
        if len(self._location.latitude)==1:
            R_ned2ecef = np.zeros((3,3)) # rotation matrix
        elif len(self._location.latitude)>1:
            R_ned2ecef = np.zeros((len(self._location.latitude), 3, 3))  # rotation matrix

        # R_ned2ecef.shape = (nlocation, 3, 3). 3x3 basis vector for each locations. 
        # Arrange basis vectors (old basis represented in new frame) in COLUMNS of R_ned2ecef. 
        # R_ned2ecef matrix converts vectors from LTP with 'NED' orientation to ECEF frame. 
        R_ned2ecef[...,0] = ux.T  # arranged in column 
        R_ned2ecef[...,1] = uy.T  # arranged in column 
        R_ned2ecef[...,2] = uz.T  # arranged in column 

        # Arrange magnetic field given in 'NED' orientation for matrix multiplication.
        B_ned = np.vstack((self._x, self._y, self._z)) # shape=(3, nlocation)
        B_ned = B_ned.T[...,np.newaxis] # correct dimension for matrix multiplication. shape=(nlocation, 3, 1)

        # Convert magnetic field vectors from LTP with 'NED' orientation at each location to ECEF frame. 
        # shape: (nlocation, 3, 3).(nlocation, 3, 1) ==> (nlocation, 3, 1)
        B_ecef = np.matmul(R_ned2ecef, B_ned)    # shape=(nlocation, 3, 1)

        return B_ecef

    def to_ltp(self, orientation, magnetic=False):
        """
        Compute the magnetic field given in one orienatation into a differention orientation.
        Note that this method computes the LTP with specified orientation for each location relative 
        to which the final magnetic field is computed. If we want the magnetic field in GRANDCS,
        only one LTP with NWU orientation must be computed at the origin of the GRAND coordinate system.
        Based on this LTP, all magnetic field vectors must be computed.
        This method first converts vectors wrt the ECEF frame and then converts them again from ECEF to the final LTP frame.
        Note: 
            'to_ltp' transforms all magnetic field vectors at different locations wrt separate LTP frames, 
                each with a given orientation at the respective location.
            'to_grandcs' transforms all magnetic field vectors at different locations wrt a single GRANDCS frame. 
        """
        B_ecef = self._to_ecef()

        # Used for GRANDCS
        azimuth = 0.0
        if magnetic:
            azimuth = self._declination

        # Get the basis vectors of LTP with input orientation in ECEF frame. 
        # Arrange basis vectors (old basis represented in new frame) in ROWS of R_ecef2ltp.
        # Arranging basis vectors in rows convert vectors in ECEF to LTP.
        # ux.shape = (3, nlocation)
        ux = basis_vector( # ux is the x-direction vector of the input orientation in ECEF frame.
            orientation[0], 
            self._location.latitude, 
            self._location.longitude, 
            azimuth)
        uy = basis_vector( # uy is the y-direction vector of the input orientation in ECEF frame.
            orientation[1], 
            self._location.latitude, 
            self._location.longitude, 
            azimuth)
        uz = basis_vector( # uz is the z-direction vector of the input orientation in ECEF frame.
            orientation[2], 
            self._location.latitude, 
            self._location.longitude, 
            azimuth)

        # Place-holder for rotation matrices.
        # Rotation matrix to transform vectors from ECEF to LTP with input orientation.
        if len(self._location.latitude)==1:
            R_ecef2ltp = np.zeros((3,3)) # rotation matrix
        elif len(self._location.latitude)>1:
            R_ecef2ltp = np.zeros((len(self._location.latitude), 3, 3))  # rotation matrix

        # R_ned2ecef.shape = (nlocation, 3, 3). 3x3 basis vector for each locations. 
        # Arrange basis vectors in ROWS of R_ecef2ltp. 
        # R_ecef2ltp matrix converts vectors from ECEF frame to LTP with input orientation. 
        R_ecef2ltp[...,0,:] = ux.T # arranged in Row 
        R_ecef2ltp[...,1,:] = uy.T # arranged in Row 
        R_ecef2ltp[...,2,:] = uz.T # arranged in Row 

        # Convert magnetic field vectors from LTP with 'NED' orientation at each location to ECEF frame. 
        # shape: (nlocation, 3, 3).(nlocation, 3, 1) ==> (nlocation, 3, 1) ==> (nlocation, 3)
        B_ltp = R_ecef2ltp @ B_ecef # @=np.matmul. shape=(nlocation, 3, 1)
        B_ltp = np.squeeze(B_ltp, axis=-1)    # shape=(nlocation, 3). Remove the last axis with length 1.  

        return BfieldRepresentation(
            x=B_ltp[:,0],
            y=B_ltp[:,1],
            z=B_ltp[:,2],
            declination=self._declination, 
            inclination=self._inclination, 
            location=self._location, 
            orientation=orientation, 
            magnetic=magnetic, 
            parent=self._parent
            ) 

    def to_grandcs(self):
        """
        Transform magnetic field vectors (given in different coordinate frame) into the GRANDCS frame.
        This method first converts vectors wrt the ECEF frame and then converts them again from ECEF to GRANDCS.
        Note: 
            'to_grandcs' transforms all magnetic field vectors at different locations wrt a single GRANDCS frame. 
            'to_ltp' transforms all magnetic field vectors at different locations wrt separate LTP frames, 
                each with a given orientation at the respective location.
        """
        grd_origin_lat, grd_origin_lon, grd_origin_height, grd_reference = grandcs_origin()
        gdcs_origin_geoid = Geodetic(
            latitude=grd_origin_lat, 
            longitude=grd_origin_lon, 
            height=grd_origin_height, 
            reference=grd_reference)   # 'ELLIPSOID'

        # Transform given magnetic field vectors into the ECEF frame.
        B_ecef = self._to_ecef()

        # Compute magnetic field declination at the GRANDCS origin.
        geoM = Geomagnet(
            model=self._parent.model, 
            obstime=self._parent.obstime,
            latest_epoch=self._parent.latest_epoch
            )
        B_gdcs_origin = geoM.field(location=gdcs_origin_geoid)
        azimuth       = B_gdcs_origin.declination # Declination angle at the GRANDCS origin.         

        # Get the basis vectors of GRANDCS in ECEF frame. 
        # Arrange basis vectors (old basis represented in new frame) in ROWS.
        # Arranging basis vectors in rows convert vectors in ECEF to LTP.
        # Magnetic north is used in GRANDCS (NWU orientation) instead of the geographic north.
        # uN.shape = (3, nlocation) 
        uN = basis_vector( # uN is the x-direction basis of GRANDCS in ECEF frame. 
            'N', 
            gdcs_origin_geoid.latitude, 
            gdcs_origin_geoid.longitude, 
            azimuth) # use geomagnetic north instead of the geographic north.
        uW = basis_vector( # uW is the y-direction basis of GRANDCS in ECEF frame.
            'W', 
            gdcs_origin_geoid.latitude, 
            gdcs_origin_geoid.longitude, 
            azimuth) # use geomagnetic north instead of the geographic north.
        uU = basis_vector( # uU is the z-direction basis of GRANDCS in ECEF frame.
            'U', 
            gdcs_origin_geoid.latitude, 
            gdcs_origin_geoid.longitude, 
            azimuth) # use geomagnetic north instead of the geographic north.

        # Place-holder for rotation matrices.
        # Rotation matrix to transform vectors from ECEF to GRAND coordinate system.
        R_ecef2gdcs = np.zeros((3,3)) # rotation matrix

        # R_ecef2gdcs.shape = (3, 3). 
        # Arrange basis vectors in ROWS of R_ecef2gdcs. 
        # R_ecef2gdcs matrix converts vectors from ECEF frame to LTP with input orientation. 
        R_ecef2gdcs[0] = uN.flatten() # arranged in Row. flatten shape: (3,1) ==> (3,) 
        R_ecef2gdcs[1] = uW.flatten() # arranged in Row. flatten shape: (3,1) ==> (3,)
        R_ecef2gdcs[2] = uU.flatten() # arranged in Row. flatten shape: (3,1) ==> (3,)

        # Convert magnetic field vectors from LTP with 'NED' orientation at each location to ECEF frame. 
        # shape: (3, 3).(3, nlocation) ==> (3, nlocation)
        B_ecef = np.squeeze(B_ecef, axis=-1).T  # shape=(nlocation, 3, 1) ==> (nlocation, 3) ==> (3, nlocation).  
        B_gdcs = R_ecef2gdcs @ B_ecef # @=np.matmul. shape=(3, 3).(3, nlocation) ==> (3, nlocation)

        return BfieldRepresentation(
            x=B_gdcs[0],
            y=B_gdcs[1],
            z=B_gdcs[2],
            declination=self._declination, 
            inclination=self._inclination, 
            location=self._location, 
            orientation='NWU', 
            magnetic=True, 
            parent=self._parent
            )

class Geomagnet:
    """
    contact: Ramesh Koirala (rkoirala.astro@gmail.com)
    
    Proxy to a geomagnetic model. 'WMMHR' is used as a default model.
    Get the geo-magnetic field components on any geodetic location of Earth at any 
    given time after 1945. For location, either provide latitude (deg), longitude (deg), 
    and height (m) above WGS84 ellipsoid or provide location in ECEF, Geodetic, or GRAND coordinate system. 
    TypeError will occur if location is not provided. 
    For obstime, provide time in isoformat ('2026-12-31') or in datetime.date(2026, 12, 31). 
    If obstime is not provided, a default value ('2026-12-31') is used.

    __init__
        Instantiate Geomagnet class and load model's parameter data based on input model and obstime.
        Parameters data for the input model and obstime are downloaded during __init__. 
    
    field:
        Compute and get information about the geomagnetic field. The result is returned as BfieldRepresentation.
        BfieldRepresentation has x [nT], y [nT], z [nT], declination [deg], inclination [deg], and location [Geodetic].
        x, y, and z are computed wrt to LTP frame with 'NED' orientation at each location. 

    Usage example:
    from grand.geo.geomagnet import Geomagnet
    geo = Geomagnet(model='WMMHR', obstime='2026-12-31') # default values are used if input is not given
    # style 1
    B   = geo.field(latitude=27, longitude=85, height=1300, reference="Ellipsoid") # Number, list, or np.ndarray is accepted
    # style 2
    geod = Geodetic(latitude=[..], longitude=[..], height=[..], reference="Geoid")
    B    = geo.field(location=geod)
    # Geomagnetic field info
    B.x
    B.y
    B.z
    B.declination
    B.inclination
    B.location
    B.to_ltp("NWU") # from "NED" to "NWU" orientation at each location
    B.to_grandcs()  # from given orientation to the GRANDCS.
    """
    def __init__(self,
        model: str = _default_model,
        obstime: Union[str, datetime.date] = _default_obstime,
        latest_epoch: int     = 2025,
        ):
        if not isinstance(obstime, (str, datetime.date)):
            raise TypeError(
                f"obstime {obstime} is of type {type(obstime)}. Provide obstime in string or datetime.date type.\
                Example: '2025-01-19' or datetime.date(2025, 1, 19)."
                )
        if not isinstance(model, str):
            raise TypeError(f"model {model} is of type {type(obstime)}. Provide model as string.")

        if not isinstance(latest_epoch, int):
            raise TypeError(f"latest_epoc {latest_epoch} is of type {type(latest_epoch)}. Provide latest_epoch as integer.")

        self.model        = model 
        self.obstime      = obstime
        self.latest_epoch = latest_epoch

        obstime    = date_decimal(obstime)
        epoch_year = int(obstime - (obstime%5)) # obstime is in decimal year. Coefficients are updated every 5 years.
        if epoch_year%5 != 0:
            raise ValueError("Epoch year must be a multiple of 5. Given epoch year is %s."%epoch_year)

        coeff_file = get_filename(model, epoch_year, self.latest_epoch)

        self.obstime_decimal      = obstime

        # Load coefficients for the given model and obstime.
        self._n_max, self._Gnm, self._Hnm, self._dGnm, self._dHnm = load_coefficients(coeff_file) # {Gnm, Hnm}.shape = (n_max+1, n_max+1)
        # linearly interpolate IGRF coefficients to compute dGnm and dHnm. Default values of dGnm and dHnm in IGRF model are all zeros for previous years.
        #if interpolate:
        if model=="IGRF" and epoch_year<self.latest_epoch: # interpolate coefficients.
            n_max1, Gnm1, Hnm1, dGnm1, dHnm1 = load_coefficients(coeff_file1) # {Gnm, Hnm}.shape = (n_max+1, n_max+1)
            # make sure arrays are of the same size. Truncate or add trailing zeros to make the size of Gnm1 and Hnm1 equal to the size of Gnm and Hnm, respectively.
            if n_max<n_max1:
                Gnm1 = Gnm1[:n_max+1, :n_max+1] # Truncate Gnm1 to the size of Gnm. Note that coefficients at higher n and m have smaller effect on the model.
                Hnm1 = Hnm1[:n_max+1, :n_max+1] # Truncate Hnm1 to the size of Hnm. Note that coefficients at higher n and m have smaller effect on the model.
            elif n_max>n_max1:
                Gnm1 = np.pad(Gnm1, (0, n_max-n_max1)) # pads zeros on the increased size.
                Hnm1 = np.pad(Hnm1, (0, n_max-n_max1)) # pads zeros on the increased size.

            self._dGnm = (Gnm1 - self._Gnm)/5.0  # compute rate of change of gnm assuming a linear increase/decrease.
            self._dHnm = (Hnm1 - self._Hnm)/5.0  # compute rate of change of hnm assuming a linear increase/decrease.

        # Compute time adjusted coefficients
        self._Gnmt  = time_adjusted_gauss_coefficients(self._Gnm, self._dGnm, obstime, epoch_year)
        self._Hnmt  = time_adjusted_gauss_coefficients(self._Hnm, self._dHnm, obstime, epoch_year)

        # computed normalization factor of the associated legendre polynomial. Compute only once for all 'theta'.
        self._S     = get_S(self._n_max) 

    def field(self, 
        longitude: Union[Number, list, np.ndarray] = None, 
        latitude: Union[Number, list, np.ndarray]  = None, 
        height: Union[Number, list, np.ndarray]    = None, 
        reference: Union["GEOID", "ELLIPSOID"]     = None,
        location: Union[ECEF, Geodetic, LTP, GRANDCS] = None,
    ):
        """ 
        Based on WMM2025 report. 
        Latitude is replaced with colatitude angle. It is common to use colatitude for geomagnetic calculation. 
        So, trignometric functions in WMM report are replaced with their co-functions. 
        Example: sin(latitude) is replaced with cos(colatitude) and vice-versa. 
        """ 
        # Make sure the location is in the correct format. i.e ECEF, Geodetic, GeodeticRepresentation,
        # or GRAND cs. OR latitude=deg, longitude=deg, height=meter.
        if ((latitude is not None) 
            and (longitude is not None) 
            and (height is not None)
            and isinstance(latitude, (Number, list, np.ndarray))
            ):
            if reference is None:
                raise Exception("Provide 'reference' based on which height is computed. 'Geoid' and 'Ellipsoid' are accepted reference.")
            geod = Geodetic(latitude=latitude, longitude=longitude, height=height, reference=reference)                
        elif isinstance(location, (ECEF, Geodetic, GeodeticRepresentation, LTP, GRANDCS)):
            geod = Geodetic(location)
        else:
            raise TypeError(
                f"Provide location in ECEF, Geodetic, LTP, or GRAND coordinate system instead of type {type(location)}.\n \
                Location can also be given as latitude=deg, longitude=deg, height=meter."
            )

        geod = Geodetic(geod, reference="Ellipsoid") # Imp: Compute height wrt ellipsoid (if in geoid) or pass if already wrt ellipsoid.
        self._location = geod # save location to use in other methods.
        
        # Schmidt Legendre Polynomial requires 'theta' from spherical coordinate system. Geodetic coordinate system is not spherical.
        # Output: deg, deg, m. phi is longitude and theta is colatitude angle.
        phi_arr, theta_arr, r_arr = geodetic_to_geocentric(
                                        geod.longitude, 
                                        geod.latitude, 
                                        geod.height     # height wrt ellipsoid is required.
                                        )  
        
        self._Bx, self._By, self._Bz = _field_kernel(
            self._n_max,
            np.asarray(phi_arr, dtype=np.float64),
            np.asarray(theta_arr, dtype=np.float64),
            np.asarray(r_arr, dtype=np.float64),
            np.asarray(geod.latitude, dtype=np.float64),
            self._Gnmt,
            self._Hnmt,
            self._S,
        )

        # calculate magnetic declination and inclination
        # Equation 19 of WMM 2025 report
        H = np.sqrt(self._Bx*self._Bx + self._By*self._By)
        I = np.arctan2(self._Bz, H)
        D = np.arctan2(self._By, self._Bx)

        self._declination = D
        self._inclination = I

        return BfieldRepresentation(
            x=self._Bx, 
            y=self._By, 
            z=self._Bz, 
            declination=self._declination, 
            inclination=self._inclination, 
            location=self._location,
            parent=self, # required to get 'model' and 'obstime' from an instance of this ('field') method.
            )
