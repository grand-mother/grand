from typing import Union
import numpy as np
from numbers import Number
from numba import njit
import rasterio
from rasterio.transform import rowcol
from numba import njit     # njit is equivalent to jit(nopython=True)
from .egm import EGM2008
from .coordinates import Geodetic, ECEF, LTP, GRANDCS, CartesianRepresentation
from grand import path_grandlib_data

"""
Topography code for GRANDlib.
This code computes the elevation of the ground wrt to EGM2008 geoid or WGS84 ellipsoid.
Copernicus data, maintained by European Space Agency, are used for elevation. Default elevation is wrt EGM2008. 

contact: Ramesh Koirala (rkoirala.astro@gmail.com)
"""
datadir = path_grandlib_data("topography")

def _bilinear_interpolation(dx, dy, v00, v10, v01, v11):
    """
    Function to linearly interpolate values to find a value at (x,y). 
    Values of the function at four corners surrounding (x,y) are given.
    Four corners: (x0,y0), (x0,y1), (x1,y0), and (x1,y1)
    Values at corners: v00, v01, v10, and v11
    dx = x - x0
    dy = y - y0
    """
    # Interpolate in x-direction at y=y0
    # v1 = v00 + (v10 - v00) * (x - x0) / (x1 - x0) # along the line where y=y0
    v1 = v00 + (v10 - v00) * dx  # dx = x-x0, x1-x0 = 1
    # Interpolate in x-direction at y=y1
    # v2 = v01 + (v11 - v01) * (x - x0) / (x1 - x0) # along the line where y=y1
    v2 = v01 + (v11 - v01) * dx # dx = x-x0, x1-x0 = 1
    # Finally, interpolate in y-direction at x=x
    # v = v1 + (v2 - v1) * (y - y0) / (y1 - y0)  # along the line where x=x
    v = v1 + (v2 - v1) * dy

    return v

#def get(name: str, tag: str = "101") -> bytes:
#    """Get a BLOB from the store."""
#    # RK: provide proper link where topography data are stored.
#    base = "https://github.com/grand-mother/store/releases/download"
#    url = f"{base}/{tag}/{name}.gz"
#    try:
#        with urllib.request.urlopen(url) as f:
#            return zlib.decompress(f.read(), wbits=31)
#    except:
#        raise Exception("Unable to load data.")

def _get_filename(latitude: Number = None, longitude: Number = None, data_source="Copernicus"):
    # Range: [-90, 90] SN for latitude, [-180,180] WE for longitude.
    # Availabe data: [-56,61] SN for latitude, [-179,179] for longitude.
    longitude = (longitude + 180.0) % 360.0 - 180.0    # file system prefers [-180,180] range instead of [0,360].
    ns = "s" if latitude < 0 else "n"
    ew = "w" if longitude < 0 else "e"

    # Base integer value of longitude and latitude
    lat_base = np.abs(np.floor(latitude))
    lon_base = np.abs(np.floor(longitude))
    
    #if data_source.upper()=="NASA":
    if data_source.upper()=="COPERNICUS":
        if (longitude < -179) or (longitude > 179) or (latitude < -56) or (latitude > 61):
            raise ValueError(f"Latitude must be between -56 and 61, and longitude must be between -179 and 179. \
            Provided latitude is {latitude} and longitude is {longitude}.")

        #return f"NASADEM_NC/NASADEM_NC_{ns}{lat_base:02.0f}{ew}{lon_base:03.0f}.nc"

        #elif data_source.upper()=="COPERNICUS":
        ns = ns.upper()
        ew = ew.upper()
        
        return datadir / f"Copernicus_DSM_30_{ns}{lat_base:02.0f}{ew}{lon_base:03.0f}.tif"
            
def _get_data(
        latitude: float = None, 
        longitude: float = None, 
        filename: str = None):
    """
    Update the cache of topography data.
    Download data from: https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3
                        Imp: Manually select location with 0.001 off from the intended latitude and longitude tile.
                        Ex: for a tile of latitude=[28,29] and longitude=[85,86], use [27.999, 29.001] and [84.999, 86.001].
    Locally saved as .../grand/data/topography/Copernicus_DSM_30_N*E*.tif
    Example: Copernicus_DSM_30_N28E085.tif ==> Latitude=[28,29]; longitude=[85,86]
    """
    #import xarray as xr    # to read NASADEM...nc (netCDF4) files.

    if filename is None:
        filename = _get_filename(latitude, longitude)

    if "Copernicus" in str(filename):
        # Open Copernicus topography file using rasterio
        data = rasterio.open(filename)
        
    return data


class Topography:
    """
    Proxy to topography data.

    To Run:
    topo = Topography()
    loc  = Geodetic(latitude=[...], longitude=[...], height=[...])
    elev = topo.elevation(loc, reference='ellipsoid')
    """
    def __init__(self):
        self.stored_data = {}
        self.data = None       # data from the latest file
        self.grid_lat = None   # latitude list from the latest file
        self.grid_lon = None   # longitude list from the latest file
        self.grid_hgt = None   # height list from the latest file
        self.last_fname  = ""  # name of the latest file
        self.egm = None
        
    def elevation(
            self,
            coordinates = None,
            latitude = None,
            longitude= None,
            reference: str = "GEOID",  # options: 'GEOID', LOCAL', 'ELLIPSOID
    ):
        """Get the topography elevation, w.r.t. sea level, w.r.t the
        ellipsoid or in local coordinates. The default reference is
        w.r.t sea level (GEOID).
        """
        if isinstance(reference, str):
            reference = reference.upper()

            if reference == "LOCAL":
                if not isinstance(coordinates, (LTP, GRANDCS)):
                    raise ValueError("not an LTP or GRANDCS frame")
                elevation = self._local_elevation(coordinates)
            else:
                if (coordinates is not None) and isinstance(coordinates, (Geodetic, ECEF, LTP, GRANDCS)):
                    geod_cs   = Geodetic(coordinates)
                    latitude  = geod_cs.latitude
                    longitude = geod_cs.longitude
                elif (latitude is not None) and (longitude is not None):
                    latitude  = latitude
                    longitude = longitude
                    
                elevation = self._global_elevation(
                    latitude  = latitude,
                    longitude = longitude,
                    reference = reference)

            if elevation.size == 1:
                elevation = elevation[0]

            return elevation
        else:
            raise Exception(f"Provide a proper reference instead of {reference}. Allowed references are Geoid, Local, and Ellipsoid.")

    # def _global_elevation(self, latitude, longitude, reference="GEOID"):
    #     """
    #     Get the topography elevation w.r.t. mean sea level (geoid) or w.r.t. the ellipsoid.
    #     Default reference is EGM2008 (geoid) for Copernicus data.
    #     """
    #     reference = reference.upper()
    #     if reference=="ELLIPSOID":
    #         # Instantiate EGM2008 to compute elevation wrt to WGS84 ellipsoid.
    #         if self.egm == None:
    #             self.egm = EGM2008() # default EGM2008 file is used.
    #         egm = self.egm

    #     # Compute the geodetic coordinates
    #     if not isinstance(latitude, np.ndarray):
    #         if isinstance(latitude, Number):
    #             latitude  = np.array((latitude,))
    #             longitude = np.array((longitude,))
    #         elif isinstance(latitude, list):
    #             latitude  = np.array(latitude)
    #             longitude = np.array(longitude)
                
    #     # Return the topography elevation
    #     elevation = np.zeros(latitude.size)

    #     # RK: To Do: make this loop more efficient.
    #     # Compute elevation at the given latitude and longitude.
    #     for indx, lat in enumerate(latitude):
    #         lon   = longitude[indx] 
    #         fname = _get_filename(lat, lon)
    #         if fname not in self.stored_data.keys():
    #             self.stored_data[fname] = _get_data(filename=fname)

    #         # Load new data only if the elevation file has changed.
    #         if (fname!=self.last_fname):
    #             self.data       = self.stored_data[fname]#.read(1)
    #             self.height     = self.data.read(1)
    #             self.last_fname = fname
                
    #         # indexing starts at upper left of the lon-lat tile.
    #         i, j   = rowcol(self.data.transform, lon, lat, op=float) # returns location index as float (i=row, j=col)
    #         i0, j0 = np.int32(np.floor(i)), np.int32(np.floor(j))
    #         i1, j1 = i0+1, j0+1
    #         di, dj = i-i0, j-j0   # fractional parts
    #         if j1 >= self.height.shape[1]:
    #             print(f"Latitude={lat}, Longitude={lon}")
    #             print(f"Tile={fname.name}")
    #             print(f"i={i}, j={j}")
    #             print(f"j0={j0}, j1={j1}")

    #         # Get the four corner elevation values surrounding the provided latitude and longitude.
    #         h00 = self.height[i0, j0]
    #         h10 = self.height[i1, j0]
    #         h01 = self.height[i0, j1]
    #         h11 = self.height[i1, j1]

    #         # Interpolate elevations given at position (i0, j0), (i1, j0), (i0, j1), and (i1, j1)
    #         #    to get elevation at lat, lon. Copernicus provides elevation wrt EGM2008 geoid.
    #         h = _bilinear_interpolation(di, dj, h00, h10, h01, h11)
            
    #         if reference=="ELLIPSOID":
    #             h += egm.geoid_undulation(lat, lon)  # h = H + N (ellipsoid = geoid + undulation)

    #         elevation[indx] = h

    #     return elevation



    def _global_elevation(self, latitude, longitude, reference="GEOID"):
        """
        Get the topography elevation w.r.t. mean sea level (geoid)
        or w.r.t. the WGS84 ellipsoid.

        Handles bilinear interpolation across neighboring Copernicus tiles,
        including tile boundaries and missing neighboring tiles.
        """
        reference = reference.upper()

        if reference not in ("GEOID", "ELLIPSOID"):
            raise ValueError(f"Unknown reference: {reference}")

        if reference == "ELLIPSOID":
            if self.egm is None:
                self.egm = EGM2008()
            egm = self.egm

        # Convert coordinates to numpy arrays and broadcast them
        latitude = np.atleast_1d(np.asarray(latitude, dtype=float))
        longitude = np.atleast_1d(np.asarray(longitude, dtype=float))
        latitude, longitude = np.broadcast_arrays(latitude, longitude)

        latitude = latitude.ravel()
        longitude = longitude.ravel()

        elevation = np.zeros(latitude.size)

        def is_valid_pixel(value, nodata):
            """Check that a pixel contains a valid elevation."""
            if np.ma.is_masked(value):
                return False

            value = float(value)

            if not np.isfinite(value):
                return False

            if nodata is not None and np.isclose(value, nodata):
                return False

            return True

        for indx, (lat, lon) in enumerate(zip(latitude, longitude)):

            fname = _get_filename(lat, lon)

            # Load the current tile if necessary
            if fname not in self.stored_data:
                self.stored_data[fname] = _get_data(filename=fname)

            if fname != self.last_fname:
                self.data = self.stored_data[fname]
                self.height = self.data.read(1)
                self.last_fname = fname

            base_transform = self.data.transform

            # Fractional pixel coordinates in the current tile
            i, j = rowcol(base_transform, lon, lat, op=float)

            i0, j0 = int(np.floor(i)), int(np.floor(j))
            i1, j1 = i0 + 1, j0 + 1

            di, dj = i - i0, j - j0

            def get_pixel(row, col):
                """
                Retrieve a pixel from the current tile or a neighboring tile.
                Candidate tiles are checked using their actual raster bounds.
                """

                height, width = self.height.shape

                # First, try the current tile
                if 0 <= row < height and 0 <= col < width:
                    value = self.height[row, col]

                    if is_valid_pixel(value, self.data.nodata):
                        return value

                # Geographic coordinates of the requested pixel center
                x, y = base_transform * (col + 0.5, row + 0.5)

                # Search the 3x3 neighborhood of possible tiles.
                # This also handles pixels just across tile boundaries.
                lat0 = int(np.floor(y))
                lon0 = int(np.floor(x))

                candidate_tiles = []

                # Try the original tile first
                candidate_tiles.append(fname)

                for lat_tile in (lat0, lat0 - 1, lat0 + 1):
                    for lon_tile in (lon0, lon0 - 1, lon0 + 1):
                        pixel_fname = _get_filename(lat_tile + 0.5, lon_tile + 0.5)

                        if pixel_fname not in candidate_tiles:
                            candidate_tiles.append(pixel_fname)

                # Test each candidate using its own affine transform
                for pixel_fname in candidate_tiles:

                    # Skip tiles that are not available on disk
                    if not pixel_fname.is_file():
                        continue

                    # Load the tile only once
                    if pixel_fname not in self.stored_data:
                        self.stored_data[pixel_fname] = _get_data(filename=pixel_fname)

                    neighbor_data = self.stored_data[pixel_fname]

                    ni, nj = rowcol(neighbor_data.transform, x, y, op=float)
                    ni, nj = int(np.floor(ni)), int(np.floor(nj))

                    neighbor_height, neighbor_width = neighbor_data.shape

                    # Check whether the requested pixel belongs to this tile
                    if not (0 <= ni < neighbor_height and 0 <= nj < neighbor_width):
                        continue

                    pixel = neighbor_data.read(1, window=((ni, ni + 1), (nj, nj + 1)))[0, 0]

                    if is_valid_pixel(pixel, neighbor_data.nodata):
                        return pixel

                raise IndexError(
                    f"No valid tile contains pixel at "
                    f"longitude={x}, latitude={y}. "
                    f"Candidate tiles: {candidate_tiles}"
                )

            # Retrieve the four pixels needed for bilinear interpolation
            h00 = get_pixel(i0, j0)
            h10 = get_pixel(i1, j0)
            h01 = get_pixel(i0, j1)
            h11 = get_pixel(i1, j1)

            # Bilinear interpolation
            h = _bilinear_interpolation(di, dj, h00, h10, h01, h11)

            # Convert geoid elevation to ellipsoid elevation if necessary
            if reference == "ELLIPSOID":
                h += egm.geoid_undulation(lat, lon)

            elevation[indx] = h

        return elevation


    def _local_elevation(self,
        arg: Union[LTP, GRANDCS] = None, # local coordinates in LTP frame.
        max_iterations: int = 5,
        tolerance: float = 1e-3) -> float:
        """
        Compute ground elevation at a single point in local coordinates.
        
        Idea:
            Find the latitude, longitude, and height whose projection onto the 
            local frame's xy-plane equals the location's x and y coordinates.
            The z coordinate wrt the local frame of the corresponding (lat, lon, height) 
            is the required local elevation. 
        Args:
            arg: location in the LTP or GRANDCS frame
            
        Returns:
            Elevation in meters (NaN if outside terrain grid)
        """
        if not isinstance(arg, (LTP, GRANDCS)):
            raise TypeError(f"Provide input coordinates in LTP and GRANDCS coordinate system instead of {type(arg)}.")

        # copy 'arg' to use in iteration.
        # iterate until x and y components of tmp_arg and arg are close to each other.
        tmp_arg   = LTP(
            x=arg.x,
            y=arg.y,
            z=np.zeros(len(arg.x)), # Use coordinates of xy-plane of the LTP frame as the first guess.
            frame=arg,
         )  

        for _ in range(max_iterations):
            # Get ground altitude for current guess
            geod = Geodetic(tmp_arg, reference='Ellipsoid')
            
            # Get terrain elevation wrt ellipsoid
            elevation = self._global_elevation(
                latitude=geod.latitude, 
                longitude=geod.longitude, 
                reference='Ellipsoid')

            # set height of the geod with the new elevation wrt ellipsoid
            geod.height = elevation
            
            # Transform 'geod' into the corresponding local coordinates
            ltp = LTP(geod, frame=arg) # use same frame of the input location.
            
            # Check for convergence
            dx = arg.x - ltp.x
            dy = arg.y - ltp.y

            within_tolerance = (abs(dx) < tolerance) & (abs(dy) < tolerance)
            if within_tolerance.all():
                break
            
            # Update local coordinates if geoid projection is not close to the input LTP locations.
            tmp_arg.x += dx
            tmp_arg.y += dy
            
        return ltp.z
    
    def distance(
            self,
            position: Union[ECEF, Geodetic, LTP, GRANDCS],
            direction: Union[CartesianRepresentation, ECEF],
            max_distance: float = 100000, # 100 km
            max_iteration: int = 10000,
    ):
        """
        Compute distance along a particle trajectory where it hits the nearest ground/mountain.
        Distance resolution is 10 cm along the particle trajectory.

        Parameters
        -----------
        position: Union[ECEF, Geodetic, LTP, GRANDCS]
                  Initial ECEF/cartesian position.
        
        direction: CartesianRepresentation in ECEF frame
                  Propagation vector

        maximum_distance: float in meter
                 Maximum distance along the particle trajectory to find interaction point.
        """
        # Transform 
        position = ECEF(position)
        if isinstance(direction, (CartesianRepresentation, ECEF)):
            # TODO: Convert direction vector given in any known coordinate frame to ECEF frame.
            #       direction must be in ECEF frame for lib.grand_topography_distance()
            pass
        else:
            raise TypeError("Direction must be in CartesianRepresentation in ECEF frame.")
        
        # Normalize the direction vector. Unit vector is required.
        norm = np.linalg.norm(direction)
        direction = direction / norm
        
        dist = 0
        ds   = 0 # sum of ds is the distance travelled by the particle.
        for _ in range(max_iteration):
            pos_geod = Geodetic(position, reference='ellipsoid')
            height   = pos_geod.height
            elevtn   = self.elevation(pos_geod, reference='ellipsoid')
            dist    += ds #distance along the trajectory
            if height<elevtn: # break 'for' loop, if height is below elevation of the ground, aka hitting the ground.
                break
            
            # adaptive step size. Min step size = 10cm or half of particle height - ground elevation.
            ds = max(0.1, 0.5*abs(height-elevtn)) 
            position += ds*direction # change position of a particle along its trajectory.
            
        return dist
