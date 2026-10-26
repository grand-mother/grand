import numpy as np
from numbers import Number
from grand import path_grandlib_data # To import .pgm file for EGM.

def bilinear_interpolation(dx, dy, v00, v10, v01, v11):
    """
    Linearly interpolate function to find value at (x,y). Values of the function at four corners surrounding (x,y) are given.
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

class EGM2008:
    """
    Earth Gravitational Model 2008. Compute geoid undulation.
    Interpolate EGM2008 geoid height from 2.5' grid (bilinear).
    File: egm2008-2_5.pgm   (https://geographiclib.sourceforge.io/C++/doc/geoid.html#geoidinst)
          downloads: wget https://sourceforge.net/projects/geographiclib/files/geoids-distrib/egm2008-2_5.tar.bz2
    PGM = Portable Gray Map
    """
    
    #def __init__(self, pgm_path: str = ".../data/topography/egm2008-2_5.pgm"):
    def __init__(self, pgm_path: str = ""):
        if pgm_path=="":
            pgm_path = path_grandlib_data("topography")/"egm2008-2_5.pgm"

        if not pgm_path.exists():
            raise FileNotFoundError(f"PGM file not found: {pgm_path}")
        
        # Load PGM (Portable Gray Map) - simple binary format
        with open(pgm_path, 'rb') as f:
            # --- Read header ---
            magic = f.readline().strip()
            if magic not in [b'P5', b'P2']:
                raise ValueError("Not a valid PGM file")
            
            # Skip comments and read offset and scale values. geoid undulation = offset + scale*pixel
            line = f.readline().strip()
            while line.startswith(b'#'):
                line = f.readline().strip()
                if line.startswith(b'# Offset'): # [offset=-108, scale=0.003]
                    self._offset = float(line.split()[-1])
                if line.startswith(b'# Scale'):  # [offset=-108, scale=0.003]
                    self._scale = float(line.split()[-1])
            
            # Dimensions (number of pixels) are given right after comments.
            width, height = map(int, line.split())
            maxval = int(f.readline().strip()) # maxval=65535 for 16-bit, 255 for 8-bit.
            
            # Read binary pixel data
            dtype = '>u2' if maxval > 255 else np.uint8  # maxval=65535 for 16-bit, 255 for 8-bit.
            data = np.fromfile(f, dtype=dtype, count=width*height) # DO NOT use dtype='uint16' or 'float16' etc. Use '>u2'.
            data = data.reshape((height, width))
        
        self._data = data.astype(np.float64)  # Convert to float for math
        self._width, self._height = width, height
        
        # Grid parameters (2.5 arcminutes)
        self._dlat = 2.5 / 60.0   # degrees
        self._dlon = 2.5 / 60.0 
        
        # Latitude: from +90 to -90 (top to bottom)
        # Longitude: from 0 to 360 (left to right)
        self._lat_max = 90.0 
        self._lon_min = 0 

        #print(f"Loaded EGM2008 2.5' grid: {width}×{height}, spacing={self.dlat:.5f}°")

    def _geoid_undulation_1location(self, latitude: float, longitude: float) -> float:
        """
        Interpolate geoid undulation N (meters) at (lat, lon) using bilinear interpolation.
        
        Args:
            lat: latitude in degrees (-90 to 90)
            lon: longitude in degrees (-180 to 180 or 0 to 360)
        
        Returns:
            N: Geoid undulation in meters
        """
        #lon = (lon + 180.0) % 360.0 - 180.0
        lon = (longitude + 360.0) % 360.0
        i = (self._lat_max - latitude) / self._dlat # Row index (i): latitude from north
        j = (lon - self._lon_min) / self._dlon # Column index (j): longitude from west
        
        # Floor indices
        i0 = int(np.floor(i)) # lat
        j0 = int(np.floor(j)) # lon

        # Clamp to grid bounds
        i0 = max(0, min(i0, self._height - 2))  # necessary for boundary, like north pole and south pole.
        j0 = max(0, min(j0, self._width - 2))   # necessary for boundary, like north pole and south pole.
        i1 = i0 + 1
        j1 = j0 + 1
        
        # Fractional parts
        di = i - i0
        dj = j - j0
                
        # Get 4 corner values (raw PGM units)
        v00 = self._data[i0, j0]
        v10 = self._data[i1, j0]
        v01 = self._data[i0, j1]
        v11 = self._data[i1, j1]
        
        # Bilinear interpolation
        v = bilinear_interpolation(di, dj, v00, v10, v01, v11)

        # Convert from PGM units to meters
        # N = offset + scale*pixel  [offset=-108, scale=0.003]
        N = self._offset + self._scale*v
        
        return N

    def geoid_undulation(self, latitude, longitude):
        """
        Compute geoid undulation.
        
        Args:
            lat: latitude in degrees (-90 to 90)
            lon: longitude in degrees (-180 to 180 or 0 to 360)
        
        Returns:
            N: Geoid undulation in meters
        """
        if isinstance(latitude, Number) and isinstance(longitude, Number):
            return self._geoid_undulation_1location(latitude, longitude)

        elif isinstance(latitude, (list, np.ndarray)) and isinstance(longitude, (list, np.ndarray)):
            N_arr = np.asarray([self._geoid_undulation_1location(latitude[i], longitude[i]) for i in range(len(latitude))])
            return N_arr

        else:
            raise TypeError("Input latitude and longitude is of a wrong type. It must be Number, list, or np.ndarray.")
