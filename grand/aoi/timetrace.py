## The grandlib classes following https://docs.google.com/document/d/1P0AwR3U3MVZyU1ewIobWkJPZmVkxKCAw/edit

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import hilbert

from grand.geo.coordinates import *


@dataclass
class Timetrace3D:
    """A class for holding x,y,z single antenna traces over time"""

    n_points: int = 0
    """The trace length"""

    time_step: float = 0
    """[ns] n_points x step = total timetrace length"""

    t0: np.datetime64 = field(default_factory=lambda: np.datetime64(0, 'ns'))
    """Start time of the trace as unix time with nanoseconds"""

    trigger_time: np.datetime64 = field(default_factory=lambda: np.datetime64(0, 'ns'))
    """Trigger time as unix time with nanoseconds"""

    t_bin_size: float = 2
    """The size of the time bin - the time resolution in ns"""

    du_id: int = -1
    """The Detector Unit ID"""

    ## Trace vector in X
    # trace_x: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Trace vector in Y
    # trace_y: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    ## Trace vector in Z
    # trace_z: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    # ToDo: Allow empty constructor in CartesianRepresentation?
    ## Trace 3D vector (x,y,z)
    _trace: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(0, np.float32), y=np.zeros(0, np.float32), z=np.zeros(0, np.float32)))
    # _trace1: list = None
    # trace: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    t_vector: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    """The time vector [ns] - generated from the trace length, t0 and t_bin_size"""

    ## *** Hilbert envelopes are currently NOT DEFINED in the data coming from hardware
    _hilbert_trace: CartesianRepresentation = field(default_factory=lambda: CartesianRepresentation(x=np.zeros(0, np.float32), y=np.zeros(0, np.float32), z=np.zeros(0, np.float32)))
    # ## Hilbert envelope vector in X
    # hilbert_trace_x: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    # ## Hilbert envelope vector in X
    # hilbert_trace_y: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))
    # ## Hilbert envelope vector in X
    # hilbert_trace_z: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float))

    ## ToDo: add additional quantities from the doc?

    ## ToDo: add additional quantities from the trees?

    def calculate_t_vector(self, time_offset):
        """Calculation of the time vector - should be called manually when all the necessary parameters of the Timetrace3D are set

        Parameters
        ----------
        time_offset : float, optional
            Offset added to the time axis, in nanoseconds.
        """
        # ToDo: t0 is at the moment the trigger time, not the start time...
        self.t_vector = np.arange(self.trace.x.size)*self.t_bin_size+(self.t0-time_offset).astype(int)

    def get_value_at_time(self, time_offset):
        """Get the signal value at a certain time. Returns 0, if nothing measured at this time

        Parameters
        ----------
        t : float
            Time, in nanoseconds.

        Returns
        -------
        ndarray, shape (3,)
            The three components at that time, interpolated.
        """
        # If a signal was measured for the requested time value, return it
        if np.any(self.t_vector == time_offset):
            return self.trace[:,np.where(self.t_vector == time_offset)[0][0]]
        # Otherwise, return 0
        else:
            return np.zeros(3, np.float32)

    def get_hilbert_value_at_time(self, time_offset):
        """Get the signal Hilbert envelope's value at a certain time. Returns 0, if nothing measured at this time

        Parameters
        ----------
        t : float
            Time, in nanoseconds.

        Returns
        -------
        ndarray, shape (3,)
            The three components at that time, interpolated.
        """
        # If a signal was measured for the requested time value, return it
        if np.any(self.t_vector == time_offset):
            return self.hilbert_trace[:,np.where(self.t_vector == time_offset)[0][0]]
        # Otherwise, return 0
        else:
            return np.zeros(3, np.float32)

    @property
    def trace(self):
        """Trace 3D vector (x,y,z)

        Returns
        -------
        ndarray, shape (3, n_samples)
            The three-component trace.
        """
        return self._trace

    @trace.setter
    def trace(self, v):
        r"""Sets the three-component trace.

        Parameters
        ----------
        v : array_like
            Samples, shape ``(3, n_samples)``.
        """
        self._trace = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

    @property
    def hilbert_trace(self):
        """Hilbert envelope 3D vector (x,y,z) - not defined in the hardware

        Returns
        -------
        ndarray, shape (3, n_samples)
            Its Hilbert envelope.
        """
        # Calculate the hilbert envelope if not yet calculated
        if len(self._hilbert_trace[0]) == 0:
            hx = np.abs(hilbert(self.trace.x))
            hy = np.abs(hilbert(self.trace.y))
            hz = np.abs(hilbert(self.trace.z))
            self._hilbert_trace = CartesianRepresentation(x=hx, y=hy, z=hz)

        return self._hilbert_trace

    @hilbert_trace.setter
    def hilbert_trace(self, v):
        r"""Sets the Hilbert envelope of the trace.

        Parameters
        ----------
        v : array_like
            Envelope, shape ``(3, n_samples)``.
        """
        self._hilbert_trace = CartesianRepresentation(x=v[0], y=v[1], z=v[2])

@dataclass
class Voltage(Timetrace3D):
    """A class for holding voltage traces + additional information"""

    ## GPS time of the trigger - why would we want it? We have already _trigger_time in Timetrace3D, that is GPS time + nanoseconds
    # _GPS_trigtime: np.uint32 = 0
    is_triggered: bool = True
    """Is this a triggered trace? - not sure if it should be here or in Timetrace3D, or perhaps further up in the event"""

@dataclass
class Efield(Timetrace3D):
    """A class for holding Efield traces + additional information"""

    eta: float = 0
    """Polarization angle of the reconstructed Efield in the shower plane [deg]"""

    a_ratio: float = 0
    """Ratio of the geomagnetic to charge excess contributions"""


## Exception risen if the TTree already exists
class TreeExists(Exception):
    r"""Raised when writing to a tree that is already present in the file.

    """
    pass

