"""Geometry of the shower relative to the array."""

from .angles import (eta, distance_source_antenna, omega, sin_geomag_angle)
from .footprint import (compute_core, generate_cone_surface_vectors)

__all__ = ['eta', 'distance_source_antenna', 'omega', 'sin_geomag_angle', 'compute_core', 'generate_cone_surface_vectors']
