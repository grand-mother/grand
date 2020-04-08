from __future__ import annotations

from typing import Any, Optional, Tuple, Union

from astropy.coordinates import BaseRepresentation, CartesianRepresentation
import astropy.units as u
import numpy
try:
    from scipy.spatial.transform import Rotation as _Rotation
except ImportError:
    _Rotation = None


class Rotation(_Rotation):
    def __mul__(self, other: Any):
        try:
            return self.apply(other)
        except NotImplementedError:
            return NotImplemented

    def apply(self,
        other: Union[_Rotation, BaseRepresentation, u.Quantity, numpy.ndarray],
        inverse: bool=False)                                                   \
        -> Union[Rotation, BaseRepresentation, u.Quantity, numpy.ndarray]:

        def apply(v):
            return super(self.__class__, self)                                 \
                .apply(v, inverse)                                             \
                .reshape(other.shape)

        if isinstance(other, _Rotation):
            if inverse:
                return super(self.__class__, self.inverse).__mul__(other)
            else:
                return super().__mul__(other)
        elif isinstance(other, BaseRepresentation):
            r = other.represent_as(CartesianRepresentation)
            matrix = self.matrix if not inverse else self.inverse.matrix
            r = r.transform(matrix)
            return r.represent_as(other)
        elif isinstance(other, u.Quantity):
            return apply(other.value) * other.unit
        elif isinstance(other, numpy.ndarray):
            return apply(other)
        else:
            raise NotImplementedError

    @classmethod
    def from_euler(cls, seq: str, *angles: u.Quantity) -> Rotation:
        if angles[0].shape:
            angles = angles[0].to_value(u.rad)
        else:
            angles = tuple(a.to_value(u.rad) for a in angles)
        return super().from_euler(seq, angles)

    def euler_angles(self, seq: str, unit: Union[str, u.Unit]=u.rad)           \
        -> u.Quantity:

        unit = u.Unit(unit)
        if unit is u.deg:
            return super().as_euler(seq, degrees=True) * unit
        else:
            return super().as_euler(seq) * u.rad

    @classmethod
    def from_rotvec(cls, rotvec: u.Quantity) -> Rotation:
        return super().from_rotvec(rotvec.to_value(u.rad))

    @classmethod
    def align_vectors(cls,
        a: Union[BaseRepresentation, u.Quantity, numpy.ndarray],
        b: Union[BaseRepresentation, u.Quantity, numpy.ndarray],
        weights: Optional[numpy.ndarray]=None)                                 \
        -> Tuple[Rotation, float]:

        def harmonize(x):
            if isinstance(x, BaseRepresentation):
                xyz = x.represent_as(CartesianRepresentation).xyz
                ux = xyz.unit
                x = xyz.value.T
            elif isinstance(x, u.Quantity):
                ux = x.unit
            else:
                ux = u.dimensionless_unscaled
            return x, ux

        a, ua = harmonize(a)
        b, ub = harmonize(b)
        b = b * (1 * ub).to_value(ua)
        return super().align_vectors(a, b, weights)

    @property
    def inverse(self) -> Rotation:
        return super().inv()

    @property
    def matrix(self) -> numpy.ndarray:
        m = super().as_matrix()
        if m.shape[0] == 1:
            m = m.reshape(3, 3)
        return m

    @property
    def rotvec(self) -> u.Quantity:
        return super().as_rotvec() * u.rad

    @property
    def magnitude(self) -> u.Quantity:
        mag = super().magnitude()
        if isinstance(mag, numpy.ndarray) and mag.size == 1:
            mag = mag[0]
        return mag * u.rad
