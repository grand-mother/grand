from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import NamedTuple

from astropy.coordinates import CartesianRepresentation
import astropy.units as u
import h5py
import numpy

__all__ = ["Shower", "Field"]


_logger = getLogger(__name__)


class Field(NamedTuple):
    r: CartesianRepresentation
    t: u.Quantity
    E: CartesianRepresentation


class Shower:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def load(cls, path, version=0):
        _logger.info(f"Loading shower data from {path}")

        path = Path(path)
        if path.is_dir():
            loadder = "_from_dir"
        else:
            loadder = f"_from_{path.suffix[1:]}"

        try:
            load = getattr(cls, loadder)
        except AttributeError:
            raise ValueError(f"Invalid data format {path.suffix}")
        else:
            self = load(path, version)

        if hasattr(self, "fields"):
            _logger.info(f"Loaded {len(self.fields)} field(s) from {path}")

        return self

    @classmethod
    def _from_hdf5(cls, path, version):
        kwargs = {}

        with h5py.File(path, "r") as f:
            group = f[f"montecarlo/shower/{version}"]
            for k in list(group.attrs.keys()):
                if k.endswith(".unit"):
                    continue
                v = group.attrs[k]
                if type(v) == numpy.bytes_:
                    v = str(v.decode())
                else:
                    try:
                        unit = group.attrs[k + ".unit"]
                    except KeyError:
                        pass
                    else:
                        v *= u.Unit(unit)
                kwargs[k] = v

            try:
                subgroup = group["fields"]
            except KeyError:
                pass
            else:
                fields = OrderedDict()
                kwargs["fields"] = fields

                for antenna, dset in subgroup.items():
                    _logger.debug(f"Loading field for antenna {antenna}")
                    position = dset.attrs["position"]
                    unit = dset.attrs["position.units"]
                    r = CartesianRepresentation(
                        position[0] * u.Unit(unit[0]),
                        position[1] * u.Unit(unit[1]),
                        position[2] * u.Unit(unit[2]))

                    unit = dset.attrs["units"]
                    t = dset[0,:] * u.Unit(unit[0])
                    E = CartesianRepresentation(dset[1,:] * u.Unit(unit[1]),
                                                dset[2,:] * u.Unit(unit[2]),
                                                dset[3,:] * u.Unit(unit[3]))
                    fields[int(antenna)] = Field(r, t, E)

        return cls(**kwargs)

    def dump(self, path, version=0):
        path = Path(path)
        if path.suffix != ".hdf5":
            raise ValueError("Invalid data format {path.suffix}")

        _logger.info(f"Dumping shower data to {path}")

        info = {}
        for k, v in self.__dict__.items():
            if k != "fields":
                if type(v) == u.Quantity:
                    info[k] = v.value
                    info[k + ".unit"] = numpy.string_(v.unit)
                else:
                    info[k] = numpy.string_(v)

        compression = {"compression": "gzip", "compression_opts": 9}

        with h5py.File(path, "w") as f:
            basename = f"montecarlo/shower/{version}"
            group = f.require_group(basename)
            for k, v in info.items():
                group.attrs[k] = v

            if hasattr(self, "fields"):
                subgroup = group.require_group("fields")
                for antenna, field in self.fields.items():
                    _logger.debug(f"Dumping field for antenna {antenna}")
                    r_unit, t_unit, E_unit = "m", "ns", "uV/m"
                    data = numpy.zeros((4, len(field.t)), dtype="f4")
                    data[0,:] = field.t.to_value(t_unit)
                    toval = lambda x: x.to_value(E_unit)
                    data[1,:] = toval(field.E.x)
                    data[2,:] = toval(field.E.y)
                    data[3,:] = toval(field.E.z)
                    dset = subgroup.require_dataset(str(antenna),
                        shape=data.shape, data=data, dtype=data.dtype,
                        **compression)
                    dset.attrs["columns"] = ("t", "Ex", "Ey", "Ez")
                    dset.attrs["units"] = (t_unit, E_unit, E_unit, E_unit)
                    dset.attrs["position"] = (field.r.x.to_value(r_unit),
                                              field.r.y.to_value(r_unit),
                                              field.r.z.to_value(r_unit))
                    dset.attrs["position.columns"] = ("x", "y", "z")
                    dset.attrs["position.units"] = (r_unit, r_unit, r_unit)
        try:
            n = len(self.fields)
        except AttributeError:
            pass
        else:
            _logger.info(f"Dumped {n} field(s) to {path}")
