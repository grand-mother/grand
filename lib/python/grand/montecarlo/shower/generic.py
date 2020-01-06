from __future__ import annotations

from collections import OrderedDict
from logging import getLogger
from pathlib import Path
from typing import NamedTuple, Optional, Union

from astropy.coordinates import CartesianRepresentation
import astropy.units as u
import numpy

from ...import io

__all__ = ["Shower", "Field"]


_logger = getLogger(__name__)


class Field(NamedTuple):
    r: CartesianRepresentation
    t: u.Quantity
    E: CartesianRepresentation


class Shower:
    def __init__(self, **kwargs) -> None:
        self.fields:Optional[OrderedDict] = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def load(cls, path:Union[Path, str], version: int=0) -> Shower:
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

        if self.fields is not None:
            _logger.info(f"Loaded {len(self.fields)} field(s) from {path}")

        return self

    @classmethod
    def _from_hdf5(cls, path: Path, version: int) -> Shower:
        kwargs = {}

        with io.open(path) as root:
            shower_node = root[f"montecarlo/shower/{version}"]
            for name, data in shower_node.elements:
                kwargs[name] = data

            try:
                fields_node = shower_node["fields"]
            except KeyError:
                pass
            else:
                fields: OrderedDict = OrderedDict()
                kwargs["fields"] = fields

                for antenna_node in fields_node:
                    antenna = int(antenna_node.name)
                    _logger.debug(f"Loading field for antenna {antenna}")
                    r = antenna_node.read("r")
                    t = antenna_node.read("t")
                    E = antenna_node.read("E")
                    fields[antenna] = Field(r, t, E)

        return cls(**kwargs)

    def dump(self, path: Union[Path, str], version: int=0) -> None:
        path = Path(path)
        if path.suffix != ".hdf5":
            raise ValueError("Invalid data format {path.suffix}")

        _logger.info(f"Dumping shower data to {path}")

        with io.open(path, "w") as root:
            shower_node = root.branch(f"montecarlo/shower/{version}")
            for k, v in self.__dict__.items():
                if k != "fields" and (k[0] != "_"):
                    shower_node.write(k, v)

            if self.fields is not None:
                for antenna, field in self.fields.items():
                    _logger.debug(f"Dumping field for antenna {antenna}")
                    with shower_node.branch(f"fields/{antenna}") as n:
                        n.write("r", field.r, unit="m")
                        n.write("t", field.t, unit="ns")
                        n.write("E", field.E, unit="uV/m")

        if self.fields is not None:
            n = len(self.fields)
            _logger.info(f"Dumped {n} field(s) to {path}")
