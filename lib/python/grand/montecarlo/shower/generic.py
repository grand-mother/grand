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
    def load(cls, source: Union[Path, str, io.DataNode]) -> Shower:
        if type(source) == io.DataNode:
            filename = f"{source.filename}:{source.path}"
            loader = "_from_datanode"
        else:
            filename = f"{source}:/"
            source = Path(source)
            if source.is_dir():
                loader = "_from_dir"
            else:
                loader = f"_from_datafile"

        _logger.info(f"Loading shower data from {filename}")

        try:
            load = getattr(cls, loader)
        except AttributeError:
            raise ValueError(f"Invalid data format")
        else:
            self = load(source)

        if self.fields is not None:
            _logger.info(f"Loaded {len(self.fields)} field(s) from {filename}")

        return self

    @classmethod
    def _from_datafile(cls, path: Path) -> Shower:
        with io.open(path) as root:
            return cls._from_datanode(root)

    @classmethod
    def _from_datanode(cls, node: io.DataNode) -> Shower:
        kwargs = {}
        for name, data in node.elements:
            kwargs[name] = data

        try:
            fields_node = node["fields"]
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

    def dump(self, source: Union[Path, str, io.DataNode]) -> None:
        if type(source) == io.DataNode:
            self._to_datanode(source)
        else:
            with io.open(source, "w") as root:
                self._to_datanode(root)

    def _to_datanode(self, node: io.DataNode):
        _logger.info(f"Dumping shower data to {node.filename}:{node.path}")

        for k, v in self.__dict__.items():
            if k != "fields" and (k[0] != "_"):
                node.write(k, v)

        if self.fields is not None:
            for antenna, field in self.fields.items():
                _logger.debug(f"Dumping field for antenna {antenna}")
                with node.branch(f"fields/{antenna}") as n:
                    n.write("r", field.r, unit="m")
                    n.write("t", field.t, unit="ns")
                    n.write("E", field.E, unit="uV/m")

            n = len(self.fields)
            _logger.info(f"Dumped {n} field(s) to {node.filename}:{node.path}")
