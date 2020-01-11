from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import cast, MutableMapping, Optional, Union

from astropy.coordinates import CartesianRepresentation
import astropy.units as u
import numpy

from ...import io

__all__ = ["Field", "FieldsCollection", "ShowerEvent"]


_logger = getLogger(__name__)


@dataclass
class Field:
    r: CartesianRepresentation
    t: u.Quantity
    E: CartesianRepresentation

    @classmethod
    def load(cls, node):
        _logger.debug(f"Loading field from {node.filename}:{node.path}")
        r = node.read("r", dtype="f8")
        t = node.read("t", dtype="f8")
        E = node.read("E", dtype="f8")
        return cls(r, t, E)

    def dump(self, node):
        _logger.debug(f"Dumping field to {node.filename}:{node.path}")
        node.write("r", self.r, unit="m", dtype="f4")
        node.write("t", self.t, unit="ns", dtype="f4")
        node.write("E", self.E, unit="uV/m", dtype="f4")


class FieldsCollection(OrderedDict, MutableMapping[int, Field]):
    pass


@dataclass
class ShowerEvent:
    energy: Optional[u.Quantity] = None
    zenith: Optional[u.Quantity] = None
    azimuth: Optional[u.Quantity] = None
    primary: Optional[str] = None

    fields: Optional[FieldsCollection] = None

    @classmethod
    def load(cls, source: Union[Path, str, io.DataNode]) -> ShowerEvent:
        if type(source) == io.DataNode:
            source = cast(io.DataNode, source)
            filename = f"{source.filename}:{source.path}"
            loader = "_from_datanode"
        else:
            source = cast(Union[Path, str], source)
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
    def _from_datafile(cls, path: Path) -> ShowerEvent:
        with io.open(path) as root:
            return cls._from_datanode(root)

    @classmethod
    def _from_datanode(cls, node: io.DataNode) -> ShowerEvent:
        kwargs = {}
        for name, data in node.elements:
            kwargs[name] = data

        try:
            fields_node = node["fields"]
        except KeyError:
            pass
        else:
            fields:OrderedDict = OrderedDict()
            kwargs["fields"] = fields

            for antenna_node in fields_node:
                antenna = int(antenna_node.name)
                fields[antenna] = Field.load(antenna_node)

        return cls(**kwargs)

    def dump(self, source: Union[Path, str, io.DataNode]) -> None:
        if type(source) == io.DataNode:
            source = cast(io.DataNode, source)
            self._to_datanode(source)
        else:
            source = cast(Union[Path, str], source)
            with io.open(source, "w") as root:
                self._to_datanode(root)

    def _to_datanode(self, node: io.DataNode):
        _logger.info(f"Dumping shower data to {node.filename}:{node.path}")

        for f in fields(self):
            k = f.name
            if k != "fields" and (k[0] != "_"):
                v = getattr(self, k)
                if v is not None:
                    node.write(k, v)

        if self.fields is not None:
            for antenna, field in self.fields.items():
                with node.branch(f"fields/{antenna}") as n:
                    field.dump(n)

            m = len(self.fields)
            _logger.info(f"Dumped {m} field(s) to {node.filename}:{node.path}")
