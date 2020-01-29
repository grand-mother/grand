from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import cast, MutableMapping, Optional, Union

from astropy.coordinates import BaseCoordinateFrame, CartesianRepresentation
import astropy.units as u
import numpy

from ..pdg import ParticleCode
from ..antenna import ElectricField
from ...import io

__all__ = ["FieldsCollection", "ShowerEvent"]


_logger = getLogger(__name__)


class FieldsCollection(OrderedDict, MutableMapping[int, ElectricField]):
    pass


@dataclass
class ShowerEvent:
    energy: Optional[u.Quantity] = None
    zenith: Optional[u.Quantity] = None
    azimuth: Optional[u.Quantity] = None
    primary: Optional[ParticleCode] = None

    frame: Optional[BaseCoordinateFrame] = None
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
                fields[antenna] = ElectricField.load(antenna_node)

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
