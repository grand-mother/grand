from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import cast, MutableMapping, Optional, Union
from datetime import datetime

import np as np

from grand.simu.shower.pdg import ParticleCode
from grand.simu import ElectricField, Voltage
from grand.io import io_node as io
from grand.io.root.event.shower import ShowerEventSimdataTree
from grand.geo.coordinates import (
    Geodetic,
    LTP,
    GRANDCS,
    CartesianRepresentation,
)  # RK
from fontTools.ttLib.tables import D_S_I_G_

__all__ = ["CollectionEntry", "FieldsCollection", "ShowerEvent"]


logger = getLogger(__name__)


@dataclass
class CollectionEntry:
    electric: Optional[ElectricField] = None
    voltage: Optional[Voltage] = None

    @classmethod
    def load(cls, node: io.DataNode) -> CollectionEntry:
        try:
            subnode = node["electric"]
        except KeyError:
            electric = None
        else:
            electric = ElectricField.load(subnode)

        try:
            subnode = node["voltage"]
        except KeyError:
            voltage = None
        else:
            voltage = Voltage.load(node)

        return cls(electric, voltage)

    def dump(self, node: io.DataNode) -> None:
        if self.electric is not None:
            self.electric.dump(node.branch("electric"))
        if self.voltage is not None:
            self.voltage.dump(node.branch("voltage"))


class FieldsCollection(OrderedDict, MutableMapping[int, CollectionEntry]):
    pass


@dataclass
class ShowerEvent:
    energy: Optional[float] = None
    zenith: Optional[float] = None
    azimuth: Optional[float] = None
    primary: Optional[ParticleCode] = None
    frame: Optional[Union[GRANDCS, LTP]] = None
    core: Optional[CartesianRepresentation] = None
    geomagnet: Optional[CartesianRepresentation] = None
    maximum: Optional[CartesianRepresentation] = None
    ground_alt: float = 0.0
    fields: Optional[FieldsCollection] = None

    def load_root(self, d_shower):
        self.energy = d_shower.prim_energy
        self.zenith = d_shower.shower_zenith
        self.zenith = 0
        self.azimuth = d_shower.shower_azimuth
        self.azimuth = 0
        self.primary = d_shower.prim_type
        #TODO: DC1 find information in file ROOT
        self.localize(39.5, 90.5)
        xmax = d_shower.xmax_pos_shc
        xmax= np.array([150750.  ,    0.,  15560.], dtype=np.float32)
        logger.debug(xmax)
        self.maximum = LTP(x=xmax[0], y=xmax[1], z=xmax[2], frame=self.frame)

    @classmethod
    def load(cls, source: Union[Path, str, io.DataNode]) -> ShowerEvent:
        baseclass = cls
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
                if not hasattr(cls, loader):
                    # Detection of the simulation engine. Lazy imports are used
                    # in order to avoid circular references
                    from grand.simu import CoreasShower
                    from grand.simu import ZhairesShower

                    if CoreasShower.check_dir(source):
                        baseclass = CoreasShower
                    elif ZhairesShower.check_dir(source):
                        baseclass = ZhairesShower
            else:
                loader = "_from_datafile"
        logger.info(f"Loading shower data from {filename}")
        try:
            load = getattr(baseclass, loader)
        except AttributeError as load_exit:
            raise NotImplementedError("Invalid data format") from load_exit
        else:
            self = load(source)
        if self.fields is not None:
            logger.info(f"Loaded {len(self.fields)} field(s) from {filename}")
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
            l_fields: OrderedDict = OrderedDict()
            kwargs["fields"] = l_fields

            for antenna_node in fields_node:
                antenna = int(antenna_node.name)
                l_fields[antenna] = CollectionEntry.load(antenna_node)

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
        logger.info(f"Dumping shower data to {node.filename}:{node.path}")

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
            logger.info(f"Dumped {m} field(s) to {node.filename}:{node.path}")

    def localize(
        self,
        latitude,
        longitude,
        height=0,
        declination: Optional[float] = None,
        obstime: Union[str, datetime] = "2020-01-01",
    ) -> None:
        location = Geodetic(latitude=latitude, longitude=longitude, height=height)  # RK
        self.frame = LTP(
            location=location,
            orientation="NWU",
            magnetic=True,
            declination=declination,
            obstime=obstime,
        )

    def shower_frame(self):
        # Idea: Change the basis vectors by vectors pointing towards evB, evvB, and ev
        ev = self.core - self.maximum
        ev /= np.linalg.norm(ev)
        ev = ev.T[0]  # [[x], [y], [z]] --> [x, y, z]
        evB = np.cross(ev, self.geomagnet.T[0])
        evB /= np.linalg.norm(evB)
        evvB = np.cross(ev, evB)

        # change these unit vector from 'NWU' LTP frame to ECEF frame.
        # RK TODO: Going back to ECEF frame is a common process for vectors.
        #          Develop a function to do this.
        ev = np.matmul(self.frame.basis.T, ev)
        evB = np.matmul(self.frame.basis.T, evB)
        evvB = np.matmul(self.frame.basis.T, evvB)
        self.frame.basis = np.vstack((evB, evvB, ev))

        return self.frame
