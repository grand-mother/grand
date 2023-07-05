from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from logging import getLogger
#from pathlib import Path
from typing import cast, MutableMapping, Optional, Union
from datetime import datetime

import numpy as np

from grand.sim.shower.pdg import ParticleCode
#from grand.basis.type_trace import ElectricField, Voltage
#from grand.dataio import io_node as io

# TODO: when the unused import grand.io.root_trees is defined test coverage indicate test on it
# from grand.io.root_trees import ShowerEventSimdataTree
from grand.geo.coordinates import (
    Geodetic,
    LTP,
    GRANDCS,
    CartesianRepresentation,
)  # RK
#from fontTools.ttLib.tables import D_S_I_G_

#__all__ = ["CollectionEntry", "FieldsCollection", "ShowerEvent"]
__all__ = ["ShowerEvent"]

logger = getLogger(__name__)

"""
#RK: From previous version of grandlib. This part is no longer used.
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

"""
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
    origin_geoid: Optional[list, np.ndarray] = None
    fields: Optional[OrderedDict] = None

    # Since ROOT is the only format in which GRAND data will be stored in, 
    # so the code to deal with other formats are deleted.
    def load_root(self, d_shower):
        self.energy = d_shower.energy_primary
        self.zenith = d_shower.zenith
        self.azimuth = d_shower.azimuth
        self.primary = d_shower.primary_type
        if self.origin_geoid is not None:
            origin_geoid = Geodetic(
                latitude=self.origin_geoid[0],
                longitude=self.origin_geoid[1],
                height=self.origin_geoid[2])
        else:
            raise Exception("Provide origin_geoid for this shower. Example: shower.origin_geoid=TRun.origin_geoid")
        self.grand_ref_frame = GRANDCS(location=origin_geoid)    # used to define antenna position.
        # define a shower core in GRANDCS. shower_core_pos are given in GRANDCS.
        self.core = GRANDCS(
            x=d_shower.shower_core_pos[0],
            y=d_shower.shower_core_pos[1],
            z=d_shower.shower_core_pos[2],
            location=origin_geoid)  #RK: add obstime=TShowerSim.event_date. Make sure obstime is in string or datetime format.
        # define a shower frame. DU positions and Xmax is defined wrt this frame.
        self.frame = LTP(
            location=self.core,
            orientation="NWU",
            magnetic=True)          #RK: add obstime=TShowerSim.event_date. Make sure obstime is in string or datetime format.

        #logger.info(f"Site position long lat: {s_pos}")
        logger.info(f"Site origin [lat, long, height]: {origin_geoid}")
        xmax = d_shower.xmax_pos_shc
        logger.info(f"xmax in shower coordinate: {xmax}")
        self.maximum = LTP(x=xmax[0], y=xmax[1], z=xmax[2], frame=self.frame)

    """
    # Since ROOT is the only format in which GRAND data will be stored in, 
    # so the code to deal with other formats are deleted. These commented 
    # part of the code is not used anywhere in the grandlib.
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
                    from grand.sim.shower.coreas import CoreasShower
                    from grand.sim.shower.zhaires import ZhairesShower

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
    """
