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
from ..antenna import ElectricField, Voltage
from ...import io
from ...tools.coordinates import ECEF, LTP, Rotation

__all__ = ['CollectionEntry', 'FieldsCollection', 'ShowerEvent']


_logger = getLogger(__name__)


@dataclass
class CollectionEntry:
    electric: Optional[ElectricField] = None
    voltage: Optional[Voltage] = None

    @classmethod
    def load(cls, node: DataNone) -> CollectionEntry:
        try:
            subnode = node['electric']
        except KeyError:
            electric = None
        else:
            electric = ElectricField.load(subnode)

        try:
            subnode = node['voltage']
        except KeyError:
            voltage = None
        else:
            voltage = Voltage.load()

        return cls(electric, voltage)

    def dump(self, node:DataNode) -> None:
        if self.electric is not None:
            self.electric.dump(node.branch('electric'))
        if self.voltage is not None:
            self.voltage.dump(node.branch('voltage'))


class FieldsCollection(OrderedDict, MutableMapping[int, CollectionEntry]):
    pass


@dataclass
class ShowerEvent:
    energy: Optional[u.Quantity] = None
    zenith: Optional[u.Quantity] = None
    azimuth: Optional[u.Quantity] = None
    primary: Optional[ParticleCode] = None

    frame: Optional[BaseCoordinateFrame] = None

    core: Optional[CartesianRepresentation] = None
    geomagnet: Optional[CartesianRepresentation] = None
    maximum: Optional[CartesianRepresentation] = None

    fields: Optional[FieldsCollection] = None

    @classmethod
    def load(cls, source: Union[Path, str, io.DataNode]) -> ShowerEvent:
        baseclass = cls

        if type(source) == io.DataNode:
            source = cast(io.DataNode, source)
            filename = f'{source.filename}:{source.path}'
            loader = '_from_datanode'
        else:
            source = cast(Union[Path, str], source)
            filename = f'{source}:/'
            source = Path(source)
            if source.is_dir():
                loader = '_from_dir'

                if not hasattr(cls, loader):
                    # Detection of the simulation engine. Lazy imports are used
                    # in order to avoid circular references
                    from .coreas import CoreasShower
                    from .zhaires import ZhairesShower

                    if CoreasShower._check_dir(source):
                        baseclass = CoreasShower
                    elif ZhairesShower._check_dir(source):
                        baseclass = ZhairesShower
            else:
                loader = f'_from_datafile'

        _logger.info(f'Loading shower data from {filename}')

        try:
            load = getattr(baseclass, loader)
        except AttributeError:
            raise NotImplementedError(f'Invalid data format')
        else:
            self = load(source)

        if self.fields is not None:
            _logger.info(f'Loaded {len(self.fields)} field(s) from {filename}')

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
            fields_node = node['fields']
        except KeyError:
            pass
        else:
            fields:OrderedDict = OrderedDict()
            kwargs['fields'] = fields

            for antenna_node in fields_node:
                antenna = int(antenna_node.name)
                fields[antenna] = CollectionEntry.load(antenna_node)

        return cls(**kwargs)

    def dump(self, source: Union[Path, str, io.DataNode]) -> None:
        if type(source) == io.DataNode:
            source = cast(io.DataNode, source)
            self._to_datanode(source)
        else:
            source = cast(Union[Path, str], source)
            with io.open(source, 'w') as root:
                self._to_datanode(root)

    def _to_datanode(self, node: io.DataNode):
        _logger.info(f'Dumping shower data to {node.filename}:{node.path}')

        for f in fields(self):
            k = f.name
            if k != 'fields' and (k[0] != '_'):
                v = getattr(self, k)
                if v is not None:
                    node.write(k, v)

        if self.fields is not None:
            for antenna, field in self.fields.items():
                with node.branch(f'fields/{antenna}') as n:
                    field.dump(n)

            m = len(self.fields)
            _logger.info(f'Dumped {m} field(s) to {node.filename}:{node.path}')

    def localize(self, latitude: u.Quantity, longitude: u.Quantity,
            height: Optional[u.Quantity]=None,
            declination: Optional[u.Quantity]=None,
            obstime: Union[datetime, Time, str, None]=None) -> None:

        if height is None:
            height = 0 * u.m

        location = ECEF(latitude, longitude, height,
                        representation_type='geodetic')
        self.frame = LTP(location=location, orientation='NWU', magnetic=True,
                         declination=declination, obstime=obstime)

    def shower_frame(self) -> BaseCoordinateFrame:
        ev = self.core - self.maximum
        ev /= ev.norm()
        evB = ev.cross(self.geomagnet)
        evB /= evB.norm()
        evvB = ev.cross(evB)

        r = Rotation.from_matrix(numpy.array((evB.xyz, evvB.xyz, ev.xyz)).T)
        return self.frame.rotated(r)

    def transform(self, representation: BaseRepresentation,
                        frame: BaseCoordinateFrame) -> BaseCoordinateFrame:
        if frame == self.frame:
            return self.frame.realize_frame(representation)
        else:
            coordinates = self.frame._replicate(representation, copy=False)
            return coordinates.transform_to(frame)
