from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
import re
from typing import Any, Dict, Optional

from astropy.coordinates import BaseCoordinateFrame, CartesianRepresentation,  \
                                PhysicsSphericalRepresentation
import astropy.units as u
import h5py
import numpy

from .generic import CollectionEntry, FieldsCollection, ShowerEvent
from ..antenna import ElectricField
from ..pdg import ParticleCode
from ...tools.coordinates import ECEF, LTP

__all__ = ['InvalidAntennaName', 'ZhairesShower']


logger = getLogger(__name__)


class InvalidAntennaName(ValueError):
    pass


class ZhairesShower(ShowerEvent):
    @classmethod
    def _check_dir(cls, path: Path) -> bool:
        try:
            info_file = path.glob('*.sry').__next__()
        except StopIteration:
            return False
        return True

    @classmethod
    def _from_dir(cls, path: Path) -> ZhairesShower:
        if not path.exists():
            raise FileNotFoundError(path)

        positions = {}
        ant_file = path / 'antpos.dat'
        if ant_file.exists():
            pattern = re.compile('A([0-9]+)$')
            with ant_file.open() as f:
                for line in f:
                    if not line: continue
                    words = line.split()

                    match = pattern.search(words[1])
                    if match is None:
                        raise InvalidAntennaName(words[1])
                    antenna = int(match.group(1))

                    positions[antenna] = CartesianRepresentation(
                        x = float(words[2]) * u.m,
                        y = float(words[3]) * u.m,
                        z = float(words[4]) * u.m
                    )

        fields: Optional[FieldsCollection] = None
        raw_fields = {}
        for field_path  in path.glob('a*.trace'):
            antenna = int(field_path.name[1:].split('.', 1)[0])
            logger.debug(f'Loading trace for antenna {antenna}')
            data = numpy.loadtxt(field_path)
            uVm = u.uV / u.m
            t  = data[:,0] * u.ns
            Ex = data[:,1] * uVm
            Ey = data[:,2] * uVm
            Ez = data[:,3] * uVm
            electric = ElectricField(
                t,
                CartesianRepresentation(Ex, Ey, Ez),
                positions[antenna]
            )
            raw_fields[antenna] = CollectionEntry(electric)

        if raw_fields:
            fields = FieldsCollection()
            for key in sorted(raw_fields.keys()):
                fields[key] = raw_fields[key]

        inp: Dict[str, Any] = {}
        inp['core'] = CartesianRepresentation(0, 0, 0, unit='m')
        try:
            sry_path = path.glob('*.sry').__next__()
        except StopIteration:
            raise FileNotFoundError(path / '*.sry')
        else:
            def parse_primary(string: str) -> ParticleCode:
                return {
                    'Proton': ParticleCode.PROTON,
                    'Iron': ParticleCode.IRON
                }[string.strip()]

            def parse_quantity(string: str) -> u.Quantity:
                words = string.split()
                return float(words[0]) * u.Unit(words[1])

            def parse_frame_location(string: str) -> BaseCoordinateFrame:
                lat, lon = string.split('Long:')
                lat = parse_quantity(lat[:-2])
                lon = parse_quantity(lon[:-3])
                return ECEF(lat, lon, 0 * u.m, representation_type='geodetic')

            def parse_date(string: str) -> datetime:
                return datetime.strptime(string.strip(), '%d/%b/%Y')

            def parse_frame_direction(string: str) -> BaseCoordinateFrame:
                inp['_origin'] = inp['frame']

                string = string.strip()
                if string == 'Local magnetic north':
                    return 'NWU'
                else:
                    raise NotImplementedError(string)

            def parse_geomagnet_intensity(string: str) -> u.Quantity:
                return float(string.split()[0]) << u.uT

            def parse_geomagnet_angles(string: str) -> CartesianRepresentation:
                intensity = inp['geomagnet']
                inclination, _, _, declination, _ = string.split()
                theta = (90 + float(inclination)) << u.deg
                inp['_declination'] = float(declination) << u.deg
                spherical = PhysicsSphericalRepresentation(theta=theta,
                    phi=0 << u.deg, r=intensity)
                return spherical.represent_as(CartesianRepresentation)

            def parse_maximum(string: str) -> CartesianRepresentation:
                _, _, *xyz = string.split()
                x, y, z = map(float, xyz)
                return CartesianRepresentation(x * u.km, y * u.km, z * u.km)

            converters = (
                ('(Lat', 'frame', parse_frame_location),
                ('Date', '_obstime', parse_date),
                ('Primary particle', 'primary', parse_primary),
                ('Primary energy', 'energy', parse_quantity),
                ('Primary zenith angle', 'zenith', parse_quantity),
                ('Primary azimuth angle', 'azimuth', parse_quantity),
                ('Zero azimuth direction', 'frame', parse_frame_direction),
                ('Geomagnetic field: Intensity:', 'geomagnet',
                    parse_geomagnet_intensity),
                ('I:', 'geomagnet', parse_geomagnet_angles),
                ('Location of max.(Km)', 'maximum', parse_maximum)
            )

            i = 0
            tag, k, convert = converters[i]
            with sry_path.open() as f:
                for line in f:
                    start = line.find(tag)
                    if start < 0: continue

                    inp[k] = convert(line[start+len(tag)+1:])
                    i = i + 1
                    try:
                        tag, k, convert = converters[i]
                    except IndexError:
                        break

        origin = inp.pop('_origin')
        declination = inp.pop('_declination')
        obstime = inp.pop('_obstime')
        orientation = inp['frame']
        inp['frame'] = LTP(location=origin, orientation=orientation,
                           declination=declination, obstime=obstime)

        return cls(fields=fields, **inp)


    @classmethod
    def _from_datafile(cls, path: Path) -> ZhairesShower:
        with h5py.File(path, 'r') as fd:
            if not 'RunInfo.__table_column_meta__' in fd['/']:
                return super()._from_datafile(path)

            for name in fd['/'].keys():
                if not name.startswith('RunInfo'):
                    break

            event = fd[f'{name}/EventInfo']
            antennas = fd[f'{name}/AntennaInfo']
            traces = fd[f'{name}/AntennaTraces']

            fields = FieldsCollection()

            pattern = re.compile('([0-9]+)$')
            for tag, x, y, z, *_ in antennas:
                tag = tag.decode()
                antenna = int(pattern.search(tag)[1])
                r = CartesianRepresentation(
                    float(x), float(y), float(z), unit=u.m)
                tmp = traces[f'{tag}/efield'][:]
                efield = tmp.view('f4').reshape(tmp.shape + (-1,))
                t = numpy.asarray(efield[:,0], 'f8') << u.ns
                Ex = numpy.asarray(efield[:,1], 'f8') << u.uV / u.m
                Ey = numpy.asarray(efield[:,2], 'f8') << u.uV / u.m
                Ez = numpy.asarray(efield[:,3], 'f8') << u.uV / u.m
                E = CartesianRepresentation(Ex, Ey, Ez, copy=False),

                fields[antenna] = CollectionEntry(
                    electric=ElectricField(t = t, E = E, r = r))

            primary = {
                'Fe^56'  : ParticleCode.IRON,
                'Gamma'  : ParticleCode.GAMMA,
                'Proton' : ParticleCode.PROTON
            }[event[0, 'Primary'].decode()]

            geomagnet = PhysicsSphericalRepresentation(
                theta = float(90 + event[0, 'BFieldIncl']) << u.deg,
                phi = 0 << u.deg,
                r = float(event[0, 'BField']) << u.uT)

            try:
                latitude = event[0, 'Latitude'] << u.deg
                longitude = event[0, 'Longitude'] << u.deg
                declination = event[0, 'BFieldDecl'] << u.deg
                obstime = datetime.strptime(event[0, 'Date'].strip(),
                                            '%d/%b/%Y')
            except ValueError:
                frame = None
            else:
                origin = ECEF(latitude, longitude, 0 * u.m,
                              representation_type='geodetic')
                frame = LTP(location=origin, orientation='NWU',
                            declination=declination, obstime=obstime)

            return cls(
                energy = float(event[0, 'Energy']) << u.EeV,
                zenith = (180 - float(event[0, 'Zenith'])) << u.deg,
                azimuth = -float(event[0, 'Azimuth']) << u.deg,
                primary = primary,

                frame = frame,
                core = CartesianRepresentation(0, 0, 0, unit='m'),
                geomagnet = geomagnet.represent_as(CartesianRepresentation),
                maximum = CartesianRepresentation(*event[0, 'XmaxPosition'],
                                                  unit='m'),

                fields = fields
            )
