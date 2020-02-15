from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from logging import getLogger
import os
from pathlib import Path
from typing import Dict, Optional

import astropy.constants
from astropy.coordinates import CartesianRepresentation
from astropy.time import Time
import astropy.units as u
import numpy

from .generic import FieldsCollection, ShowerEvent
from ..antenna import ElectricField
from ..pdg import ParticleCode
from ...tools.coordinates import ECEF, LTP

__all__ = ["CoreasShower"]


logger = getLogger(__name__)


"""CORSIKA particle Id to PDG code"""
_id_to_code: Dict[int, ParticleCode] = {
    14:   ParticleCode.PROTON,
    5626: ParticleCode.IRON
}


class CoreasShower(ShowerEvent):
    @classmethod
    def _check_dir(cls, path: Path) -> bool:
        try:
            info_file = path.glob("*.reas").__next__()
        except StopIteration:
            return False
        return True

    @classmethod
    def _from_dir(cls, path: Path) -> CoreasShower:
        if not path.exists():
            raise FileNotFoundError(path)

        positions = {}
        try:
            info_file = path.glob("*.info").__next__()
        except StopIteration:
            pass
        else:
            with info_file.open() as f:
                for line in f:
                    if not line: continue
                    words = line.split()
                    if words[0] == "ANTENNA":
                        antenna = int(words[1])
                        positions[antenna] = CartesianRepresentation(
                            x = float(words[2]) * u.m,
                            y = float(words[3]) * u.m,
                            z = float(words[4]) * u.m
                        )

        fields: Optional[FieldsCollection] = None
        raw_fields = {}
        try:
            fields_path = path.glob("*_coreas").__next__()
        except StopIteration:
            pass
        else:
            cgs2si = (
                astropy.constants.c / (u.m / u.s)).value * 1E+02 * u.uV / u.m
            for antenna_path in fields_path.glob("*.dat"):
                antenna = int(antenna_path.name[5:].split(".", 1)[0])
                logger.debug(f"Loading trace for antenna {antenna}")
                data = numpy.loadtxt(antenna_path)
                t  = data[:,0] * u.ns
                Ex = data[:,1] * cgs2si
                Ey = data[:,2] * cgs2si
                Ez = data[:,3] * cgs2si
                raw_fields[antenna] = ElectricField(
                    t,
                    CartesianRepresentation(Ex, Ey, Ez),
                    positions[antenna]
                )

            fields = FieldsCollection()
            for key in sorted(raw_fields.keys()):
                fields[key] = raw_fields[key]

        inp = {}
        try:
            inp_path = path.glob("inp/*.inp").__next__()
        except StopIteration:
            raise FileNotFoundError(path / "inp/*.inp")
        else:
            converters = {
                "ERANGE": ("energy", lambda x: float(x) * u.GeV),
                "THETAP": ("zenith", lambda x: float(x) * u.deg),
                "PHIP":   ("azimuth", lambda x: float(x) * u.deg),
                "PRMPAR": ("primary", lambda x: _id_to_code[int(x)])
            }

            with inp_path.open() as f:
                for line in f:
                    if not line: continue
                    words = line.split()
                    try:
                        tag, convert = converters[words[0]]
                    except KeyError:
                        pass
                    else:
                        inp[tag] = convert(words[1])

        try:
            reas_path = path.glob("*.reas").__next__()
        except StopIteration:
            raise FileNotFoundError(path / "*.reas")
        else:
            tags = (
                "CoreCoordinateNorth", "CoreCoordinateWest",
                "CoreCoordinateVertical", "DistanceOfShowerMaximum"
            )

            index, values = 0, numpy.empty(len(tags))
            target = tags[index]
            with reas_path.open() as f:
                for line in f:
                    try:
                        tag, value = line.split(" = ")
                    except ValueError:
                        continue
                    if tag != target: continue

                    values[index] = float(value.split(";", 1)[0])

                    index += 1
                    try:
                        target = tags[index]
                    except IndexError:
                        break

            core = CartesianRepresentation(values[0], values[1], values[2],
                                           unit="cm")
            distance = values[3] * u.cm
            theta, phi = inp["zenith"], inp["azimuth"]
            ct, st = numpy.cos(theta), numpy.sin(theta)
            direction = CartesianRepresentation( # XXX is this correct?
                st * numpy.cos(phi), st * numpy.sin(phi), ct)
            inp["maximum"] = core + distance * direction

        return cls(fields=fields, **inp)


    def localize(self, latitude: u.Quantity, longitude: u.Quantity,
                 height: Optional[u.Quantity]=None,
                 obstime: Union[datetime, Time, str, None]=None) -> None:

        if height is None:
            height = 0 * u.m

        location = ECEF(latitude, longitude, height,
                        representation_type="geodetic")
        self.frame = LTP(location=location, orientation="NWU", magnetic=True,
                         obstime=obstime)
        # XXX Is this the frame used by CoREAS?
