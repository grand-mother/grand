from __future__ import annotations

from logging import getLogger
from pathlib import Path
import re
from typing import Optional

from astropy.coordinates import CartesianRepresentation
import astropy.units as u
import numpy

from .generic import Field, FieldsCollection, ShowerEvent
from ..pdg import ParticleCode

__all__ = ["InvalidAntennaName", "ZhairesShower"]


logger = getLogger(__name__)


class InvalidAntennaName(ValueError):
    pass


class ZhairesShower(ShowerEvent):
    @classmethod
    def _from_dir(cls, path: Path) -> ZhairesShower:
        if not path.exists():
            raise FileNotFoundError(path)

        positions = {}
        ant_file = path / "antpos.dat"
        if ant_file.exists():
            pattern = re.compile("A([0-9]+)$")
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
        for field_path  in path.glob("a*.trace"):
            antenna = int(field_path.name[1:].split(".", 1)[0])
            logger.debug(f"Loading trace for antenna {antenna}")
            data = numpy.loadtxt(field_path)
            uVm = u.uV / u.m
            t  = data[:,0] * u.ns
            Ex = data[:,1] * uVm
            Ey = data[:,2] * uVm
            Ez = data[:,3] * uVm
            raw_fields[antenna] = Field(
                positions[antenna],
                t,
                CartesianRepresentation(Ex, Ey, Ez)
            )

        if raw_fields:
            fields = FieldsCollection()
            for key in sorted(raw_fields.keys()):
                fields[key] = raw_fields[key]

        inp = {}
        try:
            sry_path = path.glob("*.sry").__next__()
        except StopIteration:
            raise FileNotFoundError(path / "*.sry")
        else:
            def parse_primary(string: str) -> ParticleCode:
                return {
                    "Proton": ParticleCode.PROTON,
                    "Iron": ParticleCode.IRON
                }[string.strip()]

            def parse_quantity(string: str) -> u.Quantity:
                words = string.split()
                return float(words[0]) * u.Unit(words[1])

            converters = (
                ("Primary particle", "primary", parse_primary),
                ("Primary energy", "energy", parse_quantity),
                ("Primary zenith angle", "zenith", parse_quantity),
                ("Primary azimuth angle", "azimuth", parse_quantity)
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

        return cls(fields=fields, **inp)
