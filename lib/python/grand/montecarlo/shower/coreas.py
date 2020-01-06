from collections import OrderedDict
from logging import getLogger
import os
from pathlib import Path

import astropy.constants
from astropy.coordinates import CartesianRepresentation
import astropy.units as u
import numpy

from .generic import Field, Shower

__all__ = ["CoreasShower"]


logger = getLogger(__name__)


"""CORSIKA particles ids"""
_id_to_name = {
    14:   "p",
    5626: "Fe"
}


class CoreasShower(Shower):
    @classmethod
    def _from_dir(cls, path: Path) -> Shower:
        if not path.exists():
            raise FileNotFoundError(path)

        positions = {}
        try:
            info_file = path.glob("*.info").__next__()
        except StopIteration:
            pass
        else:
            with info_file.open() as f:
                lines = f.read().split(os.linesep)
            for line in lines:
                if not line: continue
                words = line.split()
                if words[0] == "ANTENNA":
                    antenna = int(words[1])
                    positions[antenna] = CartesianRepresentation(
                        x = float(words[2]) * u.m,
                        y = float(words[3]) * u.m,
                        z = float(words[4]) * u.m
                    )

        fields = {}
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
                fields[antenna] = Field(
                    positions[antenna],
                    t,
                    CartesianRepresentation(Ex, Ey, Ez)
                )

            ordered = OrderedDict()
            for key in sorted(fields.keys()):
                ordered[key] = fields[key]
            fields = ordered

        inp = {}
        try:
            inp_path = path.glob("inp/*.inp").__next__()
        except StopIteration:
            raise FileNotFoundError(path / "inp/*.inp")
        else:
            with inp_path.open() as f:
                lines = f.read().split(os.linesep)
            for line in lines:
                if not line: continue
                words = line.split()
                try:
                    tag, convert = {
                        "ERANGE": ("energy", lambda x: float(x) * u.GeV),
                        "THETAP": ("zenith", lambda x: float(x) * u.deg),
                        "PHIP":   ("azimuth", lambda x: float(x) * u.deg),
                        "PRMPAR": ("primary", lambda x: _id_to_name[int(x)])
                    }[words[0]]
                except KeyError:
                    pass
                else:
                    inp[tag] = convert(words[1])

        return cls(fields=fields, **inp)
