"""!
Extract informations from ZHAires simulation
"""

from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
import re
from typing import Any, Dict, Optional

import h5py
import numpy

from grand.simu.shower.generic import CollectionEntry, FieldsCollection, ShowerEvent
from grand.simu.antenna import ElectricField
from grand.simu.shower.pdg import ParticleCode
from ...tools.coordinates import ECEF, LTP, Geodetic
from ...tools.coordinates import (
    CartesianRepresentation,
    SphericalRepresentation,
)

__all__ = ["InvalidAntennaName", "ZhairesShower"]


logger = getLogger(__name__)


class InvalidAntennaName(ValueError):
    pass


class ZhairesShower(ShowerEvent):
    @classmethod
    def check_dir(cls, path: Path) -> bool:
        try:
            _ = path.glob("*.sry").__next__()
        except StopIteration:
            return False
        return True

    @classmethod
    def _from_dir(cls, path: Path) -> ZhairesShower:
        """!
        Extract informations about ZHAires simulation from xxx.sry file

        @note:
          Zhaires has a fixed coordinate frame at a location with 'NWU'
          orientation at the sea level:
              - 'N' is the magnetic north,
              - 'W' is 90 deg west from 'N',
              - 'U' is upward towards zenith.

        @param cls: class dict
        @param path (TBD): path of simulation
        """

        def parse_primary(string: str) -> ParticleCode:
            return {"Proton": ParticleCode.PROTON, "Iron": ParticleCode.IRON}[string.strip()]

        def parse_quantity(string: str):
            words = string.split()
            return float(words[0])

        def parse_frame_location(string: str):
            lat, lon = string.split("Long:")
            lat_f = parse_quantity(lat[:-2])
            lon_f = parse_quantity(lon[:-3])
            # Rk. Based on shower-event.py, reference of height is wrt ellipsoid instead of geoid.
            geodetic = Geodetic(latitude=lat_f, longitude=lon_f, height=0, reference="ELLIPSOID")
            return ECEF(geodetic)

        def parse_date(string: str) -> datetime:
            return datetime.strptime(string.strip(), "%d/%b/%Y")

        def parse_frame_direction(string: str):
            inp["_origin"] = inp["frame"]

            string = string.strip()
            if string == "Local magnetic north":
                return "NWU"
            else:
                raise NotImplementedError(string)

        def parse_geomagnet_intensity(string: str):  # -> u.Quantity:
            return float(string.split()[0]) * 1e-3  # uT --> nT

        def parse_geomagnet_angles(string: str) -> CartesianRepresentation:
            intensity = inp["geomagnet"]
            inclination, _, _, declination, _ = string.split()
            theta = 90 + float(inclination)  # deg
            inp["_declination"] = float(declination)  # deg
            # phi=0 because x-axis is parallel to the magnetic north.
            spherical = SphericalRepresentation(theta=theta, phi=0, r=intensity)
            return CartesianRepresentation(spherical)

        def parse_maximum(string: str) -> CartesianRepresentation:
            _, _, *xyz = string.split()
            x, y, z = map(float, xyz)

            ## Xmax is given as CartesianRepresentation defined in the shower frame.
            # Later (below) Xmax is saved wrt LTP frame making it independent of shower info.
            ## "Previously: Dirty hack by OMH for now" -> not necessary now. RK.
            try:
                inp_file = path.glob("*.inp").__next__()
                logger.info("### zhaires.py: reading groundaltitude from. inp file.")
                with open(inp_file, encoding="UTF-8") as f:
                    for line in f:
                        if "GroundAltitude" in line:
                            ground_alt = float(line.split()[1])  # m
                            inp["ground_alt"] = ground_alt
            except StopIteration as parse_maximum_exit:
                raise FileNotFoundError(path / "*.inp") from parse_maximum_exit
            return CartesianRepresentation(x=1000 * x, y=1000 * y, z=1000 * z)  # RK. km --> m

        if not path.exists():
            raise FileNotFoundError(path)
        inp: Dict[str, Any] = {}
        try:
            sry_path = path.glob("*.sry").__next__()
        except StopIteration as from_dir_exit:
            raise FileNotFoundError(path / "*.sry") from from_dir_exit
        else:
            converters = (
                ("(Lat", "frame", parse_frame_location),
                ("Date", "_obstime", parse_date),
                ("Primary particle", "primary", parse_primary),
                ("Primary energy", "energy", parse_quantity),
                ("Primary zenith angle", "zenith", parse_quantity),
                ("Primary azimuth angle", "azimuth", parse_quantity),
                ("Zero azimuth direction", "frame", parse_frame_direction),
                (
                    "Geomagnetic field: Intensity:",
                    "geomagnet",
                    parse_geomagnet_intensity,
                ),
                ("I:", "geomagnet", parse_geomagnet_angles),
                # deprecated 19.04.08:("Location of max.(Km)", "maximum", parse_maximum),
                ("Pos. Max.", "maximum", parse_maximum),
            )
            i = 0
            tag, k, convert = converters[i]
            with sry_path.open() as f:
                for line in f:
                    start = line.find(tag)
                    if start < 0:
                        continue
                    inp[k] = convert(line[start + len(tag) + 1 :])
                    i = i + 1
                    try:
                        tag, k, convert = converters[i]
                    except IndexError:
                        # end of list converters
                        break
        origin = inp.pop("_origin")
        declination = inp.pop("_declination")
        obstime = inp.pop("_obstime")
        orientation = inp["frame"]
        ground_alt = inp["ground_alt"]
        # RK. x, y and z are given in ECEF.
        ecef = ECEF(
            x=origin[0][0],
            y=origin[1][0],
            z=origin[2][0],
        )
        inp["frame"] = LTP(
            location=ecef,
            orientation=orientation,
            declination=declination,
            obstime=obstime,
        )
        inp["core"] = LTP(x=0, y=0, z=ground_alt, frame=inp["frame"])
        # RK. Save Xmax in LTP frame. It will be easier to convert to antenna frame.
        #    But it takes more space (about 8 times/antenna).
        Xmax = inp["maximum"]
        inp["maximum"] = LTP(x=Xmax.x, y=Xmax.y, z=Xmax.z, frame=inp["frame"])  # RK

        # Positions are in LTP frame with origin at shower core. Usually shower frame has 'NWU' orientation,
        # where N=magnetic north. Defined in ..../grand/tests/simulation/data/zhaires/*.sry file.
        positions = {}
        ant_file = path / "antpos.dat"  # Ex: 1 A0  0.00000E+00  2.70450E+02  2.90000E+03
        if ant_file.exists():
            pattern = re.compile("A([0-9]+)$")
            with ant_file.open() as f:
                for line in f:
                    if not line:
                        continue
                    words = line.split()
                    match = pattern.search(words[1])
                    if match is None:
                        raise InvalidAntennaName(words[1])
                    antenna = int(match.group(1))
                    positions[antenna] = CartesianRepresentation(
                        x=float(words[2]),  # m, # x-coordinate from shower core.
                        y=float(words[3]),  # m, # y-coordinate from shower core.
                        z=float(words[4]),  # m, # z-coordinate from shower core.
                    )
                    # print("### Warning: Forcing antenna height = 0m")
                    # RK. Note: this is time consuming but useful.
                    #     CartesianRepresentation~200Bytes/antenna, LTP~900Bytes/antenna.
                    # positions[antenna] = LTP(
                    #    x = float(words[2]), #* u.m, # x-coordinate in LTP frame.
                    #    y = float(words[3]), #* u.m, # y-coordinate in LTP frame.
                    #    z = float(words[4]), #* u.m, # z-coordinate in LTP frame.
                    #    frame = inp['frame']
                    # )
        fields: Optional[FieldsCollection] = None
        raw_fields = {}
        for field_path in path.glob("a*.trace"):
            # Example field_path => ..../grand/tests/simulation/data/zhaires/a1.trace
            #                    =>    time [ns]      Ex [uVm]    Ey [uVm]   Ez [uVm]
            #                    => -1.1463000E+04  -5.723E-05  -1.946E-04  4.324E-04
            antenna = int(field_path.name[1:].split(".", 1)[0])
            # logger.debug(f"Loading trace for antenna {antenna}")
            data = numpy.loadtxt(field_path)
            t = data[:, 0] * 1.0e-9  # ns --> s
            Ex = data[:, 1]  # uVm
            Ey = data[:, 2]  # uVm
            Ez = data[:, 3]  # uVm
            electric = ElectricField(
                t, CartesianRepresentation(x=Ex, y=Ey, z=Ez), positions[antenna]
            )
            raw_fields[antenna] = CollectionEntry(electric)
        if raw_fields:
            fields = FieldsCollection()
            for key in sorted(raw_fields.keys()):
                fields[key] = raw_fields[key]
        return cls(fields=fields, **inp)

    @classmethod
    def _from_datafile(cls, path: Path):
        with h5py.File(path, "r") as fd:
            if not "RunInfo.__table_column_meta__" in fd["/"]:
                return super()._from_datafile(path)
            last_name = ""
            for name in fd["/"].keys():
                last_name = name
                if not name.startswith("RunInfo"):
                    break

            event = fd[f"{last_name}/EventInfo"]
            antennas = fd[f"{last_name}/AntennaInfo"]
            traces = fd[f"{last_name}/AntennaTraces"]

            fields = FieldsCollection()

            pattern = re.compile("([0-9]+)$")
            for tag, x, y, z, *_ in antennas:
                tag = tag.decode()
                # TODO: mypy indicate type error ... disable
                antenna = int(pattern.search(tag)[1])  # type: ignore[index]
                r = CartesianRepresentation(x=float(x), y=float(y), z=float(z))  # RK
                tmp = traces[f"{tag}/efield"][:]
                efield = tmp.view("f4").reshape(tmp.shape + (-1,))
                t = numpy.asarray(efield[:, 0], "f8") * 1.0e-9  # ns --> s
                Ex = numpy.asarray(efield[:, 1], "f8")  # uV/m
                Ey = numpy.asarray(efield[:, 2], "f8")  # uV/m
                Ez = numpy.asarray(efield[:, 3], "f8")  # uV/m
                E = CartesianRepresentation(x=Ex, y=Ey, z=Ez)
                fields[antenna] = CollectionEntry(electric=ElectricField(t=t, E=E, r=r))

            primary = {
                "Fe^56": ParticleCode.IRON,
                "Gamma": ParticleCode.GAMMA,
                "Proton": ParticleCode.PROTON,
            }[event[0, "Primary"].decode()]

            geomagnet = SphericalRepresentation(
                theta=float(90 + event[0, "BFieldIncl"]),  # deg
                phi=0,  # deg
                r=float(event[0, "BField"]) * 1e3,  # uT --> nT
            )
            try:
                latitude = event[0, "Latitude"]  # deg
                longitude = event[0, "Longitude"]  # deg
                declination = event[0, "BFieldDecl"]  # deg
                obstime = datetime.strptime(event[0, "Date"].strip(), "%d/%b/%Y")
            except ValueError:
                frame = None
            else:
                geodetic = Geodetic(latitude=latitude, longitude=longitude, height=0.0)
                origin = ECEF(geodetic)
                frame = LTP(
                    location=origin,
                    orientation="NWU",
                    declination=declination,
                    obstime=obstime,
                )

            return cls(
                energy=float(event[0, "Energy"]) * 1.0e9,  # EeV --> GeV
                zenith=(180 - float(event[0, "Zenith"])),  # deg,
                azimuth=-float(event[0, "Azimuth"]),  # deg,
                primary=primary,
                frame=frame,
                core=LTP(x=0, y=0, z=event["GroundAltitude"], frame=frame),
                geomagnet=CartesianRepresentation(geomagnet),
                maximum=LTP(
                    x=event[0, "XmaxPosition"][0],
                    y=event[0, "XmaxPosition"][1],
                    z=event[0, "XmaxPosition"][2],
                    frame=frame,
                ),  # RK TODO: Check if x,y,z are defined in frame.
                fields=fields,
            )
