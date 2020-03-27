"""
Unit tests for the grand.simulation.shower module
"""

from collections import OrderedDict
from pathlib import Path
import tarfile
import unittest

from astropy.coordinates import CartesianRepresentation, SphericalRepresentation
import astropy.units as u
import numpy

from grand import store, io, LTP
from grand.simulation import CoreasShower, ElectricField, ShowerEvent,         \
                             ZhairesShower
from grand.simulation.pdg import ParticleCode
from grand.simulation.shower.generic import CollectionEntry
from tests import TestCase


class ShowerTest(TestCase):
    """Unit tests for the shower module"""

    path = Path("shower.hdf5")

    def tearDown(self):
        self.path.unlink()

    @staticmethod
    def get_data(tag):
        """Get test data from the store
        """
        path = Path(__file__).parent / f"data/{tag}"

        if not path.exists():
            tgz_name = f"{tag}-test.tar"
            tgz_path = path / (tgz_name + ".gz")
            tgz = store.get(tgz_name)
            path.mkdir(parents=True)
            with tgz_path.open("wb") as f: f.write(tgz)
            with tarfile.open(tgz_path, "r|*") as tar: tar.extractall(path)
            tgz_path.unlink()

        return path

    def assertField(self, a, b):
        self.assertEqual(a.voltage, None)
        self.assertEqual(b.voltage, None)
        a, b = a.electric, b.electric
        self.assertCartesian(a.r, b.r, 4)
        self.assertQuantity(a.t, b.t, 7)
        self.assertCartesian(a.E, b.E, 5)

    def test_generic(self):
        settings = {
            "primary" : ParticleCode.PROTON,
            "energy"  : 1E+18 * u.eV,
            "zenith"  : 85 * u.deg,
            "azimuth" : 0 * u.deg
        }
        shower = ShowerEvent(**settings)
        shower.dump(self.path)
        tmp = ShowerEvent.load(self.path)
        for k in settings.keys():
            self.assertEqual(getattr(shower, k), getattr(tmp, k))
        self.tearDown()

        fields = OrderedDict()
        electric = ElectricField(
            numpy.array((0, 1, 2)) * u.ns,
            CartesianRepresentation(
                numpy.array((1, 0, 0)) * u.uV / u.m,
                numpy.array((0, 1, 0)) * u.uV / u.m,
                numpy.array((0, 0, 1)) * u.uV / u.m
            ),
            CartesianRepresentation(1 * u.m, 2 * u.m, 3 * u.m),
        )
        fields[1] = CollectionEntry(electric)
        shower = ShowerEvent(fields=fields, **settings)
        shower.dump(self.path)
        tmp = ShowerEvent.load(self.path)

        def compare_showers():
            a, b = shower.fields[1], tmp.fields[1]
            self.assertField(a, b)

        compare_showers()

        nodepath = "montecarlo/shower"
        with io.open(self.path, "w") as root:
            node = root.branch(nodepath)
            shower.dump(node)

        with io.open(self.path) as root:
            node = root[nodepath]
            tmp = ShowerEvent.load(node)

        compare_showers()

    def test_coreas(self):
        path = self.get_data("coreas")
        shower = ShowerEvent.load(path)
        self.assertIs(shower.frame, None)

        shower = CoreasShower.load(path)
        shower.dump(self.path)
        tmp = shower.load(self.path)

        a, b = shower.fields[9], tmp.fields[9]
        self.assertField(a, b)

        pos0 = CoreasShower._parse_coreas_bins(path, 9000)
        pos1 = CoreasShower._parse_list(path, 9000)
        self.assertIsNotNone(pos0)
        for i, (antenna0, r0) in enumerate(pos0):
            antenna1, r1 = pos1[i]
            self.assertEqual(antenna0, antenna1)
            self.assertQuantity(r0, r1)

    def test_zhaires(self):
        path = self.get_data("zhaires")
        shower = ShowerEvent.load(path)
        self.assertIsInstance(shower.frame, LTP)
        self.assertIsNotNone(shower.geomagnet)
        self.assertQuantity(shower.geomagnet.norm(), 54.021 << u.uT)
        spherical = shower.geomagnet.represent_as(SphericalRepresentation)
        self.assertQuantity(spherical.lat, -57.43 << u.deg)
        self.assertIsNotNone(shower.core)
        self.assertIsNotNone(shower.maximum)
        self.assertQuantity(shower.frame.declination, 0.72 << u.deg)

        shower = ZhairesShower.load(path)
        shower.dump(self.path)
        tmp = shower.load(self.path)

        a, b = shower.fields[9], tmp.fields[9]
        self.assertField(a, b)

        frame = shower.shower_frame()
        ev = shower.core - shower.maximum
        ev /= ev.norm()
        evB = ev.cross(shower.geomagnet)
        evB /= evB.norm()
        ev = shower.transform(ev, frame).cartesian.xyz.value
        self.assertArray(ev, numpy.array((0, 0, 1)))
        evB = shower.transform(evB, frame).cartesian.xyz.value
        self.assertArray(evB, numpy.array((1, 0, 0)))


if __name__ == "__main__":
    unittest.main()
