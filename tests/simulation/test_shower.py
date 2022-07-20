"""
Unit tests for the grand.simu.shower module
"""

from collections import OrderedDict
from pathlib import Path
import tarfile
import unittest

import numpy

from grand import store, io, LTP, CartesianRepresentation, SphericalRepresentation
from grand.simu import CoreasShower, ElectricField, ShowerEvent, ZhairesShower
from grand.simu.shower.pdg import ParticleCode
from grand.simu.shower.generic import CollectionEntry
from tests import TestCase


class ShowerTest(TestCase):
    """Unit tests for the shower module"""

    path = Path("shower.hdf5")

    def tearDown(self):
        self.path.unlink()

    @staticmethod
    def get_data(tag):
        """Get test data from the store"""
        path = Path(__file__).parent / f"data/{tag}"

        if not path.exists():
            tgz_name = f"{tag}-test.tar"
            tgz_path = path / (tgz_name + ".gz")
            tgz = store.get(tgz_name)
            path.mkdir(parents=True)
            with tgz_path.open("wb") as f:
                f.write(tgz)
            with tarfile.open(tgz_path, "r|*") as tar:
                tar.extractall(path)
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
            "primary": ParticleCode.PROTON,
            "energy": 1e9,  # * u.GV,
            "zenith": 85,  # * u.deg,
            "azimuth": 0,  # * u.deg
        }
        shower = ShowerEvent(**settings)
        shower.dump(self.path)
        tmp = ShowerEvent.load(self.path)
        for k in settings.keys():
            self.assertEqual(getattr(shower, k), getattr(tmp, k))
        self.tearDown()

        fields = OrderedDict()
        electric = ElectricField(
            numpy.array((0, 1, 2)),  # * u.ns,
            CartesianRepresentation(
                x=numpy.array((1, 0, 0)),  # * u.uV / u.m,
                y=numpy.array((0, 1, 0)),  # * u.uV / u.m,
                z=numpy.array((0, 0, 1)),  # * u.uV / u.m
            ),
            CartesianRepresentation(x=1, y=2, z=3),
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
        self.assertAlmostEqual(numpy.linalg.norm(shower.geomagnet), 0.054021)
        spherical = SphericalRepresentation(shower.geomagnet)
        self.assertQuantity(spherical.theta, numpy.array([147.43]))
        self.assertIsNotNone(shower.core)
        self.assertIsNotNone(shower.maximum)
        self.assertEqual(shower.frame.declination, 0.72)

        shower = ZhairesShower.load(path)
        shower.dump(self.path)
        tmp = shower.load(self.path)

        a, b = shower.fields[9], tmp.fields[9]
        self.assertField(a, b)

        frame = shower.shower_frame()
        ev = shower.core - shower.maximum
        ev /= numpy.linalg.norm(ev)
        ev = ev.T[0]
        evB = numpy.cross(ev, shower.geomagnet.T[0])
        evB /= numpy.linalg.norm(evB)
        evvB = numpy.cross(ev, evB)  # evB is already transposed.

        evB = LTP(x=evB[0], y=evB[1], z=evB[2], frame=shower.frame)
        evB = evB.ltp_to_ltp(frame)
        self.assertArray(evB, numpy.array((1, 0, 0)), 7)
        evvB = LTP(x=evvB[0], y=evvB[1], z=evvB[2], frame=shower.core)
        evvB = evvB.ltp_to_ltp(frame)
        self.assertArray(evvB, numpy.array((0, 1, 0)), 7)
        ev = LTP(
            x=ev[0], y=ev[1], z=ev[2], frame=shower.core
        )  # TODO: this step is redundant as ev is already in LTP. Fix this.
        ev = ev.ltp_to_ltp(frame)
        self.assertArray(ev, numpy.array((0, 0, 1)), 7)


if __name__ == "__main__":
    unittest.main()
