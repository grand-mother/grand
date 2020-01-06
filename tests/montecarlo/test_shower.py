"""
Unit tests for the grand.montecarlo.shower module
"""

from collections import OrderedDict
from pathlib import Path
import tarfile
import unittest

from astropy.coordinates import CartesianRepresentation
import astropy.units as u
import numpy

from grand import store, io
from grand.montecarlo.shower import CoreasShower, Field, Shower


class ShowerTest(unittest.TestCase):
    """Unit tests for the Shower class"""

    path = Path("shower.hdf5")

    def tearDown(self):
        self.path.unlink()

    def assertQuantity(self, a, b, tol=9):
        """Check that two astropy.Quantities are consistent"""
        n = a.size
        b = b.to_value(a.unit)
        a = a.value
        if n > 1:
            for i, ai in enumerate(a): self.assertAlmostEqual(ai, b[i], tol)
        else:
            self.assertAlmostEquals(a, b, tol)


    def assertCartesian(self, a, b, tol=9):
        """Check that two CartesianRepresentations are consistent"""
        self.assertQuantity(a.x, b.x, tol)
        self.assertQuantity(a.y, b.y, tol)
        self.assertQuantity(a.z, b.z, tol)


    def test_generic(self):
        settings = {
            "primary" : "p",
            "energy"  : 1E+18 * u.eV,
            "zenith"  : 85 * u.deg,
            "azimuth" : 0 * u.deg
        }
        shower = Shower(**settings)
        shower.dump(self.path)
        tmp = shower.load(self.path)
        for k in settings.keys():
            self.assertEquals(getattr(shower, k), getattr(tmp, k))
        self.tearDown()

        fields = OrderedDict()
        fields[1] = Field(
            CartesianRepresentation(1 * u.m, 2 * u.m, 3 * u.m),
            numpy.array((0, 1, 2)) * u.ns,
            CartesianRepresentation(
                numpy.array((1, 0, 0)) * u.uV / u.m,
                numpy.array((0, 1, 0)) * u.uV / u.m,
                numpy.array((0, 0, 1)) * u.uV / u.m
            )
        )
        shower = Shower(fields=fields, **settings)
        shower.dump(self.path)
        tmp = Shower.load(self.path)

        def compare_showers():
            a, b = shower.fields[1], tmp.fields[1]
            self.assertCartesian(a.r, b.r)
            self.assertQuantity(a.t, b.t)
            self.assertCartesian(a.E, b.E)

        compare_showers()

        nodepath = "montecarlo/shower"
        with io.open(self.path, "w") as root:
            node = root.branch(nodepath)
            shower.dump(node)

        with io.open(self.path) as root:
            node = root[nodepath]
            tmp = Shower.load(node)

        compare_showers()


    def test_coreas(self):
        # Get the test data from the store
        path = Path(__file__).parent / "data/coreas"
        if not path.exists():
            tgz_name = "coreas-test.tar"
            tgz_path = path / (tgz_name + ".gz")
            tgz = store.get(tgz_name)
            path.mkdir(parents=True)
            with tgz_path.open("wb") as f: f.write(tgz)
            with tarfile.open(tgz_path, "r|*") as tar: tar.extractall(path)
            tgz_path.unlink()

        # Test the CoREAS loader
        shower = CoreasShower.load(path)
        shower.dump(self.path)
        tmp = shower.load(self.path)

        a, b = shower.fields[9], tmp.fields[9]
        self.assertCartesian(a.r, b.r, 8)
        self.assertQuantity(a.t, b.t, 7)
        self.assertCartesian(a.E, b.E, 5)


if __name__ == "__main__":
    unittest.main()
