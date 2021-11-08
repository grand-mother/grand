'''
Unit tests for the grand.simulation.antenna module
'''

from pathlib import Path
import unittest

#import astropy.units as u
#from astropy.coordinates import CartesianRepresentation
import numpy

from grand import ECEF, CartesianRepresentation, LTP
from grand.simulation import Antenna, ElectricField, MissingFrameError,        \
                             TabulatedAntennaModel
from tests import TestCase


class AntennaTest(TestCase):
    '''Unit tests for the antenna module'''

    path = Path('antenna.hdf5')

    def tearDown(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    @property
    def model(self):
        try:
            return self._model
        except AttributeError:
            pass

        path = Path('HorizonAntenna_EWarm_leff_loaded.npy')
        if path.exists():
            self._model = TabulatedAntennaModel.load(path)
            return self._model
        else:
            self.skipTest(f'missing {path}')

    def test_tabulated(self):
        t = self.model.table
        shape = (281, 72, 91)

        self.assertEquals(t.frequency.size, shape[0])
        self.assertEquals(t.phi.size, shape[1])
        self.assertEquals(t.theta.size, shape[2])
        self.assertEqual(t.resistance.shape, shape)
        self.assertEqual(t.reactance.shape, shape)
        self.assertEqual(t.leff_theta.shape, shape)
        self.assertEqual(t.leff_phi.shape, shape)
        self.assertEqual(t.phase_theta.shape, shape)
        self.assertEqual(t.phase_phi.shape, shape)

        self.model.dump(self.path)
        tr = TabulatedAntennaModel.load(self.path).table

        self.assertQuantity(t.frequency, tr.frequency)
        self.assertQuantity(t.phi, tr.phi)
        self.assertQuantity(t.theta, tr.theta)
        self.assertQuantity(t.resistance, tr.resistance)
        self.assertQuantity(t.reactance, tr.reactance)
        self.assertQuantity(t.leff_theta, tr.leff_theta)
        self.assertQuantity(t.leff_phi, tr.leff_phi)
        self.assertQuantity(t.phase_theta, tr.phase_theta)
        self.assertQuantity(t.phase_phi, tr.phase_phi)

    def test_antenna(self):
        ts, delta, Es = 502.5, 5, 100
        t = numpy.linspace(0, 2000, 20001)*1.e-9
        E1 = numpy.zeros(t.shape)
        E1[(t >= ts - 0.5 * delta) & (t <= ts + 0.5 * delta)] = Es

        #t *= u.ns
        E0 = numpy.zeros(t.shape) #* u.uV / u.m
        #E1 *= u.uV / u.m
        E = CartesianRepresentation(x=E1, y=E0, z=E0)

        #direction = CartesianRepresentation(x=1, y=0, z=1)
        xmax    =  LTP(x=1, y=0, z=1, location=ECEF(x=0,y=0,z=0), declination=0, orientation='NWU')
        def check(voltage):
            imin, imax = numpy.argmin(voltage.V), numpy.argmax(voltage.V)
            t0 = 0.5 * (voltage.t[imax] + voltage.t[imin])
            Vpp = voltage.V[imax] - voltage.V[imin]

            self.assertLess(t0 - ts, delta)
            self.assertGreater(Vpp, 6E-02 * Es)

        antenna = Antenna(model=self.model, frame=ECEF(x=0,y=0,z=0))
        field = ElectricField(t, E, frame=ECEF(x=0,y=0,z=0))
        check(antenna.compute_voltage(xmax, field))
        with self.assertRaises(MissingFrameError) as context:
            antenna.compute_voltage(xmax, field, frame=frame)

        antenna = Antenna(model=self.model)
        field = ElectricField(t, E)
        check(antenna.compute_voltage(xmax, field))
        with self.assertRaises(MissingFrameError) as context:
            antenna.compute_voltage(xmax, field, frame=frame)

        antenna = Antenna(model=self.model, frame=ECEF(x=0,y=0,z=0))
        check(antenna.compute_voltage(xmax, field, frame=ECEF(x=0,y=0,z=0)))
        with self.assertRaises(MissingFrameError) as context:
            antenna.compute_voltage(xmax, field)


if __name__ == '__main__':
    unittest.main()
