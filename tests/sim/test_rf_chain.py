'''
Unit tests for the grand.sim.detector.rf_chain module. RK.
'''

import numpy as np
import unittest
from tests import TestCase

import grand.sim.detector.rf_chain as grfc


def test_compute_for_freqs():
    vf = grfc.VGAFilter()
    vf.compute_for_freqs(vf.freqs_in)
    dbs11 = vf.sparams[:, 1]
    degs11 = vf.sparams[:, 2]
    mags11 = 10 ** (dbs11 / 20)
    res11 = mags11 * np.cos(np.deg2rad(degs11))
    assert np.allclose(res11, grfc.db2reim(dbs11, np.deg2rad(degs11))[0])
    
class RFChainTest(TestCase):

    def test_LowNoiseAmplifier(self):
        obj = grfc.LowNoiseAmplifier()
        self.assertEqual(np.sum(obj.dbs11), 0)
        self.assertEqual(np.sum(obj.dbs21), 0)
        self.assertEqual(np.sum(obj.dbs12), 0)
        self.assertEqual(np.sum(obj.dbs22), 0)
        self.assertEqual(np.sum(obj.s11), 0)
        self.assertEqual(np.sum(obj.s21), 0)
        self.assertEqual(np.sum(obj.s12), 0)
        self.assertEqual(np.sum(obj.s22), 0)
        self.assertEqual(np.sum(obj.ABCD_matrix), 0)
        obj.compute_for_freqs(obj.freqs_in)
        self.assertNotEqual(np.sum(obj.dbs11), 0)
        self.assertNotEqual(np.sum(obj.dbs21), 0)
        self.assertNotEqual(np.sum(obj.dbs12), 0)
        self.assertNotEqual(np.sum(obj.dbs22), 0)
        self.assertNotEqual(np.sum(obj.s11), 0)
        self.assertNotEqual(np.sum(obj.s21), 0)
        self.assertNotEqual(np.sum(obj.s12), 0)
        self.assertNotEqual(np.sum(obj.s22), 0)
        self.assertNotEqual(np.sum(obj.ABCD_matrix), 0)

    def test_BalunAfterLNA(self):
        obj = grfc.BalunAfterLNA()
        self.assertEqual(np.sum(obj.s11), 0)
        self.assertEqual(np.sum(obj.s21), 0)
        self.assertEqual(np.sum(obj.s12), 0)
        self.assertEqual(np.sum(obj.s22), 0)
        self.assertEqual(np.sum(obj.ABCD_matrix), 0)
        obj.compute_for_freqs(obj.freqs_in)
        self.assertNotEqual(np.sum(obj.s11), 0)
        self.assertNotEqual(np.sum(obj.s21), 0)
        self.assertNotEqual(np.sum(obj.s12), 0)
        self.assertNotEqual(np.sum(obj.s22), 0)
        self.assertNotEqual(np.sum(obj.ABCD_matrix), 0)

    def test_Cable(self):
        obj = grfc.Cable()
        self.assertEqual(np.sum(obj.dbs11), 0)
        self.assertEqual(np.sum(obj.dbs21), 0)
        self.assertEqual(np.sum(obj.dbs12), 0)
        self.assertEqual(np.sum(obj.dbs22), 0)
        self.assertEqual(np.sum(obj.s11), 0)
        self.assertEqual(np.sum(obj.s21), 0)
        self.assertEqual(np.sum(obj.s12), 0)
        self.assertEqual(np.sum(obj.s22), 0)
        self.assertEqual(np.sum(obj.ABCD_matrix), 0)
        obj.compute_for_freqs(obj.freqs_in)
        self.assertNotEqual(np.sum(obj.dbs11), 0)
        self.assertNotEqual(np.sum(obj.dbs21), 0)
        self.assertNotEqual(np.sum(obj.dbs12), 0)
        self.assertNotEqual(np.sum(obj.dbs22), 0)
        self.assertNotEqual(np.sum(obj.s11), 0)
        self.assertNotEqual(np.sum(obj.s21), 0)
        self.assertNotEqual(np.sum(obj.s12), 0)
        self.assertNotEqual(np.sum(obj.s22), 0)
        self.assertNotEqual(np.sum(obj.ABCD_matrix), 0)

    def test_VGAFilter(self):
        obj = grfc.VGAFilter()
        self.assertEqual(np.sum(obj.dbs11), 0)
        self.assertEqual(np.sum(obj.dbs21), 0)
        self.assertEqual(np.sum(obj.dbs12), 0)
        self.assertEqual(np.sum(obj.dbs22), 0)
        self.assertEqual(np.sum(obj.s11), 0)
        self.assertEqual(np.sum(obj.s21), 0)
        self.assertEqual(np.sum(obj.s12), 0)
        self.assertEqual(np.sum(obj.s22), 0)
        self.assertEqual(np.sum(obj.ABCD_matrix), 0)
        obj.compute_for_freqs(obj.freqs_in)
        self.assertNotEqual(np.sum(obj.dbs11), 0)
        self.assertNotEqual(np.sum(obj.dbs21), 0)
        self.assertNotEqual(np.sum(obj.dbs12), 0)
        self.assertNotEqual(np.sum(obj.dbs22), 0)
        self.assertNotEqual(np.sum(obj.s11), 0)
        self.assertNotEqual(np.sum(obj.s21), 0)
        self.assertNotEqual(np.sum(obj.s12), 0)
        self.assertNotEqual(np.sum(obj.s22), 0)
        self.assertNotEqual(np.sum(obj.ABCD_matrix), 0)

    def test_BalunBeforeADC(self):
        obj = grfc.BalunBeforeADC()
        self.assertEqual(np.sum(obj.s11), 0)
        self.assertEqual(np.sum(obj.s21), 0)
        self.assertEqual(np.sum(obj.s12), 0)
        self.assertEqual(np.sum(obj.s22), 0)
        self.assertEqual(np.sum(obj.ABCD_matrix), 0)
        obj.compute_for_freqs(obj.freqs_in)
        self.assertNotEqual(np.sum(obj.s11), 0)
        self.assertNotEqual(np.sum(obj.s21), 0)
        self.assertNotEqual(np.sum(obj.s12), 0)
        self.assertNotEqual(np.sum(obj.s22), 0)
        self.assertNotEqual(np.sum(obj.ABCD_matrix), 0)

    def test_Zload(self):
        obj = grfc.Zload()
        self.assertEqual(np.sum(obj.s), 0)
        self.assertEqual(np.sum(obj.Z_load), 0)
        obj.compute_for_freqs(obj.freqs_in)
        self.assertNotEqual(np.sum(obj.s), 0)
        self.assertNotEqual(np.sum(obj.Z_load), 0)

    def test_RFChain(self):
        obj = grfc.RFChain()
        self.assertEqual(np.sum(obj.Z_ant), 0)
        self.assertEqual(np.sum(obj.Z_in), 0)
        self.assertEqual(np.sum(obj.V_out_RFchain), 0)
        self.assertEqual(np.sum(obj.I_out_RFchain), 0)
        self.assertEqual(np.sum(obj.total_ABCD_matrix), 0)
        obj.compute_for_freqs(obj.lna.freqs_in)
        self.assertNotEqual(np.sum(obj.Z_ant), 0)
        self.assertNotEqual(np.sum(obj.Z_in), 0)
        obj.vout_f(np.ones(obj.Z_in.shape))
        self.assertNotEqual(np.sum(obj.V_out_RFchain), 0)
        self.assertNotEqual(np.sum(obj.I_out_RFchain), 0)
        self.assertNotEqual(np.sum(obj.total_ABCD_matrix), 0)

if __name__ == "__main__":
    unittest.main()



