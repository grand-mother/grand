"""
Unit tests for the grand.basis.signal module
"""
import unittest
from pathlib import Path
from tests import TestCase
import numpy as np

import grand.basis.signal as sig
import grand.dataio.root_files as RFile
from grand import grand_get_path_root_pkg


class SignalTest(TestCase):
    """Unit tests for the signal module"""

    efield_file = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
    shape = (96, 3, 999)   # (du, xyz, traces)
    TestCase.assertTrue((efield_file).exists())
    E = RFile.get_file_event(str(efield_file))

    def test_get_filter(self):
        filtered = sig.get_filter(time=self.E.traces_time[0], trace=self.E.traces[0,0], fr_min=30e6, fr_max=250e6)
        self.assertEqual(self.E.traces[0,0].shape, filtered.shape)

    def test_get_peakamptime_norm_hilbert(self):
        t_max, v_max, idx_max, norm_hilbert_amp = sig.get_peakamptime_norm_hilbert(self.E.traces_time, self.E.traces)
        self.assertEqual(len(t_max), self.E.du_count)
        self.assertEqual(len(v_max), self.E.du_count)
        self.assertEqual(len(idx_max), self.E.du_count)
        self.assertEqual(norm_hilbert_amp.shape[0], self.E.du_count)

if __name__ == "__main__":
    unittest.main()
