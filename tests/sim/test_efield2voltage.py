"""
Unit tests for the grand.sim.efield2voltage module. 

Jun 19, 2023.
"""
import os
import unittest
from tests import TestCase
from pathlib import Path

from grand import grand_get_path_root_pkg
from grand import Efield2Voltage

class Efield2VoltageTest(TestCase):
    """Unit tests for the shower module"""

    infile = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
    outfile = Path(grand_get_path_root_pkg()) / "data" / "test_voltage.root"

    def test_Efield2Voltage(self):
        os.remove(self.outfile)
        self.assertFalse(self.outfile.exists())

        master = Efield2Voltage(self.infile, self.outfile)
        self.assertTrue(master.params["add_noise"])
        self.assertTrue(master.params["add_rf_chain"])
        self.assertTrue(master.params["lst"]==18)

        master.compute_voltage()    # saves automatically

        self.assertTrue(self.outfile.exists())

if __name__ == "__main__":
    unittest.main()
