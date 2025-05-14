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
    """Unit tests for the module to compute voltage from electric field."""

    infile = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
    outfile = Path(grand_get_path_root_pkg()) / "data" / "test_voltage1.root"

    def test_Efield2Voltage(self):
        if self.outfile.exists():
            os.remove(str(self.outfile))
        self.assertFalse(self.outfile.exists())

        master = Efield2Voltage(str(self.infile), str(self.outfile))
        self.assertTrue(master.params["add_noise"])
        self.assertTrue(master.params["add_rf_chain"])
        self.assertTrue(master.params["lst"]==18)

        master.compute_voltage()    # saves automatically

        self.assertTrue(self.outfile.exists())
        os.remove(str(self.outfile))

if __name__ == "__main__":
    unittest.main()
