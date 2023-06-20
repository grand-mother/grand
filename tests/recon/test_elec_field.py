"""
Unit tests for the grand.recons.elec_field module
"""

import unittest
from tests import TestCase
from pathlib import Path

from grand.recon.elec_field import EstimateEfield
from grand import grand_get_path_root_pkg

class ElecFieldTest(TestCase):
    """Unit tests for the pipeline module"""

    filename = Path(grand_get_path_root_pkg()) / "grand" / "recon" / "elec_field.py"

    def test_efield(self):
        efield = EstimateEfield(params={})
        self.assertTrue((self.filename).exists())

if __name__ == "__main__":
    unittest.main()
