"""
Unit tests for the grand.recons.params_shower module
"""

import unittest
from tests import TestCase
from pathlib import Path

from grand.recons.params_shower import EstimateParamsShower
from grand import grand_get_path_root_pkg

class ParamsTest(TestCase):
    """Unit tests for the pipeline module"""

    filename = Path(grand_get_path_root_pkg()) / "grand" / "recons" / "params_shower.py"

    def test_params(self):
        params = EstimateParamsShower(params={})
        self.assertTrue((self.filename).exists())

if __name__ == "__main__":
    unittest.main()
