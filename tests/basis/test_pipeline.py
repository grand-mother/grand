"""
Unit tests for the grand.io.protocol module
"""

import unittest
from tests import TestCase
from pathlib import Path
import os
from grand.io.protocol import InvalidBLOB, get
from grand.io.pipeline import Pipeline
from grand import grand_get_path_root_pkg


class PipelineTest(TestCase):
    """Unit tests for the pipeline module"""

    def test_add(self):
        input_file = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
        output_file= Path(grand_get_path_root_pkg()) / "data" / "test_voltage.root"

        self.assertTrue((input_file).exists())
        try:
            os.remove(output_file)
        except:
            pass
        self.assertFalse((output_file).exists())

        pipeline = Pipeline()
        pipeline.Add("reader", 
                    f_input=str(input_file)) # filename = str, list of str
        pipeline.Add("efield2voltage", 
                    add_noise=True, 
                    add_rf_chain=True, 
                    lst=18,
                    seed=0,
                    padding_factor=1.2)
        pipeline.Add("writer", 
                    f_output=str(output_file))

        self.assertTrue((output_file).exists())

if __name__ == "__main__":
    unittest.main()