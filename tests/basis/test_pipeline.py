"""
Unit tests for the grand.dataio.protocol module
"""

import unittest
from tests import TestCase
from pathlib import Path
import tempfile
from grand.basis.pipeline import Pipeline
from grand import grand_get_path_root_pkg


class PipelineTest(TestCase):
    """Unit tests for the pipeline module"""

    def test_add(self):
        # The committed sample, and a temporary output: it read the untracked
        # data/test_efield.root and wrote into data/ (#271)
        input_file = (Path(grand_get_path_root_pkg()) / "sim2root" / "Common"
                      / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
        output_file = Path(tempfile.mkdtemp()) / "test_voltage.root"

        self.assertTrue((input_file).exists())
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
        
        #os.remove(output_file)

if __name__ == "__main__":
    unittest.main()