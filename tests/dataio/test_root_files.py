"""
Unit tests for the grand.dataio.root_files module
"""

import unittest
from tests import TestCase
from pathlib import Path
import os

import grand.dataio.root_files as RFile
from grand import grand_get_path_root_pkg


class RootFilesTest(TestCase):
    """Unit tests for the root_files module"""

    efield_file = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
    voltage_file= Path(grand_get_path_root_pkg()) / "data" / "test_voltage.root"
    shape = (96, 3, 999)

    def test_get_file_event(self):
        self.assertTrue((self.efield_file).exists())
        self.assertTrue((self.voltage_file).exists())

        E = RFile.get_file_event(str(self.efield_file))
        V = RFile.get_file_event(str(self.voltage_file))
        self.assertTrue(isinstance(E.run_number, int) and isinstance(V.run_number, int))
        self.assertTrue(isinstance(E.event_number, int) and isinstance(V.event_number, int))
        self.assertTrue((E.traces.shape == self.shape) and (V.traces.shape == self.shape))
        self.assertTrue((E.traces_time.size == self.shape[0]*self.shape[-1]) and (V.traces_time.size == self.shape[0]*self.shape[-1]))
        self.assertTrue((E.sig_size == self.shape[-1]) and (V.sig_size == self.shape[-1]))
        self.assertTrue((E.t_bin_size == 0.5) and (V.t_bin_size == 0.5))
        self.assertTrue(len(E.du_id)==self.shape[0] and len(V.du_id)==self.shape[0])
        self.assertTrue((E.du_count==self.shape[0]) and (V.du_count==self.shape[0]))
        self.assertTrue(E.du_xyz.size==self.shape[0]*self.shape[1])
        self.assertTrue((E.f_name == str(self.efield_file)) and (V.f_name == str(self.voltage_file)))
        self.assertTrue((E.tag == "efield") and (V.tag == "voltage"))

        E.get_event(0)
        V.get_event(0)
        self.assertTrue(isinstance(E.run_number, int) and isinstance(V.run_number, int))
        self.assertTrue(isinstance(E.event_number, int) and isinstance(V.event_number, int))
        self.assertTrue((E.traces.shape == self.shape) and (V.traces.shape == self.shape))
        self.assertTrue((E.traces_time.size == self.shape[0]*self.shape[-1]) and (V.traces_time.size == self.shape[0]*self.shape[-1]))
        self.assertTrue((E.sig_size == self.shape[-1]) and (V.sig_size == self.shape[-1]))
        self.assertTrue((E.t_bin_size == 0.5) and (V.t_bin_size == 0.5))
        self.assertTrue(len(E.du_id)==self.shape[0] and len(V.du_id)==self.shape[0])
        self.assertTrue((E.du_count==self.shape[0]) and (V.du_count==self.shape[0]))
        self.assertTrue(E.du_xyz.size==self.shape[0]*self.shape[1])
        self.assertTrue((E.f_name == str(self.efield_file)) and (V.f_name == str(self.voltage_file)))
        self.assertTrue((E.tag == "efield") and (V.tag == "voltage"))

    def test_FileEfield(self):
        self.assertTrue((self.efield_file).exists())

        E = RFile.FileEfield(str(self.efield_file))
        self.assertTrue(isinstance(E.run_number, int))
        self.assertTrue(isinstance(E.event_number, int))
        self.assertTrue(E.traces.shape == self.shape)
        self.assertTrue(E.sig_size == self.shape[-1])
        self.assertTrue(E.t_bin_size == 0.5)
        self.assertTrue(len(E.du_id)==self.shape[0])
        self.assertTrue(E.du_count==self.shape[0])
        self.assertTrue(E.du_xyz.size==self.shape[0]*self.shape[1])
        self.assertTrue(E.f_name == str(self.efield_file))
        self.assertTrue(E.tag == "efield")
        self.assertTrue(E.get_du_count()==self.shape[0])
        self.assertTrue(E.get_nb_events()==1)
        self.assertTrue(E.get_size_trace()==self.shape[-1])
        self.assertTrue(isinstance(E.get_sampling_freq_mhz()[0], float) and E.get_sampling_freq_mhz()[0]>0)


    def test_FileVoltage(self):
        self.assertTrue((self.voltage_file).exists())

        V = RFile.FileVoltage(str(self.voltage_file))
        self.assertTrue(isinstance(V.run_number, int))
        self.assertTrue(isinstance(V.event_number, int))
        self.assertTrue(V.traces.shape == self.shape)
        self.assertTrue(V.sig_size == self.shape[-1])
        self.assertTrue(V.t_bin_size == 0.5)
        self.assertTrue(len(V.du_id)==self.shape[0])
        self.assertTrue(V.du_count==self.shape[0])
        self.assertTrue(V.du_xyz is None)
        self.assertTrue(V.f_name == str(self.voltage_file))
        self.assertTrue(V.tag == "voltage")
        self.assertTrue(V.get_du_count()==self.shape[0])
        self.assertTrue(V.get_nb_events()==1)
        self.assertTrue(V.get_size_trace()==self.shape[-1])
        self.assertTrue(V.get_sampling_freq_mhz()>0)

        # voltage file is produced by test_pipeline.py and is removed here.
        os.remove(self.voltage_file)

if __name__ == "__main__":
    unittest.main()
