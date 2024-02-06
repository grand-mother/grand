"""
Unit tests for the grand.dataio.root_files module
"""

import unittest
from tests import TestCase
from pathlib import Path
import os

import grand.dataio.root_files as RFile
import grand.dataio.root_trees as groot
from grand import grand_get_path_root_pkg

#TODO: (JMC) almost all tests are broken by new version of GRANDROOT file, needs to have a set of coherent ROOT for test 
 

G_efield_file = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"

class RootFilesTest(TestCase):
    """Unit tests for the root_files module"""

    # event_number = 1, run_number = 0
    efield_file = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"
    voltage_file= Path(grand_get_path_root_pkg()) / "data" / "test_voltage.root"
    shape = (96, 3, 999)

    def test_fileeventbase(self):
        self.assertTrue((self.efield_file).exists())
        E = groot.TEfield(str(self.efield_file))
        eventbase = RFile._FileEventBase(E)
        self.assertTrue(hasattr(eventbase, 'run_number'))  
        self.assertTrue(hasattr(eventbase, 'event_number'))
        self.assertTrue(eventbase.run_number is None)
        eventbase.load_event_idx(0)
        self.assertTrue(eventbase.run_number==0)
        self.assertTrue(eventbase.event_number==1)
        self.assertFalse(eventbase.load_next_event()) # No more event. Return False.
        eventbase._load_event_identifier(1,0)
        self.assertTrue(eventbase.get_du_count()==96)
        self.assertTrue(eventbase.get_nb_events()==1)
        self.assertTrue(eventbase.get_size_trace()==999)
        self.assertTrue(eventbase.get_sampling_freq_mhz()==2000)

    def test_get_file_event1(self):
        if not (self.voltage_file).exists():
            os.system(f"python ../../scripts/grand_sim_e2v.py {self.efield_file} -o {self.voltage_file}")
        self.assertTrue((self.efield_file).exists())
        self.assertTrue((self.voltage_file).exists())

        E = RFile.get_file_event(str(self.efield_file))
        V = RFile.get_file_event(str(self.voltage_file))
        self.assertTrue(isinstance(E.run_number, int) and isinstance(V.run_number, int))
        self.assertTrue(isinstance(E.event_number, int) and isinstance(V.event_number, int))
        self.assertTrue((E.traces.shape == self.shape) and (V.traces.shape == self.shape))
        #self.assertTrue((E.traces_time.size == self.shape[0]*self.shape[-1]) and (V.traces_time.size == self.shape[0]*self.shape[-1]))
        self.assertTrue((E.traces.shape[-1] == self.shape[-1]) and (V.traces.shape[-1] == self.shape[-1]))
        #self.assertTrue((E.t_bin_size == 0.5) and (V.t_bin_size == 0.5))
        self.assertTrue(len(E.du_id)==self.shape[0] and len(V.du_id)==self.shape[0])
        self.assertTrue((E.du_count==self.shape[0]) and (V.du_count==self.shape[0]))
        #self.assertTrue(E.du_xyz.size==self.shape[0]*self.shape[1])
        #self.assertTrue((E.f_name == str(self.efield_file)) and (V.f_name == str(self.voltage_file)))
        #self.assertTrue((E.tag == "efield") and (V.tag == "voltage"))

    def test_FileEfield(self):
        self.assertTrue((self.efield_file).exists())

        E = RFile.FileEfield(str(self.efield_file))
        self.assertTrue(isinstance(E.run_number, int))
        self.assertTrue(isinstance(E.event_number, int))
        self.assertTrue(E.traces.shape == self.shape)
        self.assertTrue(E.traces.shape[-1] == self.shape[-1])
        #self.assertTrue(E.t_bin_size == 0.5)
        self.assertTrue(len(E.du_id)==self.shape[0])
        self.assertTrue(E.du_count==self.shape[0])
        #self.assertTrue(E.du_xyz.size==self.shape[0]*self.shape[1])
        #self.assertTrue(E.f_name == str(self.efield_file))
        #self.assertTrue(E.tag == "efield")
        #self.assertTrue(E.get_du_count()==self.shape[0])
        #self.assertTrue(E.get_nb_events()==1)
        #self.assertTrue(E.get_size_trace()==self.shape[-1])
        #self.assertTrue(isinstance(E.get_sampling_freq_mhz()[0], float) and E.get_sampling_freq_mhz()[0]>0)


    def test_FileVoltage(self):
        if not (self.voltage_file).exists():
            os.system(f"python ../../scripts/grand_sim_e2v.py {self.efield_file} -o {self.voltage_file}")
        self.assertTrue((self.voltage_file).exists())

        V = RFile.FileVoltage(str(self.voltage_file))
        self.assertTrue(isinstance(V.run_number, int))
        self.assertTrue(isinstance(V.event_number, int))
        self.assertTrue(V.traces.shape == self.shape)
        self.assertTrue(V.traces.shape[-1] == self.shape[-1])
        #self.assertTrue(V.t_bin_size == 0.5)
        self.assertTrue(len(V.du_id)==self.shape[0])
        self.assertTrue(V.du_count==self.shape[0])
        #self.assertTrue(V.du_xyz is None)
        #self.assertTrue(V.f_name == str(self.voltage_file))
        #self.assertTrue(V.tag == "voltage")
        #self.assertTrue(V.get_du_count()==self.shape[0])
        #self.assertTrue(V.get_nb_events()==1)
        #self.assertTrue(V.get_size_trace()==self.shape[-1])
        #self.assertTrue(V.get_sampling_freq_mhz()>0)

        # voltage file is produced by test_pipeline.py and is removed here.
        #os.remove(self.voltage_file)

    def test_get_file_event2(self):
        self.assertTrue((self.efield_file).exists())
        efield = RFile.get_file_event(str(self.efield_file))
        self.assertTrue(hasattr(efield, 'traces'))

def test_get_obj_handling3dtraces():
    ef_root = RFile.get_file_event(str(G_efield_file))
    ef_obj = ef_root.get_obj_handling3dtraces()
    assert ef_obj.get_size_trace() == ef_root.sig_size


if __name__ == "__main__":
    unittest.main()
