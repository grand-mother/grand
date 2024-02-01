"""
Unit test for basis/traces_event.py
"""
import os
import unittest
from pathlib import Path
from tests import TestCase
import numpy as np

from grand.basis.traces_event import Handling3dTraces

class EventTracesTest(TestCase):
    def __init__(self, *args):
        super().__init__(*args)
        self.tr3d = Handling3dTraces()
        self.size_tot = 3 * 3 * 4
        self.traces = np.arange(self.size_tot, dtype=np.float32).reshape((3, 3, 4))
        self.du_id = np.arange(3)
        self.t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
        self.f_samp_mhz = 2000
        self.tr3d.init_traces(self.traces, self.du_id, self.t_start_ns, self.f_samp_mhz)

    def get_traces(self):
        self.size_tot = 3 * 3 * 4
        self.traces = np.arange(self.size_tot, dtype=np.float32).reshape((3, 3, 4))
        self.du_id = np.arange(3)
        self.t_start_ns = np.arange(3, dtype=np.float32) * 10 + 100
        self.f_samp_mhz = 2000
        self.tr3d.init_traces(self.traces, self.du_id, self.t_start_ns, self.f_samp_mhz)

    def test_init_traces(self):
        assert self.tr3d.traces[0, 0, 0] == 0.0
        assert self.tr3d.traces[2, 2, 3] == self.size_tot - 1
        assert self.tr3d.du_id[-1] == 2
        assert np.allclose(self.tr3d.t_start_ns, np.array([100, 110, 120]))

    def test_init_network(self):
        du_pos = np.arange(int(len(self.du_id)*3)).reshape((len(self.du_id), 3))
        self.tr3d.init_network(du_pos) # du_pos: float[nb_DU, 3]
        assert self.tr3d.network.du_pos.shape[0]==len(self.du_id)
        
    def test_define_t_samples(self):
        # RK: add more/better tests.
        self.get_traces()
        assert np.allclose(self.tr3d.t_samples[0], np.array([100.0, 100.5, 101.0, 101.5]))
        self.tr3d.f_samp_mhz = 1000
        self.tr3d.t_samples = np.zeros(0)
        self.tr3d._define_t_samples()
        assert np.allclose(self.tr3d.t_samples[-1], np.array([120.0, 121, 122.0, 123]))
        
    def test_set_unit_axis(self):
        # RK: add more/better tests.
        self.tr3d.set_unit_axis()
        #assert self.tr3d.unit_trace=="TBD"
        #assert self.tr3d.axis_name=="idx"    

    def test_define_t_samples(self):
        # RK: add more/better tests.
        self.get_traces()
        self.tr3d._define_t_samples()
        assert self.tr3d.t_samples.size!=0

    #def test_reduce_nb_du(self):
    #    # RK: add more/better tests.
    #    self.get_traces()
    #    self.tr3d.reduce_nb_du(2)
    #    assert len(self.tr3d.network.du_id)==2

    #def test_delta_t_ns(self):
    #    # RK: add more/better tests.
    #    self.get_traces()
    #    assert self.tr3d.delta_t_ns() == 1e3 / self.tr3d.f_samp_mhz
    #    assert self.tr3d.delta_t_ns() == 1.0

    def test_get_max_norm(self):
        # RK: add more/better tests.
        max_norm = self.tr3d.get_max_norm()
        assert len(max_norm)==len(self.du_id)

    def test_get_norm(self):
        # RK: add more/better tests.
        trnorm = self.tr3d.get_norm() # (nb_du, n_traces)
        assert trnorm.shape==(len(self.du_id), self.traces.shape[-1])

    def test_get_tmax_vmax(self):
        # RK: add more/better tests.
        tmax, vmax = self.tr3d.get_tmax_vmax()
        assert len(tmax)==len(self.du_id)
        assert len(vmax)==len(self.du_id)

    def test_get_nb_du(self):
        # RK: add more/better tests.
        nb_du = self.tr3d.get_nb_du()
        assert nb_du==len(self.du_id)

    def test_get_size_trace(self):
        # RK: add more/better tests.
        assert self.tr3d.get_size_trace()==self.traces.shape[-1]

    def test_get_extended_traces(self):
        # RK: add more/better tests.
        common_time, extended_traces = self.tr3d.get_extended_traces()
        assert extended_traces.shape[0]==len(self.du_id)

if __name__ == "__main__":
    unittest.main()
