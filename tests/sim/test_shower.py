"""
Unit tests for the grand.sim.shower module. 

Rewrote on Jun 19, 2023.
Removed test on older version of CoREAS and ZHAiRES to hdf file.
"""
import unittest
from tests import TestCase
import numpy as np
from pathlib import Path

from grand import ShowerEvent
from grand import grand_get_path_root_pkg
from grand import GRANDCS, LTP
import grand.dataio as groot

class ShowerTest(TestCase):
    """Unit tests for the shower module"""

    # The committed sample: data/test_efield.root is untracked (#271)
    path = (Path(grand_get_path_root_pkg()) / "sim2root" / "Common"
            / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")

    def test_showerevent(self):
        self.tevents = groot.TEfield(str(self.path / "efield_1618-13790_L0_0000.root"))   # traces and du_pos are stored here
        self.trun = groot.TRun(str(self.path / "run_1_L0_0000.root"))                      # site_long, site_lat info is stored here. Used to define shower frame.
        self.tshower = groot.TShower(str(self.path / "shower_1618-13790_L0_0000.root"))    # shower info (like energy, theta, phi, xmax etc) are stored here.
        self.events_list = self.tevents.get_list_of_events() # [[evt0, run0], [evt1, run0], ...[evt0, runN], ...]
        self.event_number = self.events_list[0][0]
        self.run_number = self.events_list[0][1]
        self.tevents.get_event(self.event_number, self.run_number)           # update traces, du_pos etc for event with event_idx.
        self.tshower.get_event(self.event_number, self.run_number)           # update shower info (theta, phi, xmax etc) for event with event_idx.
        self.trun.get_run(self.run_number)

        shower = ShowerEvent()
        shower.origin_geoid  = self.trun.origin_geoid # [lat, lon, height]
        shower.load_root(self.tshower)                # calculates grand_ref_frame, shower_frame, Xmax in shower_frame LTP etc

        self.assertTrue(isinstance(shower.energy, np.float32))
        self.assertTrue(isinstance(shower.zenith, np.float32))
        self.assertTrue(isinstance(shower.azimuth, np.float32))
        self.assertTrue(isinstance(shower.primary, str))
        self.assertTrue(isinstance(shower.grand_ref_frame, GRANDCS))
        self.assertTrue(isinstance(shower.core, GRANDCS))
        self.assertTrue(isinstance(shower.frame, LTP))
        self.assertTrue(isinstance(shower.maximum, LTP))

if __name__ == "__main__":
    unittest.main()
