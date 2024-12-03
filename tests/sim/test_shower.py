"""
Unit tests for the grand.sim.shower module. 

Rewrote on Jun 19, 2023.
Removed test on older version of CoREAS and ZHAiRES to hdf file.
"""
import unittest
from tests import TestCase
import numpy as np
from pathlib import Path

#from grand import ShowerEvent
from grand.sim.shower.gen_shower import ShowerEvent

from grand import grand_get_path_root_pkg
#from grand import GRANDCS, LTP
from grand.geo.coordinates import (
    GRANDCS,
    LTP
)
import grand.dataio.root_trees as groot

class ShowerTest(TestCase):
    """Unit tests for the shower module"""

    path = Path(grand_get_path_root_pkg()) / "data" / "test_efield.root"

    def test_showerevent(self):
        self.tevents = groot.TEfield(str(self.path))                # traces and du_pos are stored here
        self.trun = groot.TRun(str(self.path))                      # site_long, site_lat info is stored here. Used to define shower frame.
        self.tshower = groot.TShower(str(self.path))                # shower info (like energy, theta, phi, xmax etc) are stored here.
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
