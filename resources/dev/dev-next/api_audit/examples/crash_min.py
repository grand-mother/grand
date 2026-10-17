import gc
import os
import sys

import ROOT
from grand.aoi.event import Event
from grand.aoi.event_list import EventList
from grand.dataio import TADC

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def peek():
    """Opens the sample's files through an Event that is dropped on return."""
    event = Event()
    event.directory = SAMPLE


gc.disable()                  # hold the garbage collector until the new tree exists
if "list" in sys.argv:
    events = EventList(SAMPLE)
peek()
print("current ROOT directory:", os.path.basename(ROOT.gDirectory.GetName()))
tadc = TADC()                 # a new tree, in memory
print("the new tree is in:", os.path.basename(tadc._tree.GetDirectory().GetName()))
gc.collect()                  # the dropped Event is freed, as the collector may do at any time
tadc.du_id = [1]
tadc.fill()
print("filled:", tadc.get_number_of_entries(), "entry")
