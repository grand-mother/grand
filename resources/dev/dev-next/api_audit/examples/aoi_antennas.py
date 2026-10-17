import os

import numpy as np
from grand.aoi.event_list import EventList

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"

event = EventList(SAMPLE).get_event(13790, 1)
event.site = "GP13"
try:
    event.fill_antennas(gp300_workaround=False)
    print(len(event.antennas), "antennas; the first:",
          [(int(np.asarray(a.id).ravel()[0]), round(float(np.asarray(a.position.x).ravel()[0]), 1)) for a in event.antennas[:2]])
except Exception as err:
    print(type(err).__name__ + ":", err)
