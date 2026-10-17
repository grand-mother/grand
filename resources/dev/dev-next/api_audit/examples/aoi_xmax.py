import os
from grand.aoi.event_list import EventList

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"

events = EventList(SAMPLE)
for number in (13790, 1618):
    event = events.get_event(number, 1)
    xmax = event.simshower.Xmaxpos
    print("event %d: Xmax seen from the shower core, x y z [m]:" % number,
          [round(float(c[0]), 1) for c in (xmax.x, xmax.y, xmax.z)])
print("ground altitude of the site [m]:", float(event.trun.origin_geoid[2]))
