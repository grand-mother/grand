import os
import shutil

from grand.aoi.event_list import EventList

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"
shutil.copytree(SAMPLE, "input")   # work on a copy: the example must not write into the sample

os.mkdir("output")
with open("output/notes.txt", "w") as notes:
    notes.write("results of an earlier analysis\n")
print("output holds", sorted(os.listdir("output")))

event = EventList("input").get_event(13790, 1)
event.write(out_dir="output", overwrite=True)
print("after event.write(out_dir='output', overwrite=True), output holds", sorted(os.listdir("output")))
