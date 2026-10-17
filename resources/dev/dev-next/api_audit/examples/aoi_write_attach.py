import os
import shutil

import numpy as np
from grand.aoi import Efield, Event, Voltage
from grand.aoi.event_list import EventList

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"
shutil.copytree(SAMPLE, "input")   # work on a copy: the example must not write into the sample


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


def tree_file(tree):
    return "None" if tree is None else "a %s in %s" % (type(tree).__name__, os.path.relpath(tree.file_name))


def new_event():
    """An event built in memory, as in examples/aoi/event_generation.py."""
    event = Event()
    event.run_number, event.event_number = 1, 1
    efield, voltage = Efield(), Voltage()
    for trace in (efield, voltage):
        trace.du_id, trace.n_points = 1, 4
        trace.trace = [np.zeros(4), np.ones(4), np.zeros(4)]
    event.efields, event.voltages = [efield], [voltage]
    return event


built = new_event()
built.write_efields("built.root")
print("1. built in memory, after write_efields: event.tefield is", tree_file(built.tefield))
attempt("   write_run(\"run.root\")", lambda: built.write_run("run.root") or "written")

read = EventList("input").get_event(13790, 1)
print("2. read from files: event.tefield is", tree_file(read.tefield))
attempt("   write_efields(\"read.root\", overwrite=True)", lambda: read.write_efields("read.root", overwrite=True) or "written")
print("   then event.tefield is", tree_file(read.tefield))
