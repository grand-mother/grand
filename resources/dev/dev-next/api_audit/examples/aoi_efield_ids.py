import numpy as np
from grand.aoi import Efield, Event, Voltage
from grand.dataio import TEfield


def trace(kind, unit):
    t = kind()
    t.du_id, t.n_points = unit, 4
    t.trace = [np.zeros(4), np.ones(4), np.zeros(4)]
    return t


event = Event()
event.run_number, event.event_number = 1, 1
event.efields = [trace(Efield, 7), trace(Efield, 8)]   # electric fields at units 7 and 8
event.voltages = [trace(Voltage, 3)]                   # a voltage at unit 3 only
event.write_efields("efield.root")

tefield = TEfield("efield.root")
tefield.get_entry(0)
print("efields of units [7, 8] written; the file says du_id =", list(tefield.du_id),
      "with", len(tefield.trace), "traces")
