from grand.aoi.event import Event
from grand.dataio import TRun, TShower

run = TRun()
run.run_number, run.origin_geoid = 1, [40.0, 90.0, 1000.0]
run.fill()
shower = TShower()
shower.run_number, shower.event_number = 1, 1
shower.zenith, shower.azimuth = -9.37, 0.0          # out of range, as data_storing.py's random values can be
shower.xmax_pos_shc = [1000.0, 0.0, 5000.0]
shower.fill()
shower.build_index("run_number", "event_number")
print("written: zenith", shower.zenith)

event = Event()
event.trun, event.tshower = run, shower
event.run_number, event.event_number = 1, 1
try:
    event.fill_event_from_shower_tree()
    print("read back: zenith", float(event.shower.zenith))
except Exception as err:
    print("read back ->", type(err).__name__ + ":", err)
