import os
from grand.aoi.event import Event
from grand.aoi.event_list import EventList
from grand.dataio import TRawVoltage

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        result = call()
        print(label, "->", result if isinstance(result, str) else repr(result))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


def describe(event):
    return "an Event" if isinstance(event, Event) else repr(event)


def four_channels():
    raw = TRawVoltage()                       # one unit, four channels holding 0, 1, 2, 3
    raw.run_number, raw.event_number = 1, 1
    raw.du_id, raw.du_seconds, raw.du_nanoseconds = [1], [1_700_000_000], [0]
    raw.trace_ch = [[[0.0] * 4, [1.0] * 4, [2.0] * 4, [3.0] * 4]]
    raw.fill()
    raw.build_index("run_number", "event_number")
    event = Event()
    event.tvoltage, event.run_number, event.event_number = raw, 1, 1
    event.fill_event_from_voltage_tree(use_trawvoltage=True, trawvoltage_channels=[1, 2, 3, 0])
    trace = event.voltages[0].trace
    return "x, y, z start with %s" % [float(trace.x[0]), float(trace.y[0]), float(trace.z[0])]


kept = []   # keeps the Event of case 3 open: releasing it can crash case 6 on dev-next (#312)


def both_numbers():
    event = Event()
    kept.append(event)
    event.directory = SAMPLE
    return event.fill_event_from_trees(entry_number=0, event_number=13790, run_number=1)


events = EventList(SAMPLE)   # run 1: events 13790 and 1618
attempt("1. Event().fill_event_from_trees()", lambda: Event().fill_event_from_trees())
attempt("2. events.get_event(entry_number=0, event_number=13790)",
        lambda: describe(events.get_event(entry_number=0, event_number=13790)))
attempt("3. event.fill_event_from_trees(entry_number=0, event_number=13790, run_number=1)", both_numbers)
attempt("4. events.get_event(99999, 1)", lambda: describe(events.get_event(99999, 1)))
attempt("5. EventList(123).event_list", lambda: EventList(123).event_list)
attempt("6. four channels, trawvoltage_channels=[1, 2, 3, 0]", four_channels)
