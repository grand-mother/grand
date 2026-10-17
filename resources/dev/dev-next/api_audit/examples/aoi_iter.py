import os
from grand.aoi.event_list import EventList

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


def numbers(events):
    return [int(event.event_number) for event in events]


attempt("EventList(SAMPLE)                  ", lambda: numbers(EventList(SAMPLE)))
attempt("EventList(SAMPLE, start_entry=1)   ", lambda: numbers(EventList(SAMPLE, start_entry=1)))
attempt("EventList(SAMPLE, start_event=1618)", lambda: numbers(EventList(SAMPLE, start_event=1618)))
attempt("EventList(SAMPLE, start_entry=5)   ", lambda: numbers(EventList(SAMPLE, start_entry=5)))
