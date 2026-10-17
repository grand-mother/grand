import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", repr(call()))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


from grand.dataio import TADC, TRun

tadc = TADC(SAMPLE + "adc_1618-13790_L1_0000.root")
trun = TRun(SAMPLE + "run_1_L0_0000.root")
attempt("tadc.has_event(13790.0, 1.0)  ", lambda: tadc.has_event(13790.0, 1.0))
attempt("tadc.get_event(13790.0, 1.0)  ", lambda: tadc.get_event(13790.0, 1.0) or int(tadc.event_number))
attempt("tadc.get_event('13790', '1')  ", lambda: tadc.get_event("13790", "1") or int(tadc.event_number))
attempt("trun.has_run(1.0)             ", lambda: trun.has_run(1.0))
attempt("trun.get_run(1.0)             ", lambda: trun.get_run(1.0) or int(trun.run_number))
