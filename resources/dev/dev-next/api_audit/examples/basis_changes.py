import logging
import numpy as np
from grand.basis.traces_event import Handling3dTraces
from grand.basis.type_trace import Voltage
from grand.manage_log import create_output_for_logger


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


one = Handling3dTraces()
one.init_traces(np.arange(48.0).reshape(1, 3, 16))            # one antenna
attempt("1. get_tmax_vmax for one antenna", lambda: one.get_tmax_vmax(hilbert=False, interpol="no"))
two = Handling3dTraces()
two.init_traces(np.ones((2, 3, 4)), f_samp_mhz=np.float32(2000))
attempt("2. f_samp_mhz after init_traces(..., f_samp_mhz=np.float32(2000))", lambda: repr(two.f_samp_mhz))
attempt("3. init_traces(..., f_samp_mhz=0)", lambda: Handling3dTraces().init_traces(np.ones((2, 3, 4)), f_samp_mhz=0))
attempt("4. Voltage(t=np.arange(4.0), V=np.zeros((4, 3)))", lambda: Voltage(t=np.arange(4.0), V=np.zeros((4, 3))).V.shape)
log = logging.getLogger("grand")
create_output_for_logger("info", log_file="run.log", log_stdout=False)
log.info("first")
create_output_for_logger("info", log_file="run.log", log_stdout=False)
log.info("second")
attempt("5. after create_output_for_logger twice: handlers", lambda: len(log.handlers))
attempt("   run.log holds", lambda: open("run.log").read().count("first") * ["first"] + open("run.log").read().count("second") * ["second"])
