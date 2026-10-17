import numpy as np
from grand.sim.detector.rf_chain import RFChain, VGAFilter
from grand.sim.efield2voltage import Efield2Voltage
import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"


def attempt(label, call):
    try:
        print(label, "->", call())
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


attempt("1. RFChain().compute_for_freqs([-10.0, 50.0, 100.0])", lambda: RFChain().compute_for_freqs(np.array([-10.0, 50.0, 100.0])))
attempt("2. VGAFilter(gain=False).gain", lambda: VGAFilter(gain=False).gain)


def flags_as_numbers():
    e = Efield2Voltage(SAMPLE, seed=1)
    e.params.update(add_noise=0, add_rf_chain=0, resample_to_mhz=0, extend_to_us=0,
                    calibration_smearing_sigma=0, add_jitter_ns=0)
    e.compute_voc_event(event_number=1618, run_number=1)
    return "computed"


attempt("3. Efield2Voltage with params add_noise=0, add_rf_chain=0", flags_as_numbers)
