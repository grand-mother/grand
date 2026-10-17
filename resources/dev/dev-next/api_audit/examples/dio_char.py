import os

SAMPLE = os.environ["GRAND_ROOT"] + "/sim2root/Common/sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000/"


def attempt(label, call):
    try:
        print(label, "->", repr(call()))
    except Exception as err:
        print(label, "->", type(err).__name__ + ":", err)


import numpy as np
from grand.dataio import TADC

tadc = TADC()
tadc.adc_input_channels_ch = [[1, 2, 3, 4]]   # stored as unsigned char, as in the data format
tadc.test_pulse_rate_divider = [3, 5]           # unsigned char too
attempt("list(tadc.adc_input_channels_ch[0])     ", lambda: list(tadc.adc_input_channels_ch[0]))
attempt("tadc.test_pulse_rate_divider[0]         ", lambda: tadc.test_pulse_rate_divider[0])
attempt("tadc.test_pulse_rate_divider == [3, 5]  ", lambda: tadc.test_pulse_rate_divider == [3, 5])
attempt("test_pulse_rate_divider = ['\\x03', '\\x05']", lambda: setattr(tadc, "test_pulse_rate_divider", ["\x03", "\x05"]))
tadc.du_id = [14, 21]
attempt("tadc.du_id == np.array([14, 21])        ", lambda: tadc.du_id == np.array([14, 21]))
attempt("tadc.du_id == (14, 21)                  ", lambda: tadc.du_id == (14, 21))
attempt("hasattr(TADC, 'du_id')                  ", lambda: hasattr(TADC, "du_id"))
