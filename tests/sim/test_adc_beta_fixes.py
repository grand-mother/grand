# -*- coding: utf-8 -*-
r"""#239: non-finite or huge voltages in the ADC step.

A NaN, inf or very large voltage was cast to the most negative int64, whose
absolute value is negative too, so saturation never clipped it; a large
positive voltage came out as the most negative count.  Saturation itself was
never reported.
"""

import logging
import pathlib
import shutil

import numpy as np
import pytest

from grand.sim.detector.adc import ADC

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"


def test_huge_voltages_saturate_with_the_right_sign(caplog):
    x = np.zeros((1, 3, 16))
    x[0, 0, 3], x[0, 1, 3] = 1e30, -1e30
    with caplog.at_level(logging.WARNING):
        out = ADC().process(x)
    assert out[0, 0, 3] == 8192 and out[0, 1, 3] == -8192
    assert "ADC saturation: 2 samples" in caplog.text


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_voltages_are_refused(bad):
    x = np.zeros((2, 3, 16))
    x[1, 2, 5] = bad
    with pytest.raises(ValueError, match=r"first at \(unit, channel, sample\) index \(1, 2, 5\)"):
        ADC().process(x)


@pytest.mark.skipif(not (ROOT / "data" / "detector").exists(), reason="needs the data model")
def test_a_non_finite_efield_is_refused(tmp_path):
    from grand import Efield2Voltage
    from grand.dataio import TEfield

    sample = tmp_path / "sample"
    shutil.copytree(SAMPLE, sample)
    for path in sample.glob("*_L1_*"):
        path.unlink()
    for prefix in ("voltage_", "adc_"):
        for path in sample.glob(prefix + "*"):
            path.unlink()
    efield = next(sample.glob("efield_*_L0_*.root"))
    copy = tmp_path / efield.name
    src, dst = TEfield(str(efield)), TEfield(str(copy))
    for entry in range(src.get_number_of_entries()):
        src.get_entry(entry)
        trace = src.trace.asnumpy().astype(np.float32)    # before copy_contents (#282)
        target_event = int(src.event_number) == 1618
        dst.copy_contents(src)
        if target_event:
            trace[0, 1, 10] = np.nan
            dst.trace = trace
        dst.fill()
    dst.write()
    src.stop_using()
    dst.stop_using()
    shutil.move(str(copy), str(efield))

    signal = Efield2Voltage(str(sample), "out.root", output_directory=str(tmp_path), seed=1)
    with pytest.raises(ValueError, match="event 1618 .* NaN or infinite samples"):
        signal.compute_voltage(event_number=1618, run_number=1)
