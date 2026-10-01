# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to the e-field → voltage → ADC chain.

Each test names its GitHub issue and fails without the fix.  They run on a
copy of a committed sim2root sample (RUN1, events 1618 and 13790), with its
level-1 files removed so that every tree is read at level 0.
"""

import pathlib
import shutil

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "detector").exists(), reason="needs the GRAND data model (data/detector)")


@pytest.fixture
def level0_sample(tmp_path):
    r"""The committed RUN1 sample at level 0 only, in a temporary folder."""
    target = tmp_path / "sample"
    shutil.copytree(SAMPLE, target)
    for path in target.glob("*_L1_*"):
        path.unlink()
    for prefix in ("voltage_", "adc_"):
        for path in target.glob(prefix + "*"):
            path.unlink()
    return target


def _voltage(sample, event_number, **params):
    r"""The voltage of one event with the given processing switches."""
    from grand import Efield2Voltage

    signal = Efield2Voltage(str(sample), seed=1)
    signal.params.update(params)
    signal.compute_voltage_event(event_number=event_number, run_number=1)
    signal.final_resample()
    return np.array(signal.vout)


def test_nut_and_gaa_chains_apply_without_noise_or_the_main_chain(level0_sample):
    r"""#227: ``--no_noise --no_rf_chain --rf_chain_nut`` returned V_oc unchanged."""
    off = dict(add_noise=False, add_rf_chain=False)
    voc = _voltage(level0_sample, 1618, **off)
    nut = _voltage(level0_sample, 1618, add_rf_chain_nut=True, **off)
    gaa = _voltage(level0_sample, 1618, add_rf_chain_gaa=True, **off)
    assert not np.allclose(nut, voc)
    assert not np.allclose(gaa, voc)
    assert not np.allclose(nut, gaa)


def test_an_event_the_input_does_not_hold_is_refused(level0_sample, tmp_path):
    r"""#238: a missing (event, run) wrote the last loaded event with an empty trace."""
    from grand import Efield2Voltage

    signal = Efield2Voltage(str(level0_sample), "out.root", output_directory=str(tmp_path), seed=1)
    signal.params["add_noise"] = False
    for event_number, run_number in ((999, 1), (1618, 7)):
        with pytest.raises(KeyError, match="no event %d in run %d" % (event_number, run_number)):
            signal.compute_voltage(event_number=event_number, run_number=run_number)
    with pytest.raises(Exception, match="event_idx"):
        signal.compute_voltage(event_idx=-1)
    assert not (tmp_path / "out.root").exists()
    # NumPy integers are integers
    signal.compute_voltage(event_number=np.int64(1618), run_number=np.uint32(1))
    assert (tmp_path / "out.root").exists()
