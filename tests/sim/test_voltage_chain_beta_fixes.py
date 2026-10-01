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


def _rewrite_shower(sample, **changes):
    r"""Rewrites the shower file of `sample`, setting the given fields in every entry."""
    from grand.dataio import TShower

    path = next(sample.glob("shower_*_L0_*.root"))
    source = TShower(str(path))
    rows = []
    for entry in range(source.get_number_of_entries()):
        source.get_entry(entry)
        rows.append({name: getattr(source, name) for name in ("event_number", "zenith", "azimuth",
                                                                "energy_primary", "xmax_grams",
                                                                "xmax_pos_shc", "xmax_pos", "core_alt",
                                                                "shower_core_pos", "primary_type",
                                                                "magnetic_field", "direction")})
    source.stop_using()
    path.unlink()
    target = TShower(str(path))
    for row in rows:
        row.setdefault("run_number", 1)
        row.update(changes)
        for name, value in row.items():
            setattr(target, name, value)
        target.fill()
    target.write()
    target.stop_using()


def test_a_shower_tree_without_the_event_is_refused(level0_sample, tmp_path):
    r"""#247: a missing shower entry silently reused the previous event's shower."""
    from grand import Efield2Voltage

    _rewrite_shower(level0_sample, run_number=2)
    signal = Efield2Voltage(str(level0_sample), "out.root", output_directory=str(tmp_path), seed=1)
    signal.params["add_noise"] = False
    with pytest.raises(KeyError, match="shower tree has no entry for event"):
        signal.compute_voltage(event_number=13790, run_number=1)


def test_the_run_tree_is_read_at_the_level_of_the_efield(level0_sample):
    r"""#237: an L0 efield with an L1 run tree used the L1 sampling time (2× amplitude)."""
    from grand import Efield2Voltage

    off = dict(add_noise=False, add_rf_chain=False)
    clean = _voltage(level0_sample, 13790, **off)
    for path in SAMPLE.glob("run*_L1_*"):          # the L1 run trees, 2 ns sampling
        shutil.copy(path, level0_sample)
    assert np.array_equal(_voltage(level0_sample, 13790, **off), clean)

    for path in level0_sample.glob("run_*_L0_*"):  # no run tree at the efield's level
        path.unlink()
    with pytest.raises(FileNotFoundError, match="no run file at that level"):
        Efield2Voltage(str(level0_sample))


def test_a_resampled_voltage_is_not_saved_and_the_script_refuses_it(level0_sample, tmp_path):
    r"""#229: a resampled voltage was saved with the input's t_bin_size, so convert_voltage2adc
    resampled it a second time (all zeros at 250 MHz).  The voltage file cannot record a new
    rate, so saving is refused, by the Python API as by the script; in memory it still works."""
    import subprocess
    import sys

    from grand import Efield2Voltage

    signal = Efield2Voltage(str(level0_sample), "out.root", output_directory=str(tmp_path), seed=1)
    signal.params.update(add_noise=False, add_rf_chain=False, resample_to_mhz=500)
    with pytest.raises(ValueError, match="GRANDlib: Efield2Voltage.save_voltage: .*resampled to 500"):
        signal.compute_voltage(event_number=1618, run_number=1)
    assert not (tmp_path / "out.root").exists()

    full = _voltage(level0_sample, 1618, add_noise=False, add_rf_chain=False)
    resampled = _voltage(level0_sample, 1618, add_noise=False, add_rf_chain=False, resample_to_mhz=500)
    assert resampled.shape[-1] * 4 == full.shape[-1]     # 2 GHz -> 500 MHz, in memory

    # The input's own rate is not a resampling: saved as usual
    signal = Efield2Voltage(str(level0_sample), "same.root", output_directory=str(tmp_path), seed=1)
    signal.params.update(add_noise=False, add_rf_chain=False, resample_to_mhz=2000)
    signal.compute_voltage(event_number=1618, run_number=1)
    assert (tmp_path / "same.root").exists()

    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_efield2voltage.py"),
                           str(level0_sample), "--target_sampling_rate_mhz", "500"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "--target_sampling_rate_mhz is not supported" in done.stderr


@pytest.mark.parametrize("xmax", [[np.nan] * 3, [0.0, 0.0, 0.01]], ids=["nan", "at_the_core"])
def test_an_event_without_a_usable_xmax_is_refused(level0_sample, tmp_path, xmax):
    r"""#228: a NaN Xmax crashed in the antenna lookup; one at the core gave ~1e-13 uV."""
    from grand import Efield2Voltage

    _rewrite_shower(level0_sample, xmax_pos_shc=xmax)
    signal = Efield2Voltage(str(level0_sample), "out.root", output_directory=str(tmp_path), seed=1)
    signal.params["add_noise"] = False
    with pytest.raises(ValueError, match="no usable Xmax position"):
        signal.compute_voltage(event_number=13790, run_number=1)
