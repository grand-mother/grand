# -*- coding: utf-8 -*-
r"""Tests for the offline T1 trigger and its use in ``convert_voltage2adc`` (#139).

The traces are synthetic, in ADC counts at 2 ns per sample.  With the default
parameters: T1 at 100, T2 at 50, a quiet time of 256 samples before the T1
crossing, a window of 256 samples after it, T2 crossings less than 10 ns
(5 samples) apart, and 2 to 8 crossings.
"""

import ast
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from grand.sim.detector.trigger import (DEFAULT_T1_CONFIG,
                                        extract_trigger_parameters,
                                        t1_channel_trigger, t1_du_triggers,
                                        t1_trigger_flags)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = (ROOT / "sim2root" / "Common"
          / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")
N = 1024


def _load(name, path):
    r"""Imports a script by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _voltage2adc():
    r"""Imports ``scripts/convert_voltage2adc.py`` without needing psutil."""
    sys.modules.setdefault("psutil", types.ModuleType("psutil"))
    return _load("convert_voltage2adc", ROOT / "scripts" / "convert_voltage2adc.py")


def pulse(start, n_crossings, step=2, amplitude=200, n=N):
    r"""A trace crossing +T1 `n_crossings` times, every `step` samples."""
    trace = np.zeros(n, dtype=int)
    for k in range(n_crossings):
        trace[start + k * step] = amplitude
        trace[start + k * step + 1] = -amplitude
    return trace


# ------------------------------------------------------------ one channel

def test_the_defaults_are_those_of_the_offline_script():
    assert DEFAULT_T1_CONFIG == {
        "t_quiet": 512, "t_period": 512, "t_sepmax": 10, "nc_min": 2,
        "nc_max": 8, "q_min": 0, "q_max": 255, "th1": 100, "th2": 50,
        "t_pretrig": 960, "t_overlap": 64, "t_posttrig": 1024,
        # the offline script's t_sepmax rule; the alternatives are off (#233)
        "sepmax_inclusive": 0, "sepmax_ends_count": 0}


def test_a_clear_pulse_above_threshold_triggers():
    info = extract_trigger_parameters(pulse(400, 4))
    assert info["index_T1_crossing"] == 400
    assert list(info["index_T2_crossing"]) == [400, 402, 404, 406]
    assert info["NC"] == 4
    assert t1_channel_trigger(pulse(400, 4))


def test_noise_below_threshold_does_not_trigger():
    rng = np.random.default_rng(139)
    noise = np.clip(rng.normal(0, 30, N), -99, 99).astype(int)
    with pytest.raises(ValueError, match="No T1 crossing"):
        extract_trigger_parameters(noise)
    assert not t1_channel_trigger(noise)
    # Exactly at the threshold is not above it.
    assert not t1_channel_trigger(np.full(N, 100))


def test_only_the_positive_polarity_crosses_t1():
    assert not t1_channel_trigger(-np.abs(pulse(400, 4)))
    assert t1_channel_trigger(np.abs(pulse(400, 4)) * np.tile([1, 0], N // 2))


def test_the_number_of_crossings_must_be_within_nc_min_and_nc_max():
    assert not t1_channel_trigger(pulse(400, 1))    # NC = 1 < nc_min
    assert t1_channel_trigger(pulse(400, 2))
    assert t1_channel_trigger(pulse(400, 8))
    assert not t1_channel_trigger(pulse(400, 9))    # NC = 9 > nc_max
    assert t1_channel_trigger(pulse(400, 9), {"nc_max": 9})


def test_crossings_must_be_less_than_t_sepmax_apart():
    # 4 samples = 8 ns < 10 ns
    assert extract_trigger_parameters(pulse(400, 2, step=4))["NC"] == 2
    # 5 samples = 10 ns, not < 10 ns: the channel is rejected.
    with pytest.raises(ValueError, match="Tsepmax"):
        extract_trigger_parameters(pulse(400, 2, step=5))
    assert not t1_channel_trigger(pulse(400, 2, step=5))
    assert t1_channel_trigger(pulse(400, 2, step=5), {"t_sepmax": 12})


def test_the_quiet_time_needs_t_quiet_over_two_samples_before_the_crossing():
    half_quiet = DEFAULT_T1_CONFIG["t_quiet"] // 2          # 256 samples
    with pytest.raises(ValueError, match="Not enough data"):
        extract_trigger_parameters(pulse(half_quiet - 1, 3))
    assert t1_channel_trigger(pulse(half_quiet, 3))
    assert t1_channel_trigger(pulse(100, 3), {"t_quiet": 200})


def test_only_crossings_inside_the_window_are_counted():
    window = DEFAULT_T1_CONFIG["t_period"] // 2              # 256 samples
    trace = pulse(300, 2)
    # A later burst just outside the window, which would otherwise break
    # t_sepmax, is ignored.
    trace[300 + window:300 + window + 8] = pulse(0, 4, n=8)
    info = extract_trigger_parameters(trace)
    assert info["NC"] == 2 and t1_channel_trigger(trace)
    # With a wider window it is seen, too far from the first two.
    with pytest.raises(ValueError, match="Tsepmax"):
        extract_trigger_parameters(trace, {"t_period": 2 * DEFAULT_T1_CONFIG["t_period"]})


def test_an_unknown_parameter_is_refused():
    with pytest.raises(KeyError, match="th3"):
        t1_channel_trigger(pulse(400, 3), {"th3": 1})


# ------------------------------------------------------------ every DU

def _event(triggering):
    r"""Five DUs with pulses only in the (DU, channel) pairs given."""
    traces = np.zeros((5, 3, N), dtype=int)
    for du, channel in triggering:
        traces[du, channel] = pulse(400, 3)
    return traces


def test_every_du_is_evaluated_not_only_the_first():
    traces = _event([(3, 2), (4, 1)])
    assert list(t1_du_triggers(traces)) == [False, False, False, True, True]
    flags = t1_trigger_flags(traces)
    assert flags.dtype == np.ushort and list(flags) == [0, 0, 0, 1, 1]


def test_a_du_triggers_on_any_of_the_chosen_channels():
    traces = _event([(0, 0), (1, 1), (2, 2)])
    assert list(t1_du_triggers(traces)) == [True, True, True, False, False]
    assert list(t1_du_triggers(traces, channels=(0, 1))) == [True, True, False, False, False]


def test_the_offline_script_finds_an_entry_triggered_by_a_later_du():
    r"""Before #139 its ``__main__`` looked at ``trace_ch[0]`` only."""
    script = _load("T1_trigger_offline", ROOT / "scripts" / "T1_trigger_offline.py")
    events = [_event([]), _event([(2, 1)]), _event([(0, 0)])]

    class FakeTADC:
        def get_number_of_entries(self):
            return len(events)

        def get_entry(self, k):
            self.trace_ch = events[k]

    assert script.triggered_entries(FakeTADC()) == [1, 2]
    assert script.dict_trigger_parameter == DEFAULT_T1_CONFIG


# ------------------------------------------------------------ the converter

def test_the_converter_leaves_the_trigger_off_by_default(monkeypatch):
    script = _voltage2adc()
    args = script.manage_args(["some_dir"])
    assert args.t1_trigger is False and args.t1_param is None
    args = script.manage_args(["some_dir", "--t1_trigger", "--t1_param", "th1=120",
                               "--t1_param", "nc_max=10"])
    config = script.t1_config_from_params(args.t1_param)
    assert args.t1_trigger and config["th1"] == 120 and config["nc_max"] == 10
    assert script.t1_config_from_params(None) == DEFAULT_T1_CONFIG
    for bad in ("th1", "th3=1", "th1=high"):
        with pytest.raises(ValueError, match="--t1_param"):
            script.t1_config_from_params([bad])


def test_the_converter_writes_one_flag_per_du():
    script = _voltage2adc()
    tadc = types.SimpleNamespace()
    flags = script.apply_t1_trigger(tadc, _event([(1, 0)]), dict(DEFAULT_T1_CONFIG))
    assert list(tadc.trigger_flag) == [0, 1, 0, 0, 0]
    assert tadc.trigger_flag is flags


def _has_psutil():
    r"""Whether the real psutil, which the script run below needs, exists.

    Checked in a child process: other tests put a stand-in in sys.modules.
    """
    return subprocess.run([sys.executable, "-c", "import psutil"],
                          capture_output=True).returncode == 0


def _dump(path):
    r"""Every branch of every entry of every tree in a ROOT file, as text."""
    import ROOT as R

    f = R.TFile.Open(str(path))
    out = {}
    for key in f.GetListOfKeys():
        tree = f.Get(key.GetName())
        rows = []
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            row = {}
            for branch in tree.GetListOfBranches():
                value = getattr(tree, branch.GetName())
                try:
                    value = [list(x) if hasattr(x, "__len__") and not isinstance(x, str)
                             else x for x in value]
                except TypeError:
                    pass
                row[branch.GetName()] = repr(value)
            rows.append(row)
        out[key.GetName()] = rows
    f.Close()
    return out


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="the sample run is not present")
@pytest.mark.skipif(not _has_psutil(), reason="psutil is not installed")
def test_the_option_changes_only_trigger_flag(tmp_path):
    r"""Without --t1_trigger trigger_flag stays as it was (empty); with it,
    it holds one flag per DU, and nothing else in the output changes."""
    env = dict(os.environ)
    # This checkout's grand, not whichever one is installed.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    outputs = {}
    for name, extra in (("default", []), ("t1", ["--t1_trigger"])):
        run = tmp_path / name
        shutil.copytree(SAMPLE, run, ignore=shutil.ignore_patterns("adc_*"))
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "convert_voltage2adc.py"),
                               str(run), *extra],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr[-2000:]
        outputs[name] = _dump(run / "adc_1618-13790_L1_0000.root")

    default, t1 = outputs["default"]["tadc"], outputs["t1"]["tadc"]
    assert len(default) == len(t1) == 2
    for row_default, row_t1 in zip(default, t1):
        assert row_default["trigger_flag"] == "[]"
        n_du = len(ast.literal_eval(row_t1["du_id"]))
        flags = ast.literal_eval(row_t1["trigger_flag"])
        assert len(flags) == n_du and set(flags) <= {0, 1}
        row_default.pop("trigger_flag")
        row_t1.pop("trigger_flag")
        assert row_default == row_t1


def _t1_script(*args, cwd):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "T1_trigger_offline.py"), *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def test_t1_script_parses_its_arguments(tmp_path):
    r"""#183: -h was opened as a file, and no argument gave IndexError."""
    helped = _t1_script("-h", cwd=tmp_path)
    assert helped.returncode == 0 and "adc_file" in helped.stdout
    bare = _t1_script(cwd=tmp_path)
    assert bare.returncode == 2 and "usage" in bare.stderr
    bad = _t1_script(ROOT / "nothing.root", "--t1_param", "th9=1", cwd=tmp_path)
    assert bad.returncode != 0 and "GRANDlib: T1_trigger_offline" in bad.stderr

    adc = next((ROOT / "sim2root" / "Common").glob("sim_*/adc_*_L1_*.root"), None)
    if adc is None:
        pytest.skip("no committed ADC file")
    out = tmp_path / "list.txt"
    ran = _t1_script(adc, "-o", out, "--t1_param", "th1=1", "--t1_param", "th2=1", "--t1_param", "nc_min=1",
                     cwd=tmp_path)
    assert ran.returncode == 0, ran.stderr[-2000:]
    assert "triggered" in ran.stdout
    if out.exists():
        assert "'th1': 1" in out.read_text().splitlines()[0]


def test_a_nan_trace_is_refused():
    r"""#288: a NaN trace still triggered."""
    from grand.sim.detector.trigger import t1_du_triggers

    traces = np.zeros((2, 3, 2048))
    traces[1, 0, 100] = np.nan
    with pytest.raises(ValueError, match="unit 1 .*NaN"):
        t1_du_triggers(traces)


@pytest.mark.parametrize("f_mhz,start", [(60, 400), (100, 400), (200, 400), (100, 100)])
def test_clean_pulses_do_not_pass_with_the_defaults(f_mhz, start):
    r"""#233: the diagnosis recorded in known_issues (issue-t1-clean-simulations).

    A pulse before sample t_quiet/2 is rejected, and a clean pulse's T2
    crossings are at least t_sepmax = 10 ns apart, which rejects the channel.
    If the trigger group changes either rule, update the known issue.
    """
    from grand.sim.detector.trigger import t1_du_triggers

    t = np.arange(1024) * 2.0
    pulse = np.where(t >= start * 2, 850 * np.exp(-(t - start * 2) / 30)
                     * np.sin(2 * np.pi * f_mhz * 1e-3 * (t - start * 2)), 0)
    traces = np.stack([pulse, pulse, pulse])[None]
    assert not t1_du_triggers(traces).any()


def _clean_pulse(f_mhz, start=400):
    t = np.arange(1024) * 2.0
    return np.where(t >= start * 2, 850 * np.exp(-(t - start * 2) / 30)
                    * np.sin(2 * np.pi * f_mhz * 1e-3 * (t - start * 2)), 0)


def test_the_t_sepmax_rule_can_be_chosen():
    r"""#233: sepmax_inclusive and sepmax_ends_count, off by default, for the trigger group to compare."""
    from grand.sim.detector.trigger import (extract_trigger_parameters, t1_config_from_params)

    pulse = _clean_pulse(100)                  # crossings exactly 10 ns apart
    with pytest.raises(ValueError, match="Violating Tsepmax"):
        extract_trigger_parameters(pulse)      # the offline script's rule
    assert extract_trigger_parameters(pulse, {"sepmax_inclusive": 1})["NC"] > 1
    assert extract_trigger_parameters(_clean_pulse(60), {"sepmax_ends_count": 1})["NC"] == 1
    config = t1_config_from_params(["sepmax_inclusive=1", "sepmax_ends_count=1"])
    assert config["sepmax_inclusive"] == config["sepmax_ends_count"] == 1
    with pytest.raises(ValueError, match="sepmax_ends_count must be 0 or 1"):
        t1_config_from_params(["sepmax_ends_count=2"])
