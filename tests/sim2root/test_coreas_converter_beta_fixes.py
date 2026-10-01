# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to the CoREAS converter.

Each test names its GitHub issue and fails without the fix.  The converter
writes into its working folder, so it runs on a copy of
``sim2root/CoREASRawRoot`` in a temporary directory.
"""

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
COREAS = ROOT / "sim2root" / "CoREASRawRoot"

pytestmark = pytest.mark.skipif(not (COREAS / "proton" / "SIM004100.reas").exists(),
                                reason="the committed CoREAS sample is not present")

#: Lines of the full .reas that carry the shower parameters; the committed
#: SIM004100.reas lacks them, which sends the converter to the CORSIKA .inp.
EVENT_KEYS = ("ShowerZenithAngle", "ShowerAzimuthAngle", "PrimaryParticleEnergy", "PrimaryParticleType",
              "DepthOfShowerMaximum", "DistanceOfShowerMaximum", "MagneticFieldStrength",
              "MagneticFieldInclinationAngle", "GeomagneticAngle")


def _workdir(tmp_path, with_event_block):
    work = tmp_path / "CoREASRawRoot"
    shutil.copytree(COREAS, work, ignore=shutil.ignore_patterns("*.rawroot", "__pycache__"))
    if with_event_block:
        full = (work / "proton" / "SIM004100-001004105-000000001.reas").read_text().splitlines()
        block = [line for line in full if line.split("=")[0].strip() in EVENT_KEYS]
        reas = work / "proton" / "SIM004100.reas"
        reas.write_text(reas.read_text().rstrip("\n") + "\n" + "\n".join(block) + "\n")
    return work


def _convert(work):
    from sim2root.Common.raw_root_trees import RawShowerTree

    done = subprocess.run([sys.executable, "CoreasToRawROOT.py", "--file", "proton/SIM004100.reas"],
                          cwd=work, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    shower = RawShowerTree(str(work / "Coreas_004100.rawroot"))
    shower.get_entry(0)
    angles = float(shower.zenith), float(shower.azimuth)
    shower.stop_using()
    return angles


@pytest.mark.parametrize("with_event_block", [False, True], ids=["inp_fallback", "reas_block"])
def test_azimuth_is_where_the_shower_comes_from(tmp_path, with_event_block):
    r"""#209: the .inp fallback gave 180 - PHIP = -13.57 (mirrored); both paths must give +13.57.

    CORSIKA PHIP = 193.57 and the .reas ShowerAzimuthAngle = -166.43 are both the
    direction of travel; GRAND's "comes from" azimuth is 13.57 degrees, which is
    also what a plane-wave fit to the sample's antenna timing gives.
    """
    zenith, azimuth = _convert(_workdir(tmp_path, with_event_block))
    assert zenith == pytest.approx(55.0, abs=1e-3)
    assert azimuth == pytest.approx(13.57, abs=1e-3)


def test_an_unknown_xmax_distance_is_written_as_nan(tmp_path):
    r"""#228: DistanceOfShowerMaximum = -1 ("unknown") was used as -1 cm, putting Xmax at the core."""
    import numpy as np

    from sim2root.Common.raw_root_trees import RawShowerTree

    work = _workdir(tmp_path, with_event_block=True)   # its block has DistanceOfShowerMaximum = -1
    _convert(work)
    shower = RawShowerTree(str(work / "Coreas_004100.rawroot"))
    shower.get_entry(0)
    assert np.all(np.isnan(np.asarray(shower.xmax_pos_shc, dtype=float)))
    shower.stop_using()


def _damage(work, how):
    proton = work / "proton"
    traces = sorted((proton / "SIM004100_coreas").glob("raw_*.dat"))
    if how == "truncated_list":
        lines = (proton / "SIM004100.list").read_text().splitlines()
        (proton / "SIM004100.list").write_text("\n".join(lines[:100]) + "\n")
    elif how == "extra_trace":
        shutil.copy(traces[0], traces[0].with_name("raw_extra999.dat"))
    elif how == "nan_trace":
        rows = traces[0].read_text().splitlines()
        parts = rows[10].split()
        rows[10] = " ".join([parts[0], "nan", "nan", "nan"])
        traces[0].write_text("\n".join(rows) + "\n")
    elif how == "short_trace":
        rows = traces[0].read_text().splitlines()
        traces[0].write_text("\n".join(rows[:50]) + "\n")


@pytest.mark.parametrize("how, message", [
    ("truncated_list", "trace files are not in the antenna list"),
    ("extra_trace", "trace files are not in the antenna list"),
    ("nan_trace", "NaN or infinite"),
    ("short_trace", "different lengths"),
])
def test_antenna_list_and_traces_are_cross_checked(tmp_path, how, message):
    r"""#243: a list/trace mismatch, NaN or ragged traces were written silently (exit 0)."""
    work = _workdir(tmp_path, with_event_block=False)
    _damage(work, how)
    done = subprocess.run([sys.executable, "CoreasToRawROOT.py", "--file", "proton/SIM004100.reas"],
                          cwd=work, capture_output=True, text=True, timeout=900)
    assert done.returncode != 0
    assert message in done.stderr
    assert not (work / "Coreas_004100.rawroot").exists()


def _run(work, *args):
    return subprocess.run([sys.executable, "CoreasToRawROOT.py", *args], cwd=work,
                          capture_output=True, text=True, timeout=900)


def test_an_existing_output_is_refused_or_replaced(tmp_path):
    r"""#181: a second conversion appended to the previous output and failed with NotUniqueEvent."""
    work = _workdir(tmp_path, with_event_block=False)
    first = _run(work, "--file", "proton/SIM004100.reas")
    assert first.returncode == 0, first.stderr[-2000:]
    assert (work / "Coreas_004100.rawroot").is_file()

    again = _run(work, "--file", "proton/SIM004100.reas")
    assert again.returncode != 0
    assert "GRANDlib: CoreasToRawROOT:" in again.stderr and "--overwrite" in again.stderr
    assert "NotUniqueEvent" not in again.stderr

    replaced = _run(work, "--file", "proton/SIM004100.reas", "--overwrite")
    assert replaced.returncode == 0, replaced.stderr[-2000:]

    elsewhere = _run(work, "-d", "proton", "-o", str(tmp_path / "out" / "raw"))
    assert elsewhere.returncode == 0, elsewhere.stderr[-2000:]
    assert (tmp_path / "out" / "raw" / "Coreas_004100.rawroot").is_file()


def test_no_option_is_an_error(tmp_path):
    work = _workdir(tmp_path, with_event_block=False)
    assert _run(work).returncode != 0
