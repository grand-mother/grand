# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to the ZHAireS converter (issue #242).

A damaged or incomplete ZHAireS simulation used to become valid-looking data:
a missing trace file was dropped, a truncated trace was zero-padded, a missing
zenith line gave a vertical shower, and so on.  Each case below damages one
file of a copy of the committed sample, and checks that the converter now
exits non-zero, names the problem, and leaves no output file behind.
"""

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONVERTER = ROOT / "sim2root" / "ZHAireSRawRoot" / "ZHAireSRawToRawROOT.py"
SAMPLE = ROOT / "sim2root" / "ZHAireSRawRoot" / "GP300_Xi_Sib_Proton_3.8_51.6_135.4_1618"
NAME = SAMPLE.name

pytestmark = pytest.mark.skipif(not (SAMPLE / (NAME + ".sry")).exists(),
                                reason="the committed ZHAireS sample is not present")


def _copy(tmp_path):
    folder = tmp_path / NAME
    shutil.copytree(SAMPLE, folder)
    return folder


def _run(folder):
    output = folder.parent / "out.rawroot"
    done = subprocess.run([sys.executable, str(CONVERTER), str(folder), "standard", "0", "1618",
                           str(output)], cwd=folder.parent, capture_output=True, text=True, timeout=600)
    return done, output


def _edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def _remove_a2(folder):
    (folder / "a2.trace").unlink()


def _truncate_a1(folder):
    lines = (folder / "a1.trace").read_text().splitlines(keepends=True)
    (folder / "a1.trace").write_text("".join(lines[:100]))


def _nan_in_a1(folder):
    lines = (folder / "a1.trace").read_text().splitlines(keepends=True)
    lines[50] = "1.0 nan 0.0 0.0\n"
    (folder / "a1.trace").write_text("".join(lines))


def _three_columns_in_a1(folder):
    lines = (folder / "a1.trace").read_text().splitlines()
    (folder / "a1.trace").write_text("".join(" ".join(line.split()[:3]) + "\n" for line in lines))


def _no_zenith(folder):
    sry = folder / (NAME + ".sry")
    sry.write_text("".join(line for line in sry.read_text().splitlines(keepends=True)
                           if "Primary zenith angle" not in line))


def _unknown_energy_unit(folder):
    _edit(folder / (NAME + ".sry"), "Primary energy: 3.7977 EeV", "Primary energy: 3.7977 parsec")


def _empty_event_parameters(folder):
    (folder / (NAME + ".EventParameters")).write_text("")


def _nan_antenna_position(folder):
    _edit(folder / (NAME + ".sry"), "A14                    195.39", "A14                    nan")


def _duplicate_antenna_name(folder):
    _edit(folder / (NAME + ".sry"), "A21    ", "A14    ")


DAMAGE = {
    "missing_trace": (_remove_a2, "missing a<i>.trace for i in [2]"),
    "truncated_trace": (_truncate_a1, "the trace has 100 samples"),
    "nan_trace": (_nan_in_a1, "non-finite values"),
    "three_column_trace": (_three_columns_in_a1, "must have 4 columns"),
    "no_zenith": (_no_zenith, "no 'Primary zenith angle:' line"),
    "unknown_energy_unit": (_unknown_energy_unit, "unknown energy unit 'parsec'"),
    "empty_event_parameters": (_empty_event_parameters, "no 'Core Position:' line"),
    "nan_antenna_position": (_nan_antenna_position, "a position or t0 is not finite"),
    "duplicate_antenna_name": (_duplicate_antenna_name, "is also the name of antenna 1"),
}


@pytest.mark.parametrize("case", sorted(DAMAGE))
def test_a_damaged_simulation_is_refused(tmp_path, case):
    r"""#242: each damage used to give exit code 0 and a valid-looking rawroot file."""
    damage, message = DAMAGE[case]
    folder = _copy(tmp_path)
    damage(folder)
    done, output = _run(folder)
    assert done.returncode != 0
    assert "GRANDlib:" in done.stderr and message in done.stderr, done.stderr[-2000:]
    assert not output.exists()


def test_the_intact_sample_still_converts(tmp_path):
    r"""#242: the checks accept the committed sample, with its 5 antennas."""
    from sim2root.Common.raw_root_trees import RawEfieldTree

    done, output = _run(_copy(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    efield = RawEfieldTree(str(output))
    efield.get_entry(0)
    assert efield.du_count == 5
    assert list(efield.du_id) == [14, 21, 27, 28, 35]
    efield.stop_using()


def test_a_missing_event_parameters_file_warns(tmp_path):
    r"""#242: without .EventParameters the core is (0, 0, 0), as before, but now said loudly."""
    folder = _copy(tmp_path)
    (folder / (NAME + ".EventParameters")).unlink()
    done, output = _run(folder)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "WARNING" in done.stderr and "using core position (0, 0, 0)" in done.stderr
