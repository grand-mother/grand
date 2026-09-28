# -*- coding: utf-8 -*-
r"""Regression tests for four GitHub issues fixed together in September 2026.

Each test checks the behaviour the issue reported, and that what already
worked still works:

- #95: ``EventList.get_event`` with an event number the input does not hold;
- #122: noise files for ``scripts/convert_voltage2adc.py`` given a directory
  without a trailing '/';
- #123: the hadronic model and CoREAS version read from a CORSIKA log;
- #136 (found while triaging it): ``creation_datetime`` and
  ``source_datetime`` read back as integers after being set to a datetime.
"""

import contextlib
import datetime
import importlib.util
import io
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE = (ROOT / "sim2root" / "Common"
          / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")


def _load(name, path):
    r"""Imports a script or converter module by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ #95

@pytest.mark.skipif(not SAMPLE.is_dir(), reason="the sample run is not present")
def test_an_unknown_event_number_returns_none_not_a_stale_event():
    r"""Asking for an event the input does not hold gives ``None``.

    Before, a fresh list crashed in the reader ("zero-size array"), and after
    a valid event the call returned an Event labelled with the requested
    number but still holding the previous event's traces.
    """
    from grand.aoi.event_list import EventList

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        events = EventList(str(SAMPLE))
        assert events.get_event(event_number=999999, run_number=1) is None

        first = events.get_event(event_number=1618, run_number=1)
        assert (first.event_number, len(first.efields)) == (1618, 5)
        assert events.get_event(event_number=999999, run_number=1) is None

        # A valid event after a refused one still loads, with its own data.
        other = events.get_event(event_number=13790, run_number=1)
        assert (other.event_number, len(other.efields)) == (13790, 44)
    assert "No event with event number 999999" in quiet.getvalue()


# ------------------------------------------------------------------ #122

def _voltage2adc():
    r"""Imports ``scripts/convert_voltage2adc.py`` without needing psutil."""
    sys.modules.setdefault("psutil", types.ModuleType("psutil"))
    return _load("convert_voltage2adc", ROOT / "scripts" / "convert_voltage2adc.py")


def test_a_noise_directory_is_found_with_or_without_a_trailing_slash(tmp_path):
    r"""The files inside the directory, either way (#122)."""
    script = _voltage2adc()
    noise = tmp_path / "noise"
    noise.mkdir()
    for name in ("b.root", "a.root", "notes.txt"):
        (noise / name).write_text("")
    # A sibling that the old code, without the slash, would have matched.
    (tmp_path / "noise_other.root").write_text("")

    expected = [str(noise / "a.root"), str(noise / "b.root")]
    assert script.noise_files(str(noise) + "/") == expected
    assert script.noise_files(str(noise)) == expected


def test_a_path_prefix_still_works_and_nothing_found_is_an_error(tmp_path):
    r"""A prefix keeps its old meaning; an empty match no longer passes."""
    script = _voltage2adc()
    (tmp_path / "GP80_2025_x.root").write_text("")
    (tmp_path / "GP80_2024_y.root").write_text("")
    assert script.noise_files(str(tmp_path / "GP80_2025")) == [
        str(tmp_path / "GP80_2025_x.root")]
    with pytest.raises(FileNotFoundError):
        script.noise_files(str(tmp_path / "empty_dir_that_does_not_exist/"))


# ------------------------------------------------------------------ #123

def _corsika_info():
    return _load("corsika_info_funcs",
                 ROOT / "sim2root" / "CoREASRawRoot" / "CorsikaInfoFuncs.py")


@pytest.mark.parametrize("line, model", [
    # The one case the old code recognised: the output must not change.
    ("              S I B Y L L  2.3d  (ORIGINAL)", "Sibyll 2.3d"),
    (" SIBYLL 2.3c", "Sibyll 2.3c"),
    (" QGSJET-II-04 INTERACTION MODEL", "QGSJET-II-04"),
    (" WITH EPOS LHC", "EPOS LHC"),
    (" nothing about a model here", "n/a"),
])
def test_the_hadronic_model_is_read_not_matched_literally(tmp_path, line, model):
    r"""Other models and versions are named, not reported as "n/a" (#123)."""
    log = tmp_path / "run.log"
    log.write_text("CORSIKA header\n%s\nmore output\n" % line)
    with contextlib.redirect_stdout(io.StringIO()):
        assert _corsika_info().read_HADRONIC_INTERACTION(str(log)) == model


@pytest.mark.parametrize("line, version", [
    (" CoREAS V1.4 by T. Huege", "1.4"),       # unchanged
    (" CoREAS V1.4.2", "1.4.2"),
    (" CoREAS V2.0", "2.0"),
    (" no version", "n/a"),
])
def test_the_coreas_version_is_read_not_matched_literally(tmp_path, line, version):
    r"""Any version after "CoREAS V", not only 1.4 (#123)."""
    log = tmp_path / "run.log"
    log.write_text("header\n%s\n" % line)
    with contextlib.redirect_stdout(io.StringIO()):
        assert _corsika_info().read_coreas_version(str(log)) == version


# ------------------------------------------------------------------ #136

@pytest.mark.parametrize("field", ["creation_datetime", "source_datetime"])
def test_a_datetime_set_on_a_tree_reads_back_as_that_datetime(field):
    r"""Setting a datetime used to read back as its integer timestamp."""
    from grand.dataio.data_tree import DataTree

    tree = DataTree()
    moment = datetime.datetime(2025, 1, 1, 12, 0, 0)
    setattr(tree, field, moment)
    assert getattr(tree, field) == moment

    # An integer timestamp still reads back as the matching datetime.
    setattr(tree, field, int(moment.timestamp()))
    assert getattr(tree, field) == moment
