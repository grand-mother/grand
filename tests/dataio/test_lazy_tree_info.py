"""DataFile computes a tree's list of units on first use (#283)."""
import pathlib

import pytest

from grand.dataio import DataFile

SAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "sim2root" / "Common"
          / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")

pytestmark = pytest.mark.skipif(not SAMPLE.is_dir(), reason="needs the RUN1 sample")


def test_dus_are_listed_on_first_use(monkeypatch):
    calls = []
    real = DataFile._get_list_of_all_used_dus
    monkeypatch.setattr(DataFile, "_get_list_of_all_used_dus",
                        lambda self, tree: calls.append(tree) or real(self, tree))
    files = sorted(str(f) for f in SAMPLE.glob("efield_*_L0_*.root"))
    df = DataFile(files * 2)                    # a chain, as DataDirectory builds
    assert calls == []                          # not at open
    info = next(iter(df.tree_types["TEfield"].values()))
    expected = sorted(df.tefield.get_list_of_all_used_dus())
    assert sorted(info["dus"]) == expected and len(calls) == 1
    assert "dus" in info and sorted(info.get("dus")) == expected and len(calls) == 1
    run = next(iter(df.tree_types.get("TRun", {}).values()), None)
    if run is not None:
        assert "dus" not in run and run.get("dus") is None
