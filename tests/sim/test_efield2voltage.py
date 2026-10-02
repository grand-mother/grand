"""
Unit tests for the grand.sim.efield2voltage module.

Jun 19, 2023.  Rewritten on the committed sample, in a temporary folder: it
read the untracked data/test_efield.root and wrote into data/ (#271).
"""
import pathlib
import shutil

import pytest

from grand import Efield2Voltage
from grand.dataio import TVoltage

ROOT = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "detector").exists() or not SAMPLE.is_dir(),
                                reason="needs the data model and the RUN1 sample")


def test_Efield2Voltage(tmp_path):
    folder = tmp_path / "sample"
    shutil.copytree(SAMPLE, folder)
    master = Efield2Voltage(str(folder), "out_voltage.root", output_directory=str(tmp_path), seed=1)
    assert master.params["add_noise"]
    assert master.params["add_rf_chain"]
    assert master.params["lst"] == 18

    master.compute_voltage()    # saves automatically

    out = tmp_path / "out_voltage.root"
    assert out.exists()
    with TVoltage(str(out)) as voltage:
        assert sorted(voltage.get_list_of_events()) == [(1618, 1), (13790, 1)]
