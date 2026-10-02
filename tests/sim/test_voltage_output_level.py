# -*- coding: utf-8 -*-
r"""The default voltage file name carries the level of the fields it was computed from.

On a folder holding electric fields at levels 0 and 1, ``efield_level=0``
read the level-0 fields but named the output after the level-1 file,
``voltage_*_L1_*``, while the tree inside said level 0.
"""

import pathlib

import pytest

SAMPLE = (pathlib.Path(__file__).resolve().parents[2] / "sim2root" / "Common"
          / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000")

pytestmark = pytest.mark.skipif(not (SAMPLE / "efield_1618-13790_L1_0000.root").is_file(),
                                reason="the sample run with two levels is not present")


def test_default_name_follows_the_level_read(tmp_path):
    from grand import Efield2Voltage
    from grand.dataio import TVoltage

    sim = Efield2Voltage(str(SAMPLE), output_directory=str(tmp_path), seed=1, efield_level=0)
    sim.compute_voltage()

    assert [p.name for p in tmp_path.glob("*.root")] == ["voltage_1618-13790_L0_0000.root"]
    with TVoltage(str(tmp_path / "voltage_1618-13790_L0_0000.root")) as tvoltage:
        assert tvoltage.analysis_level == 0
