# -*- coding: utf-8 -*-
r"""Fixes from the dev-next beta test to High issues in ``grand.dataio``.

Each test names its GitHub issue and fails without the fix.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
AOI = ROOT / "examples" / "analysis" / "reconstructed_events_AOI"


def test_data_directory_keeps_every_file_of_a_level():
    r"""#195: files of one level with names of different lengths replaced each other."""
    from grand.dataio import DataDirectory

    d = DataDirectory(str(AOI))
    assert len(list(AOI.glob("shower_*_L1_*.root"))) == 10
    assert d.tshower.get_number_of_entries() == 10
    assert len(d.tshower.get_list_of_events()) == 10
