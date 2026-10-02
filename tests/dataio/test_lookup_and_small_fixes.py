"""The smaller dataio inconsistencies of #206, on a copy of the committed sample."""
import glob
import pathlib
import shutil
import warnings

import numpy as np
import pytest
import ROOT

from grand.basis.validate import GRANDlibWarning
from grand.dataio import TEfield, TRecons, TRun, TShower

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
SAMPLE = ROOT_DIR / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"

pytestmark = pytest.mark.skipif(not SAMPLE.is_dir(), reason="needs the RUN1 sample")


@pytest.fixture
def sample(tmp_path):
    for f in SAMPLE.glob("*.root"):
        shutil.copy(f, tmp_path)
    return tmp_path


def _one(folder, prefix):
    return glob.glob(str(folder / (prefix + "_*.root")))[0]


def test_failed_lookups_raise(sample):
    shower = TShower(_one(sample, "shower"))
    shower.get_event(1618, 1)
    energy = shower.energy_primary
    with pytest.raises(LookupError, match="get_event: no event 999 in run 1"):
        shower.get_event(999, 1)
    with pytest.raises(LookupError, match="no event 1618 in run 0"):
        shower.get_event(1618)
    with pytest.raises(IndexError, match="no entry -1; the tshower tree has 2 entries"):
        shower.get_entry(-1)
    with pytest.raises(IndexError, match="no entry 10"):
        shower.get_entry(10)
    assert shower.energy_primary == energy          # the loaded event is kept
    assert shower.has_event(1618, 1) and not shower.has_event(999, 1)

    run = TRun(_one(sample, "run"))
    with pytest.raises(LookupError, match="get_run: no run 999 in the trun tree"):
        run.get_run(999)
    assert run.has_run(1) and not run.has_run(999)

    empty = TShower()
    with pytest.raises(IndexError, match="no entry 0; the tshower tree has 0 entries"):
        empty.get_entry(0)


def test_get_entry_with_index_is_keyword_only(sample):
    shower = TShower(_one(sample, "shower"))
    with pytest.raises(TypeError, match="give run_no= and evt_no= by name"):
        shower.get_entry_with_index(1618, 1)
    assert shower.get_entry_with_index(run_no=1, evt_no=1618) > 0
    assert shower.event_number == 1618
    with pytest.raises(LookupError, match="no event 1 in run 1618"):
        shower.get_entry_with_index(run_no=1618, evt_no=1)


def test_in_memory_tree_finds_filled_events():
    shower = TShower()
    shower.run_number, shower.event_number = 1, 5
    shower.fill()
    assert shower.get_event(5, 1) > 0 and shower.has_event(5, 1)
    shower.run_number, shower.event_number = 1, 6
    shower.fill()
    assert shower.get_event(6, 1) > 0 and shower.event_number == 6


def test_number_of_events_docstring():
    assert "differs" not in TShower.get_number_of_events.__doc__


def test_wrong_tree_name_does_not_add_an_empty_tree(sample):
    name = _one(sample, "shower")
    efield = TEfield(name)
    assert efield.get_list_of_events() == []
    with pytest.warns(GRANDlibWarning, match="held no tefield tree"):
        efield.write()
    efield.stop_using()
    f = ROOT.TFile(name)
    try:
        assert [k.GetName() for k in f.GetListOfKeys()] == ["tshower"]
    finally:
        f.Close()


def test_setter_messages():
    shower = TShower()
    with pytest.raises(TypeError, match="TShower.primary_type: must be a string, got int"):
        shower.primary_type = 5
    with pytest.raises(TypeError, match="energy_primary: must be a number, got None"):
        shower.energy_primary = None
    with pytest.raises(TypeError, match="must be a number, got None"):
        shower.run_number = None
    shower.energy_primary = np.nan                 # NaN still means unknown


@pytest.mark.parametrize("field", ["zenith_pwf", "zenith_swf", "zenith_adf", "crb_zenith_adf"])
def test_recons_angles_are_radians(field):
    recons = TRecons()
    with pytest.warns(GRANDlibWarning, match="between 0 and 3.14.* radians, got 85"):
        setattr(recons, field, 85.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        setattr(recons, field, 1.2)


def test_run_du_id_matches_event_trees():
    run = TRun()
    with pytest.warns(GRANDlibWarning, match="between 0 and 65535"):
        run.du_id = [70000]
    with pytest.warns(GRANDlibWarning, match="between 0 and 65535"):
        run.du_id = [-1]
