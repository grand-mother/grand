# -*- coding: utf-8 -*-
r"""What the χ² and bound fields of `TRecons` hold (#211)."""

import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CANDIDATES = ROOT / "examples" / "analysis" / "recons_CR_candidates.root"


def test_unfilled_chi2_and_bounds_read_nan(tmp_path):
    r"""An unfilled χ² or Cramér-Rao bound read 0.0: a perfect fit, no uncertainty."""
    from grand.dataio import TRecons

    name = str(tmp_path / "recons.root")
    t = TRecons(name)
    t.run_number, t.event_number = 1, 1
    t.zenith_pwf = 1.0
    t.fill()
    t.write()
    t.stop_using()

    t = TRecons(name)
    t.get_entry(0)
    assert np.isnan(t.chi2_pwf) and np.isnan(t.chi2_adf)
    assert np.isnan(t.crb_zenith_pwf) and np.isnan(t.crb_t_s)
    assert t.zenith_pwf == pytest.approx(1.0)
    t.stop_using()


@pytest.mark.skipif(not CANDIDATES.is_file(), reason="the committed file is not present")
def test_the_committed_candidates_hold_the_raw_chi2():
    r"""``chi2_pwf`` is the raw χ², as `TRecons` documents; notebook 11 divides it."""
    import grand.analysis.fitting as fit
    from grand.dataio import TRecons

    r = TRecons(str(CANDIDATES))
    for event in r.get_list_of_events():
        r.get_event(*event)
        raw = fit.PWF_loss((r.zenith_pwf, r.azimuth_pwf), np.asarray(r.Xants),
                           np.asarray(r.peak_time), sigma=5e-9)
        assert r.chi2_pwf == pytest.approx(raw, rel=1e-4)
    r.stop_using()

