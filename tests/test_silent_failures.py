# -*- coding: utf-8 -*-
r"""#256: failures that were silent, or global state changed behind the user's back."""

import logging
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_newton_reports_non_convergence():
    from grand.analysis.physics.cherenkov_angle import newton
    from grand.basis.validate import GRANDlibWarning

    with pytest.warns(GRANDlibWarning, match="no convergence in 5 iterations"):
        assert np.isnan(newton(lambda x: x ** 2 + 1, 1.0, nstep_max=5))
    assert newton(lambda x: x ** 2 - 4, 0.0) == pytest.approx(2.0)   # started at 0
    assert newton(lambda x: x - 3, 1.0) == pytest.approx(3.0)


def test_importing_grand_leaves_complex_warnings_alone():
    r"""After ``import grand.geo.coordinates`` a ComplexWarning in user code raised."""
    import subprocess
    import sys

    code = ("import numpy as np, grand.geo.coordinates\n"
            "print(np.array([1 + 1j]).astype(float)[0])\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-1500:]
    assert done.stdout.strip() == "1.0"


def test_logging_setup_does_not_stack_handlers():
    import grand.manage_log as mlg

    roots = ["grand_test_256"]
    for _ in range(3):
        mlg.create_output_for_logger("info", log_stdout=True, log_root=roots)
    assert roots == ["grand_test_256"]
    managed = [h for h in logging.getLogger("grand_test_256").handlers
               if getattr(h, "_grand_managed", False)]
    assert len(managed) == 1


@pytest.mark.skipif(not (ROOT / "data" / "topography" / "N41E096.hgt").exists(),
                    reason="needs a topography tile")
def test_turtle_maps_are_cached_by_resolved_path():
    from grand.geo.turtle import Map

    tile = ROOT / "data" / "topography" / "N41E096.hgt"
    assert Map(str(tile)) is Map(tile)


@pytest.mark.skipif(not (ROOT / "sim2root" / "Common").is_dir(), reason="needs the samples")
def test_event_list_raises_instead_of_printing():
    from grand.aoi import EventList

    sample = ROOT / "sim2root" / "Common" / "sim_Xiaodushan_20221026_000000_RUN1_CD_ZHAireS_0000"
    events = EventList(str(sample))
    with pytest.raises(LookupError, match="no event with event number 7"):
        events.get_event(event_number=7, run_number=1)
    with pytest.raises(ValueError, match="not both"):
        events.get_event(entry_number=0, event_number=1618, run_number=1)
