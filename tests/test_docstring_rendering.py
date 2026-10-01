# -*- coding: utf-8 -*-
r"""#261, items 6, 9 and 10: docstrings that misdescribed or did not render.

Doxygen ``@param`` tags and an "Arguments" heading are not numpydoc, so
Sphinx showed them as plain text; several aoi and analysis docstrings named
the wrong argument or did not say what they return.
"""

import inspect
import pathlib
import re

import numpy as np

GRAND = pathlib.Path(__file__).resolve().parents[1] / "grand"


def test_no_doxygen_tags_or_unknown_headings():
    for path in GRAND.rglob("*.py"):
        text = path.read_text()
        assert "@param" not in text, path
        assert not re.search(r"\n\s*Arguments\n\s*-{5,}\n", text), path


def test_timetrace_documents_its_real_argument():
    from grand.aoi.timetrace import Timetrace3D

    for name in ("get_value_at_time", "get_hilbert_value_at_time"):
        method = getattr(Timetrace3D, name)
        assert list(inspect.signature(method).parameters)[1] == "time_offset"
        assert "time_offset : float" in method.__doc__ and "no interpolation" in method.__doc__


def test_cramer_rao_bounds_say_what_they_return():
    from grand.analysis.cramer_rao_bounds.cramer_rao import CRB_ADF_SWF, CRB_PWF

    assert "standard deviations" in CRB_PWF.__doc__ and "radians" in CRB_PWF.__doc__
    assert "shape (8,)" in CRB_ADF_SWF.__doc__


def test_shower_direction_vector_is_the_propagation_direction():
    from grand.analysis.coords.array_shower import shower_direction_vector
    from grand.dataio.xmax_frame import arrival_direction

    np.testing.assert_allclose(shower_direction_vector(np.radians(60), np.radians(30)),
                               -np.asarray(arrival_direction(60, 30)))
    assert "propagates" in shower_direction_vector.__doc__
