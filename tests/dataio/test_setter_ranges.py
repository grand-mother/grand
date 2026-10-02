# -*- coding: utf-8 -*-
r"""#267: tree setters that stored out-of-range values without a word."""

import pytest

from grand.basis.validate import GRANDlibWarning
from grand.dataio import TEfield, TRun, TShower


@pytest.mark.parametrize("make, field, value, match", [
    (TRun, "origin_geoid", [95.0, 10.0, 0.0], "latitude"),
    (TRun, "origin_geoid", [45.0, 400.0, 0.0], "longitude"),
    (TEfield, "du_nanoseconds", [2_000_000_000], "999999999"),
    (TShower, "azimuth", 400.0, "between 0 and 360"),
    (TShower, "azimuth", -30.0, "between 0 and 360"),
])
def test_out_of_range_values_are_warned_about(make, field, value, match):
    tree = make()
    with pytest.warns(GRANDlibWarning, match=match):
        setattr(tree, field, value)


def test_a_bool_in_a_numeric_field_is_refused():
    with pytest.raises(TypeError, match="TShower.zenith: must be a number"):
        TShower().zenith = True


def test_a_wrong_type_in_a_string_field_names_that_field():
    with pytest.raises(TypeError, match="TShower.primary_type: must be a string"):
        TShower().primary_type = 5
