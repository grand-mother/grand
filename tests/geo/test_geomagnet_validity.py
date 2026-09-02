# -*- coding: utf-8 -*-
r"""The date range the shipped geomagnetic model actually covers.

``data/geomagnet/IGRF13.COF`` is IGRF-13, defined from 1900 to 2025.  Dates at
or beyond the end of that range fail, so a simulation of recent data cannot
evaluate the field at the time the data were taken.

The default ``obstime`` throughout :mod:`grand.geo.coordinates` is the literal
string ``"2020-01-01"``, which is inside the range -- which is why this went
unnoticed for over a year after the model lapsed.
"""

import pytest

from grand import Geodetic, LTP

SITE = Geodetic(latitude=40.98, longitude=93.95, height=1200.0)


def _magnetic_frame(obstime):
    r"""Returns a magnetic-north frame at `obstime`, or raises.

    Parameters
    ----------
    obstime : str
        Date to evaluate the declination at.

    Returns
    -------
    LTP
        The frame.
    """
    return LTP(x=1000.0, y=0.0, z=0.0, location=SITE,
               orientation='ENU', magnetic=True, obstime=obstime)


@pytest.mark.parametrize('obstime', ['2020-01-01', '2024-06-01'])
def test_dates_inside_the_model_range_work(obstime):
    r"""A magnetic frame can be built for a date the model covers."""
    _magnetic_frame(obstime)


@pytest.mark.parametrize('obstime', ['2025-01-01', '2026-01-01'])
def test_dates_beyond_the_model_range_fail(obstime):
    r"""Dates from 2025 onward raise, because IGRF-13 ends there.

    Pinned rather than xfailed: this is the current, reproducible behaviour,
    and the test is how anyone will know when shipping IGRF-14 has fixed it --
    at that point this test fails and should be updated to the new range.

    See :ref:`issue-geomagnetic-model-expired`.
    """
    with pytest.raises(Exception) as excinfo:
        _magnetic_frame(obstime)
    assert 'missing data' in str(excinfo.value) or 'IGRF' in str(excinfo.value), (
        'failed for an unexpected reason: %s' % excinfo.value)


def test_the_default_obstime_is_a_stale_fixed_date():
    r"""The default epoch is a hard-coded past date, not the present.

    That is why the expiry went unnoticed: everything works on the default and
    only fails when a caller passes a real date.
    """
    import inspect
    from grand.geo.coordinates import LTP as LTPClass

    signature = inspect.signature(LTPClass.__init__)
    default = signature.parameters['obstime'].default
    assert str(default).startswith('20'), (
        'obstime default is no longer a fixed date: %r. If it now follows the '
        'clock, this test should be removed and the model range checked '
        'instead.' % (default,))
