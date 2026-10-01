# -*- coding: utf-8 -*-
r"""#286: recons_ADF returned its starting point, silently, for impossible input.

A zero amplitude made the loss infinite everywhere; a source at or below the
antennas made the minimizer run for minutes; both ended with the start values.
"""

import numpy as np
import pytest


@pytest.fixture
def event():
    import grand.analysis.constants as cons
    from grand.analysis.fitting.adf import ADF_parameters
    from grand.analysis.fitting.spherical import compute_Xsource_cartesian_coords

    rng = np.random.default_rng(0)
    antennas = np.column_stack([rng.uniform(-3000, 3000, 12), rng.uniform(-3000, 3000, 12),
                                cons.groundAltitude + rng.uniform(-20, 20, 12)])
    zenith, azimuth = np.deg2rad(75.0), np.deg2rad(40.0)
    source = compute_Xsource_cartesian_coords(zenith, azimuth, 40000.0)[0]
    amplitudes = ADF_parameters(zenith, azimuth, 2.0, 5.0e7, antennas, source)[-1]
    return zenith, azimuth, amplitudes, antennas, source


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_amplitudes_are_refused(event, value):
    from grand.analysis.fitting.adf import recons_ADF

    zenith, azimuth, amplitudes, antennas, source = event
    amplitudes = amplitudes.copy()
    amplitudes[2] = value
    with pytest.raises(ValueError, match=r"antennas \[2\]"):
        recons_ADF(zenith, azimuth, amplitudes, antennas, source)


def test_a_source_below_the_antennas_is_refused(event):
    from grand.analysis.fitting.adf import recons_ADF

    zenith, azimuth, amplitudes, antennas, source = event
    below = np.array([0.0, 0.0, antennas[:, 2].max() - 10.0])
    with pytest.raises(ValueError, match="must lie above the antennas"):
        recons_ADF(zenith, azimuth, amplitudes, antennas, below)
