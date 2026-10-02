# -*- coding: utf-8 -*-
r"""#261, item 7: grand.basis functions that did not do what they documented."""

import numpy as np
import pytest

from grand.basis.du_network import DetectorUnitNetwork


def _network(positions):
    network = DetectorUnitNetwork("square")
    network.init_pos_id(np.asarray(positions, dtype=float))
    return network


def test_the_surface_of_a_plain_array():
    # A 1 km square: np.cross on 2-D vectors failed under NumPy 2
    square = _network([[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0]])
    assert square.get_surface() == pytest.approx(1.0)


def test_fewer_than_three_units_have_no_surface():
    assert _network([[0, 0, 0], [1000, 0, 0]]).get_surface() == 0


def test_the_largest_distance_between_units():
    square = _network([[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0]])
    assert square.get_max_dist_du() == pytest.approx(np.sqrt(2.0))
    assert _network([[0, 0, 0]]).get_max_dist_du() == 0.0


def test_snr_and_noise_returns_what_it_says():
    from grand.basis.traces_event import Handling3dTraces

    rng = np.random.default_rng(1)
    traces = rng.normal(0, 1, (2, 3, 1000))
    traces[:, 0, 300] += 50
    event = Handling3dTraces()
    event.init_traces(traces, du_id=[1, 2], t_start_ns=np.zeros(2), f_samp_mhz=2000)
    snr, v_max, noise = event.get_snr_and_noise()
    np.testing.assert_allclose(snr, v_max / noise)
    assert "v_max" in Handling3dTraces.get_snr_and_noise.__doc__
