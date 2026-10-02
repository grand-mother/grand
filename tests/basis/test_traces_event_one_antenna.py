# -*- coding: utf-8 -*-
r"""#287: Handling3dTraces on one-antenna events.

np.squeeze dropped the antenna axis, so get_tmax_vmax and get_snr_and_noise
crashed on one antenna (and interpol="no" returned 0-d arrays); an unknown
interpol raised "No active exception to reraise".
"""

import numpy as np
import pytest

from grand.basis.traces_event import Handling3dTraces


def _event(n):
    rng = np.random.default_rng(1)
    t = Handling3dTraces()
    t.init_traces(rng.normal(size=(n, 3, 256)), du_id=list(range(n)),
                  t_start_ns=np.arange(n) * 10.0, f_samp_mhz=2000)
    return t


@pytest.mark.parametrize("n", [1, 2, 5])
def test_one_value_per_antenna(n):
    t = _event(n)
    for kwargs in ({}, {"interpol": "no"}, {"interpol": "auto"}, {"hilbert": False}):
        tmax, vmax = t.get_tmax_vmax(**kwargs)
        assert np.shape(tmax) == (n,) and np.shape(vmax) == (n,), kwargs
    snr, vmax, noise = t.get_snr_and_noise()
    assert np.shape(snr) == np.shape(vmax) == np.shape(noise) == (n,)


def test_unknown_interpolation_is_refused():
    with pytest.raises(ValueError, match="'interpol' must be"):
        _event(1).get_tmax_vmax(interpol="cubic")
