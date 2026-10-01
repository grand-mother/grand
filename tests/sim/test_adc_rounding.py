# -*- coding: utf-8 -*-
r"""Exact ADC counts (#270): values were never pinned, so ``np.floor`` for
``np.trunc`` (a bias of -1 count on negative samples) passed."""

import numpy as np

from grand.sim.detector.adc import ADC


def test_counts_truncate_towards_zero():
    adc = ADC()
    lsb = adc.max_voltage / adc.max_bit_value      # µV per count
    volts = np.array([0.5, -0.5, 1.5, -1.5, 0.0]) * lsb
    counts = adc._digitize(volts.reshape(1, 1, -1))
    assert counts.ravel().tolist() == [0, 0, 1, -1, 0]


def test_full_scale():
    adc = ADC()
    counts = adc._digitize(np.array([adc.max_voltage, -adc.max_voltage]).reshape(1, 1, -1))
    assert counts.ravel().tolist() == [adc.max_bit_value, -adc.max_bit_value]
