# -*- coding: utf-8 -*-
r"""#261, items 4, 5 and 8: sim docstrings checked against the code."""

import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "detector").exists(), reason="needs the data model")


def test_every_params_key_is_documented():
    from grand.sim.efield2voltage import PARAM_DEFAULTS, Efield2Voltage

    for key in PARAM_DEFAULTS:
        assert "``%s``" % key in Efield2Voltage.__doc__, key


def test_vout_f_says_what_shape_it_takes():
    from grand.sim.detector.rf_chain import RFChain

    chain = RFChain()
    freqs = np.linspace(30, 250, 50)
    chain.compute_for_freqs(freqs)
    assert chain.vout_f(np.ones((3, freqs.size))).shape == (3, freqs.size)
    with pytest.raises(ValueError, match=r"voc_f must have shape \(3, 50\)"):
        chain.vout_f(np.ones((2, 3, freqs.size)))
    assert "no effect" in RFChain.__init__.__doc__


def test_constructors_document_no_missing_parameters():
    from grand.sim.detector import rf_chain

    for name in ("MatchingNetwork", "gaa_frontend0db", "LowNoiseAmplifier", "BalunBeforeADC", "Zload"):
        doc = getattr(rf_chain, name).__init__.__doc__
        assert ":param" not in doc and "size_sig" not in doc.replace("``size_sig`` it documented", ""), name
