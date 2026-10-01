# -*- coding: utf-8 -*-
r"""#255: rf_chain configuration errors.

A component missing from rf_chain_config.xml printed an error and then
failed with ``NameError: name 'Nonec' is not defined``; an invalid axis
printed an error and returned None, which callers passed on until an
unrelated TypeError.  The configuration was parsed at import.
"""

import pytest

from grand.sim.detector import rf_chain


def test_filenames_resolve_for_every_axis_spelling():
    for axis in (0, "0", "X"):
        assert rf_chain.get_axis_filename("MatchingNetwork", axis).endswith("MatchingNetworkX.s2p")


def test_a_missing_component_raises_a_clear_error():
    with pytest.raises(KeyError, match="Nope is missing from"):
        rf_chain.get_axis_filename("Nope", 0)


def test_an_invalid_axis_raises_a_clear_error():
    with pytest.raises(ValueError, match="invalid axis 7 for MatchingNetwork"):
        rf_chain.get_axis_filename("MatchingNetwork", 7)


def test_an_invalid_vga_gain_raises_without_assert():
    with pytest.raises(ValueError, match="-5, 0, 5 or 20 dB"):
        rf_chain.VGAFilter(gain=3)._set_name_data_file()


def test_the_configuration_is_read_once_on_first_use():
    rf_chain._config.cache_clear()
    assert rf_chain._config.cache_info().currsize == 0
    rf_chain.get_axis_filename("LNA", 1)
    rf_chain.get_axis_filename("LNA", 2)
    assert rf_chain._config.cache_info().misses == 1
