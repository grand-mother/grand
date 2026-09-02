# -*- coding: utf-8 -*-
r"""Pins which physical arm each tabulated antenna file describes.

The GRANDlib Handbook states that the "East--West arm" is "denoted as EW or X
arm" and the "South--North" arm as "SN or Y arm".  The code assigns them the
other way round: :class:`~grand.sim.detector.antenna_model.AntennaModel`
reads ``Light_GP300Antenna_nec_Xarm_leff.npz`` into ``leff_sn`` and
``..._nec_Yarm_leff.npz`` into ``leff_ew``.

One of the two is wrong, and it is decidable from the data.  The HFSS files are
named after the physical arm (``EWarm``, ``SNarm``) and carry no X/Y ambiguity,
so the NEC and MATLAB files can be identified by correlating their patterns
against those.  The measurement says **X is the south-north arm and Y is the
east-west arm**, which is what the code does and the opposite of what the
handbook says.  It is also what :class:`~grand.geo.coordinates.GRANDCS`
implies, since its ``x`` axis points north.

Getting this backwards swaps two of the three output channels for every event
simulated with ``du_type='GP300_nec'`` or ``'GP300_mat'``, which would look
like a physics result rather than a bug.
"""

import numpy as np
import pytest

from grand import grand_add_path_data

#: 60-200 MHz, where the arms have most of their response.
BAND = (60.0, 200.0)


def _pattern(name):
    r"""Returns the frequency axis and the response magnitudes of one file.

    Parameters
    ----------
    name : str
        File name under ``data/detector/``.

    Returns
    -------
    freq_mhz : ndarray, shape (221,)
        Frequency axis.
    magnitudes : ndarray, shape (2, 361, 91, 221)
        ``|leff_theta|`` and ``|leff_phi|``, indexed by azimuth, zenith and
        frequency as the files store them.
    """
    data = np.load(grand_add_path_data("detector/" + name))
    return data["freq_mhz"], np.stack([np.abs(data["leff_theta"]),
                                       np.abs(data["leff_phi"])])


def _correlation(a, b):
    r"""Returns the Pearson correlation of two response patterns in band.

    Comparing whole patterns rather than the position of a peak: a peak
    position is one number and could agree by chance, where the full
    azimuth-zenith-frequency pattern of a wrong pairing does not.
    """
    (freq, pa), (_, pb) = a, b
    sel = (freq >= BAND[0]) & (freq <= BAND[1])
    x = pa[:, :, :, sel].ravel().astype(float)
    y = pb[:, :, :, sel].ravel().astype(float)
    x = x - x.mean()
    y = y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


@pytest.fixture(scope="module")
def patterns():
    r"""Returns the six tabulated patterns, loaded once."""
    return {
        "hfss_EW": _pattern("Light_GP300Antenna_EWarm_leff.npz"),
        "hfss_SN": _pattern("Light_GP300Antenna_SNarm_leff.npz"),
        "nec_X": _pattern("Light_GP300Antenna_nec_Xarm_leff.npz"),
        "nec_Y": _pattern("Light_GP300Antenna_nec_Yarm_leff.npz"),
        "mat_X": _pattern("Light_GP300Antenna_mat_Xarm_leff.npz"),
        "mat_Y": _pattern("Light_GP300Antenna_mat_Yarm_leff.npz"),
    }


def test_the_two_hfss_arms_are_independent(patterns):
    r"""The reference arms are orthogonal, so the comparison below can decide.

    If ``EWarm`` and ``SNarm`` were similar to each other, correlating an
    unknown file against them would be uninformative.  They are 90 degrees
    apart and correlate at about zero, which is what makes the test work.
    """
    value = _correlation(patterns["hfss_EW"], patterns["hfss_SN"])
    assert abs(value) < 0.1, (
        'the two HFSS arms correlate at %+.3f; they are meant to be '
        'perpendicular, so the identification below cannot be trusted' % value)


@pytest.mark.parametrize("x_file,y_file", [("nec_X", "nec_Y"),
                                           ("mat_X", "mat_Y")])
def test_x_is_the_south_north_arm(patterns, x_file, y_file):
    r"""``Xarm`` matches ``SNarm`` and ``Yarm`` matches ``EWarm``.

    This is the assignment :class:`~grand.sim.detector.antenna_model.AntennaModel`
    makes.  If it ever fails, either the data files were renamed or the loader
    was "corrected" to follow the handbook, and the X and Y channels of every
    NEC or MATLAB simulation have been swapped.
    """
    x_vs_sn = _correlation(patterns[x_file], patterns["hfss_SN"])
    x_vs_ew = _correlation(patterns[x_file], patterns["hfss_EW"])
    y_vs_ew = _correlation(patterns[y_file], patterns["hfss_EW"])
    y_vs_sn = _correlation(patterns[y_file], patterns["hfss_SN"])

    assert x_vs_sn > x_vs_ew, (
        '%s looks like the east-west arm (%.3f) rather than south-north '
        '(%.3f)' % (x_file, x_vs_ew, x_vs_sn))
    assert y_vs_ew > y_vs_sn, (
        '%s looks like the south-north arm (%.3f) rather than east-west '
        '(%.3f)' % (y_file, y_vs_sn, y_vs_ew))
    assert x_vs_sn > 0.5 and y_vs_ew > 0.5, (
        'the identification is weak: %s vs SN = %.3f, %s vs EW = %.3f'
        % (x_file, x_vs_sn, y_file, y_vs_ew))


def test_the_loader_follows_the_measurement():
    r"""``AntennaModel`` puts the X file in ``leff_sn`` and the Y file in ``leff_ew``.

    Asserted against the source rather than against the loaded arrays, because
    the arrays are what the previous test already checked; what could silently
    change is the pairing in the loader.
    """
    import inspect

    from grand.sim.detector.antenna_model import AntennaModel

    source = inspect.getsource(AntennaModel.__init__)
    for variant in ("nec", "mat"):
        x_line = "Light_GP300Antenna_%s_Xarm_leff.npz" % variant
        y_line = "Light_GP300Antenna_%s_Yarm_leff.npz" % variant
        assert x_line in source and y_line in source, (
            'the %s files are no longer loaded by name' % variant)
        # The assignment that follows each path must be the one measured above.
        after_x = source.split(x_line, 1)[1].split("\n", 2)[1]
        after_y = source.split(y_line, 1)[1].split("\n", 2)[1]
        assert "leff_sn" in after_x, (
            '%s Xarm is no longer assigned to leff_sn; the measurement in this '
            'file says X is the south-north arm' % variant)
        assert "leff_ew" in after_y, (
            '%s Yarm is no longer assigned to leff_ew; the measurement in this '
            'file says Y is the east-west arm' % variant)


def test_output_channel_order_is_sn_ew_z():
    r"""``voc[:, 0]`` is SN, ``voc[:, 1]`` is EW, ``voc[:, 2]`` is Z.

    Combined with the identification above, this is what makes the channels of
    a ``TVoltage`` trace X, Y, Z in that order.  The documentation and the
    notebooks label them that way and depend on this.
    """
    import inspect

    from grand.sim.efield2voltage import Efield2Voltage

    source = inspect.getsource(Efield2Voltage)
    for index, arm in ((0, "sn"), (1, "ew"), (2, "z")):
        needle = "self.voc[du_idx, %d] = self.ant_leff_%s" % (index, arm)
        assert needle in source, (
            'channel %d is no longer the %s arm; every XYZ label in the '
            'documentation and notebooks assumes SN, EW, Z in that order'
            % (index, arm))
