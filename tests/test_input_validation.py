# -*- coding: utf-8 -*-
r"""GRANDlib refuses invalid input with a clear message, and warns on suspicious input.

In October 2026 a review called 60 public entry points with wrong types,
signs, ranges, shapes, NaN, empty arrays and missing files: 7 refused the
input deliberately, 27 accepted it silently, 13 failed deep inside with an
unrelated exception and one called ``exit()``.  This file is that probe,
turned into tests.

The convention, from ``grand.basis.validate``: errors are the standard
``TypeError`` (wrong kind of value), ``ValueError`` (right kind, wrong value
or shape) or ``FileNotFoundError``; warnings are ``GRANDlibWarning``; every
message starts with ``GRANDlib: <function>:`` and names the argument.
"""

import os

import numpy as np
import pytest

from grand.basis import validate
from grand.basis.validate import GRANDlibWarning

PREFIX = r"^GRANDlib: "


# ------------------------------------------------------------------ helpers

def test_messages_name_the_function_and_the_argument():
    with pytest.raises(ValueError, match=r"^GRANDlib: f: 'x' must be between 0 and 1 m, got 2"):
        validate.in_range(2.0, "x", "f", 0, 1, "m")


@pytest.mark.parametrize("value, error", [("3", TypeError), (1.5, TypeError), (True, TypeError),
                                          (-1, ValueError), (2**40, ValueError)])
def test_as_integer_refuses_what_is_not_a_whole_number_in_range(value, error):
    with pytest.raises(error, match=PREFIX):
        validate.as_integer(value, "n", "f", minimum=0, maximum=2**32 - 1)


def test_as_integer_accepts_whole_floats_and_numpy_integers():
    assert validate.as_integer(3.0, "n", "f") == 3
    assert validate.as_integer(np.uint16(7), "n", "f") == 7


def test_as_array_checks_shape_length_and_finiteness():
    with pytest.raises(ValueError, match=r"must have shape \(N, 3\), got \(4, 2\)"):
        validate.as_array(np.zeros((4, 2)), "X", "f", shape=(None, 3))
    with pytest.raises(ValueError, match="at least 3 elements, got 2"):
        validate.as_array([1, 2], "t", "f", min_length=3)
    with pytest.raises(ValueError, match="must not contain NaN"):
        validate.as_array([1.0, np.nan], "t", "f", finite=True)
    with pytest.raises(TypeError, match="got the string"):
        validate.as_array("abc", "t", "f")


def test_coerce_to_dtype_refuses_silent_changes():
    with pytest.raises(TypeError, match="must be an integer, got 1.7"):
        validate.coerce_to_dtype(1.7, np.uint32, "T.n")
    with pytest.raises(ValueError, match="between 0 and 65535"):
        validate.coerce_to_dtype([-1, 3], np.uint16, "T.v")
    with pytest.raises(TypeError, match="must be a number"):
        validate.coerce_to_dtype("abc", np.float32, "T.x")
    assert validate.coerce_to_dtype(4.0, np.int32, "T.n") == 4


def test_missing_files_and_directories_are_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file: .*nope.root"):
        validate.existing_file(str(tmp_path / "nope.root"), "f", "g")
    with pytest.raises(FileNotFoundError, match="no such directory"):
        validate.existing_directory(str(tmp_path / "nope"), "d", "g")


# ------------------------------------------------------------------ coordinates

def test_coordinates_refuse_invalid_components():
    from grand import CartesianRepresentation, Geodetic, GRANDCS

    with pytest.raises(ValueError, match="'latitude' must be between -90 and 90 degrees, got 200"):
        Geodetic(latitude=200.0, longitude=92.0, height=0.0)
    with pytest.raises(TypeError, match="must all be given; missing 'longitude', 'height'"):
        Geodetic(latitude=40.0)
    with pytest.raises(TypeError, match=PREFIX):
        Geodetic(latitude="abc", longitude=92.0, height=0.0)
    with pytest.raises(ValueError, match="must have the same length, got 2, 1, 1"):
        CartesianRepresentation(x=np.array([1.0, 2.0]), y=np.array([1.0]), z=np.array([1.0]))
    with pytest.raises(TypeError, match=PREFIX):
        GRANDCS(x="a", y=0.0, z=0.0)


def test_a_nan_position_warns_and_the_callers_array_is_not_modified():
    from grand import Geodetic

    with pytest.warns(GRANDlibWarning, match="'latitude' contains NaN"):
        Geodetic(latitude=np.nan, longitude=92.0, height=0.0)
    longitude = np.array([-10.0, 20.0])
    Geodetic(latitude=np.array([1.0, 2.0]), longitude=longitude, height=np.zeros(2))
    assert longitude.tolist() == [-10.0, 20.0]


def test_topography_refuses_an_unknown_reference():
    from grand import Geodetic
    from grand.geo import topography

    with pytest.raises(ValueError, match="'reference' must be one of"):
        topography.elevation(Geodetic(latitude=40.9, longitude=96.5, height=0.0), reference="BAD")


# ------------------------------------------------------------------ data trees and files

def _set(cls, field, value):
    tree = cls()
    setattr(tree, field, value)
    return getattr(tree, field)


@pytest.mark.parametrize("field, value, error, text", [
    ("run_number", -1, ValueError, r"TRun.run_number: must be an integer between 0 and 4294967295"),
    ("run_number", 1.7, TypeError, r"TRun.run_number: must be an integer, got 1.7"),
    ("run_number", "abc", TypeError, r"TRun.run_number: must be a number"),
    ("du_xyz", [[1.0, 2.0]], ValueError, r"TRun.du_xyz: each entry must have 3 values"),
    ("du_id", [1.7], TypeError, r"TRun.du_id: must be an integer"),
])
def test_tree_fields_refuse_values_they_would_change(field, value, error, text):
    from grand.dataio import TRun

    with pytest.raises(error, match="GRANDlib: " + text):
        _set(TRun, field, value)


def test_tree_fields_warn_on_physically_impossible_values_but_store_them():
    r"""Warnings, not errors: existing files hold such placeholders and must stay readable."""
    from grand.dataio import TRun, TShower

    with pytest.warns(GRANDlibWarning, match=r"TShower.zenith: should be between 0 and 180 degrees, got 500"):
        assert _set(TShower, "zenith", 500.0) == 500.0
    with pytest.warns(GRANDlibWarning, match=r"TShower.energy_primary: should be >= 0"):
        _set(TShower, "energy_primary", -1.0)
    with pytest.warns(GRANDlibWarning, match=r"TRun.t_bin_size: should be positive"):
        _set(TRun, "t_bin_size", [-2.0])
    # NaN means "unknown": no warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", GRANDlibWarning)
        _set(TShower, "zenith", np.nan)


def test_fixed_shape_fields_name_the_expected_shape():
    from grand.dataio import TShower

    with pytest.raises(ValueError, match=r"TShower.xmax_pos: must have 3 values \(shape \(3,\)\), got shape \(2,\)"):
        _set(TShower, "xmax_pos", [1.0, 2.0])


def test_fixed_shape_fields_still_accept_the_right_number_of_values_in_another_layout():
    from grand.dataio import TShower

    shower = TShower()
    shower.direction = [[0.0, 0.6, 0.8]]
    assert np.allclose(shower.direction, [0.0, 0.6, 0.8])


def test_files_that_cannot_be_read_say_why(tmp_path):
    from grand.dataio import DataDirectory, DataFile, TRun

    text = tmp_path / "not_root.root"
    text.write_text("hello")
    with pytest.raises(OSError, match="it is not a ROOT file"):
        TRun(str(text))
    with pytest.raises(OSError, match="it is not a ROOT file"):
        DataFile(str(text))
    with pytest.raises(FileNotFoundError, match="DataFile: no such file"):
        DataFile(str(tmp_path / "missing.root"))
    with pytest.raises(FileNotFoundError, match="DataDirectory: no such directory"):
        DataDirectory(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="its directory .* does not exist"):
        TRun(str(tmp_path / "no_such_dir" / "run.root"))
    # A new file in an existing directory is how trees are written: still allowed
    TRun(str(tmp_path / "new_run.root"))
    assert os.path.isfile(tmp_path / "new_run.root")


def test_event_list_raises_instead_of_exiting(tmp_path):
    from grand.aoi.event_list import EventList

    with pytest.raises(TypeError, match="EventList: the input must be"):
        EventList(123)
    with pytest.raises(FileNotFoundError, match="EventList: no such file or directory"):
        EventList(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match=r"EventList: no ROOT files \(\*.root\)"):
        EventList(str(tmp_path))


def test_xmax_frame_helpers_check_shapes_and_angles():
    from grand.dataio import xmax_frame

    with pytest.raises(ValueError, match="'xmax_pos_shc' must be three numbers"):
        xmax_frame.xmax_above_ground([1.0, 2.0], 60.0, 30.0, 1264.0)
    with pytest.raises(ValueError, match="'zenith' must be between 0 and 180 degrees"):
        xmax_frame.propagation_direction(200.0, 30.0)


# ------------------------------------------------------------------ reconstruction

@pytest.fixture(scope="module")
def antennas():
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(-3000, 3000, 6), rng.uniform(-3000, 3000, 6),
                            np.full(6, 1231.0)])


def test_reconstruction_checks_its_antennas_and_times(antennas):
    pytest.importorskip("iminuit")
    from grand.analysis.fitting import plane_wave as pw, spherical as sw

    times = pw.PWF_model(np.deg2rad([60, 30]), antennas)
    with pytest.raises(ValueError, match="PWF_semianalytical: needs at least 3 antennas, got 2"):
        pw.PWF_semianalytical(antennas[:2], times[:2])
    with pytest.raises(ValueError, match=r"'tants' must have one value per antenna \(6\), got 5"):
        pw.PWF_semianalytical(antennas, times[:5])
    with pytest.raises(ValueError, match=r"'Xants' must have shape \(N, 3\)"):
        pw.PWF_semianalytical(antennas[:, :2], times)
    with pytest.raises(ValueError, match="'tants' must not contain NaN"):
        pw.PWF_semianalytical(antennas, np.r_[times[:5], np.nan])
    with pytest.raises(ValueError, match="'sigma' must be positive"):
        pw.PWF_semianalytical(antennas, times, sigma=-1.0)
    with pytest.raises(ValueError, match="recons_swf: needs at least 4 antennas, got 3"):
        sw.recons_swf(1.0, 0.5, times[:3], antennas[:3])


def test_energy_proxy_refuses_a_zero_geomagnetic_angle():
    from grand.analysis.energy_reco.voltage import recons_energy_from_voltage

    with pytest.raises(ValueError, match="'sin_alpha' is 0"):
        recons_energy_from_voltage(1e3, 0.0)


def test_cramer_rao_bounds_refuse_too_few_antennas_and_warn_when_degenerate():
    pytest.importorskip("iminuit")
    from grand.analysis.cramer_rao_bounds import CRB_PWF

    with pytest.raises(ValueError, match="CRB_PWF: needs at least 3 antennas, got 2"):
        CRB_PWF(1.0, 0.5, np.array([[0, 0, 1231.0], [1000, 0, 1231.0]]))
    on_a_line = np.array([[0, 0, 1231.0], [1000, 0, 1231.0], [2000, 0, 1231.0]])
    with pytest.warns(GRANDlibWarning, match="barely constrain the parameters"):
        CRB_PWF(1.0, 0.5, on_a_line)


def test_signal_extraction_checks_channels_and_sampling():
    from grand.analysis.signals import extraction as ex

    with pytest.raises(ValueError, match="channel 5 does not exist: 'trace' has 3 channels"):
        ex.convert_voltage_to_ADC(np.zeros((3, 100)), channels=[5])
    with pytest.raises(ValueError, match="'trace' has no samples"):
        ex.get_peak_amplitude(np.zeros((3, 0)), channels=[0])
    with pytest.raises(ValueError, match="'dt_ns' must be positive"):
        ex.get_peak_time(np.ones((3, 10)), 0, channels=[0], dt_ns=0)


@pytest.mark.parametrize("channels", [[True, False, True], slice(0, 3, 2), [0, 2], np.array([0, 2])],
                         ids=["mask", "slice", "list", "array"])
def test_adc_conversion_selects_the_same_channels_as_numpy(channels):
    r"""#263: a mask converted every channel, and a slice or an int raised TypeError."""
    from grand.analysis.signals import extraction as ex

    trace = np.arange(30, dtype=float).reshape(3, 10) * 100
    converted = ex.convert_voltage_to_ADC(trace, channels)
    scale = 1e-6 * 8192 / 0.9
    np.testing.assert_allclose(converted[[0, 2]], trace[[0, 2]] * scale)
    np.testing.assert_array_equal(converted[1], trace[1])
    one = ex.convert_voltage_to_ADC(trace, 1)
    np.testing.assert_allclose(one[1], trace[1] * scale)
    np.testing.assert_array_equal(one[[0, 2]], trace[[0, 2]])
    with pytest.raises(ValueError, match="needs one value per channel"):
        ex.convert_voltage_to_ADC(trace, [True, False])


@pytest.mark.parametrize("rate", [np.float32(2000), np.int64(2000), [2000, 2000], np.array([2000., 2000.])],
                         ids=["float32", "int64", "list", "array"])
def test_traces_accept_a_numpy_sampling_rate(rate):
    r"""#264: a NumPy scalar rate was kept as a scalar, and apply_bandpass failed with IndexError."""
    from grand.basis.traces_event import Handling3dTraces

    traces = Handling3dTraces()
    traces.init_traces(np.random.default_rng(0).normal(size=(2, 3, 256)), f_samp_mhz=rate)
    np.testing.assert_array_equal(traces.f_samp_mhz, [2000.0, 2000.0])
    traces.apply_bandpass(50, 200)


def test_traces_refuse_a_wrong_sampling_rate():
    from grand.basis.traces_event import Handling3dTraces

    traces = np.zeros((2, 3, 16))
    with pytest.raises(ValueError, match="one rate, or one per unit"):
        Handling3dTraces().init_traces(traces, f_samp_mhz=[2000., 2000., 2000.])
    with pytest.raises(TypeError, match="got a bool"):
        Handling3dTraces().init_traces(traces, f_samp_mhz=True)


# ------------------------------------------------------------------ simulation

def test_simulation_inputs_are_checked():
    from grand.basis.traces_event import Handling3dTraces
    from grand.sim.detector.antenna_model import AntennaModel
    from grand.sim.detector.rf_chain import RFChain
    from grand.sim.noise import galaxy

    freqs = np.linspace(30, 250, 64)
    with pytest.raises(ValueError, match="'f_lst' \\(local sidereal time\\) must satisfy"):
        galaxy.galactic_noise(30.0, 1024, freqs, 2)
    with pytest.raises(ValueError, match="'freqs_mhz' must not be negative"):
        galaxy.galactic_noise(18.0, 1024, -freqs, 2)
    with pytest.raises(ValueError, match="'du_type' must be one of"):
        galaxy.galactic_noise(18.0, 1024, freqs, 2, du_type="BAD")
    with pytest.raises(ValueError, match="AntennaModel: 'du_type' must be one of"):
        AntennaModel(du_type="BAD")
    with pytest.raises(ValueError, match="'vga_gain' must be one of -5, 0, 5 or 20 dB"):
        RFChain(vga_gain=-100)
    with pytest.raises(ValueError, match=r"'traces' must have shape \(n_du, 3, n_samples\)"):
        Handling3dTraces().init_traces(np.zeros((3, 100)))


def test_checks_keep_accepting_what_the_functions_always_accepted():
    r"""Inputs that worked before the checks were added still work.

    A covariance matrix as ``sigma`` (its off-diagonal terms are zero), the
    channels as a slice, and the source position in the ``(1, 3)`` shape
    ``compute_Xsource_cartesian_coords`` returns.
    """
    from grand.analysis import fitting as fit
    from grand.analysis.signals import extraction as ext

    rng = np.random.default_rng(0)
    antennas = np.c_[rng.uniform(-1e3, 1e3, (8, 2)), np.full(8, 1264.0)]
    times = antennas @ np.array([0.3, 0.1, -0.9]) / 3e8
    covariance = np.diag(np.full(8, 25e-18))
    assert np.allclose(fit.PWF_semianalytical(antennas, times, sigma=covariance),
                       fit.PWF_semianalytical(antennas, times))
    with pytest.raises(ValueError, match="diagonal of 'sigma'"):
        fit.PWF_semianalytical(antennas, times, sigma=-covariance)

    trace = rng.normal(size=(3, 100))
    assert ext.get_peak_amplitude(trace, slice(0, 3)) == ext.get_peak_amplitude(trace, [0, 1, 2])

    source = fit.compute_Xsource_cartesian_coords(np.deg2rad(70.0), np.deg2rad(30.0), 2e4)
    assert source.shape == (1, 3)
    amplitudes = rng.uniform(50, 100, 8)
    assert np.allclose(fit.recons_ADF(1.2, 0.5, amplitudes, antennas, source),
                       fit.recons_ADF(1.2, 0.5, amplitudes, antennas, source[0]))


def test_text_that_reads_as_a_number_is_converted_with_a_warning():
    r"""``"1618"`` was always stored as 1618 (sim2root passes values read from
    file names and the command line); refusing it broke the ZHAireS converter's
    one-argument mode and ``sim2root.py -la``.  Other text is still refused."""
    from grand.dataio import TRun, TShower

    with pytest.warns(GRANDlibWarning, match="stored as the number it reads as"):
        assert validate.coerce_to_dtype("1618", np.uint32, "T.n") == 1618
    with pytest.warns(GRANDlibWarning):
        shower = TShower()
        shower.event_number = "1618"
    assert shower.event_number == 1618
    with pytest.warns(GRANDlibWarning):
        run = TRun()
        run.origin_geoid = ["41.5", "93.94", "1264.0"]
    assert np.allclose(run.origin_geoid, [41.5, 93.94, 1264.0])
    with pytest.raises(TypeError, match="must be an integer"):
        with pytest.warns(GRANDlibWarning):
            validate.coerce_to_dtype("1.7", np.uint32, "T.n")
    with pytest.raises(TypeError, match="must be a number"):
        validate.coerce_to_dtype(["1", "x"], np.float32, "T.x")
