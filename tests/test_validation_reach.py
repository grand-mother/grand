# -*- coding: utf-8 -*-
r"""#267: the remaining entry points the validation work had not reached."""

import numpy as np
import pytest


def test_particle_code_takes_names_and_lists_them():
    from grand.sim.shower.pdg import ParticleCode

    assert ParticleCode("proton") == ParticleCode.PROTON == ParticleCode(2212)
    with pytest.raises(ValueError, match="PROTON \\(2212\\)"):
        ParticleCode("protn")


@pytest.mark.parametrize("params, match", [
    (["th1=50", "th2=100"], "th2"),
    (["nc_min=10", "nc_max=2"], "nc_min"),
    (["t_period=0"], "t_period"),
])
def test_t1_parameters_are_checked(params, match):
    from grand.sim.detector.trigger import t1_config_from_params, t1_channel_trigger

    with pytest.raises(ValueError, match=match):
        t1_config_from_params(params)
    config = {key.split("=")[0]: int(key.split("=")[1]) for key in params}
    with pytest.raises(ValueError, match=match):
        t1_channel_trigger(np.zeros(2048), config)


def test_voltage_checks_its_shapes():
    from grand.basis.type_trace import Voltage

    Voltage(t=np.arange(4.0), V=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="one per time"):
        Voltage(t=np.arange(4.0), V=np.zeros((3, 5)))


def test_a_tuple_of_unit_ids_is_accepted():
    from grand.basis.du_network import DetectorUnitNetwork

    network = DetectorUnitNetwork()
    network.init_pos_id(np.zeros((2, 3)), du_id=(1, 2))
    assert list(network.idx2idt) == [1, 2]


def test_readers_name_a_missing_file(tmp_path):
    from grand.dataio.root_files import get_handling3dtraces

    missing = tmp_path / "missing.root"
    with pytest.raises(FileNotFoundError, match="no such file: .*missing.root"):
        get_handling3dtraces(str(missing))
    assert not missing.exists()


def test_protocol_refuses_no_name_without_a_request():
    from grand.dataio import protocol

    with pytest.raises(TypeError, match="non-empty file name"):
        protocol.get(None)


def test_xmax_in_site_frame_checks_angles_even_for_an_unknown_xmax():
    from grand.dataio.xmax_frame import xmax_in_site_frame

    with pytest.raises(ValueError, match="zenith"):
        xmax_in_site_frame([np.nan] * 3, 500.0, 0.0, 1000.0, [0.0, 0.0, 0.0])
