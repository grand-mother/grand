# -*- coding: utf-8 -*-
"""grand.analysis.gnn on small untrained networks: the inputs, the storage and the conventions."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import grand.analysis.constants as cons          # noqa: E402
import grand.analysis.signals as sig             # noqa: E402
from grand.analysis import gnn                   # noqa: E402
from grand.analysis.gnn import features          # noqa: E402
from grand.analysis.gnn.architecture import GNNScaleMDN   # noqa: E402

TINY = {"hidden": 8, "n_graph_layers": 2, "graph_out_channels": 8}
NORMALISATION = {"pos_x": {"scale": 1e-4}, "pos_y": {"scale": 1e-4}, "pos_z": {"scale": 1e-2},
                 "time": {"scale": 2e4}, "amplitude": {"power": 0.2, "scale": 0.125}}


def _save(directory, task="direction", name="tiny", version="v0", **inputs):
    gnn_model_card = gnn.complete_gnn_model_card({"name": name, "version": version, "task": task, "inputs": inputs,
                              "architecture": TINY, "normalisation": NORMALISATION})
    states = []
    for seed in range(2):
        torch.manual_seed(seed)
        states.append(GNNScaleMDN(gnn_model_card).state_dict())
    return gnn.save_model(str(directory), gnn_model_card, states)


def _plane_wave(theta=1.3, phi=2.0, n=12):
    rng = np.random.default_rng(0)
    Xants = np.column_stack([rng.uniform(-2000, 2000, (n, 2)), 1231 + rng.uniform(0, 20, n)])
    peak_time = Xants @ features.propagation_vector(theta, phi) / (cons.c_light / cons.n_atm) + 1e-3
    return Xants, peak_time, rng.uniform(200, 2000, n)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """A tiny direction and energy network, chosen by default."""
    _save(tmp_path / "models" / "tiny" / "v0")
    _save(tmp_path / "models" / "tiny_e" / "v0", task="energy", name="tiny_e")
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"direction": {"name": "tiny", "version": "v0"},
                                "energy": {"name": "tiny_e", "version": "v0"}, "models_dir": "models"}))
    monkeypatch.setenv(gnn.SETTINGS_ENV, str(path))
    gnn.clear_cache()
    yield str(path)
    gnn.clear_cache()


def test_default_gnn_model_card():
    gnn_model_card = gnn.complete_gnn_model_card({"task": "direction"})
    assert gnn_model_card["inputs"]["channels"] == ["x", "y"]
    assert gnn_model_card["inputs"]["node_features"] == ["pos_x", "pos_y", "pos_z", "time", "amplitude"]
    assert gnn_model_card["inputs"]["knn_k"] == 8
    assert gnn_model_card["inputs"]["delta_t_wave_speed"] == cons.c_light / cons.n_atm
    assert gnn_model_card["normalisation_method"] == "gnn_lib"         # GNN_forPaper's, by default
    assert gnn_model_card["normalisation"]["time"] == {"scale": 2e4, "power": 1.0, "offset": 0.0}


def test_knn_graph():
    pos = np.random.default_rng(1).uniform(-1000, 1000, size=(15, 3))
    pairs = set(map(tuple, features.knn_edge_index(pos, 4).T))
    assert all((j, i) in pairs for i, j in pairs)                      # both directions
    distance = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    for i in range(len(pos)):
        assert all((j, i) in pairs for j in np.argsort(distance[i])[:5])   # itself and 4 nearest


def test_make_graph():
    inputs = gnn.complete_gnn_model_card({"task": "direction", "inputs": {
        "node_features": ["pos_x", "pos_y", "pos_z", "time", "amplitude", "delta_t"]}})["inputs"]
    Xants, t, a = _plane_wave()
    graph = features.make_graph(Xants, t, a, inputs)
    x = graph["x"]
    assert x.shape == (12, 6) and graph["x_graph"].shape == (1, 3)
    np.testing.assert_allclose(x[:, :3].mean(axis=0), 0, atol=1e-9)    # positions about their mean
    assert x[:, 3].min() == 0                                           # times from the earliest peak
    np.testing.assert_array_equal(x[:, 4], a)
    np.testing.assert_allclose(x[:, 5], 0, atol=1e-12)                  # a plane wave: no residual
    np.testing.assert_allclose(graph["x_graph"][0], features.propagation_vector(1.3, 2.0), atol=1e-6)


def test_trigger_defaults_to_5_sigma():
    xy = gnn.complete_gnn_model_card({"task": "direction"})["inputs"]
    xyz = gnn.complete_gnn_model_card({"task": "direction", "inputs": {"channels": ["x", "y", "z"]}})["inputs"]
    np.testing.assert_array_equal(features.trigger([106.0, 106.1], xy), [False, True])   # 5 * 15 * sqrt(2)
    np.testing.assert_array_equal(features.trigger([129.8, 130.0], xyz), [False, True])  # 5 * 15 * sqrt(3)
    np.testing.assert_array_equal(features.trigger([100.0, 100.0], xy, sigma=[10.0, 30.0]), [True, False])


def _normalized(gnn_model_card, x):
    model = GNNScaleMDN(gnn_model_card)
    x = torch.tensor(x, dtype=torch.float32)
    return (torch.sign(x) * torch.abs(x) ** model.power * model.scale + model.offset).numpy()


def test_gnn_lib_normalisation_is_that_of_gnn_for_paper():
    r"""GNN_lib's NormaliseLayer with NormaliseLayerDeltaT of GNN_forPaper, on (pos, pos, pos, t_sm, p2p_sm, delta_t)."""
    names = ["pos_x", "pos_y", "pos_z", "time", "amplitude", "delta_t"]
    gnn_model_card = gnn.complete_gnn_model_card({"task": "direction", "inputs": {"node_features": names}})
    x = np.array([[1234.0, -567.0, 12.0, 3.2e-6, 850.0, -4e-9]])
    expected = x.copy()
    expected[:, :2] /= 10000                     # x[:, mask_pos[:, :2]] /= 10000
    expected[:, 2] /= 100                        # x[:, mask_pos[:, 2]] /= 100
    expected[:, 3] *= 20000                      # times spanning less than 1 (seconds)
    expected[:, 4] = expected[:, 4] ** (1 / 5) / 8
    expected[:, 5] /= 2.6e-8                     # NormaliseLayerDeltaT
    np.testing.assert_allclose(_normalized(gnn_model_card, x), expected, rtol=1e-5)
    assert "scale" not in GNNScaleMDN(gnn_model_card).state_dict()     # in the card, not with the weights


def test_explicit_normalisation_constants():
    gnn_model_card = gnn.complete_gnn_model_card({"task": "direction", "normalisation": NORMALISATION})
    assert gnn_model_card["normalisation_method"] == "constants"
    np.testing.assert_allclose(_normalized(gnn_model_card, [[1e4, -2e4, 100.0, 1e-6, 1024.0]])[0],
                               [1, -2, 1, 0.02, 1024 ** 0.2 / 8], rtol=1e-6)


def test_fitted_normalisation_finds_the_power_of_the_amplitude():
    rng = np.random.default_rng(0)
    n = 20000
    x = np.column_stack([rng.uniform(-3000, 3000, n),                   # positions: spread evenly
                         10 ** rng.normal(2.5, 0.5, n)])                # amplitudes: over decades
    fitted = features.fit_normalisation(x, ["pos_x", "amplitude"])
    assert fitted["pos_x"]["power"] >= 0.8 and fitted["amplitude"]["power"] <= 0.25
    gnn_model_card = gnn.complete_gnn_model_card({"task": "direction", "inputs": {"node_features": ["pos_x", "amplitude"]},
                                                  "normalisation": fitted})
    y = _normalized(gnn_model_card, x)
    np.testing.assert_allclose(y.mean(axis=0), 0, atol=1e-3)
    np.testing.assert_allclose(y.std(axis=0), 1, rtol=1e-3)
    # The amplitude comes out about symmetric, where a mean/std scaling would leave it skewed
    skew = lambda v: np.mean((v - v.mean()) ** 3) / v.std() ** 3
    assert abs(skew(y[:, 1])) < 0.3 < skew(x[:, 1])
    assert gnn.complete_gnn_model_card({"task": "direction", "normalisation": "fit"})["normalisation"] == "fit"


def test_legacy_state_dict_loads():
    model = GNNScaleMDN(gnn.complete_gnn_model_card({"task": "direction"}))
    assert {"graph_net.convs.0.nn.lins.0.weight", "fc1.0.weight", "fc2.2.bias"} <= set(model.state_dict())


def test_save_and_load(settings, tmp_path):
    model = gnn.load_model("direction")
    assert len(model.models) == 2 and model.tag == "tiny/v0"
    assert gnn.load_model("direction") is model                         # loaded once
    with open(tmp_path / "models" / "tiny" / "v0" / "seed_01.pt", "ab") as f:
        f.write(b"x")
    gnn.clear_cache()
    with pytest.raises(ValueError, match="does not match"):
        gnn.load_model("direction")


def test_choose_the_network(settings, tmp_path):
    for version in ("v2", "v10"):
        _save(tmp_path / "models" / "tiny" / version, version=version)
    _save(tmp_path / "elsewhere", name="trial")
    assert gnn.load_model("direction", "tiny/v2").tag == "tiny/v2"
    assert gnn.load_model("direction", "tiny").tag == "tiny/v10"        # latest version
    assert gnn.load_model("direction", str(tmp_path / "elsewhere")).tag == "trial/v0"
    with pytest.raises(ValueError, match="energy network"):
        gnn.load_model("direction", "tiny_e/v0")
    with pytest.raises(FileNotFoundError, match="download_gnn_models"):
        gnn.load_model("direction", "tiny/v9")


def test_recons(settings):
    Xants, t, a = _plane_wave()
    zenith, azimuth, std_zenith, std_azimuth, du_count = gnn.recons_GNN(Xants, t, a)
    assert 0 <= zenith <= np.pi and 0 <= azimuth < 2 * np.pi
    assert std_zenith > 0 and std_azimuth > 0 and du_count == 12
    # Neither the time offset nor the frame origin changes anything
    again = gnn.recons_GNN(Xants + [500, -300, 0], t + 5e-3, a)
    np.testing.assert_allclose(again[:2], (zenith, azimuth), rtol=1e-5)
    energy = gnn.recons_energy_GNN(Xants, t, a)
    assert energy.energy > 0 and energy.std_energy > 0 and energy.du_count == 12


def test_too_few_antennas_give_nan(settings, tmp_path):
    _save(tmp_path / "cut", trigger={"noise_rms_per_channel": 100.0})
    model = gnn.load_model("direction", str(tmp_path / "cut"))
    Xants, t, a = _plane_wave()
    a[:] = 100.0
    a[:4] = 5000.0                                                      # 4 antennas pass, 5 needed
    result = gnn.recons_GNN(Xants, t, a, model=model)
    assert np.isnan(result.zenith) and result.du_count == 4
    assert gnn.recons_GNN(Xants, t, a, model=model, sigma=1.0).du_count == 12
    lower = gnn.load_model("direction", str(tmp_path / "cut"), trigger={"noise_rms_per_channel": 1.0})
    assert gnn.recons_GNN(Xants, t, a, model=lower).du_count == 12


def test_direction_from_k():
    theta, phi, sigma = 1.2, 4.0, 1e-3
    k = features.propagation_vector(theta, phi)
    zenith, azimuth, std_zenith, std_azimuth = gnn.direction_from_k([k], [[sigma**2] * 3])
    assert (zenith, azimuth) == pytest.approx((theta, phi))
    assert std_zenith == pytest.approx(sigma)
    assert std_azimuth == pytest.approx(sigma / np.sin(theta))
    _, _, wider, _ = gnn.direction_from_k([k, k + [0, 0, 2e-3]], [[sigma**2] * 3] * 2)
    assert wider > std_zenith                                           # the spread of the members adds


def test_energy_from_log():
    energy, std = gnn.energy_from_log([0.0, 0.2], [0.01, 0.01], 1e17)
    assert energy == pytest.approx(1e17 * np.exp(0.1))
    assert std == pytest.approx(energy * np.sqrt(0.01 + 0.01))


def test_peaks_are_those_of_the_adf(settings):
    rng = np.random.default_rng(3)
    traces = rng.normal(0, 5, size=(6, 4, 256))
    traces[:, 1:3, 100:110] += 300
    t0 = rng.uniform(0, 500, 6)
    amps, times = gnn.peaks(traces, t0, gnn.load_model("direction"), source="data")
    np.testing.assert_array_equal(amps, [sig.get_peak_amplitude(tr, [1, 2]) for tr in traces])
    np.testing.assert_array_equal(times, [sig.get_peak_time(tr, t0[i], [1, 2]) for i, tr in enumerate(traces)])


def test_trecons_stores_the_gnn_fields(tmp_path):
    from grand.dataio import TRecons

    written = tmp_path / "recons.root"
    tree = TRecons()
    tree.run_number, tree.event_number = 1, 7
    tree.peak_amps_gnn, tree.peak_time_gnn = [300.0, 400.0], [1e-6, 2e-6]
    tree.du_count_gnn = 2
    tree.zenith_gnn, tree.azimuth_gnn, tree.std_zenith_gnn = 1.2, 5.0, 0.01
    tree.energy_em_gnn = 3e17
    tree.gnn_model = "direction=pgnn_direction_xy/v0"
    tree.fill()
    tree.write(str(written))

    back = TRecons(str(written))
    back.get_event(7, 1)
    assert (back.zenith_gnn, back.azimuth_gnn, back.std_zenith_gnn) == pytest.approx((1.2, 5.0, 0.01))
    assert back.energy_em_gnn == pytest.approx(3e17, rel=1e-6)
    assert list(back.peak_amps_gnn) == pytest.approx([300.0, 400.0])
    assert back.du_count_gnn == 2 and str(back.gnn_model) == "direction=pgnn_direction_xy/v0"
