"""Where the networks are, and how they are saved and loaded.

A network is a directory holding ``gnn_model_card.json`` (task, inputs,
normalization, architecture, training hyperparameters, provenance and the
SHA-256 of each member) and one ``seed_XX.pt`` per ensemble member.
Downloaded networks are in ``<GRAND_DATA_PATH>/gnn/<name>/<version>/``
(``data/download_gnn_models.py``).

``settings.json``, next to this module (or the file named by
``$GRAND_GNN_SETTINGS``), chooses the default network of each task.  A
script can name another one: see :func:`load_model`.

``defaults.json`` completes a new GNN model card when it is saved (:func:`save_model`),
for example with the normalization of GNN_forPaper ("gnn_lib"),
so a saved GNN model card states every value and does not depend on the defaults.
"""

import copy
import hashlib
import json
import os

import numpy as np
import torch

import grand.analysis.constants as cons
from grand import GRAND_DATA_PATH
from grand.analysis.gnn.architecture import GNNScaleMDN
from grand.analysis.gnn.features import IDENTITY, NORMALISATIONS, channel_indices

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(HERE, "settings.json")
SETTINGS_ENV = "GRAND_GNN_SETTINGS"
DEFAULTS_FILE = os.path.join(HERE, "defaults.json")
GNN_MODEL_CARD_FILE = "gnn_model_card.json"

_loaded = {}


def read_json(path):
    """Reads a JSON file, without the keys starting with "_" (comments)."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if not k.startswith("_")}
        return value

    with open(path) as f:
        return strip(json.load(f))


def load_settings(path=None):
    """GNN settings: `path`, else $GRAND_GNN_SETTINGS, else the packaged settings.json."""
    path = path or os.environ.get(SETTINGS_ENV) or SETTINGS_FILE
    settings = read_json(path)
    models_dir = settings.get("models_dir") or os.path.join(GRAND_DATA_PATH, "gnn")
    # a relative models_dir is relative to the settings file
    settings["models_dir"] = os.path.join(os.path.dirname(os.path.abspath(path)), os.path.expanduser(models_dir))
    return settings


def complete_gnn_model_card(gnn_model_card):
    """Returns `gnn_model_card` with what it does not state taken from defaults.json."""
    defaults = read_json(DEFAULTS_FILE)
    gnn_model_card = copy.deepcopy(gnn_model_card)
    inputs = {**defaults["inputs"], **gnn_model_card.get("inputs", {})}
    inputs["trigger"] = {**defaults["inputs"]["trigger"], **gnn_model_card.get("inputs", {}).get("trigger", {})}
    if inputs["delta_t_wave_speed"] is None:
        inputs["delta_t_wave_speed"] = cons.c_light / cons.n_atm
    gnn_model_card["inputs"] = inputs
    gnn_model_card["architecture"] = {**defaults["architecture"], **gnn_model_card.get("architecture", {})}
    # Normalization: a name of features.NORMALISATIONS, "fit" (the training
    # repository fits it on the training graphs and writes the constants), or
    # the constants of each feature (identity for those it omits)
    normalisation = gnn_model_card.get("normalisation", defaults["normalisation"])
    if normalisation == "fit":
        gnn_model_card["normalisation_method"] = "fit"
    else:
        if isinstance(normalisation, str):
            gnn_model_card.setdefault("normalisation_method", normalisation)
            normalisation = NORMALISATIONS[normalisation]
        gnn_model_card.setdefault("normalisation_method", "constants")
        normalisation = {name: {**IDENTITY, **normalisation.get(name, {})} for name in inputs["node_features"]}
    gnn_model_card["normalisation"] = normalisation
    if gnn_model_card["task"] == "energy":
        gnn_model_card["output"] = {**defaults["energy_output"], **gnn_model_card.get("output", {})}
    return gnn_model_card


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def save_model(directory, gnn_model_card, state_dicts):
    """Saves a network: its completed GNN model card and one seed_XX.pt per member.

    Returns
    -------
    str
        The directory.
    """
    gnn_model_card = complete_gnn_model_card(gnn_model_card)
    os.makedirs(directory, exist_ok=True)
    gnn_model_card["ensemble"] = []
    for i, state in enumerate(state_dicts):
        path = os.path.join(directory, "seed_%02d.pt" % i)
        torch.save(state, path)
        gnn_model_card["ensemble"].append({"file": os.path.basename(path), "sha256": _sha256(path)})
    with open(os.path.join(directory, GNN_MODEL_CARD_FILE), "w") as f:
        json.dump(gnn_model_card, f, indent=2)
    return directory


class GNNEnsemble:
    """The members of a saved network, ready to predict.

    Parameters
    ----------
    directory : str
        The network's directory.
    device : str
        "cpu", "cuda" or "auto".
    trigger : dict, optional
        Replaces entries of the GNN model card's trigger, for data whose noise differs
        from the simulations' (e.g. ``{"noise_rms_per_channel": 18}``).
    """

    def __init__(self, directory, device="cpu", trigger=None):
        self.directory = directory
        self.gnn_model_card = read_json(os.path.join(directory, GNN_MODEL_CARD_FILE))
        self.gnn_model_card["inputs"]["trigger"].update(trigger or {})
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.models = []
        for member in self.gnn_model_card["ensemble"]:
            path = os.path.join(directory, member["file"])
            if _sha256(path) != member["sha256"]:
                raise ValueError("%s does not match its GNN model card: download the network again" % path)
            model = GNNScaleMDN(self.gnn_model_card)
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.models.append(model.to(self.device).eval())

    @property
    def task(self):
        return self.gnn_model_card["task"]

    @property
    def inputs(self):
        return self.gnn_model_card["inputs"]

    @property
    def tag(self):
        """"name/version", as written to TRecons.gnn_model."""
        return "%s/%s" % (self.gnn_model_card["name"], self.gnn_model_card["version"])

    def channel_indices(self, source):
        """Trace indices of the network's channels in measured ("data") or simulated ("sim") traces."""
        return channel_indices(self.inputs["channels"], source)

    def predict(self, graphs):
        """Runs every member on graphs from features.make_graph.

        Returns
        -------
        value, variance : np.ndarray, shape (n_members, n_graphs, out)
        """
        from torch_geometric.data import Batch, Data

        batch = Batch.from_data_list([
            Data(x=torch.tensor(g["x"], dtype=torch.float32),
                 edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
                 x_graph=torch.tensor(g["x_graph"], dtype=torch.float32)) for g in graphs]).to(self.device)
        values, variances = [], []
        with torch.no_grad():
            for model in self.models:
                value, variance = model(batch.x, batch.edge_index, batch.batch, batch.x_graph)
                values.append(value.cpu().numpy())
                variances.append(variance.cpu().numpy())
        return np.array(values, dtype=float), np.array(variances, dtype=float)

    def __repr__(self):
        return "GNNEnsemble(%s, %s, %d members, channels %s, trigger %s)" % (
            self.tag, self.task, len(self.models), self.inputs["channels"], self.inputs["trigger"])


def find_model(task, model=None, settings=None):
    """Directory of a network.

    Parameters
    ----------
    task : str
        "direction" or "energy".
    model : str, optional
        None: the network settings.json chooses for `task`.
        "name/version", or "name" for its latest version: in the models directory.
        Otherwise the path of a network's directory.
    settings : str, optional
        Another settings file.
    """
    settings = load_settings(settings)
    if model is None:
        model = "%s/%s" % (settings[task]["name"], settings[task]["version"])
    if os.path.isdir(model):
        return model
    name, _, version = model.partition("/")
    if not version:
        versions = os.listdir(os.path.join(settings["models_dir"], name))
        version = max(versions, key=lambda v: int("".join(c for c in v if c.isdigit()) or 0))
    return os.path.join(settings["models_dir"], name, version)


def load_model(task="direction", model=None, settings=None, trigger=None):
    """The network `model` names for `task` (see :func:`find_model`), loaded once.

    Examples: ``load_model("direction")`` (the default network),
    ``load_model("direction", "pgnn_direction_xy/v1")``,
    ``load_model("energy", "/path/to/my_network")``.
    """
    directory = find_model(task, model, settings)
    if not os.path.isfile(os.path.join(directory, GNN_MODEL_CARD_FILE)):
        raise FileNotFoundError("no GNN network in %s: run data/download_gnn_models.py, "
                                "or name another network" % directory)
    key = (os.path.abspath(directory), json.dumps(trigger, sort_keys=True))
    if key not in _loaded:
        ensemble = GNNEnsemble(directory, load_settings(settings).get("device", "cpu"), trigger)
        if ensemble.task != task:
            raise ValueError("%s is a %s network, not a %s one" % (ensemble.tag, ensemble.task, task))
        _loaded[key] = ensemble
    return _loaded[key]


def clear_cache():
    """Forgets the loaded networks."""
    _loaded.clear()
