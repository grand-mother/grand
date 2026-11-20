"""Shower reconstruction with graph neural networks (GNN).

The networks of A. Ferrière (pGNN direction and energy), run on the peaks of
grand.analysis.signals (the functions of the ADF chain) and stored in
TRecons beside the PWF, SWF and ADF results.

* features: the graph a network sees (shared with the training repository);
* architecture: the network, built from its GNN model card;
* model_store: which networks (settings.json), how they are saved and loaded;
* recons: recons_GNN, recons_energy_GNN, peaks.

Needs torch and torch_geometric: pip install torch torch_geometric.
"""

try:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401
except ImportError as error:
    raise ImportError('grand.analysis.gnn needs torch and torch_geometric: pip install torch torch_geometric') from error

from .features import NODE_FEATURES, GRAPH_FEATURES, make_graph, trigger
from .model_store import (GNNEnsemble, load_model, find_model, save_model, complete_gnn_model_card, load_settings,
                          clear_cache, SETTINGS_FILE, SETTINGS_ENV)
from .recons import recons_GNN, recons_energy_GNN, peaks, direction_from_k, energy_from_log, GNNDirection, GNNEnergy
