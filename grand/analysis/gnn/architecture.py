"""The network: GNNScaleMDN of the GNN codes, built from a GNN model card.

An EdgeCNN, a mean and max readout, then two heads: the value (fc1) and its
variance (fc2).  The submodules keep the names of the GNN codes, so their
``state_dict`` files load.  The features are normalized inside the network,
``scale * sign(x) * |x|**power + offset``, with the constants of the GNN model card.
"""

import torch
import torch_geometric.nn as tgn


class GNNScaleMDN(torch.nn.Module):
    """The network a GNN model card describes (blocks ``inputs``, ``normalisation``, ``architecture``)."""

    def __init__(self, gnn_model_card):
        super().__init__()
        features = gnn_model_card["inputs"]["node_features"]
        arch = gnn_model_card["architecture"]
        out = 3 if gnn_model_card["task"] == "direction" else 1

        # Normalization constants: from the GNN model card only, not saved with the weights
        norm = [gnn_model_card["normalisation"][name] for name in features]
        for key in ("scale", "power", "offset"):
            self.register_buffer(key, torch.tensor([n[key] for n in norm], dtype=torch.float32), persistent=False)

        self.graph_net = tgn.EdgeCNN(len(features), arch["hidden"], arch["n_graph_layers"],
                                     arch["graph_out_channels"], dropout=arch["dropout"],
                                     norm=arch["norm"], jk=arch["jk"])
        width = 2 * arch["graph_out_channels"] + len(gnn_model_card["inputs"]["graph_features"])
        self.fc1 = torch.nn.Sequential(torch.nn.Linear(width, arch["graph_out_channels"]), torch.nn.ReLU(),
                                       torch.nn.Linear(arch["graph_out_channels"], out))
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(width, arch["graph_out_channels"]), torch.nn.ReLU(),
                                       torch.nn.Linear(arch["graph_out_channels"], out), torch.nn.Softplus())

    def forward(self, x, edge_index, batch, x_graph):
        """Returns (value, variance), each of shape (n_events, out)."""
        x = torch.sign(x) * torch.abs(x) ** self.power * self.scale + self.offset
        h = self.graph_net(x, edge_index, batch=batch)
        flat = torch.cat([tgn.global_mean_pool(h, batch), tgn.global_max_pool(h, batch), x_graph], dim=1)
        return self.fc1(flat), self.fc2(flat)
