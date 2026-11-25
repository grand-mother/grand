"""The graph a GNN sees: one definition for training and inference.

The training repository builds its datasets with :func:`make_graph`, and
:func:`grand.analysis.gnn.recons_GNN` calls it on measured events, so a
network sees at inference exactly what it was trained on.  The peaks come
from :mod:`grand.analysis.signals` (the functions of the ADF chain).

Which features a network uses, and in which order, is written in its GNN
model card (``inputs``); this module says how to compute each of them, and
how they can be normalized (NORMALISATIONS, fit_normalisation).
"""

import numpy as np

import grand.analysis.constants as cons

# Node features: one value per antenna.  `pos` is relative to the mean
# position of the antennas, `time` to the earliest peak; `k` is the
# plane-wave propagation vector.  To add a feature, add it here (so that
# training and inference compute it the same way) and name it in a GNN model card.
NODE_FEATURES = {
    "pos_x": lambda pos, time, amp, k, inputs: pos[:, 0],
    "pos_y": lambda pos, time, amp, k, inputs: pos[:, 1],
    "pos_z": lambda pos, time, amp, k, inputs: pos[:, 2],
    "time": lambda pos, time, amp, k, inputs: time,
    "amplitude": lambda pos, time, amp, k, inputs: amp,
    # Residual to the plane wave, mean removed
    "delta_t": lambda pos, time, amp, k, inputs: _centered(time - pos @ k / inputs["delta_t_wave_speed"]),
}

# Graph features: one value per event.
GRAPH_FEATURES = {
    "k_pwf_x": lambda k: k[0],
    "k_pwf_y": lambda k: k[1],
    "k_pwf_z": lambda k: k[2],
}

# The features that need the plane-wave fit
NEEDS_PWF = {"delta_t", "k_pwf_x", "k_pwf_y", "k_pwf_z"}

# Normalization of the node features, done inside the network:
#     x -> scale * sign(x) * |x|**power + offset
# A GNN model card chooses "normalisation": one of these names, "fit" (see
# fit_normalisation), or the constants of each feature.  The saved card
# holds the constants used, whichever way they were chosen.
IDENTITY = {"scale": 1.0, "power": 1.0, "offset": 0.0}
NORMALISATIONS = {
    # GNN_lib's NormaliseLayer, as GNN_forPaper used it (NormaliseLayerDeltaT for delta_t).
    # GNN_lib multiplies the times by 2e4 when they span less than 1, i.e. when in seconds, as here.
    "gnn_lib": {
        "pos_x": {"scale": 1e-4, "power": 1.0, "offset": 0.0},
        "pos_y": {"scale": 1e-4, "power": 1.0, "offset": 0.0},
        "pos_z": {"scale": 1e-2, "power": 1.0, "offset": 0.0},
        "time": {"scale": 2e4, "power": 1.0, "offset": 0.0},
        "amplitude": {"scale": 1 / 8, "power": 0.2, "offset": 0.0},
        "delta_t": {"scale": 1 / 2.6e-8, "power": 1.0, "offset": 0.0},
    },
}


def fit_normalisation(x, names, powers=np.arange(0.05, 2.0001, 0.05)):
    """Normalization constants fitted on the node features of training graphs.

    For each feature, the power is the one making sign(x) * |x|**power most
    Gaussian (maximum likelihood, as for a Box-Cox transform): about 0.2 for
    amplitudes spread over decades, about 1 for positions.  Then scale and
    offset give the transformed feature a mean 0 and a standard deviation 1.

    Parameters
    ----------
    x : np.ndarray, shape (n_nodes, F)
        Node features of the training graphs, unnormalized, in the order of `names`.
    names : list of str
        The node features.
    powers : array-like
        The powers tried.

    Returns
    -------
    dict
        {name: {"scale", "power", "offset"}}.
    """
    fitted = {}
    for name, column in zip(names, np.asarray(x, dtype=float).T):
        log_abs = np.log(np.abs(column[column != 0]))

        def log_likelihood(p):
            y = np.sign(column) * np.abs(column) ** p
            # Gaussian likelihood of y, and the Jacobian of x -> y on the nonzero values
            return -0.5 * len(column) * np.log(y.var()) + len(log_abs) * np.log(p) + (p - 1) * log_abs.sum()

        power = float(max(powers, key=log_likelihood))
        y = np.sign(column) * np.abs(column) ** power
        fitted[name] = {"scale": float(1 / y.std()), "power": power, "offset": float(-y.mean() / y.std())}
    return fitted


def _centered(x):
    return x - x.mean()


def channel_indices(channels, source):
    """Trace indices of the channels named in a GNN model card (e.g. ["x", "y"]).

    `source` is "data" (measured traces) or "sim" (simulated traces): their
    channels differ, see grand.analysis.constants.CHANNEL_INDEX.
    """
    return [cons.CHANNEL_INDEX[source][c] for c in channels]


def propagation_vector(theta, phi):
    """Unit vector along which the shower travels, from its "coming from" zenith and azimuth (radians).

    Same convention as the GNN codes: ``k = -sph2cart(theta, phi)``.
    """
    return -np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def trigger(peak_amps, inputs, sigma=None):
    """Antennas a network uses: those whose peak exceeds n_sigma times the noise.

    Parameters
    ----------
    peak_amps : array-like, shape (N,)
        Peak amplitudes (ADC).
    inputs : dict
        The ``inputs`` block of the GNN model card.
    sigma : float or array-like, optional
        Noise RMS of the network's channels combined, for all antennas or per
        antenna.  Default: ``noise_rms_per_channel * sqrt(number of channels)``
        of the GNN model card's trigger, i.e. 5 * 15 * sqrt(2) = 106 ADC on x, y.

    Returns
    -------
    np.ndarray of bool
    """
    trig = inputs["trigger"]
    if sigma is None:
        sigma = trig["noise_rms_per_channel"] * np.sqrt(len(inputs["channels"]))
    return np.asarray(peak_amps) > trig["n_sigma"] * np.asarray(sigma)


def knn_edge_index(Xants, k):
    """Edges linking each antenna to itself and its k nearest neighbors, in both directions.

    Same graph as ``to_undirected(knn_graph(pos, k + 1, loop=True))`` of
    PyTorch Geometric, as in the GNN codes.

    Returns
    -------
    np.ndarray, shape (2, E)
        Source and target indices, sorted by target then source.
    """
    n = len(Xants)
    distance = np.linalg.norm(Xants[:, None, :] - Xants[None, :, :], axis=-1)
    np.fill_diagonal(distance, -1.0)  # itself first
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :min(k + 1, n)]
    target = np.repeat(np.arange(n), nearest.shape[1])
    source = nearest.ravel()
    pairs = np.unique(np.concatenate([np.stack([source, target], axis=1),
                                      np.stack([target, source], axis=1)]), axis=0)
    pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]
    return pairs.T


def make_graph(Xants, peak_time, peak_amps, inputs, sigma=None):
    """Graph of one event, as the network takes it.

    The antennas that pass the trigger are kept, the plane-wave direction is
    fitted on them if the features need it, and the features are computed in
    the GNN model card's order.

    Parameters
    ----------
    Xants : array-like, shape (N, 3)
        Antenna positions (m).
    peak_time : array-like, shape (N,)
        Peak times (s), from grand.analysis.signals.get_peak_time.
    peak_amps : array-like, shape (N,)
        Peak amplitudes (ADC), from grand.analysis.signals.get_peak_amplitude.
    inputs : dict
        The ``inputs`` block of the GNN model card.
    sigma : float or array-like, optional
        Measured noise for the trigger (see :func:`trigger`).

    Returns
    -------
    dict or None
        ``x`` (node features, (n, F)), ``edge_index`` (2, E), ``x_graph``
        (graph features, (1, G)) and ``mask`` (antennas used, (N,)); None if
        fewer than ``min_antennas`` antennas pass the trigger, or if the
        plane-wave fit the features need fails (NaN).
    """
    Xants = np.asarray(Xants, dtype=float)
    peak_time = np.asarray(peak_time, dtype=float)
    peak_amps = np.asarray(peak_amps, dtype=float)
    mask = trigger(peak_amps, inputs, sigma)
    if mask.sum() < inputs["trigger"]["min_antennas"]:
        return None
    pos, time, amp = Xants[mask], peak_time[mask], peak_amps[mask]

    k = None
    if NEEDS_PWF & set(inputs["node_features"] + inputs["graph_features"]):
        from grand.analysis.fitting import PWF_semianalytical
        theta, phi = PWF_semianalytical(pos, time, c=cons.c_light, n=cons.n_atm)
        k = propagation_vector(theta, phi)
        if not np.all(np.isfinite(k)):   # the plane-wave fit failed
            return None

    pos = pos - pos.mean(axis=0)
    time = time - time.min()
    x = np.column_stack([NODE_FEATURES[name](pos, time, amp, k, inputs) for name in inputs["node_features"]])
    x_graph = np.array([[GRAPH_FEATURES[name](k) for name in inputs["graph_features"]]]).reshape(1, -1)
    return {"x": x, "edge_index": knn_edge_index(pos, inputs["knn_k"]), "x_graph": x_graph, "mask": mask}
