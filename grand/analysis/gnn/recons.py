"""Direction and energy of a shower from a graph neural network (GNN).

The counterpart of grand.analysis.fitting.recons_ADF: from the antenna
positions and the peak times and amplitudes of grand.analysis.signals, it
returns radians "coming from" (and eV), as TRecons stores them in
zenith_gnn, azimuth_gnn and energy_em_gnn.

The uncertainty combines the variance each member predicts and the spread
of the members (variance of a mixture).
"""

from typing import NamedTuple

import numpy as np

import grand.analysis.signals as sig
from grand.analysis.gnn.features import make_graph, trigger
from grand.analysis.gnn.model_store import GNNEnsemble, load_model


class GNNDirection(NamedTuple):
    """Result of recons_GNN."""
    zenith: float  # radians, "coming from"; NaN if too few antennas
    azimuth: float  # radians in [0, 2 pi), "coming from"
    std_zenith: float
    std_azimuth: float
    du_count: int  # antennas the network used


class GNNEnergy(NamedTuple):
    """Result of recons_energy_GNN."""
    energy: float  # electromagnetic energy, eV; NaN if too few antennas
    std_energy: float
    du_count: int


def peaks(traces, t0, model, source="data", dt_ns=2):
    """Peak amplitude and time of each antenna on the network's channels, with the ADF chain's functions.

    Parameters
    ----------
    traces : array-like, shape (N, channels, samples)
        Traces in ADC.
    t0 : array-like, shape (N,)
        Start time of each trace in ns (grand.analysis.signals.compute_t0).
    model : GNNEnsemble
        The network: its GNN model card names the channels.
    source : str
        "data" for measured traces, "sim" for simulated ones: their channels
        differ (grand.analysis.constants.CHANNEL_INDEX).
    dt_ns : float
        Sampling period (ns).

    Returns
    -------
    peak_amps, peak_time : np.ndarray, shape (N,)
        Amplitudes, and times in seconds.
    """
    channels = model.channel_indices(source)
    peak_amps = np.array([sig.get_peak_amplitude(trace, channels) for trace in traces])
    peak_time = np.array([sig.get_peak_time(trace, t0[i], channels, dt_ns=dt_ns) for i, trace in enumerate(traces)])
    return peak_amps, peak_time


def direction_from_k(k, var_k):
    """Zenith, azimuth and their standard deviations from the members' propagation vectors.

    Parameters
    ----------
    k : array-like, shape (M, 3)
        Propagation vector predicted by each member.
    var_k : array-like, shape (M, 3)
        Their predicted variances.
    """
    k = np.asarray(k, dtype=float)
    k_mean = k.mean(axis=0)
    variance = np.mean(var_k, axis=0) + k.var(axis=0)
    norm = np.linalg.norm(k_mean)
    source = -k_mean / norm
    v = variance / norm**2

    zenith = np.arccos(source[2])
    azimuth = np.arctan2(source[1], source[0]) % (2 * np.pi)
    st, ct, sp, cp = np.sin(zenith), np.cos(zenith), np.sin(azimuth), np.cos(azimuth)
    std_zenith = np.sqrt(ct**2 * cp**2 * v[0] + ct**2 * sp**2 * v[1] + st**2 * v[2])
    std_azimuth = np.sqrt(sp**2 * v[0] + cp**2 * v[1]) / st
    return zenith, azimuth, std_zenith, std_azimuth


def energy_from_log(y, var_y, scale_eV):
    """Energy (eV) and its standard deviation from the members' log(E / scale_eV)."""
    y = np.ravel(y)
    variance = np.mean(var_y) + y.var()
    energy = scale_eV * np.exp(y.mean())
    return energy, energy * np.sqrt(variance)


def recons_GNN(Xants, peak_time, peak_amps, model=None, sigma=None):
    """Shower direction from the GNN.

    Parameters
    ----------
    Xants : array-like, shape (N, 3)
        Antenna positions (m), as for the ADF.
    peak_time : array-like, shape (N,)
        Peak times (s) on the network's channels (see :func:`peaks`).
    peak_amps : array-like, shape (N,)
        Peak amplitudes (ADC) on the network's channels.
    model : GNNEnsemble or str, optional
        The network, or what load_model accepts: None for the default one,
        "name/version", or a directory.
    sigma : float or array-like, optional
        Measured noise for the trigger (see features.trigger).

    Returns
    -------
    GNNDirection
        (zenith, azimuth, std_zenith, std_azimuth, du_count); NaN angles if
        too few antennas pass the trigger.
    """
    if not isinstance(model, GNNEnsemble):
        model = load_model("direction", model)
    graph = make_graph(Xants, peak_time, peak_amps, model.inputs, sigma)
    if graph is None:
        return GNNDirection(np.nan, np.nan, np.nan, np.nan, int(trigger(peak_amps, model.inputs, sigma).sum()))
    value, variance = model.predict([graph])
    return GNNDirection(*direction_from_k(value[:, 0], variance[:, 0]), int(graph["mask"].sum()))


def recons_energy_GNN(Xants, peak_time, peak_amps, model=None, sigma=None):
    """Electromagnetic energy (eV) from the GNN; same inputs as :func:`recons_GNN`.

    Returns
    -------
    GNNEnergy
        (energy, std_energy, du_count); NaN energy if too few antennas pass the trigger.
    """
    if not isinstance(model, GNNEnsemble):
        model = load_model("energy", model)
    graph = make_graph(Xants, peak_time, peak_amps, model.inputs, sigma)
    if graph is None:
        return GNNEnergy(np.nan, np.nan, int(trigger(peak_amps, model.inputs, sigma).sum()))
    value, variance = model.predict([graph])
    energy, std_energy = energy_from_log(value[:, 0, 0], variance[:, 0, 0], model.gnn_model_card["output"]["scale_eV"])
    return GNNEnergy(energy, std_energy, int(graph["mask"].sum()))
