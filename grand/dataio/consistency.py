# -*- coding: utf-8 -*-
r"""Checks that the run tree and an event tree of a folder describe each other.

The conversion scripts paired an event with its run by run number and its
units by ``du_id`` deep inside the computation, so an inconsistent folder
failed far from the cause -- "operands could not be broadcast together with
shapes (0,) (44,)", ``IndexError``, ``AttributeError`` on ``None`` -- or was
accepted silently (#249).  :func:`check_event_trees` is called by
``Efield2Voltage`` and the conversion scripts after opening their input and
before computing anything.
"""

import numpy as np

from grand.basis import validate as _validate


def check_event_trees(events, run, where, folder="", event_kind="efield"):
    r"""Raises if `events` and `run` do not describe the same runs and units.

    Parameters
    ----------
    events : MotherEventTree or None
        The event tree (efield or voltage) to be converted.
    run : TRun or None
        The run tree at the same level.
    where : str
        Who asks, for messages.
    folder : str, optional
        The input folder, for messages.
    event_kind : str, optional
        The event file's prefix, for messages: "efield" or "voltage".

    Raises
    ------
    FileNotFoundError
        If either tree is missing.
    ValueError
        If an event's run is not in the run tree, an event lists a unit twice
        or one the run does not hold, or a run's ``t_bin_size`` is not one
        positive value.
    """
    if events is None:
        raise FileNotFoundError(_validate.message(
            where, "%s has no %s file (%s_*_L<level>_*.root)" % (folder, event_kind, event_kind)))
    if run is None:
        raise FileNotFoundError(_validate.message(
            where, "%s has no run file (run_*_L<level>_*.root)" % folder))

    units = {}
    for entry in range(run.get_number_of_entries()):
        run.get_entry(entry)
        number = int(run.run_number)
        bins = np.asarray(run.t_bin_size, dtype=float)
        # A run whose showers hit no antenna has no units and no bins (#91)
        if bins.size == 0 and len(run.du_id) == 0:
            units[number] = set()
            continue
        if bins.size == 0 or not np.all(np.isfinite(bins)) or not np.all(bins > 0):
            raise ValueError(_validate.message(
                where, "run %d: t_bin_size must be positive, got %s" % (number, _short(bins))))
        # The first unit's value was used for all (a known limitation)
        if not np.allclose(bins, bins[0]):
            raise ValueError(_validate.message(
                where, "run %d: t_bin_size differs between units (%s); one sampling time per "
                "run is supported" % (number, _short(np.unique(bins)))))
        units[number] = set(int(du) for du in run.du_id)

    tree = events._tree
    # Draw's buffers hold "estimate" rows, and every unit is a row: size them
    # from the unit count, as DataFile does for chains
    tree.SetEstimate(tree.GetEntries() + 1)
    count = events.draw("Length$(du_id)", "", "goff")
    rows = int(np.sum(np.frombuffer(tree.GetV1(), dtype=np.float64, count=max(count, 0))))
    tree.SetEstimate(rows + 1)
    count = events.draw("du_id:run_number:event_number", "", "goff")
    if count < 0:
        raise ValueError(_validate.message(where, "cannot read du_id, run_number and event_number "
                                                  "from the %s file" % event_kind))
    du = np.frombuffer(tree.GetV1(), dtype=np.float64, count=count).astype(np.int64)
    run_numbers = np.frombuffer(tree.GetV2(), dtype=np.float64, count=count).astype(np.int64)
    event_numbers = np.frombuffer(tree.GetV3(), dtype=np.float64, count=count).astype(np.int64)

    for run_number in np.unique(run_numbers):
        if int(run_number) not in units:
            raise ValueError(_validate.message(
                where, "the %s file has events of run %d, but the run file holds run %s"
                % (event_kind, run_number, ", ".join(str(r) for r in sorted(units)) or "none")))
    keys = np.stack([run_numbers, event_numbers], axis=1)
    for run_number, event_number in np.unique(keys, axis=0):
        mask = (run_numbers == run_number) & (event_numbers == event_number)
        ids = du[mask]
        values, counts = np.unique(ids, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(_validate.message(
                where, "event %d (run %d) lists units %s more than once"
                % (event_number, run_number, values[counts > 1].tolist())))
        missing = sorted(set(ids.tolist()) - units[int(run_number)])
        if missing:
            raise ValueError(_validate.message(
                where, "units %s of event %d (run %d) are not in the run's du_id"
                % (missing[:10], event_number, run_number)))


def _short(values):
    values = np.ravel(values)
    return str(values[:5].tolist())[:-1] + (", ...]" if values.size > 5 else "]")
