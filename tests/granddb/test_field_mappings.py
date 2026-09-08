# -*- coding: utf-8 -*-
r"""The database column mappings name fields that actually exist.

``granddb/rootdblib.py`` carries one dictionary per tree type -- ``trunToDB``,
``tshowerToDB`` and so on -- mapping a field on a GRAND tree class to a column
in the PostgreSQL catalogue.  ``granddblib`` walks those dictionaries and reads
each field off the tree with a bare ``getattr``, so a key naming something the
class does not have raises :exc:`AttributeError` at registration time rather
than being skipped.

Nothing connected the two sides.  Renaming a field in
``grand/dataio/event_trees.py`` -- which happens; ``nutrig_rhox`` arrived that
way on 2026-09-08 -- left the mapping pointing at a name that no longer
existed, and no test, lint rule or type check would notice.  This is that
check.

It found one on the day it was written: ``trunnoiseToDB`` asked for
``GalNoiseMap`` and ``GalNoiseLST``, where :class:`TRunNoise` has
``gal_noise_map`` and ``gal_noise_LST``.  Neither spelling has ever existed in
``grand/``.  It survived because no ROOT file in this repository contains a
``trunnoise`` tree, so the code path was never reached.
"""
import dataclasses

import pytest

import grand.dataio as gd
from granddb.rootdblib import RootFile

#: Mappings whose tree type no longer exists in ``grand.dataio``.  These predate
#: the ``tevent*`` -> ``t*`` renaming and describe trees GRANDlib stopped
#: writing.  They are inert -- ``granddblib`` only consults a mapping when it
#: meets a tree of that name -- and are left in place rather than deleted,
#: since they record what those columns were for.  The list is pinned so that
#: it cannot quietly grow.
RETIRED = {
    "teventshowerToDB",
    "teventshowersimdataToDB",
    "teventshowerzhairesToDB",
    "trunefieldsimdataToDB",
}

#: Not a tree mapping at all: it maps a tree's *metadata* -- name, comment,
#: analysis level -- rather than its fields, and is read from a different place.
METADATA = {"metaToDB"}


def _mappings():
    r"""Returns every ``*ToDB`` dictionary declared on :class:`RootFile`."""
    return sorted(name for name in dir(RootFile) if name.endswith("ToDB"))


def _tree_class(mapping_name):
    r"""Returns the tree class a mapping describes, or ``None``.

    Parameters
    ----------
    mapping_name : str
        For example ``"trunToDB"``.

    Returns
    -------
    type or None
        The class in :mod:`grand.dataio` whose name matches the mapping's stem,
        compared case-insensitively -- the mapping is ``trunnoiseToDB`` and the
        class is ``TRunNoise``.  ``None`` when no such class exists.
    """
    stem = mapping_name[: -len("ToDB")]
    for candidate in dir(gd):
        if candidate.lower() == stem.lower():
            return getattr(gd, candidate)
    return None


@pytest.mark.parametrize(
    "mapping_name",
    [n for n in _mappings() if n not in RETIRED and n not in METADATA],
)
def test_every_mapping_names_fields_that_exist(mapping_name):
    r"""Each key is a real field on the tree class the mapping describes.

    A key that is not raises ``AttributeError`` in ``granddblib`` when a file
    holding that tree is registered.  The failure message names the offenders
    and the nearest real fields, because the usual cause is a rename.
    """
    cls = _tree_class(mapping_name)
    assert cls is not None, (
        "%s describes a tree class that no longer exists. If the tree was "
        "retired, add the mapping to RETIRED in this file with a note." % mapping_name
    )

    declared = {field.name for field in dataclasses.fields(cls)}
    keys = [key for key in getattr(RootFile, mapping_name) if key != "table"]
    missing = [key for key in keys if key not in declared]

    assert not missing, (
        "%s maps %d field(s) that %s does not have: %s. granddblib reads these "
        "with a bare getattr, so registering such a tree would raise. The tree "
        "class has: %s"
        % (mapping_name, len(missing), cls.__name__, missing, sorted(declared))
    )


def test_the_mapping_set_is_the_one_we_know_about():
    r"""No mapping appears or disappears unnoticed.

    A new ``*ToDB`` is a new claim about the database schema, and a vanished one
    means a column set is no longer being written.  Either is worth a moment's
    attention rather than a silent diff.
    """
    live = {n for n in _mappings() if n not in RETIRED and n not in METADATA}
    assert live == {
        "trunToDB",
        "trunefieldsimToDB",
        "trunnoiseToDB",
        "trunrawvoltageToDB",
        "trunshowersimToDB",
        "trunvoltageToDB",
        "tshowerToDB",
        "tshowersimToDB",
    }, "the set of live mappings changed: %s" % sorted(live)


@pytest.mark.parametrize("mapping_name", sorted(RETIRED))
def test_retired_mappings_really_have_no_tree_class(mapping_name):
    r"""The retired list is retired for the stated reason.

    If one of these tree types comes back, the mapping stops being inert and
    belongs under the check above instead.
    """
    assert _tree_class(mapping_name) is None, (
        "%s has a tree class again -- move it out of RETIRED so its fields are "
        "checked." % mapping_name
    )


def test_every_mapping_declares_its_table():
    r"""Each live mapping says which table it writes to.

    ``granddblib`` reads ``mapping.get('table')`` and would otherwise pass
    ``None`` into the SQL it builds.
    """
    for name in _mappings():
        if name in METADATA:
            continue
        mapping = getattr(RootFile, name)
        assert mapping.get("table"), "%s has no 'table' key" % name
