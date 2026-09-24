# -*- coding: utf-8 -*-
r"""Pins the ROOT tree schema, so that a field change is visible in the diff.

Every field on a tree class becomes a branch in the files the collaboration
writes, and those files outlive the code that produced them.  A field added,
renamed or retyped is therefore a change to a data contract, not an
implementation detail -- but nothing made it visible: the fields are declared
as ordinary dataclass attributes, and a pull request that adds one looks like
a pull request that adds a variable.

This test records the schema in ``schema_snapshot.json`` and compares the
package against it.  Adding a field fails the test until the snapshot is
regenerated, which puts the change in the diff where a reviewer sees it::

    python tests/dataio/test_schema_snapshot.py --write

**This is not hypothetical.**  Two branches in flight both add NUTRIG
correlation fields to ``TADC``, by the same author, with the same type and
the same meaning, under different names -- ``nutrig_rhox``/``nutrig_rhoy`` on
``dev_nutrig_fields`` and ``correlation_x``/``correlation_y`` on
``dev_fix_root_warnings_lwp_new_fields``.  Neither branch had been merged, so
nothing compared them.  With this snapshot in place the second merge fails
with the two names side by side, which is the moment to notice that only one
of them should exist.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'schema_snapshot.json')

# The tree classes whose layout is a data contract.  Ordered as they appear
# in the data model, not alphabetically, so the snapshot reads like the
# format documentation.
TREE_CLASSES = ['TRun', 'TRunVoltage', 'TRunRawVoltage', 'TADC', 'TRawVoltage', 'TVoltage',
                'TEfield', 'TShower', 'TShowerSim', 'TRunEfieldSim',
                'TRunShowerSim', 'TRunNoise',
                # Reconstruction results, from dev_marion; approved as part of
                # the format on 2026-09-24 (RECOVERY_PLAN.md, Phase 5).
                'TRecons']


def _collect():
    r"""Returns ``{class name: {field name: type string}}`` for the tree classes.

    Returns
    -------
    dict
        One entry per class in `TREE_CLASSES` that the package defines.
        Fields are read from the dataclass definition, so the snapshot
        follows the source rather than a live ROOT file.
    """
    import dataclasses

    import grand.dataio.event_trees as et
    import grand.dataio.run_trees as rt

    found = {}
    for name in TREE_CLASSES:
        cls = getattr(et, name, None) or getattr(rt, name, None)
        if cls is None or not dataclasses.is_dataclass(cls):
            continue
        found[name] = {f.name: str(f.type) for f in dataclasses.fields(cls)
                       if not f.name.startswith('_')}
    return found


def test_schema_matches_snapshot():
    r"""The declared tree fields are exactly those in the snapshot."""
    if not os.path.exists(SNAPSHOT):
        pytest.skip('no snapshot yet; run this file with --write to create it')

    current = _collect()
    with open(SNAPSHOT) as handle:
        recorded = json.load(handle)

    for cls in sorted(set(recorded) | set(current)):
        want = set(recorded.get(cls, {}))
        have = set(current.get(cls, {}))
        added, removed = sorted(have - want), sorted(want - have)
        assert not added and not removed, (
            'schema of %s changed: added %s, removed %s.  If this is '
            'intended, regenerate with `python %s --write` so the change '
            'appears in the diff.' % (cls, added, removed, __file__))

    for cls, fields in current.items():
        for field, kind in fields.items():
            was = recorded.get(cls, {}).get(field)
            assert was == kind, (
                'type of %s.%s changed from %r to %r' % (cls, field, was, kind))


def test_every_tree_class_is_pinned():
    r"""No tree class escapes the snapshot by not being in `TREE_CLASSES`.

    The snapshot only compares the classes it is told about, so a *new* tree
    passed it unseen -- which is the largest schema change there is.  Checked
    on 2026-09-24 against ``dev_marion``, whose ``TRecons`` adds a tree of 27
    fields and left this file green.  It also found ``TRunRawVoltage``, a tree
    on the trunk that had never been pinned at all.

    Every subclass of the two mother trees defined in the data modules is a
    tree; each must be listed above and so recorded in the snapshot.
    """
    import inspect

    import grand.dataio.event_trees as et
    import grand.dataio.run_trees as rt

    mothers = (et.MotherEventTree, rt.MotherRunTree)
    defined = sorted(
        name for module in (et, rt)
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and obj.__module__ == module.__name__
        and issubclass(obj, mothers) and obj not in mothers)

    missing = [name for name in defined if name not in TREE_CLASSES]
    assert not missing, (
        'tree classes not pinned by the schema snapshot: %s.  A new tree is a '
        'new part of the data format: add it to TREE_CLASSES in %s, then '
        'regenerate with `python %s --write` so it appears in the diff.'
        % (missing, __file__, __file__))


def test_no_duplicate_meaning_in_tadc():
    r"""``TADC`` does not carry two names for the NUTRIG correlation.

    A narrow guard against the specific collision described in the module
    docstring: both spellings present at once means two branches were merged
    that should have agreed on one name first.
    """
    fields = _collect().get('TADC', {})
    nutrig = {'nutrig_rhox', 'nutrig_rhoy'} & set(fields)
    correl = {'correlation_x', 'correlation_y'} & set(fields)
    assert not (nutrig and correl), (
        'TADC carries both %s and %s for the same quantity; pick one name '
        'before merging.' % (sorted(nutrig), sorted(correl)))


if __name__ == '__main__':
    if '--write' in sys.argv:
        with open(SNAPSHOT, 'w') as handle:
            json.dump(_collect(), handle, indent=2, sort_keys=True)
            handle.write('\n')
        print('wrote %s' % SNAPSHOT)
    else:
        print(json.dumps(_collect(), indent=2, sort_keys=True))
