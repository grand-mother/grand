# -*- coding: utf-8 -*-
r"""Tree classes must be constructible with no arguments.

Regression test for a NumPy 2 incompatibility that made every run and event
tree impossible to build.  ``TTreeScalarDesc.__set__`` receives the descriptor
object itself when a dataclass field takes its default, and the guard for that
case bound ``value`` to the same array as ``inst``, so the assignment became
``inst[0] = inst``.  NumPy 1 accepted a one-element array in a scalar slot;
NumPy 2 raises ``ValueError: setting an array element with a sequence``.

Nothing in the code changed when this appeared -- the environment moved.  The
container in which CI last ran successfully dates from January 2022 and
carried NumPy 1, which is why no test caught it.
"""

import numpy as np
import pytest

import grand.dataio.event_trees as event_trees
import grand.dataio.run_trees as run_trees

RUN_TREES = ['TRun', 'TRunVoltage', 'TRunNoise', 'TRunEfieldSim',
             'TRunShowerSim']
EVENT_TREES = ['TADC', 'TRawVoltage', 'TVoltage', 'TEfield', 'TShower',
               'TShowerSim']


def _classes(module, names):
    r"""Yields ``(name, class)`` for those `names` the module defines."""
    for name in names:
        cls = getattr(module, name, None)
        if cls is not None:
            yield name, cls


@pytest.mark.parametrize('name', RUN_TREES)
def test_run_tree_constructs(name):
    r"""A run tree can be built with no arguments."""
    cls = getattr(run_trees, name, None)
    if cls is None:
        pytest.skip('%s is not defined in this version' % name)
    cls()


@pytest.mark.parametrize('name', EVENT_TREES)
def test_event_tree_constructs(name):
    r"""An event tree can be built with no arguments."""
    cls = getattr(event_trees, name, None)
    if cls is None:
        pytest.skip('%s is not defined in this version' % name)
    cls()


def test_scalar_default_is_the_declared_default():
    r"""Taking the default leaves the declared value, not an array.

    The fix skips the assignment rather than performing it, so this checks
    that skipping is right: the value installed by ``create_default`` is
    still there, and it is a scalar rather than the one-element array the
    old code would have written back.
    """
    run = run_trees.TRun()
    assert run.run_number == 0, 'default run_number was not installed'
    assert np.isscalar(run.run_number) or np.ndim(run.run_number) == 0, (
        'run_number came back as %r, not a scalar' % (run.run_number,))
