# -*- coding: utf-8 -*-
r"""Every field of every tree survives a write and a read unchanged.

``test_schema_snapshot.py`` pins *which* fields exist.  This pins that they
work.  The two fail for different reasons: the snapshot catches a field being
added or renamed in the source, and this catches a field whose declared C++
type cannot carry the values the Python side puts in it -- a ``float`` branch
declared ``int``, a nested vector flattened on the way out, a value silently
truncated by its width.

Nothing checked that before.  ``test_integration.py`` round-trips four fields
of three classes; the data model has 275 branch fields across eleven classes,
so 271 of them were written by code no test had ever read back.

The test is generic on purpose.  It reads each field's C++ type out of its
descriptor, makes up a value of that type, writes, closes the file, reopens
it, and compares.  A field added tomorrow is covered the day it is added,
without anyone remembering to extend this file -- which is the only way a
coverage claim over 275 fields stays true.
"""

import dataclasses

import numpy as np
import pytest

import grand.dataio.event_trees as event_trees
import grand.dataio.run_trees as run_trees
from grand.dataio.descriptors import (StdStringDesc, StdVectorListDesc,
                                      TTreeArrayDesc, TTreeScalarDesc)

#: The tree classes whose contents are a data contract.  Same list, and same
#: order, as the schema snapshot.
TREE_CLASSES = ['TRun', 'TRunVoltage', 'TADC', 'TRawVoltage', 'TVoltage',
                'TEfield', 'TShower', 'TShowerSim', 'TRunEfieldSim',
                'TRunShowerSim', 'TRunNoise']

#: C++ integer types, which take integer test values.  Everything else that
#: is not a string or a bool is treated as floating point.
INTEGER_TYPES = ('int', 'unsigned int', 'short', 'unsigned short',
                 'long', 'long long', 'unsigned long long',
                 'char', 'unsigned char')

#: Declared on the base classes rather than on each tree, so they do not
#: appear in ``dataclasses.fields`` of the subclass but do appear as branches.
INHERITED_BRANCHES = {'run_number', 'event_number'}

#: Fields whose C++ type is ``unsigned char``.  PyROOT hands these back as
#: one-character strings rather than as numbers; see the dedicated test at the
#: bottom of this file for what that means and why it is left alone.
CHARACTER_TYPES = ('unsigned char', 'char')


def _class_named(name):
    r"""Returns the tree class of that name, from whichever module defines it.

    Parameters
    ----------
    name : str
        Class name, for example ``'TADC'``.

    Returns
    -------
    type or None
        The class, or None if the package does not define it.
    """
    return getattr(event_trees, name, None) or getattr(run_trees, name, None)


def _branch_fields(cls):
    r"""Returns ``{field name: (kind, cpp_or_dtype, extra)}`` for one class.

    The descriptors keep their element type inside a closure -- ``factory``
    -- rather than as an attribute, so the type is recovered by calling the
    factory once and looking at what comes out.  That is why this reads a
    little indirectly.

    Parameters
    ----------
    cls : type
        A tree dataclass.

    Returns
    -------
    dict
        ``kind`` is one of ``'vector'``, ``'scalar'``, ``'array'`` or
        ``'string'``.  For vectors the second element is the C++ element
        type; for scalars and arrays it is the NumPy dtype; for arrays the
        third element is the shape.  Fields that are not branches -- plain
        dataclass attributes such as ``is_tchain`` -- are left out.
    """
    found = {}
    for field in dataclasses.fields(cls):
        if field.name.startswith('_'):
            continue
        descriptor = cls.__dict__.get(field.name)
        if isinstance(descriptor, StdVectorListDesc):
            prototype = descriptor.factory()
            found[field.name] = ('vector', prototype.vec_type, None)
        elif isinstance(descriptor, TTreeScalarDesc):
            prototype = descriptor.factory()
            found[field.name] = ('scalar', prototype.dtype, None)
        elif isinstance(descriptor, TTreeArrayDesc):
            prototype = descriptor.factory()
            found[field.name] = ('array', prototype.dtype, prototype.shape)
        elif isinstance(descriptor, StdStringDesc):
            found[field.name] = ('string', None, None)
    return found


def _value_for(kind, element_type, shape):
    r"""Returns a distinctive value of the right type for one field.

    Distinctive matters: the values must not be the defaults, or a field that
    is never written would round-trip perfectly by doing nothing.  Every
    number here is non-zero, and the floating-point ones end in ``.5`` so
    they are exact in binary and a float-to-int truncation shows up as a
    changed value rather than as rounding noise.

    Parameters
    ----------
    kind : str
        As returned by `_branch_fields`.
    element_type : str or type
        C++ element type for vectors, NumPy dtype for scalars and arrays.
    shape : tuple or None
        Shape, for arrays.

    Returns
    -------
    object
        A value the field's setter accepts.
    """
    if kind == 'string':
        return 'roundtrip'

    if kind == 'scalar':
        dtype = np.dtype(element_type)
        return dtype.type(7) if dtype.kind in 'iu' else dtype.type(12.5)

    if kind == 'array':
        dtype = np.dtype(element_type)
        values = np.arange(1, int(np.prod(shape)) + 1, dtype=dtype)
        if dtype.kind == 'f':
            values = values + 0.5
        return values.reshape(shape)

    # Vectors.  A tree holds one entry per event and each vector holds one
    # element per detector unit, so two elements stand in for two units.
    inner = element_type
    if inner.startswith('vector<'):
        inner = inner[len('vector<'):-1]
        if inner.startswith('vector<'):            # vector<vector<vector<T>>>
            innermost = inner[len('vector<'):-1]
            if innermost == 'bool':
                return [[[True, False]]]
            if innermost in INTEGER_TYPES:
                return [[[1, 2], [3, 4]]]
            return [[[1.5, 2.5], [3.5, 4.5]]]
        if inner == 'bool':
            return [[True, False]]
        if inner == 'string':
            return [['a', 'b']]
        if inner in INTEGER_TYPES:
            return [[1, 2], [3, 4]]
        return [[1.5, 2.5], [3.5, 4.5]]

    if inner == 'bool':
        return [True, False]
    if inner == 'string':
        return ['alpha', 'beta']
    if inner in INTEGER_TYPES:
        return [3, 5]
    return [1.5, 2.5]


def _flatten(value):
    r"""Returns the scalars inside an arbitrarily nested sequence, in order.

    Comparing nested vectors elementwise is otherwise a pile of special
    cases, and the shape is checked anyway by comparing lengths.

    Parameters
    ----------
    value : object
        A scalar, or any nesting of sequences of them.

    Returns
    -------
    list
    """
    if hasattr(value, '__len__') and not isinstance(value, str):
        flat = []
        for element in value:
            flat.extend(_flatten(element))
        return flat
    return [value]


def _nesting(value):
    r"""Returns the nested lengths of a sequence, or ``()`` for a scalar.

    ``_flatten`` deliberately loses the shape, so on its own it cannot tell
    ``[[1.5], [2.5]]`` from ``[[1.5, 2.5]]`` -- and a nested vector arriving
    flattened is one of the failures most worth catching, because the values
    all survive and only the grouping by detector unit is gone.

    Parameters
    ----------
    value : object
        A scalar, or any nesting of sequences of them.

    Returns
    -------
    tuple
        ``(length, shape of element 0, shape of element 1, ...)``, recursively.
    """
    if isinstance(value, str) or not hasattr(value, '__len__'):
        return ()
    return (len(value),) + tuple(_nesting(element) for element in value)


def _matches(read_back, written, element_type):
    r"""Returns True if the value read back is the value that was written.

    Parameters
    ----------
    read_back, written : object
        Values to compare, nested to any depth.
    element_type : str or type
        The field's declared type, needed only for the ``unsigned char``
        case described below.

    Returns
    -------
    bool
    """
    got, want = _flatten(read_back), _flatten(written)
    if len(got) != len(want):
        return False

    # Compared only when both sides are sequences.  A scalar field can come
    # back as a one-element array rather than as a bare number, which is a
    # presentation difference and not a change to the data, so requiring the
    # nesting to agree there would fail for a reason nobody cares about.
    both_are_sequences = all(hasattr(side, '__len__') and not isinstance(side, str)
                             for side in (read_back, written))
    if both_are_sequences and _nesting(read_back) != _nesting(written):
        return False

    for one_got, one_want in zip(got, want):
        # PyROOT maps `unsigned char` to a one-character Python string, so a
        # field written as the number 3 comes back as '\x03'.  The number is
        # intact -- ord() recovers it -- and only the type changed, so this
        # is decoded rather than treated as a failure.  The test at the
        # bottom of the file is what pins the behaviour itself.
        if isinstance(one_got, str) and not isinstance(one_want, str):
            one_got = ord(one_got)
        if isinstance(one_want, (str, bool, np.bool_)):
            if one_got != one_want:
                return False
        elif abs(float(one_got) - float(one_want)) > 1e-4:
            return False
    return True


def _round_trip(cls, path):
    r"""Writes one filled entry, reads it back, and returns both sides.

    Parameters
    ----------
    cls : type
        Tree class to exercise.
    path : str
        Where to put the ROOT file.

    Returns
    -------
    tuple
        ``(written, reopened, fields)``: the dict of values that went in, a
        fresh tree object positioned on entry 0, and the field description.
    """
    fields = _branch_fields(cls)
    tree = cls(_file_name=str(path))

    written = {}
    for name, (kind, element_type, shape) in fields.items():
        value = _value_for(kind, element_type, shape)
        setattr(tree, name, value)
        written[name] = value

    tree.fill()
    tree.write()
    tree.close_file()

    reopened = cls(_file_name=str(path))
    reopened.get_entry(0)
    return written, reopened, fields


@pytest.mark.parametrize('class_name', TREE_CLASSES)
def test_every_field_survives_a_write_and_a_read(class_name, tmp_path):
    r"""Each field comes back holding what was put into it.

    The core of the file.  Failure names the field, its declared type, what
    went in and what came out, because that quartet is usually the whole
    diagnosis.
    """
    cls = _class_named(class_name)
    assert cls is not None, '%s is no longer defined' % class_name

    written, reopened, fields = _round_trip(
        cls, tmp_path / ('%s.root' % class_name))

    damaged = []
    for name, value in written.items():
        element_type = fields[name][1]
        try:
            read_back = getattr(reopened, name)
        except Exception as error:                      # noqa: BLE001
            damaged.append('%s (%s): reading raised %s'
                           % (name, element_type, type(error).__name__))
            continue
        if not _matches(read_back, value, element_type):
            damaged.append('%s (%s): wrote %r, read %r'
                           % (name, element_type, value, list(read_back)
                              if hasattr(read_back, '__len__') else read_back))

    assert not damaged, ('%d of %d fields of %s did not survive the round '
                         'trip:\n  %s' % (len(damaged), len(written),
                                          class_name, '\n  '.join(damaged)))


@pytest.mark.parametrize('class_name', TREE_CLASSES)
def test_every_declared_field_becomes_a_branch(class_name, tmp_path):
    r"""Nothing declared in Python is missing from the file on disk.

    A field can be declared, assigned and read back within one process
    entirely in Python memory while never reaching the file.  The round-trip
    test above would catch that only if the read came from a genuinely fresh
    object, which it does -- but this states the property directly, against
    the branch list ROOT itself reports.
    """
    cls = _class_named(class_name)
    declared = set(_branch_fields(cls))

    tree = cls(_file_name=str(tmp_path / ('%s.root' % class_name)))
    tree.run_number = 1
    if hasattr(tree, 'event_number'):
        tree.event_number = 1
    tree.fill()
    tree.write()
    tree.close_file()

    reopened = cls(_file_name=str(tmp_path / ('%s.root' % class_name)))
    branches = {branch.GetName()
                for branch in reopened._tree.GetListOfBranches()}

    missing = sorted(declared - branches)
    assert not missing, ('%s declares %s but the file has no such branch'
                         % (class_name, missing))

    unexplained = sorted(branches - declared - INHERITED_BRANCHES)
    assert not unexplained, ('%s writes branches nothing declares: %s'
                             % (class_name, unexplained))


def test_the_comparison_would_notice_a_changed_value():
    r"""The guard on the two tests above: `_matches` can actually fail.

    A comparison helper that returned True unconditionally -- through a bare
    ``except``, a length mismatch swallowed, a tolerance far too loose --
    would leave 275 fields looking checked and checked by nothing.  These are
    the cases it must reject.
    """
    assert _matches([1.5, 2.5], [1.5, 2.5], 'float')
    assert not _matches([1.5, 2.5], [1.5, 9.5], 'float'), 'changed value'
    assert not _matches([1.5], [1.5, 2.5], 'float'), 'dropped element'
    assert not _matches([[1.5, 2.5]], [[1.5], [2.5]], 'vector<float>'), (
        'flattened nesting')
    assert not _matches([1.0, 2.0], [1.5, 2.5], 'float'), 'truncated to int'
    assert not _matches(['a'], ['b'], 'string'), 'changed string'


def test_unsigned_char_fields_come_back_as_characters(tmp_path):
    r"""Fourteen fields change Python type across a write, and keep their value.

    ``unsigned char`` is what the data model uses for small counters and mode
    flags, and PyROOT presents ``std::vector<unsigned char>`` as characters.
    So ``tadc.qmax_ch`` is a list of ints before the file is written and a
    list of one-character strings after it is read, and ``sum()`` over it
    works in the first case and raises ``TypeError`` in the second.

    The value is not lost -- ``ord`` recovers it exactly -- so this is a trap
    rather than a defect, and changing the declared type to ``unsigned
    short`` to avoid it would change the on-disk format for every file the
    collaboration has already written.  It is therefore pinned here rather
    than fixed: the test documents the behaviour and will fail if a future
    ROOT or a schema change alters it, which is the moment to decide.
    """
    tadc = _class_named('TADC')(_file_name=str(tmp_path / 'chars.root'))
    tadc.run_number, tadc.event_number = 1, 1
    tadc.qmax_ch = [[1, 2], [3, 4]]
    tadc.test_pulse_rate_divider = [3, 5]
    tadc.fill()
    tadc.write()
    tadc.close_file()

    reopened = _class_named('TADC')(_file_name=str(tmp_path / 'chars.root'))
    reopened.get_entry(0)

    divider = list(reopened.test_pulse_rate_divider)
    assert all(isinstance(element, str) for element in divider), (
        'unsigned char no longer reads back as characters; if PyROOT now '
        'returns integers the workaround in _matches can go')
    assert [ord(element) for element in divider] == [3, 5], (
        'the numeric value did not survive, which would be a real defect'
        ' rather than a presentation quirk')

    qmax = [[ord(element) for element in row] for row in reopened.qmax_ch]
    assert qmax == [[1, 2], [3, 4]], 'nested unsigned char lost its values'


def test_the_round_trip_is_not_vacuous(tmp_path):
    r"""A fresh tree does not already hold the values, before `get_entry`.

    The round-trip tests compare against a newly constructed object, so they
    would pass trivially if construction somehow produced the written values
    -- or if the test values happened to be the defaults.  This states the
    opposite directly: opening the file gives defaults, and only reading
    entry 0 gives back what was written.
    """
    path = tmp_path / 'vacuity.root'
    efield = _class_named('TEfield')(_file_name=str(path))
    efield.run_number, efield.event_number = 1, 1
    efield.du_id = [3, 5]
    efield.fill()
    efield.write()
    efield.close_file()

    before = _class_named('TEfield')(_file_name=str(path))
    assert list(before.du_id) == [], (
        'a freshly opened tree already holds data, so the round-trip tests '
        'may be comparing a value against itself')

    before.get_entry(0)
    assert list(before.du_id) == [3, 5], 'get_entry did not load the entry'
