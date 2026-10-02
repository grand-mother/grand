# -*- coding: utf-8 -*-
r"""#201: ``StdVectorList`` behaves like the list it claims to be.

``unsigned char`` elements came back as characters, ``+=`` replaced instead
of appending, numpy ``bool`` and ``uint64`` arrays could not be assigned and
the failed assignment emptied the field, and the ``MutableSequence`` methods
(negative indices, slices, ``del``, ``insert``, ``pop``) failed in C++ or
silently did nothing.
"""

import numpy as np
import pytest

from grand.dataio.descriptors import StdVectorList


def test_char_elements_are_numbers():
    assert list(StdVectorList("unsigned char", [1, 200, 255])) == [1, 200, 255]
    assert list(StdVectorList("char", [1, -3])) == [1, -3]
    nested = StdVectorList("vector<unsigned char>", [[1, 2], [200]])
    assert nested[1] == [200] and nested == [[1, 2], [200]]
    assert StdVectorList("unsigned char", [1, 200]).asnumpy().tolist() == [1, 200]


def test_plus_equals_appends():
    s = StdVectorList("float", [1.0, 2.0])
    s += [3.0]
    s += np.array([4.0])
    assert list(s) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("vec_type, value", [
    ("bool", np.array([False, False, True])),
    ("unsigned long long", np.array([1, 2, 2**63 + 5], dtype=np.uint64)),
])
def test_bool_and_uint64_arrays_assign(vec_type, value):
    s = StdVectorList(vec_type)
    s._assign(value)
    assert list(s) == value.tolist()


def test_a_failed_assignment_keeps_the_old_value():
    from grand.dataio import TADC
    t = TADC()
    t.gps_long = [1, 2, 3]
    with pytest.raises((TypeError, ValueError)):
        t.gps_long = ["a"]
    assert list(t.gps_long) == [1, 2, 3]


def test_sequence_methods():
    s = StdVectorList("float", [1.0, 2.0, 3.0])
    assert s[-1] == 3.0 and s[1:] == [2.0, 3.0]
    s[-1] = 9.0
    assert list(s) == [1.0, 2.0, 9.0]
    with pytest.raises(IndexError):
        s[10] = 5.0
    with pytest.raises(IndexError):
        s[-4]
    del s[0]
    s.insert(0, 5.0)
    assert s.pop() == 9.0
    s[0:1] = [7.0, 8.0]
    s.remove(8.0)
    assert list(s) == [7.0, 2.0]
    assert StdVectorList("float")[:2] == []


def test_nested_equality():
    assert StdVectorList("vector<float>", [[1.0, 2.0], [3.0]]) == [[1.0, 2.0], [3.0]]
    assert StdVectorList("vector<float>", [[1.0, 2.0], [3.0]]) != [[1.0], [2.0, 3.0]]
    three = StdVectorList("vector<vector<float>>", [[[1.0, 2.0]], [[3.0]]])
    assert three == [[[1.0, 2.0]], [[3.0]]]
    assert three != [[1.0, 2.0], [3.0]]
