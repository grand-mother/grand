"""StdVectorList's fallback chain reports what failed (#256)."""
import pytest

from grand.dataio.descriptors import StdVectorList


class _Refusing:
    r"""Stands for a ROOT vector that refuses every way of filling it."""

    def __init__(self, error):
        self.error = error

    def assign(self, *args):
        raise self.error("refused")

    def __iadd__(self, other):
        raise self.error("refused")

    def clear(self):
        pass

    def swap(self, other):
        pass

    def size(self):
        return 0

    def __len__(self):
        return 0


def test_an_overflow_is_raised_not_dropped():
    v = StdVectorList("int")
    v._vector = _Refusing(OverflowError)
    with pytest.raises(OverflowError, match="does not fit in a vector<int>"):
        v._append_raw([1, 2])


def test_the_last_fallback_names_the_first_error():
    v = StdVectorList("float")
    v._vector = _Refusing(TypeError)
    with pytest.raises(TypeError, match=r"cannot store \['a'\] in a vector<float>"):
        v._append_raw(["a"])
