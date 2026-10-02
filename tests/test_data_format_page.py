# -*- coding: utf-8 -*-
r"""``docs/dev/make_data_format.py``: the field reference covers every branch of every tree."""

import importlib.util
import pathlib

from grand.basis.fielddoc import _fields

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_data_format",
                                               ROOT / "docs" / "dev" / "make_data_format.py")
mdf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdf)


def test_every_tree_and_field_is_on_the_page():
    text = mdf.render()
    trees = [cls for _, classes in mdf._trees() for cls in classes]
    assert len(trees) >= 13
    for cls in trees:
        assert "``%s``\n~~~" % cls.__name__ in text, cls.__name__
        for field, _, _ in _fields(cls):
            if field not in mdf.NOT_BRANCHES:
                assert "   * - ``%s``" % field in text, (cls.__name__, field)


def test_units_get_their_own_column():
    assert mdf._split("float32, in GeV") == ("float32", "GeV")
    assert mdf._split("uint32") == ("uint32", "")


def test_writing_twice_changes_nothing(tmp_path):
    target = tmp_path / "data_format.rst"
    assert mdf.write(target)
    assert not mdf.write(target)
