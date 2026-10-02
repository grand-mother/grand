# -*- coding: utf-8 -*-
r"""#261, item 1: the tree fields beta testers could not interpret state their unit.

Field docstrings are attribute docstrings, which ``help()`` does not show, so
this reads them from the source, as Sphinx does.
"""

import ast
import pathlib

import pytest

DATAIO = pathlib.Path(__file__).resolve().parents[2] / "grand" / "dataio"


def _field_docs(module):
    docs = {}
    tree = ast.parse((DATAIO / module).read_text())
    for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        body = cls.body
        for field, doc in zip(body, body[1:]):
            if isinstance(field, ast.AnnAssign) and isinstance(doc, ast.Expr) \
                    and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
                docs["%s.%s" % (cls.name, field.target.id)] = doc.value.value
    return docs


@pytest.mark.parametrize("module, field, unit", [
    ("event_trees.py", "TShower.azimuth", "degrees"),
    ("event_trees.py", "TShower.magnetic_field", "µT"),
    ("event_trees.py", "TShower.core_time_ns", "Nanoseconds"),
    ("event_trees.py", "TVoltage.trace", "µV"),
    ("event_trees.py", "TEfield.trace", "µV/m"),
    ("event_trees.py", "TEfield.time_max", "ns"),
    ("run_trees.py", "TRun.origin_geoid", "degrees"),
    ("run_trees.py", "TRun.du_geoid", "meters"),
    ("run_trees.py", "TRun.first_event_time", "Unix seconds"),
    ("run_trees.py", "TRunEfieldSim.t_post", "ns"),
    ("run_trees.py", "TRunNoise.gal_noise_LST", "hours"),
    ("run_trees.py", "TRunNoise.gal_noise_sigma", "µV"),
])
def test_the_unit_is_stated(module, field, unit):
    assert unit in _field_docs(module)[field]
