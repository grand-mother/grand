# -*- coding: utf-8 -*-
r"""Pins known defects in ``sim2root/`` without importing or modifying it.

``sim2root/`` converts ZHAireS and CoREAS output into the GRAND schema.  It is
outside the lint and test gates (see ``docs/source/sim2root.rst``), and this
file does not change that -- it records two defects so that they are visible in
a test run rather than only in a documentation page.

**Why static analysis rather than importing.**  The converters import ZHAireS
and CORSIKA helpers that are not installed, use star imports, and execute work
at module scope, so importing them in a test is neither possible nor desirable.
Everything below parses the source with :mod:`ast` instead, which needs nothing
but the file.

**Why the defect is not simply fixed.**  ``dev_io_root_testmerges`` modifies
the very block these tests describe -- its diff against this branch has hunks
at lines 732, 737, 741 and 749 -- and it *adds* further calls to the same
undefined names, carrying 54 where this branch has 49.  Repairing the block
here would conflict with that branch and, if it landed afterwards, would
quietly reintroduce what had been removed.  The order has to be the other way
round: land the sim2root branches, then fix.  These tests are what makes the
defect impossible to lose track of in the meantime.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONVERTER = ROOT / 'sim2root' / 'ZHAireSRawRoot' / 'ZHAireSRawToRawROOT.py'

#: Names called inside the longitudinal-tables block that are defined nowhere
#: in the file.  They are leftovers from an HDF5-based predecessor.
ORPHANS = ('SimShower', 'HDF5handle')


needs_converter = pytest.mark.skipif(
    not CONVERTER.exists(),
    reason='sim2root/ZHAireSRawRoot/ZHAireSRawToRawROOT.py is not present')


def _tree():
    r"""Returns the parsed converter module.

    Returns
    -------
    ast.Module
        The syntax tree.
    """
    return ast.parse(CONVERTER.read_text(encoding='utf-8', errors='replace'))


def _bound_names(tree):
    r"""Returns every name the module binds anywhere.

    Deliberately generous: imports, assignments, function and class
    definitions, comprehension targets, ``for`` targets, ``with`` targets and
    arguments all count.  A name still missing after that is genuinely
    undefined rather than merely bound somewhere awkward.

    Parameters
    ----------
    tree : ast.Module
        The parsed module.

    Returns
    -------
    set of str
        The bound names.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.alias):
            names.add((node.asname or node.name).split('.')[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


@needs_converter
@pytest.mark.parametrize('orphan', ORPHANS)
def test_longitudinal_block_calls_undefined_names(orphan):
    r"""``SimShower`` and ``HDF5handle`` are used but never bound.

    Asserted as present, not absent: this is a record of a defect that has not
    been repaired, and the test is meant to *fail* on the day it is, which is
    the signal to delete this file and drop the caveat from
    ``docs/source/sim2root.rst``.
    """
    tree = _tree()
    used = {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    assert orphan in used, (
        '%s is no longer used in %s; the longitudinal block may have been '
        'repaired or removed, so this test and the sim2root documentation are '
        'stale' % (orphan, CONVERTER.name))
    assert orphan not in _bound_names(tree), (
        '%s is now bound somewhere in %s -- the defect appears to be fixed, so '
        'delete this test and the caveat in docs/source/sim2root.rst'
        % (orphan, CONVERTER.name))


@needs_converter
def test_the_block_is_unreachable_as_written():
    r"""The block is guarded by a flag that is hard-coded false.

    This is what keeps the undefined names above from being a crash rather than
    a latent one.  ``NLongitudinal`` was a parameter of the function; the
    parameter is commented out of the signature and the name is assigned
    ``False`` at module scope of the function body instead.

    If this assertion fails while the one above still passes, the block has
    become reachable and the converter will raise ``NameError`` on its first
    call.
    """
    tree = _tree()
    literals = [node.value.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id == 'NLongitudinal'
                        for t in node.targets)]
    assert literals, 'NLongitudinal is no longer assigned a literal'
    assert all(value is False for value in literals), (
        'NLongitudinal is no longer hard-coded False (%s); the block calling '
        '%s is now reachable and will raise NameError'
        % (literals, ' and '.join(ORPHANS)))


@needs_converter
def test_the_converter_still_parses():
    r"""The file is syntactically valid Python.

    Trivial, and the only check in this repository that covers it: nothing
    imports these modules, the linter does not run on them, and there is no
    other test.  A syntax error here would otherwise reach whoever next tried
    to convert a shower.
    """
    _tree()          # raises SyntaxError if not
