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

#: The abandoned copy of the ZHAireS tooling.  See the tests at the end of this
#: file and ``issue-src-outlib-conflict`` in the documentation.
OUTLIB = ROOT / 'src_outlib'
LIVE_AIRES = ROOT / 'sim2root' / 'ZHAireSRawRoot' / 'AiresInfoFunctionsGRANDROOT.py'
STALE_AIRES = OUTLIB / 'AiresInfoFunctionsGRANDROOT.py'
BROKEN = OUTLIB / 'ZHAireSRawToGRANDROOT.py'

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


# --------------------------------------------------------------------------
# src_outlib/: abandoned, and one file has not parsed since 2023
# --------------------------------------------------------------------------

needs_outlib = pytest.mark.skipif(not OUTLIB.is_dir(),
                                  reason='src_outlib/ is not present')


@needs_outlib
def test_src_outlib_still_carries_a_2023_merge_conflict():
    r"""``ZHAireSRawToGRANDROOT.py`` contains committed conflict markers.

    Landed on 2023-06-30 in "Merging master into this branch" and never
    resolved, so the file has not been valid Python since.  Nothing imports it,
    it is not packaged, and the linter does not cover ``src_outlib/``, so there
    is no path by which the syntax error reaches anyone -- which is why it
    survived.

    Asserted as **present**: this records a defect that has deliberately not
    been repaired, because four branches still touch ``src_outlib/`` and
    deleting it would turn each of their merges into a delete/modify conflict.
    The test is meant to fail on the day it is cleaned up, in Phase 10, which
    is the signal to drop it and the matching entry in ``known_issues.rst``.
    """
    if not BROKEN.exists():
        pytest.skip('the file has been removed, which is the intended fix')

    text = BROKEN.read_text(encoding='utf-8', errors='replace')
    markers = [line for line in text.split('\n')
               if line.startswith('<<<<<<<') or line.startswith('>>>>>>>')]
    assert markers, (
        '%s no longer carries conflict markers -- delete this test and the '
        'known-issues entry that cites it' % BROKEN.name)

    with pytest.raises(SyntaxError):
        ast.parse(text)


@needs_outlib
def test_the_two_aires_readers_have_diverged():
    r"""``src_outlib`` holds a stale copy of the sim2root ZHAireS reader.

    Both files are called ``AiresInfoFunctionsGRANDROOT.py``.  The sim2root one
    is live and larger; the ``src_outlib`` one is missing a series of
    ``Get*FromSry`` functions.  Editing the wrong one is an easy mistake and
    fails silently, because nothing imports the stale copy.

    Checks that the live copy is a superset by function name, which is the
    property that makes "use the sim2root one" the right advice.
    """
    if not (STALE_AIRES.exists() and LIVE_AIRES.exists()):
        pytest.skip('one of the two copies is gone, which is the intended fix')

    def functions(path):
        tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
        return {node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)}

    live, stale = functions(LIVE_AIRES), functions(STALE_AIRES)
    only_in_stale = stale - live

    assert live - stale, (
        'the two copies now define the same functions; they may have been '
        'reconciled, in which case this test and the known-issues entry are '
        'stale')
    assert not only_in_stale, (
        'src_outlib defines functions the live sim2root copy does not: %s. '
        'The advice to treat src_outlib as abandoned no longer holds -- it '
        'carries something unique that needs salvaging first.'
        % ', '.join(sorted(only_in_stale)))
