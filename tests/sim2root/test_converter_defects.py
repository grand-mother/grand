# -*- coding: utf-8 -*-
r"""Pins known defects in ``sim2root/`` without importing or modifying it.

``sim2root/`` converts ZHAireS and CoREAS output into the GRAND schema.  It is
outside the lint and test gates (see ``docs/source/sim2root.rst``), and this
file does not change that -- it records two defects so that they are visible in
a test run rather than only in a documentation page.

**Why static analysis rather than importing.**  The converters import ZHAireS
and CORSIKA helpers that are not installed, use star imports, and execute work
at module scope, so importing them in a test is neither possible nor desirable.
Most of what follows parses the source with :mod:`ast` instead, which needs
nothing but the file.

The exception is ``sim2root/CoREASRawRoot/CorsikaInfoFuncs.py``, which the
site-table tests at the end import directly.  It is a leaf module -- its whole
import list is ``re``, ``io``, ``numpy`` and ``datetime`` -- and it does
nothing at module scope, so it costs nothing to import and the defect is worth
demonstrating rather than inferring.

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

#: The CoREAS helper module holding the hard-coded site table.
CORSIKA_INFO = ROOT / 'sim2root' / 'CoREASRawRoot' / 'CorsikaInfoFuncs.py'

#: The only two sites the table knows.  Everything else raises.
KNOWN_SITES = ('Dunhuang', 'Lenghu')


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


# --------------------------------------------------------------------------
# The CoREAS site table
#
# `read_lat_long_alt` in sim2root/CoREASRawRoot/CorsikaInfoFuncs.py maps a
# site name to (latitude, longitude, altitude).  It has two problems, one
# live and one dormant, and they are next to each other in five lines of
# source.  See `issue-coreas-site-table` in the documentation.
# --------------------------------------------------------------------------

needs_corsika_info = pytest.mark.skipif(
    not CORSIKA_INFO.exists(),
    reason='sim2root/CoREASRawRoot/CorsikaInfoFuncs.py is not present')


def _read_lat_long_alt():
    r"""Returns the site-table function, imported from the converter.

    Returns
    -------
    callable
        ``read_lat_long_alt`` from
        ``sim2root/CoREASRawRoot/CorsikaInfoFuncs.py``.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location('corsika_info_funcs',
                                                  CORSIKA_INFO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_lat_long_alt


@needs_corsika_info
@pytest.mark.parametrize('site', KNOWN_SITES)
def test_the_site_table_answers_for_the_two_sites_it_knows(site):
    r"""Dunhuang and Lenghu return three numbers, as callers expect."""
    latitude, longitude, altitude = _read_lat_long_alt()(site)

    assert 30.0 < latitude < 45.0, 'latitude outside the Gobi, in degrees'
    assert 90.0 < longitude < 100.0, 'longitude outside the Gobi, in degrees'
    assert altitude > 0.0


@needs_corsika_info
def test_the_site_table_raises_for_every_other_site():
    r"""Any site the table does not know crashes on an empty unpacking.

    The ``else`` branch is ``latitude, longitude, altitude = []``, so an
    unrecognised site does not fall back, warn, or return ``None`` -- it
    raises ``ValueError: not enough values to unpack``, from inside a
    conversion run, with a message that names neither the site nor the table.

    Xiaodushan is the case that matters: it is a real GRAND site, the ZHAireS
    fixtures in this repository are simulations of it, and a CoREAS
    simulation of the same site cannot be converted.

    This is pinned rather than fixed because ``dev_io_root_testmerges`` is in
    flight over ``sim2root/``; see the module docstring. The fix is two lines
    -- raise something that names the site, or read the site list from a data
    file -- and belongs after that branch lands.
    """
    read_lat_long_alt = _read_lat_long_alt()

    with pytest.raises(ValueError) as raised:
        read_lat_long_alt('Xiaodushan')

    assert 'unpack' in str(raised.value), (
        'the failure mode changed; if the table now raises deliberately, '
        'this test and the known-issues entry are stale')
    assert 'Xiaodushan' not in str(raised.value), (
        'the error now names the site, which is the fix -- update this test')


@needs_corsika_info
def test_the_site_table_altitudes_are_in_centimetres():
    r"""The table's altitudes are 100x the site altitude in metres.

    Dunhuang is at 1142 m and the table says 114200; Lenghu is at 2800 m and
    the table says 280000. A comment on each line says ``# alt in cm``, so
    this is deliberate and matches CORSIKA's own units -- but the value is
    handed to ``RawShower.site_alt``, whose other producer (the ZHAireS
    reader) writes metres, and no unit is recorded anywhere in the schema.

    It is currently dormant. ``CoreasToRawROOT.py`` overwrites the altitude
    with the observation level in metres three lines after reading it, which
    it has done since ``0694fa9`` (2024-11-04, "save obs level as site
    altitude"). Before that the centimetre value reached the output, and it
    is still visible in the April 2024 fixture: ``run_1_L0_0000.root`` of the
    Dunhuang set carries ``origin_geoid = [40.14, 94.66, 114200]`` where the
    ZHAireS fixtures carry ``1264``.

    So the table is a landmine rather than a bug: deleting or reordering the
    override reintroduces a silent factor of 100 in an altitude. This test
    states the unit so that the next person to touch those lines has to
    notice it.
    """
    read_lat_long_alt = _read_lat_long_alt()

    for site, metres in (('Dunhuang', 1142.0), ('Lenghu', 2800.0)):
        _, _, altitude = read_lat_long_alt(site)
        assert altitude / metres == pytest.approx(100.0, abs=1.0), (
            '%s is at %.0f m and the table says %r, which is neither metres '
            'nor centimetres; the unit may have been changed without the '
            'callers being updated' % (site, metres, altitude))


@needs_corsika_info
def test_the_altitude_override_is_still_in_place():
    r"""``CoreasToRawROOT.py`` still overwrites the centimetre altitude.

    The line that keeps the test above dormant. If it goes, the site table's
    centimetres reach ``site_alt`` and then ``origin_geoid``, and every
    CoREAS conversion is wrong by 100x in the array origin -- silently,
    because nothing downstream checks the magnitude of an altitude.

    Read statically: the converter cannot be imported, and the property is
    about the source anyway.
    """
    converter = ROOT / 'sim2root' / 'CoREASRawRoot' / 'CoreasToRawROOT.py'
    if not converter.exists():
        pytest.skip('the CoREAS converter is not present')

    text = converter.read_text(encoding='utf-8', errors='replace')
    lines = [line.strip() for line in text.splitlines()]

    try:
        read_at = next(number for number, line in enumerate(lines)
                       if line.startswith('latitude, longitude, altitude ='
                                          ' read_lat_long_alt'))
        override_at = next(number for number, line in enumerate(lines)
                           if line.startswith('altitude = CorePosition[2]'))
    except StopIteration:
        pytest.fail('the site table is read or overridden differently now; '
                    'check whether the centimetre value still reaches '
                    'RawShower.site_alt')

    assert override_at > read_at, (
        'the override no longer follows the table read, so the centimetre '
        'altitude may now be what gets written')
    assert override_at - read_at < 10, (
        'the override has drifted %d lines from the read; anything between '
        'them that uses `altitude` is using centimetres'
        % (override_at - read_at))


# --------------------------------------------------------------------------
# `read_list_of_params` in sim2root/CoREASRawRoot/CorsikaInfoFuncs.py raised
# `UnboundLocalError` for any keyword that was not in the file.  Its result was
# held in a local named `list`, and assigning that name anywhere in the body
# makes it local throughout, so `return list` on the not-found path referred to
# a variable that had never been assigned rather than to the builtin.  The
# error named neither the keyword nor the file, and arrived from inside a
# conversion run.  Fixed September 2026; these tests keep it fixed.
# --------------------------------------------------------------------------

def _corsika_info():
    r"""Returns ``CorsikaInfoFuncs`` as an imported module.

    Returns
    -------
    module
        ``sim2root/CoREASRawRoot/CorsikaInfoFuncs.py``, loaded by path.  It is
        a leaf module that does nothing at import, so this is cheap.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location('corsika_info_funcs',
                                                  CORSIKA_INFO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inp_file(tmp_path, with_parallel):
    r"""Writes a minimal CORSIKA ``.inp``, with or without a PARALLEL card.

    THIN is written before THINH deliberately.  Lookup is a substring test, so
    ``"THIN" in "THINH ..."`` is true and a file listing THINH first would have
    ``read_list_of_params(f, "THIN")`` return the hadronic values.  That trap
    is not what these tests are about, but it is why the order here is fixed.
    """
    lines = [
        'RUNNR   100001',
        'ECUTS   0.05 0.05 0.001 0.001',
        'THIN    1.E-6 1.E2 0.',
        'THINH   1.E0 1.E2',
    ]
    if with_parallel:
        lines.append('PARALLEL 1000. 10000. 1 F')
    path = tmp_path / 'SIM100001.inp'
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


@needs_corsika_info
def test_an_absent_keyword_returns_none_rather_than_raising(tmp_path):
    r"""The regression itself: absence is a value, not an exception.

    ``is not list`` is checked as well as ``is None`` because the obvious wrong
    repair is to stop the ``UnboundLocalError`` by making the name refer to the
    builtin -- renaming the local, or initialising it to ``list``.  That would
    return the type object, and on Python 3.9 and later ``list[0]`` is a
    generic alias rather than an error, so the absence would reach the ROOT
    trees as ``list[0]`` instead of failing.  Trading a loud error for a silent
    wrong number is the one outcome worse than the original.
    """
    module = _corsika_info()
    values = module.read_list_of_params(_inp_file(tmp_path, False), 'PARALLEL')

    assert values is not list, (
        'an absent keyword now returns the builtin list; the caller will store '
        'list[0], which is a generic alias and not an error')
    assert values is None


@needs_corsika_info
def test_a_present_keyword_still_reads_its_values(tmp_path):
    r"""The fix must not have cost the ordinary case."""
    module = _corsika_info()
    values = module.read_list_of_params(_inp_file(tmp_path, True), 'PARALLEL')

    assert isinstance(values, list)
    assert [float(value) for value in values[:2]] == [1000.0, 10000.0]


@needs_corsika_info
def test_a_required_keyword_that_is_missing_names_itself_and_the_file(tmp_path):
    r"""``read_required_list_of_params`` fails where the absence happens.

    The point of the helper is the message: before it, a missing ECUTS reached
    ``ecuts[3]`` and raised somewhere else entirely, naming neither the
    keyword nor the file it was expected in.
    """
    module = _corsika_info()
    path = tmp_path / 'no-ecuts.inp'
    path.write_text('RUNNR   100001\n')

    with pytest.raises(ValueError) as raised:
        module.read_required_list_of_params(str(path), 'ECUTS')

    message = str(raised.value)
    assert 'ECUTS' in message
    assert 'no-ecuts.inp' in message


@needs_corsika_info
def test_the_coreas_converter_tolerates_a_missing_parallel_card():
    r"""Issue #147: a non-parallel CoREAS run writes no PARALLEL card.

    Read statically -- the converter cannot be imported.  The branch on
    ``parallel is None`` is what makes the absence survivable.

    ``147-add-option-to-read-in-non-parallel-coreas-sims-in-sim2root`` proposed
    a bare ``try/except`` around the subscript instead.  That does work -- the
    ``UnboundLocalError`` is an exception like any other -- but it catches
    every other failure too, including an unreadable file, and writes -1 for
    all of them; and it leaves ECUTS, THIN and THINH raising the same
    unhelpful error.
    """
    converter = ROOT / 'sim2root' / 'CoREASRawRoot' / 'CoreasToRawROOT.py'
    if not converter.exists():
        pytest.skip('the CoREAS converter is not present')

    text = converter.read_text(encoding='utf-8', errors='replace')

    assert 'if parallel is None:' in text, (
        'the missing-PARALLEL branch is gone; a non-parallel CoREAS run will '
        'subscript None')
    assert 'except' not in text.split('parallel = read_list_of_params')[1][:400], (
        'the PARALLEL handling is back to catching an exception that a missing '
        'keyword does not raise')


# --------------------------------------------------------------------------
# The antenna-list readers in CorsikaInfoFuncs assumed `np.genfromtxt` returns
# a table.  It returns a one-dimensional array for a single-row file, so a
# simulation with one antenna raised `IndexError: too many indices for array`
# from `file[:, 5]`, naming neither antennas nor the file.  Fixed September
# 2026 with `_read_antenna_list`; these tests keep the row axis in place.
# --------------------------------------------------------------------------

def _antenna_list(tmp_path, rows):
    r"""Writes a CoREAS ``.list`` with `rows` antennas.

    The format is ``AntennaPosition = x y z name``, with positions in
    centimetres, which is what the readers divide by 100.
    """
    path = tmp_path / 'SIM000001.list'
    path.write_text(''.join(
        'AntennaPosition = %.2f %.2f %.2f ant%d\n' % (-310078.36 + i * 100,
                                                      -700000.0, 120000.0, i)
        for i in range(rows)))
    return str(path)


@needs_corsika_info
@pytest.mark.parametrize('rows', [1, 2, 5])
def test_the_antenna_list_reads_at_any_length(tmp_path, rows):
    r"""One antenna is a list like any other.

    Parametrised from one upward because one is the case that used to raise:
    `np.genfromtxt` drops the row axis for a single-row file, and every reader
    in the module indexes ``[:, 5]``.
    """
    module = _corsika_info()
    path = _antenna_list(tmp_path, rows)

    info = module.antenna_positions_dict(path)
    assert len(info['name']) == rows
    assert list(info['name'])[0] == 'ant0'

    position = module.get_antenna_position(path, 'ant0')
    assert position is not None, 'ant0 is in the file it was read from'
    x, y, z = position
    assert x == pytest.approx(-3100.7836)   # centimetres to metres
    assert y == pytest.approx(-7000.0)
    assert z == pytest.approx(1200.0)


@needs_corsika_info
def test_a_single_antenna_no_longer_raises_index_error(tmp_path):
    r"""Pinned on its own, naming the error it used to raise.

    The parametrised test above would still pass if someone reintroduced the
    bug for the one-row case only and the fixture happened to have two rows.
    """
    module = _corsika_info()
    path = _antenna_list(tmp_path, 1)
    try:
        module.antenna_positions_dict(path)
    except IndexError as error:
        pytest.fail('a one-antenna list raises again: %s' % error)
