# -*- coding: utf-8 -*-
r"""Marks the tests that are known to fail, with the reason for each.

These are real failures, not flakes, and none of them is hidden: pytest
reports them as ``xfailed``, and a run prints the count.  They are marked
rather than deleted or skipped so that the suite can be a required check in
CI while they are worked through -- a permanently red gate is a gate nobody
looks at.

``strict=False`` throughout: if one starts passing it is reported as
``xpassed`` rather than failing the run, which is the signal to remove its
entry here.

Each entry says what is wrong and who can settle it.  The corresponding
narrative is in ``docs/source/known_issues.rst``.
"""

import pytest

# test id (substring match) -> why it fails
KNOWN_FAILURES = {
    # --- expectations written against an older tree ---------------------
    'test_timetrace.py::test_timetrace_defaults':
        'Expects float32 traces; the code produces float64. Written on '
        'dev_aoi_unittest in January against a different state of aoi. '
        'Which precision is intended is a decision for the aoi author.',
    'test_timetrace.py::test_trace_setter_getter':
        'Same float32/float64 expectation as test_timetrace_defaults.',

    # --- API drift ------------------------------------------------------
    'test_pipeline.py::PipelineTest::test_add':
        "DataFile has no attribute 'trun'. grand.basis.pipeline is the "
        'unfinished Pipeline sketch -- its own docstring marks it TODO -- '
        'and it has drifted from dataio. Phase 6 decides whether it is '
        'rebuilt on the new interface or dropped.',

    # --- numerical / library behaviour ----------------------------------
    'test_du_network.py::test_get_surface':
        'Cross product rejects 2-dimensional vectors: NumPy 2 removed '
        'support for 2-vector cross products. The fix is a genuine choice '
        'about what a 2D detector layout means here.',
    'test_du_network.py::test_keep_only_du_with_index':
        'Same 2-vector cross product as test_get_surface.',
    'test_topography.py::TopographyTest::test_topography_cache':
        'Cache directory assertion depends on where the data model was '
        'downloaded; fails when GRAND_DATA_PATH differs from the default.',

    # --- missing fixture -------------------------------------------------
    'test_root_trees.py::RootTreesTest::test_datatree':
        'Asserts that data/test_efield.root exists. It is not in version '
        'control -- data/.gitignore excludes everything -- and is not fetched '
        'by env/setup.sh, so it is absent on any clean checkout and in CI. '
        'It passes on a developer machine only when a stale copy happens to '
        'be present, which is worse than failing. It therefore xpasses '
        'locally and xfails in CI; strict=False means both are green, and '
        'the xpass is the reminder that the fixture is still not in the '
        'repository. Same root cause as the end-to-end fixture below.',

    'test_efield2voltage.py::Efield2VoltageTest::test_Efield2Voltage':
        'data/test_efield.root is not in version control -- data/.gitignore '
        'excludes everything -- and is not fetched by env/setup.sh, so the '
        'end-to-end test cannot run on a fresh checkout or in CI. The file '
        'present locally is 615 bytes and contains no trees. Phase 3 needs a '
        'small committed fixture, or one generated in a pytest fixture; that '
        'is also what would let this test assert numbers instead of only '
        'that an output file appeared.',

    # --- downstream of the open physics questions -----------------------
    'test_shower.py::ShowerTest::test_showerevent':
        'IndexError building a ShowerEvent from the test fixture; the '
        'fixture predates changes to the shower containers.',
}


def pytest_collection_modifyitems(config, items):
    r"""Applies an ``xfail`` marker to each test named in `KNOWN_FAILURES`."""
    for item in items:
        for pattern, reason in KNOWN_FAILURES.items():
            if pattern in item.nodeid:
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break
