# -*- coding: utf-8 -*-
r"""Finding a file in the local directories.

``DataManager.get_file(name)`` is what granddb is for.  It looks through the
configured local directories first, then through remote repositories, and hands
back a path.  The local half is the half that can be tested without a server,
and it is also the half every lookup goes through first, so it runs far more
often than the remote code that has all the machinery.

The readme documents four behaviours here -- recursive search, restriction to a
named repository, restriction to a directory within one, and ``None`` when the
file is nowhere.  None of them was checked.  A fifth, the fallback when the
directory asked for does not exist, arrived with ``dev_database``'s last commit
in June 2025 and is pinned below because nothing else records that it is
deliberate rather than an accident of control flow.
"""
import textwrap

import pytest

from granddb.datamanager import DataManager


@pytest.fixture
def catalogue(tmp_path):
    r"""Returns a data manager over a small tree of files.

    The layout, all under one configured local directory::

        incoming/top.root
        incoming/sub/deeper/buried.root
        incoming/sub/twice.root
        incoming/sub/deeper/twice.root

    Returns
    -------
    tuple
        ``(manager, incoming_path)``.
    """
    incoming = tmp_path / "incoming"
    deeper = incoming / "sub" / "deeper"
    deeper.mkdir(parents=True)

    (incoming / "top.root").write_text("top")
    (deeper / "buried.root").write_text("buried")
    (incoming / "sub" / "twice.root").write_text("shallow copy")
    (deeper / "twice.root").write_text("deep copy")

    config = tmp_path / "config.ini"
    config.write_text(textwrap.dedent("""\
        [general]
        provider = "Testing"

        [directories]
        localdir = ["%s"]
        """ % incoming))

    return DataManager(str(config)), incoming


def test_a_file_in_the_configured_directory_is_found(catalogue):
    r"""The simple case: the file sits where the config file points."""
    manager, incoming = catalogue
    found = manager.get_file("top.root")
    assert found is not None, "top.root should be found in the configured localdir"
    assert str(incoming / "top.root") == str(found)


def test_the_search_descends_into_subdirectories(catalogue):
    r"""``localdir`` names a root, not a flat folder.

    The readme says the search covers "localdirs (and subdirs)", and the
    implementation uses ``rglob``.  Worth pinning because a reader could
    reasonably expect a single-level listing, and because narrowing this later
    would break lookups silently rather than loudly.
    """
    manager, incoming = catalogue
    found = manager.get_file("buried.root")
    assert found is not None, "a file two directories down should still be found"
    assert str(found).endswith("sub/deeper/buried.root")


def test_a_missing_file_gives_none_rather_than_raising(catalogue):
    r"""Absence is a return value, not an exception.

    Callers branch on it -- ``register_file`` and the scripts both do -- so this
    is part of the contract rather than an implementation detail.
    """
    manager, _ = catalogue
    assert manager.get_file("nothing-here.root") is None


def test_a_directory_can_be_named_to_restrict_the_search(catalogue):
    r"""Passing a path searches only there.

    The readme calls this a "one time shot" lookup for a file in a location the
    config file does not mention.
    """
    manager, incoming = catalogue
    deeper = incoming / "sub" / "deeper"
    found = manager.get_file("twice.root", "localdir", str(deeper))
    assert found is not None
    assert str(found).endswith("sub/deeper/twice.root"), (
        "the search should have been confined to %s, but found %s" % (deeper, found))


def test_a_directory_that_does_not_exist_falls_back_to_the_configured_one(catalogue):
    r"""Naming a missing directory is not fatal.

    ``dev_database``'s final commit, June 2025, changed this from a warning that
    left the caller with nothing into a fallback onto the path from the config
    file -- "Will use path defined in config.ini!".  That is a deliberate
    kindness rather than an accident, and nothing but this test says so.
    """
    manager, incoming = catalogue
    found = manager.get_file("top.root", "localdir", str(incoming / "no-such-dir"))
    assert found is not None, (
        "a non-existent search path should fall back to the configured localdir")
    assert str(found) == str(incoming / "top.root")
