# -*- coding: utf-8 -*-
r"""The data manager reads its ini file the way its readme says it does.

:class:`granddb.datamanager.DataManager` is the entry point of the data
catalogue: given a file name it looks through local directories, then through
remote repositories, and hands back a path.  All of that is driven by an ini
file whose format ``granddb/readme.md`` documents in prose and nothing checked.

These tests exercise the parsing only.  Nothing here contacts a database or a
remote host: the readme states that the ``[database]`` and ``[registerer]``
sections are optional, and the tests below rely on that being true -- which is
itself worth pinning, because it is the difference between granddb being
testable and not.
"""
import textwrap

import pytest

from granddb.datamanager import DataManager


def _config(tmp_path, body):
    r"""Writes an ini file and returns its path as a string.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's per-test temporary directory.
    body : str
        Ini contents, dedented before writing.

    Returns
    -------
    str
        Path to the file, which is what ``DataManager`` expects.
    """
    path = tmp_path / "config.ini"
    path.write_text(textwrap.dedent(body))
    return str(path)


@pytest.fixture
def incoming(tmp_path):
    r"""Returns a directory to use as the incoming folder.

    ``DataManager`` requires the first entry of ``localdir`` to exist and be
    writable, so it cannot simply be a name.
    """
    path = tmp_path / "incoming"
    path.mkdir()
    return path


def test_it_builds_without_a_database_section(tmp_path, incoming):
    r"""The readme says the database section is optional.  It is.

    This is load-bearing for the rest of granddb's tests: if a data manager
    could not be built without PostgreSQL, nothing in this package could be
    tested without one either.
    """
    dm = DataManager(_config(tmp_path, """\
        [general]
        provider = "Testing"

        [directories]
        localdir = ["%s"]
        """ % incoming))

    assert dm.database() is None, "no [database] section should mean no database"
    assert dm.provider() == "Testing"


def test_the_first_local_directory_is_the_incoming_folder(tmp_path, incoming):
    r"""``localdir``'s first entry is where fetched files are put.

    The readme is explicit that the first path is special, and the ordering is
    otherwise invisible -- a reader would reasonably assume the list is a set.
    """
    second = tmp_path / "elsewhere"
    second.mkdir()
    dm = DataManager(_config(tmp_path, """\
        [general]
        provider = "Testing"

        [directories]
        localdir = ["%s", "%s"]
        """ % (incoming, second)))

    assert str(incoming) in dm.incoming(), (
        "the incoming folder should be the first localdir, not %s" % dm.incoming())


def test_declared_repositories_are_read_and_localdir_is_added(tmp_path, incoming):
    r"""Repositories come from the file, plus an implicit local one.

    ``localdir`` is not written in the ``[repositories]`` section but behaves as
    one, which is the sort of thing that is obvious once known and puzzling
    until then.
    """
    dm = DataManager(_config(tmp_path, """\
        [general]
        provider = "Testing"

        [directories]
        localdir = ["%s"]

        [repositories]
        CC = ["ssh","cca.in2p3.fr",22,["/sps/grand/"]]
        WEB = ["https","github.com",443,["/grand-mother/data/"]]
        """ % incoming))

    names = list(dm.repositories())
    assert "CC" in names and "WEB" in names, "declared repositories missing: %s" % names
    assert "localdir" in names, (
        "the local directories should appear as a repository; got %s" % names)


def test_credentials_cannot_carry_a_password(tmp_path, incoming):
    r"""A credentials entry is a login and a key file, and nothing else.

    ``granddb/readme.md``: "For security reasons you will not be allowed to
    provide sensitive information as password in this file."  That promise is
    kept by :class:`Credentials` taking only ``(name, user, keyfile)`` and
    setting its password to the empty string, so there is nowhere for one to
    come from -- it is asked for interactively instead.  Pinned here because a
    third positional argument would quietly turn the ini file into a place
    people keep passwords.
    """
    import inspect

    from granddb.datamanager import Credentials

    parameters = list(inspect.signature(Credentials.__init__).parameters)
    assert parameters == ["self", "name", "user", "keyfile"], (
        "Credentials takes %s; a password parameter would let the ini file "
        "carry one" % parameters)

    assert Credentials("CC", "login", "").password() == "", (
        "a freshly parsed credential should hold no password")


def test_two_managers_do_not_share_credentials(tmp_path):
    r"""Each data manager reads its own file and keeps its own credentials.

    ``DataManager`` is a plain class whose ``_credentials``, ``_directories``
    and ``_repositories`` were declared as class-level ``{}`` and ``[]``, which
    in Python means one object shared by every instance.  ``__init__`` then
    mutated them, so a second manager inherited the first one's logins,
    directories and repositories -- and the first gained the second's.

    Two managers built from unrelated config files must not see each other.
    """
    first = tmp_path / "one"
    second = tmp_path / "two"
    for path in (first, second):
        (path / "incoming").mkdir(parents=True)

    alpha = DataManager(_config(first, """\
        [general]
        provider = "Alpha"

        [directories]
        localdir = ["%s"]

        [credentials]
        ALPHA = ["alice", ""]
        """ % (first / "incoming")))

    beta = DataManager(_config(second, """\
        [general]
        provider = "Beta"

        [directories]
        localdir = ["%s"]

        [credentials]
        BETA = ["bob", ""]
        """ % (second / "incoming")))

    assert "BETA" not in alpha._credentials, (
        "the first manager picked up the second's credentials: %s"
        % sorted(alpha._credentials))
    assert "ALPHA" not in beta._credentials, (
        "the second manager picked up the first's credentials: %s"
        % sorted(beta._credentials))
    assert alpha._credentials is not beta._credentials, (
        "both managers share one credentials dictionary")
