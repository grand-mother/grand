# -*- coding: utf-8 -*-
r"""Every tree GRANDlib writes says which code wrote it (issue #137).

``modification_software`` and ``modification_software_version`` existed in
the metadata of every tree, but were always written empty.  They now hold
"GRANDlib" and the package version with the git branch and commit it ran
from, so a simulation file can be traced to the code that produced it.
"""

import shutil
import subprocess

import pytest

from grand import provenance

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _repo(path):
    r"""Makes a git repository at ``path`` with one committed file."""
    def git(*args):
        subprocess.run(["git", "-C", str(path)] + list(args), check=True,
                       capture_output=True)
    path.mkdir(parents=True, exist_ok=True)
    git("init", "-q", "-b", "main")
    (path / "a.txt").write_text("one\n")
    git("add", "a.txt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "one")
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


@needs_git
def test_a_checkout_is_described_by_branch_commit_and_changes(tmp_path):
    r"""The branch and full commit, and whether tracked files were changed."""
    commit = _repo(tmp_path / "repo")
    assert provenance.git_state(str(tmp_path / "repo")) == dict(
        commit=commit, branch="main", modified=False)
    assert provenance.describe(str(tmp_path / "repo")) == (
        "%s (git main %s)" % (provenance.package_version(), commit))

    (tmp_path / "repo" / "a.txt").write_text("two\n")
    assert provenance.describe(str(tmp_path / "repo")).endswith(", modified)")


@needs_git
def test_outside_a_checkout_only_the_version_is_given(tmp_path):
    r"""No commit is claimed for a copy that is not a checkout's top.

    A copy installed inside some other repository must not report that
    repository's commit as its own.
    """
    _repo(tmp_path / "other")
    inside = tmp_path / "other" / "site-packages"
    inside.mkdir()
    assert provenance.git_state(str(inside)) is None
    assert provenance.describe(str(inside)) == provenance.package_version()
    assert provenance.git_state(str(tmp_path / "nothing")) is None


def test_a_new_tree_records_the_code_that_wrote_it(tmp_path):
    r"""The metadata is written into the file and reads back.

    A tool that derives a tree may still set its own, as before.
    """
    from grand.dataio import TRun

    tree = TRun()
    tree.run_number = 1
    tree.fill()
    tree.write(str(tmp_path / "run.root"))
    back = TRun(str(tmp_path / "run.root"))
    assert back.modification_software == "GRANDlib"
    assert back.modification_software_version == provenance.current()
    assert back.modification_software_version.startswith(provenance.package_version())

    tree = TRun()
    tree.modification_software = "extract_events.py"
    assert tree.modification_software == "extract_events.py"
