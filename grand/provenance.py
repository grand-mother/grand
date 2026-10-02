# -*- coding: utf-8 -*-
r"""Which GRANDlib produced a file: the package version and its git commit.

Every tree GRANDlib creates records this in its ``modification_software`` and
``modification_software_version`` metadata, so a simulated or
converted file says which code wrote it.  The package version alone is not
enough: it moves only on release, and most files are written from a git
checkout between releases.
"""

import functools
import importlib.metadata
import os
import subprocess

#: What ``modification_software`` records for trees GRANDlib creates.
SOFTWARE_NAME = "GRANDlib"


def package_version():
    r"""Returns the installed GRANDlib version, or ``"unknown"``.

    Read from the package metadata rather than a constant, so it cannot drift
    from what was installed.  ``"unknown"`` rather than an error when the
    package is not installed: running from a source tree is legitimate.

    Returns
    -------
    str
    """
    try:
        return importlib.metadata.version("grand")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git(root, *args):
    r"""Runs git in ``root``; returns its stripped output, or None on failure."""
    try:
        done = subprocess.run(["git", "-C", root] + list(args), capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_state(root=None):
    r"""Describes the git checkout GRANDlib runs from, if it runs from one.

    Parameters
    ----------
    root : str, optional
        The repository to describe.  Defaults to the one holding this package.

    Returns
    -------
    dict or None
        ``commit`` (full hash), ``branch`` (``"HEAD"`` when detached) and
        ``modified`` (True when tracked files differ from the commit).  None
        when ``root`` is not the top of a git checkout -- for example an
        installed copy -- so that a copy installed inside some other
        repository never reports that repository's commit.
    """
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or os.path.realpath(top) != os.path.realpath(root):
        return None
    commit = _git(root, "rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return dict(commit=commit,
                branch=_git(root, "rev-parse", "--abbrev-ref", "HEAD") or "HEAD",
                modified=bool(status))


def describe(root=None):
    r"""Returns the version string written into files.

    Parameters
    ----------
    root : str, optional
        Passed to :func:`git_state`.

    Returns
    -------
    str
        For example ``"0.2.0 (git dev-next 1aeac037153a..., modified)"``, or
        the package version alone outside a git checkout.
    """
    version = package_version()
    state = git_state(root)
    if state is None:
        return version
    return "%s (git %s %s%s)" % (version, state["branch"], state["commit"],
                                 ", modified" if state["modified"] else "")


@functools.lru_cache(maxsize=1)
def current():
    r"""Returns :func:`describe` for this package, computed once per process.

    Trees are created often; git runs only the first time.

    Returns
    -------
    str
    """
    return describe()
