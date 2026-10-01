"""Safe extraction of tar archives.

``tarfile.extractall`` without a filter trusts the archive: a member named
``../x`` or ``/etc/x``, or a symbolic link followed by a file under it, is
written outside the target directory.  Archives reach GRANDlib from download
servers and from collaborators (ZHAireS events are shipped as ``.tgz``), so they
are checked here before anything is written.
"""

import os
import tarfile

from grand.basis import validate as _validate

__all__ = ["safe_extract"]


def _inside(path, root):
    r"""Whether ``path`` lies in ``root`` (both absolute, symbolic links resolved)."""
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def check_members(tar, dest):
    r"""Raises ValueError if extracting ``tar`` into ``dest`` would write outside it.

    Refused: absolute names, names that climb out with ``..``, symbolic or hard
    links whose target is outside ``dest``, and device or FIFO members.

    Parameters
    ----------
    tar : tarfile.TarFile
        The open archive.
    dest : str or os.PathLike
        The directory it is to be extracted into.
    """
    where = "safe_extract"
    dest = os.path.abspath(dest)
    for member in tar.getmembers():
        name = member.name
        target = os.path.join(dest, name)
        if os.path.isabs(name) or not _inside(target, dest):
            raise ValueError(_validate.message(
                where, "archive member %r would be written outside %s" % (name, dest)))
        if member.issym():
            link = os.path.join(os.path.dirname(target), member.linkname)
            if os.path.isabs(member.linkname) or not _inside(link, dest):
                raise ValueError(_validate.message(
                    where, "archive member %r is a link to %r, outside %s"
                    % (name, member.linkname, dest)))
        elif member.islnk():
            link = os.path.join(dest, member.linkname)
            if os.path.isabs(member.linkname) or not _inside(link, dest):
                raise ValueError(_validate.message(
                    where, "archive member %r is a hard link to %r, outside %s"
                    % (name, member.linkname, dest)))
        elif not (member.isfile() or member.isdir()):
            raise ValueError(_validate.message(
                where, "archive member %r is a device or FIFO; refused" % name))


def safe_extract(archive, dest, mode="r:*"):
    r"""Extracts a tar archive into ``dest``, refusing members that escape it.

    Every member is checked first (see :func:`check_members`), so a refused
    archive writes nothing.  Where the standard library offers it (Python 3.12,
    and backported to 3.10.12 and 3.11.4), the ``"data"`` extraction filter is
    applied as well.

    Parameters
    ----------
    archive : str or os.PathLike
        Path to the ``.tar``, ``.tgz`` or ``.tar.gz`` file.
    dest : str or os.PathLike
        Directory to extract into; created if missing.
    mode : str, optional
        Mode for :func:`tarfile.open`.

    Returns
    -------
    list of tarfile.TarInfo
        The members extracted.

    Raises
    ------
    ValueError
        If any member would be written outside ``dest``.
    """
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, mode) as tar:
        check_members(tar, dest)
        members = tar.getmembers()
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:                                      # pragma: no cover - old Pythons
            tar.extractall(dest)
    return members
