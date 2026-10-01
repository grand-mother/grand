"""One writer per ROOT file, across processes.

Several processes appending to one ROOT file -- two batch jobs with the same
output name, or a resubmitted job -- used to interleave their writes: events
were lost while every process reported success, or the file was left
unreadable (grand-mother/grand#281).  ROOT does no locking of its own.

A process that writes a file now holds an exclusive ``flock`` on the file
itself from the moment it opens the file for writing until it closes it.
Another process that tries to write the file meanwhile is refused with a clear
message.  A process that opened the file earlier, for reading, and wants to
write it after someone else did is refused too: its view of the file is out of
date, and writing would corrupt the file.  Readers take no lock, so reading is
never blocked.

Where ``flock`` is unavailable (Windows, or a file system without locks) the
checks are skipped and the behaviour is the old one.
"""

import errno
import os
import time
from logging import getLogger

from grand.basis import validate as _validate

try:
    import fcntl
except ImportError:                                  # pragma: no cover - Windows
    fcntl = None

logger = getLogger(__name__)

#: realpath -> file descriptor holding the exclusive lock, for files this process writes
_held = {}
#: realpath -> (device, inode, size, mtime) when this process last opened the file
_seen = {}

_NO_LOCKS = (errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL)

#: Seconds to wait for a lock before refusing.  A reader holds its shared lock
#: only while opening a file, so a writer that meets one waits for it instead
#: of refusing: refusing made processes started together all give up.
WAIT = 2.0


def _flock(fd, mode):
    r"""``flock(fd, mode | LOCK_NB)``, retried for up to ``WAIT`` seconds."""
    deadline = time.monotonic() + WAIT
    while True:
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _key(name):
    return os.path.realpath(os.fspath(name))


def _signature(path):
    r"""Identity, size, time and leading bytes of a file.

    The leading bytes hold ROOT's file and directory headers, rewritten on
    every write: file times alone are too coarse to tell two writes apart.
    """
    st = os.stat(path)
    with open(path, "rb") as f:
        head = f.read(512)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, head)


def note_opened(name):
    r"""Records the state of a file this process is opening."""
    path = _key(name)
    try:
        _seen[path] = _signature(path)
    except OSError:
        _seen.pop(path, None)


class opening:
    r"""Context manager around opening an existing file for reading.

    Refuses, with a clear message, a file another process is writing (what
    ROOT would read is half written), and records the file's state while no
    other process can change it.
    """

    def __init__(self, name, where):
        self.name, self.where, self.fd = name, where, None

    def __enter__(self):
        """Takes a shared lock for the open, or refuses."""
        path = _key(self.name)
        if fcntl is not None and path not in _held:
            try:
                self.fd = os.open(path, os.O_RDONLY)
                _flock(self.fd, fcntl.LOCK_SH)
            except BlockingIOError:
                os.close(self.fd)
                self.fd = None
                raise OSError(_validate.message(
                    self.where, "cannot open %s: another process is writing it; wait for it "
                    "to finish" % self.name)) from None
            except OSError:
                if self.fd is not None:
                    os.close(self.fd)
                self.fd = None
        note_opened(self.name)
        return self

    def __exit__(self, *exc):
        """Drops the shared lock."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        return False


def lock_for_writing(name, where, fresh=False):
    r"""Takes the exclusive write lock on ``name``, or raises OSError.

    Parameters
    ----------
    name : str
        The ROOT file, which must exist.
    where : str
        Who asks, for the error message.
    fresh : bool, optional
        The file is about to be opened (or was just created) by this call's
        caller, so this process holds no earlier view of it to go stale.
    """
    path = _key(name)
    if fcntl is None or path in _held:
        return
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return                                       # ROOT reports a file it cannot open
    try:
        _flock(fd, fcntl.LOCK_EX)
    except BlockingIOError:
        os.close(fd)
        raise OSError(_validate.message(
            where, "%s is being written by another process; wait for it to finish, or "
            "write to another file (several processes must not write one file)" % name)) from None
    except OSError as error:
        os.close(fd)
        if error.errno in _NO_LOCKS:
            logger.debug("no file locking for %s: %s", path, error)
            return
        raise
    seen = _seen.get(path)
    if not fresh and seen is not None and seen != _signature(path):
        os.close(fd)
        raise OSError(_validate.message(
            where, "%s was changed by another process after this one opened it; writing now "
            "would corrupt it. Open it again (a new tree object) and write then" % name))
    _held[path] = fd


def is_current(name):
    r"""Whether a file this process holds open still is the file on disk.

    False when it was removed, or replaced or changed by something other than
    this process since it was opened: a tree reopening the name must then
    open it afresh, not reuse the open copy (#236).  Unknown files, and files
    this process is writing, count as current.
    """
    path = _key(name)
    if path in _held:
        return True
    if not os.path.exists(path):
        return False
    seen = _seen.get(path)
    if seen is None:
        return True
    try:
        return _signature(path) == seen
    except OSError:
        return False


def release(name):
    r"""Releases the write lock on ``name`` and forgets its state; the file was closed."""
    path = _key(name)
    _seen.pop(path, None)
    fd = _held.pop(path, None)
    if fd is not None:
        os.close(fd)                                 # closing the descriptor drops the lock

