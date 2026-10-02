# -*- coding: utf-8 -*-
r"""Checks on the downloaded data model (antenna, RF chain, noise, topography).

A missing data-model file failed with whatever library read it -- numpy's
``FileNotFoundError``, ``BadZipFile``, ``EOFError``, an ``IndexError`` deep in
the RF chain, a GULL or TURTLE ``LibraryError`` -- none pointing at the cure;
and a damaged file that still parsed changed the voltages silently (#279).

``data/download_data_grand.py`` now writes a manifest of the files it
installs, with their sizes and SHA-256 sums.  :func:`check` is called by the
loaders before reading a file: a missing file, or one whose size differs from
the manifest, raises with one actionable message.  ``python -m
grand.basis.data_model`` verifies the whole installation (sizes and sums), or
with ``--write`` records the current one as the reference.
"""

import argparse
import hashlib
import json
import os
import sys

from grand import GRAND_DATA_PATH
from grand.basis import validate as _validate

#: Where the manifest lives, beside the data model it describes.
MANIFEST = os.path.join(GRAND_DATA_PATH, "data_model_manifest.json")

#: The directories the data-model archive installs.
DIRECTORIES = ("detector", "noise", "topography")

_REMEDY = ("run `python data/download_data_grand.py` (env/setup.sh does), or set GRAND_ROOT to "
           "an installation that has it")
_manifest = None


def _load_manifest():
    r"""Returns ``{relative path: {"size", "sha256"}}``, or {} without a manifest."""
    global _manifest
    if _manifest is None:
        try:
            with open(MANIFEST) as handle:
                _manifest = json.load(handle).get("files", {})
        except (OSError, ValueError):
            _manifest = {}
    return _manifest


def check(path, where):
    r"""Returns `path` if the data-model file is there and has its recorded size.

    Parameters
    ----------
    path : str or os.PathLike
        The file about to be read.
    where : str
        Who reads it, for the message.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If the manifest records another size for it: the file is damaged or
        from another version of the data model.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(_validate.message(
            where, "data model incomplete: %s is missing; %s" % (path, _REMEDY)))
    entry = _load_manifest().get(os.path.relpath(os.path.abspath(path), GRAND_DATA_PATH))
    if entry is not None and os.path.getsize(path) != entry["size"]:
        raise ValueError(_validate.message(
            where, "data model damaged: %s has %d bytes, the installed data model %d; %s"
            % (path, os.path.getsize(path), entry["size"], _REMEDY)))
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root=GRAND_DATA_PATH, version=None):
    r"""Records the files under the data-model directories of `root`.

    Returns
    -------
    int
        The number of files recorded.
    """
    files = {}
    for directory in DIRECTORIES:
        for base, _, names in os.walk(os.path.join(root, directory)):
            for name in names:
                path = os.path.join(base, name)
                files[os.path.relpath(path, root)] = {"size": os.path.getsize(path), "sha256": _sha256(path)}
    with open(os.path.join(root, os.path.basename(MANIFEST)), "w") as handle:
        json.dump({"version": version, "files": files}, handle, indent=1, sort_keys=True)
    global _manifest
    _manifest = None
    return len(files)


def verify(root=GRAND_DATA_PATH, sums=False):
    r"""Returns the problems of the installation in `root`, as messages (empty if none).

    Without a manifest, only checks that the data-model directories exist.
    """
    problems = ["%s/ is missing" % d for d in DIRECTORIES if not os.path.isdir(os.path.join(root, d))]
    try:
        with open(os.path.join(root, os.path.basename(MANIFEST))) as handle:
            files = json.load(handle).get("files", {})
    except (OSError, ValueError):
        return problems
    for rel, entry in sorted(files.items()):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            problems.append("%s is missing" % rel)
        elif os.path.getsize(path) != entry["size"]:
            problems.append("%s has %d bytes, expected %d" % (rel, os.path.getsize(path), entry["size"]))
        elif sums and _sha256(path) != entry["sha256"]:
            problems.append("%s differs from the installed data model" % rel)
    return problems


def main(argv=None):
    r"""Command line: verify the data model, or record the current one."""
    parser = argparse.ArgumentParser(description="Verify the GRAND data model, or record it as the reference.")
    parser.add_argument("--write", action="store_true", help="record the current files as the reference")
    args = parser.parse_args(argv)
    if args.write:
        print("recorded %d files in %s" % (write_manifest(), MANIFEST))
        return 0
    problems = verify(sums=True)
    for problem in problems:
        print("data model: " + problem)
    if not os.path.exists(MANIFEST):
        print("no manifest at %s; only the directories were checked" % MANIFEST)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
