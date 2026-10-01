# -*- coding: utf-8 -*-
r"""Archives are extracted only inside their target directory.

A member named ``../x`` or ``/x``, or a link pointing outside, used to be
written wherever it pointed: ``tarfile.extractall`` without a filter trusts the
archive.  Each crafted archive below must be refused, with nothing written,
and an ordinary archive must still extract.
"""

import io
import os
import pathlib
import subprocess
import sys
import tarfile

import pytest

from grand.basis.archive import safe_extract

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPRESS = ROOT / "sim2root" / "ZHAireSRawRoot" / "ZHAireSCompressEvent.py"


def _archive(path, members):
    r"""Writes a .tgz holding ``members``: (name, kind, payload) tuples."""
    with tarfile.open(path, "w:gz") as tar:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                data = payload.encode()
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload
                tar.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = payload
                tar.addfile(info)
    return path


EVIL = {
    "parent": [("../escaped", "file", "x")],
    "nested_parent": [("a/../../escaped", "file", "x")],
    "absolute": [("/tmp/grand_test_escaped_absolute", "file", "x")],
    "symlink_out": [("lnk", "symlink", ".."), ("lnk/escaped", "file", "x")],
    "symlink_absolute": [("lnk", "symlink", "/tmp")],
    "hardlink_out": [("hl", "hardlink", "../outside")],
}


@pytest.mark.parametrize("case", sorted(EVIL))
def test_an_escaping_member_is_refused_and_nothing_is_written(tmp_path, case):
    dest = tmp_path / "dest"
    archive = _archive(tmp_path / "evil.tgz", [("ok.txt", "file", "fine")] + EVIL[case])
    with pytest.raises(ValueError, match="GRANDlib: safe_extract:"):
        safe_extract(archive, dest)
    assert not (tmp_path / "escaped").exists()
    assert not (dest / "ok.txt").exists()          # checked before writing anything


def test_an_ordinary_archive_extracts(tmp_path):
    archive = _archive(tmp_path / "ok.tgz", [
        ("detector", "dir", None), ("detector/a.txt", "file", "A"),
        ("detector/link", "symlink", "a.txt"), ("noise/b.txt", "file", "B")])
    members = safe_extract(archive, tmp_path / "data")
    assert (tmp_path / "data/detector/a.txt").read_text() == "A"
    assert (tmp_path / "data/noise/b.txt").read_text() == "B"
    assert os.path.islink(tmp_path / "data/detector/link")
    assert len(members) == 4


def test_zhaires_uncompress_refuses_an_escaping_archive(tmp_path):
    event = tmp_path / "event"
    event.mkdir()
    _archive(event / "evil.tgz", [("../escaped", "file", "x"), ("legit.txt", "file", "y")])
    done = subprocess.run([sys.executable, str(COMPRESS), str(event), "uncompress"],
                          capture_output=True, text=True, timeout=120)
    assert not (tmp_path / "escaped").exists()
    assert "outside" in done.stderr, done.stderr[-1500:]


def test_zhaires_uncompress_still_extracts_an_event(tmp_path):
    event = tmp_path / "event"
    event.mkdir()
    _archive(event / "Run_1.tgz", [("Run_1.sry", "file", "summary"), ("a0.trace", "file", "1 2 3 4")])
    done = subprocess.run([sys.executable, str(COMPRESS), str(event), "uncompress"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    assert (event / "Run_1.sry").read_text() == "summary"
    assert (event / "a0.trace").exists()
