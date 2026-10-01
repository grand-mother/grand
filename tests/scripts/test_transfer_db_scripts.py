# -*- coding: utf-8 -*-
r"""#245: scripts/pipeline/get_files_from_db.py and get_files_list.py.

Small files were moved without a word, missing files dropped silently, a
misnamed database gave "[]" and exit 0, and a mistyped path created an empty
database before failing on "no such table".
"""

import json
import pathlib
import sqlite3
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "scripts" / "pipeline"


def _database(tmp_path, files):
    path = tmp_path / "17_GP80_dbfile.db"
    db = sqlite3.connect(str(path))
    db.execute("CREATE TABLE gfiles (id INTEGER, target TEXT)")
    db.execute("CREATE TABLE transfer (id INTEGER, tag INTEGER, success INTEGER)")
    for i, name in enumerate(files):
        db.execute("INSERT INTO gfiles VALUES (?, ?)", (i, str(name)))
        db.execute("INSERT INTO transfer VALUES (?, 17, 1)", (i,))
    db.commit()
    db.close()
    return path


def _run(script, *args):
    return subprocess.run([sys.executable, str(PIPELINE / script), *map(str, args)],
                          capture_output=True, text=True, timeout=120)


def test_moves_and_missing_files_are_reported(tmp_path):
    data = tmp_path / "a" / "b" / "c"
    data.mkdir(parents=True)
    big, small = data / "big.bin", data / "small.bin"
    big.write_bytes(b"x" * 300000)
    small.write_bytes(b"x" * 10)
    db = _database(tmp_path, [big, small, data / "gone.bin"])
    done = _run("get_files_from_db.py", "-d", db)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [str(big)]
    assert "moved (256 kB or less): %s" % small in done.stderr
    assert "missing: %s" % (data / "gone.bin") in done.stderr
    assert (tmp_path / "a" / "crap" / "small.bin").exists()


def test_bad_databases_are_refused_without_creating_one(tmp_path):
    for script in ("get_files_from_db.py", "get_files_list.py"):
        done = _run(script, "-d", tmp_path / "nonsense.db")
        assert done.returncode != 0 and "<tag>_<site>_" in done.stderr
        typo = tmp_path / "17_GP80_typo.db"
        done = _run(script, "-d", typo)
        assert done.returncode != 0 and "no such database" in done.stderr
        assert not typo.exists()
