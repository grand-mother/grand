"""Shared access to an observatory transfer database: <tag>_<site>_dbfile.db.

Used by get_files_from_db.py and get_files_list.py.  The database is opened
read-only: a mistyped path used to create an empty database and then fail
with "no such table" (#245).
"""
import re
import sqlite3
import sys
from pathlib import Path

NAME = re.compile(r"(\d+)_([A-Za-z0-9]+)_")


def transferred_files(db, limit=None):
    r"""Returns the target paths of the files the transfer database marks as transferred.

    Exits with status 2 and a message on stderr if the database name does not
    follow ``<tag>_<site>_...``, or it is missing or not an SQLite database.
    """
    match = NAME.match(Path(db).name)
    if not match:
        sys.exit("%s: the database name must look like <tag>_<site>_dbfile.db, got %s"
                 % (Path(sys.argv[0]).name, Path(db).name))
    tag, _site = match.groups()
    if not Path(db).is_file():
        sys.exit("%s: no such database: %s" % (Path(sys.argv[0]).name, db))
    try:
        connection = sqlite3.connect("file:%s?mode=ro" % Path(db).resolve(), uri=True)
        query = ("SELECT target AS file FROM gfiles, transfer WHERE gfiles.id = transfer.id "
                 "AND transfer.tag = ? AND transfer.success = 1")
        if limit is not None:
            query += " LIMIT %d" % int(limit)
        rows = connection.execute(query, (int(tag),)).fetchall()
        connection.close()
    except sqlite3.DatabaseError as error:
        sys.exit("%s: cannot read %s: %s" % (Path(sys.argv[0]).name, db, error))
    return [row[0] for row in rows]
