# This will return the list of files transfered in a batch from an observatory
# The tag and site are extracted from the database file name
# The full path to the database must be provided
# The database name must be : <tag>_<site>_dbfile.db
#
# Files of 256 kB or less are moved to <parent>/<parent>/<parent>/crap/ and
# left out of the list, as before; every move, and every listed file that does
# not exist, is now reported on stderr (#245).
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _transfer_db import transferred_files  # noqa: E402

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--database", help="Database file to use", required=True)
args = argParser.parse_args()

existing_files = []
for name in transferred_files(args.database):
    file_path = Path(name)
    if not file_path.exists():
        print("missing: %s" % name, file=sys.stderr)
        continue
    if file_path.stat().st_size > 262144:  # 256 kB in bytes
        existing_files.append(name)
        continue
    # Move small files to the relative crap directory
    crap_dir = file_path.parent.parent.parent / "crap"
    crap_dir.mkdir(exist_ok=True)
    dest_path = crap_dir / file_path.name
    shutil.move(str(file_path), str(dest_path))
    print("moved (256 kB or less): %s -> %s" % (name, dest_path), file=sys.stderr)

print(json.dumps(existing_files))
