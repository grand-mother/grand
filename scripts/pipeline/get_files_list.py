# This will return the list of files transfered in a batch from an observatory
# The tag and site are extracted from the database file name
# The full path to the database must be provided
# The database name must be : <tag>_<site>_dbfile.db
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _transfer_db import transferred_files  # noqa: E402

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--database", help="Database file to use", required=True)
args = argParser.parse_args()

for name in transferred_files(args.database, limit=10):
    print(name)
