# This will return the list of files transfered in a batch from an observatory
# The tag and site are extracted from the database file name
# The full path to the database must be provided
# The database name must be : <tag>_<site>_dbfile.db
import sqlite3
import argparse, re, json
from pathlib import Path

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--database", help="Database file to use", required=True)
args = argParser.parse_args()
db=args.database
full_path = Path(db)
file_name = full_path.name

match = re.match(r"(\d+)_([A-Za-z0-9]+)_", file_name)
if match:
    tag, site = match.groups()
    connection = sqlite3.connect(db)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute("SELECT target as file FROM gfiles, transfer WHERE gfiles.id = transfer.id AND transfer.tag = "+tag+" AND transfer.success=1 LIMIT 10;")
    rows = cursor.fetchall()
    connection.close()
    files = [row[0] for row in rows]
#    for row in rows:
#        print(row[0])
else:
    files=[]

print(json.dumps(files))
#print(files)