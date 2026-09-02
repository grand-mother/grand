# This will return the list of files transfered in a batch from an observatory
# The tag and site are extracted from the database file name
# The full path to the database must be provided
# The database name must be : <tag>_<site>_dbfile.db
import sqlite3
import argparse, re, json
from pathlib import Path
import shutil

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
    cursor.execute("SELECT target as file FROM gfiles, transfer WHERE gfiles.id = transfer.id AND transfer.tag = "+tag+" AND transfer.success=1  ;")
    rows = cursor.fetchall()
    connection.close()
    # Filter only existing files
    #files = [row[0] for row in rows if Path(row[0]).exists()]


    existing_files = []
    missing_files = []
    small_files = []

    for row in rows:
        file_path = Path(row[0])
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > 262144:  # 256 kB in bytes
                existing_files.append(row[0])
            else:
                # Move small files to the relative crap directory
                crap_dir = file_path.parent.parent.parent / "crap"
                crap_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
                dest_path = crap_dir / file_path.name
                shutil.move(str(file_path), str(dest_path))
                small_files.append(row[0])
        else:
            missing_files.append(row[0])
else:
    existing_files = []
    missing_files = []
    small_files = []
    files=[]

# Should we add a warning for small files/missing files ?



print(json.dumps(existing_files))
#print(json.dumps(files))
#print(files)
