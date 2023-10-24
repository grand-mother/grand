import sys
from granddatalib import DataManager

if len(sys.argv) < 2:
    print("No file")
    exit(1)
else:
    print(f"Reading file {sys.argv[1]}")


dm = DataManager()
