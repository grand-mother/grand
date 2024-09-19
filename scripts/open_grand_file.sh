#!/bin/bash

# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [-p] <filename>"
    echo "-p        run standard Python shell instead of IPython"
    exit 1
fi

# Check for the -p option
if [ "$1" == "-p" ]; then
    # Assign the filename argument to a variable
    FILENAME=$2
    # Start the Python shell with the specified import and command
    python3 -i -c "from grand.dataio.root_trees import *; f = DataFile('$FILENAME'); print('\n\033[0;31mOpened file $FILENAME as f\033[0m\n'); f.print()"
else
    # Assign the filename argument to a variable
    FILENAME=$1
    # Start IPython with the specified import and command
    ipython -i -c "from grand.dataio.root_trees import *; f = DataFile('$FILENAME'); print('\n\033[0;31mOpened file $FILENAME as f\033[0m\n'); f.print()"
fi