#!/bin/bash

# Check if a filename argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [-p] <dirname>"
    echo "-p        run standard Python shell instead of IPython"
    exit 1
fi

# Check for the -p option
if [ "$1" == "-p" ]; then
    # Assign the filename argument to a variable
    DIRNAME=$2
    # Start the Python shell with the specified import and command
    python3 -i -c "from grand.dataio.root_trees import *; d = DataDirectory('$DIRNAME'); print('\n\033[0;31mOpened directory $DIRNAME as d\033[0m\n'); d.print()"
else
    # Assign the filename argument to a variable
    DIRNAME=$1
    # Start IPython with the specified import and command
    ipython -i -c "from grand.dataio.root_trees import *; d = DataDirectory('$DIRNAME'); print('\n\033[0;31mOpened directory $DIRNAME as d\033[0m\n'); d.print()"
fi