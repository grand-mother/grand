#!/bin/bash
#
# Compiles the external C libraries.  Sourced by env/setup.sh, so it returns a
# status rather than exiting, and does not use `set -e` -- that would leak into
# the caller's shell.

echo "Install external lib gull and turtle"
echo "===================================="
cd $GRAND_ROOT/src

# test conda case, for amd64
if [ -n "$CONDA_PREFIX" ]
then
	echo "Add conda path to env variable C_INCLUDE_PATH and LIBRARY_PATH"
    export C_INCLUDE_PATH=$C_INCLUDE_PATH:$CONDA_PREFIX/include
    export LIBRARY_PATH=$LIBRARY_PATH:$CONDA_PREFIX/lib
fi
#
./install_ext_lib.bash
_grand_lib_status=$?

cd $GRAND_ROOT

# Return the build's status, not the `cd`'s.  This used to end on the cd, so a
# failed compile was reported as success to env/setup.sh.
return $_grand_lib_status
