#!/bin/bash

echo "Install external lib gull and turtle"
echo "===================================="
cd $GRAND_ROOT/src

# test conda case, for amd64 
if [ ! -z $CONDA_PREFIX ]
then
	echo "Add conda path to env variable C_INCLUDE_PATH and LIBRARY_PATH"
    export C_INCLUDE_PATH=$C_INCLUDE_PATH:$CONDA_PREFIX/include
    export LIBRARY_PATH=$LIBRARY_PATH:$CONDA_PREFIX/lib
fi   
# 
./install_ext_lib.bash
cd $GRAND_ROOT
