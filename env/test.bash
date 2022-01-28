#!/bin/bash

echo $1
#echo "FIN"
export GRAND_ROOT=$1
echo "GRAND_ROOT"
echo $GRAND_ROOT
export PATH=$PATH:$GRAND_ROOT/quality
export PYTHONPATH=$PYTHONPATH:$GRAND_ROOT
cd $GRAND_ROOT/src
./install_ext_lib.bash
cd -
