#!/bin/bash
echo "=========="
echo $1
echo "=========="
export GRAND_ROOT=$PWD/..
echo "GRAND_ROOT"
echo $GRAND_ROOT
export PATH=$PATH:$GRAND_ROOT/quality
export PYTHONPATH=$PYTHONPATH:$GRAND_ROOT
#cd $GRAND_ROOT/src
cd ../src
./install_ext_lib.bash
cd ..
