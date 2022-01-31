#!/bin/bash

export GRAND_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Set var GRAND_ROOT="$GRAND_ROOT
export PATH=$PATH:$GRAND_ROOT/quality
echo "update PATH with grand quality"
export PYTHONPATH=$PYTHONPATH:$GRAND_ROOT
echo "update PYTHONPATH with grand source"

echo "Install external lib gull and turtle"
cd $GRAND_ROOT/src
./install_ext_lib.bash
cd $GRAND_ROOT
