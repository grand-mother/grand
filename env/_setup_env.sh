#!/bin/bash

export GRAND_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Set var GRAND_ROOT="$GRAND_ROOT
echo "=============================="

export PATH=$PATH:$GRAND_ROOT/quality
echo "add grand/quality to PATH"
echo "=============================="

export PATH=$PATH:$GRAND_ROOT/scripts
echo "add grand/scripts to PATH "
echo "=============================="

export PYTHONPATH=$PYTHONPATH:$GRAND_ROOT
echo "add grand to PYTHONPATH"
echo "=============================="
