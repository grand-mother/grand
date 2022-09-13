#!/bin/bash

export GRAND_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Set var GRAND_ROOT="$GRAND_ROOT
echo "=============================="

export PATH=$GRAND_ROOT/quality:$PATH
echo "add grand/quality to PATH"
echo "=============================="

export PATH=$GRAND_ROOT/scripts:$PATH
echo "add grand/scripts to PATH "
echo "=============================="

export PYTHONPATH=$GRAND_ROOT:$PYTHONPATH
echo "add grand to PYTHONPATH"
echo "=============================="
