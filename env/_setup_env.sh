#!/bin/bash

export GRAND_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
echo "Set var GRAND_ROOT="$GRAND_ROOT
echo "=============================="
export PATH=$PATH:$GRAND_ROOT/quality
echo "update PATH with grand quality"
echo "=============================="

export PYTHONPATH=$PYTHONPATH:$GRAND_ROOT
echo "update PYTHONPATH with grand source"
echo "=============================="
