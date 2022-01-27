#! /bin/bash

local prefix="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# add to env variable define root pf package
export GRAND_ROOT=$prefix
export PATH=$PATH:$GRAND_ROOT/quality
export PYTHONPATH=$PYTHONPATH:$GRAND_ROOT

cd $GRAND_ROOT/src
./install_ext_lib.bash
cd -
	 