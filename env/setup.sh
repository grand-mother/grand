#! /bin/bash

call_path=$PWD
script_full_path=$(dirname "${BASH_SOURCE[0]}")

cd $script_full_path

. _setup_env.sh
. _setup_lib.sh
_setup_vs_code.py

# download data model for GRAND 
data/download_data_grand.py

cd $call_path