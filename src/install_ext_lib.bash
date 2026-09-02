#! /bin/bash
#
# Compiles TURTLE and GULL and installs the results into the package.
#
# `set -e` is correct here and not in env/setup.sh: this script is *executed*,
# so aborting on the first failure ends the script and nothing else.  Without
# it, a failed `make` was followed by `cp` regardless, and the script's exit
# status became the status of the last `cp` -- so a build failure was reported
# as success and only surfaced later as "No module named 'grand._core'".
set -e

#make clean
make

cp build/grand/_core.abi3.so ../grand
cp build/lib/*.so ../lib
