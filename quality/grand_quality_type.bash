#!/bin/bash

cd $GRAND_ROOT
mkdir -p user/grand/stubs
MYPYPATH=user/grand/stubs
mypy --config-file=tests/mypy.ini --exclude grand/simulation/shower/coreas.py --show-error-codes -p grand > quality/report_type.txt
echo "================================== check type : mypy report"
cat quality/report_type.txt