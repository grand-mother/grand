#!/bin/bash

cd $GRAND_ROOT
mkdir -p user/grand/stubs
MYPYPATH=user/grand/stubs
mypy --config-file=tests/mypy.ini --show-error-codes --pretty --html-report quality/html_mypy -p grand > quality/report_type.txt
echo "================================== check type : mypy report"
cat quality/report_type.txt