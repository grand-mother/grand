#!/bin/bash

cd $GRAND_ROOT
mkdir -p user/grand/stubs
MYPYPATH=user/grand/stubs
mypy --config-file=tests/mypy.ini --pretty --html-report quality/html_mypy -p grand > quality/report_type.txt