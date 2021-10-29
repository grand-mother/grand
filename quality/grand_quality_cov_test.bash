#!/bin/bash

cd $GRAND_ROOT
coverage erase
rm -f quality/report_*.*
coverage run -m pytest tests 
coverage xml -i -o quality/report_coverage.xml
coverage html -i -d quality/html_coverage

