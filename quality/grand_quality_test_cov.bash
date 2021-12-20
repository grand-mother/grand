#!/bin/bash


# touch something

cd $GRAND_ROOT
coverage erase
rm -f quality/report_coverage*
rm -rf quality/html_cov*
coverage run --source=grand -m pytest tests -v
coverage xml -i -o quality/report_coverage.xml
coverage html -i -d quality/html_coverage
coverage report
