#!/bin/bash

cd $GRAND_ROOT
FILE=quality/report_coverage.xml
if [ -f "$FILE" ]; then
	echo "Skip test coverage already done"	
else	
	coverage run --source=grand -m pytest tests -v
    coverage xml -i -o quality/report_coverage.xml    
fi 
coverage report
exit 0
