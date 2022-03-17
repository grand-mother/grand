#!/bin/bash

FILE=sonar.properties
if [ -f "$FILE" ]; then
	cd $GRAND_ROOT
	grand_quality_test_cov.bash
	pylint grand --rcfile quality/pylint.conf --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" --output=quality/report_pylint.txt --exit-zero
	cat quality/report_pylint.txt
	sonar-scanner -Dproject.settings=$FILE
fi 
# always ok for github CI
exit 0
