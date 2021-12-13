#!/bin/bash

FILE=sonar.properties
if [ -f "$FILE" ]; then
	mkdir sonar
	cd sonar         
	curl -O https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.6.2.2472-linux.zip
	unzip  sonar-scanner-cli-4.6.2.2472-linux.zip
	PATH=$PATH:$GRAND_ROOT/sonar/sonar-scanner-4.6.2.2472-linux/bin
	cd $GRAND_ROOT
	grand_quality_test_cov.bash
	pylint grand --rcfile quality/pylint.conf --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" --output=quality/report_pylint.txt -r y --exit-zero
	sonar-scanner -Dproject.settings=$FILE
fi 
# always ok for github CI
exit 0
