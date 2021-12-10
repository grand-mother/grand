#!/bin/bash

if [ $1 eq 0 ]; then
	mkdir sonar
	cd sonar         
	curl -O https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-4.6.2.2472-linux.zip
	unzip  sonar-scanner-cli-4.6.2.2472-linux.zip
	PATH=$PATH:$GRAND_ROOT/sonar/sonar-scanner-4.6.2.2472-linux/bin
	cd $GRAND_ROOT
	grand_quality_test_cov.bash
	grand_quality_analysis.bash
	sonar-scanner -Dproject.settings=quality/sonar.properties
fi 
