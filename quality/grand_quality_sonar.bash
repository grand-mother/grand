#!/bin/bash

cd $GRAND_ROOT
sonar-scanner -Dproject.settings=quality/sonar_grand_in2p3.properties

