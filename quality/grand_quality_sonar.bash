#!/bin/bash

cd $GRAND_ROOT
sonar-scanner -Dproject.settings=quality/sonar_grand_local.properties

