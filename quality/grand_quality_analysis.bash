#!/bin/bash

cd $GRAND_ROOT
#pylint grand --rcfile quality/pylint.conf -r y --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" | tee quality/report_pylint.txt
pylint grand --rcfile quality/pylint.conf --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" --output=quality/report_pylint.txt -r y
status=$?
#pylint grand --rcfile quality/pylint.conf -r y
cat quality/report_pylint.txt
exit $status