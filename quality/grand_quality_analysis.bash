#!/bin/bash

cd $GRAND_ROOT
pylint grand --rcfile quality/pylint.conf -r y --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" | tee quality/report_pylint.txt
echo $?

