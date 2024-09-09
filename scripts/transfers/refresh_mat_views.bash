#!/bin/bash
uname -r |grep el9 >/dev/null
el9=$?

cd /pbs/home/p/prod_grand/softs/grand
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
if [ "$el9" -ne 0 ]; then
  conda activate /sps/grand/software/conda/grandlib_2304
else
  conda activate /sps/grand/software/conda/grandlib_2409
fi
source env/setup.sh
cd /pbs/home/p/prod_grand/scripts/transfers
python3 /pbs/home/p/prod_grand/softs/grand/granddb/refresh_mat_views.py