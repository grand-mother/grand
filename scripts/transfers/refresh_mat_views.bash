#!/bin/bash -l
cd /pbs/home/p/prod_grand/softs/grand
export PLATFORM=redhat-9-x86_64
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate /sps/grand/software/conda/grandlib_2409
source env/setup.sh
#cd /pbs/home/p/prod_grand/scripts/transfers
python3 /pbs/home/p/prod_grand/softs/grand/granddb/refresh_mat_views.py