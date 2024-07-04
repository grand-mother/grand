#!/bin/bash
cd /pbs/home/p/prod_grand/softs/grand
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate /sps/grand/software/conda/grandlib_2304
source env/setup.sh
cd /pbs/home/p/prod_grand/scripts/transfers
python3 /pbs/home/p/prod_grand/softs/grand/granddb/refresh_mat_views.py