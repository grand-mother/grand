#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
source ${SCRIPT_DIR}/pipeline_setup.bash
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh


cd ${grandlib_path}
conda activate ${conda_lib}
export PATH=${conda_lib}/bin/:$PATH
source env/setup.sh
${conda_lib}/bin/python3 granddb/refresh_mat_views.py -c ${default_config}



