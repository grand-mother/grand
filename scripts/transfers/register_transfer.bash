#!/bin/bash -l
default_config="/pbs/home/p/prod_grand/scripts/transfers/softs/grand/scripts/transfers/config-prod.ini"
register_transfers='python3 /pbs/home/p/prod_grand/softs/grand/scripts/transfers/register_transfers.py'

while getopts ":d:t:c:" option; do
  case $option in
    d)
      db=${OPTARG};;
    t)
      tag=${OPTARG};;
    c)
      config=${OPTARG};;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    ?) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done

if [ -z ${config} ]; then
  config=$default_config
fi

cd /pbs/home/p/prod_grand/softs/grand
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate /sps/grand/software/conda/grandlib_2409

source env/setup.sh
cd /pbs/home/p/prod_grand/softs/grand/scripts/transfers
export PATH=/sps/grand/software/conda/grandlib_2409/bin/:$PATH

#${register_transfers} -d ${db} -t ${tag} -c ${config}
${register_transfers} -d ${db} -t ${tag} -c ${config}