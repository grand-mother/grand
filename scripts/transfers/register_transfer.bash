#!/bin/bash -l
register_transfers='python3 /pbs/home/p/prod_grand/scripts/transfers/register_transfers.py'

while getopts ":d:t:" option; do
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
export PATH=/sps/grand/software/conda/grandlib_2409/bin/:$PATH

#${register_transfers} -d ${db} -t ${tag} -c ${config}
${register_transfers} -d ${db} -t ${tag}