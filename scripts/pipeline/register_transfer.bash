#!/bin/bash
# SLURM options:
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --mem=2000
#SBATCH --time=0-01:00:00
#SBATCH --mail-user=fleg@lpnhe.in2p3.fr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --licenses=sps


while getopts ":d:t:c:g:" option; do
  case $option in
    d)
      db=${OPTARG};;
    t)
      tag=${OPTARG};;
    c)
      config=${OPTARG};;
    g)
      grand_path=${OPTARG};;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    ?) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done

#Get the pipeline configuration
setup_file="${grand_path}/scripts/pipeline/pipeline_setup.bash"
source $setup_file

if [ -z ${config} ]; then
  config=$default_config
fi

register_transfers="${python_interpreter} ${register_transfers_py}"

cd ${grand_path}
source ${conda_init}
conda activate ${conda_lib}

source env/setup.sh
cd ${grand_path}/scripts/transfers
export PATH=${conda_lib}/bin/:$PATH

#${register_transfers} -d ${db} -t ${tag} -c ${config}
echo "launch ${register_transfers} -d ${db} -t ${tag} -c ${config}"

${register_transfers} -d ${db} -t ${tag} -c ${config}