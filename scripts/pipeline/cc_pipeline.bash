#!/bin/bash -l
# SLURM options:
#SBATCH --job-name=test-pipeline
#S BATCH --partition=htc_daemon
#SBATCH --partition=htc
#SBATCH --ntasks=2
#SBATCH --mem=4000
#SBATCH --account=grand
#SBATCH --time=0-06:45:00
#SBATCH --mail-user=fleg@lpnhe.in2p3.fr
#SBATCH --mail-type=FAIL
#SBATCH --licenses=sps
#SBATCH --output=/sps/grand/prod_grand/DB_TESTS/TEST_SLURM/data/gp80/logs/%x_%j.log
#SBATCH --error=/sps/grand/prod_grand/DB_TESTS/TEST_SLURM/data/gp80/logs/%x_%j.err


dbfile=$1

if [ "${dbfile}" == "" ]; then
  dbfile="/sps/grand/prod_grand/DB_TESTS/TEST_SLURM/20250905224520_GP80_dbfile.db"
fi

echo "Running with database ${dbfile}"
timestamp=$(basename "${dbfile}" | cut -d'_' -f1)
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate /pbs/home/p/prod_grand/.conda/envs/snakemake
cd /sps/grand/prod_grand/DB_TESTS/grand/scripts/pipeline/
echo "snakemake --directory /sps/grand/prod_grand/DB_TESTS/TEST_SLURM/data/gp80/logs/ \
   --snakefile  /sps/grand/prod_grand/DB_TESTS/grand/scripts/pipeline/Snakefile \
   --config dbfile=${dbfile} \
   --cores 2 --profile ccin2p3 --keep-going  --latency-wait 60  --printshellcmds "
snakemake --directory /sps/grand/prod_grand/DB_TESTS/TEST_SLURM/data/gp80/logs/ \
   --snakefile  /sps/grand/prod_grand/DB_TESTS/grand/scripts/pipeline/Snakefile \
   --config dbfile=${dbfile} \
   --cores 2 --profile ccin2p3 --keep-going  --latency-wait 60  --printshellcmds # --verbose 



#/sps/grand/data/gp80/logs/20260303214752_GP80_dbfile.db
