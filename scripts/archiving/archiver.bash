#!/bin/bash
# SLURM options:
#SBATCH --job-name=archive-data-1m
#SBATCH --output=/sps/grand/prod_grand/archiving/archiving-1m.log
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --time=0-10:00:00
#SBATCH --mail-user=fleg@lpnhe.in2p3.fr
#SBATCH --mail-type=FAIL,TIME_LIMIT,INVALID_DEPEND
#SBATCH --licenses=sps
/pbs/home/p/prod_grand/softs/grand/scripts/archiving/archive_grandraw.bash -d 1
