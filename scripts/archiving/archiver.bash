#!/bin/bash
# SLURM options:
# S BATCH --job-name=archive-data-1m
# S BATCH --output=/sps/grand/prod_grand/archiving/archiving-1m.log
# S BATCH --partition=htc
# S BATCH --ntasks=1
# S BATCH --mem=8000
# S BATCH --time=0-10:00:00
# S BATCH --mail-user=fleg@lpnhe.in2p3.fr
# S BATCH --mail-type=FAIL,TIME_LIMIT,INVALID_DEPEND
# S BATCH --licenses=sps
delay=1
mail_user='fleg@lpnhe.in2p3.fr'
mail_type='START,END,FAIL,TIME_LIMIT,INVALID_DEPEND'

read year month << DATE_COMMAND
 $(date --date="TODAY -${delay} month" "+%Y %m")
DATE_COMMAND

read y m << DATE_COMMAND
 $(date --date="TODAY +1 month" "+%Y %m")
DATE_COMMAND

nextdate=${y}-${m}-15T00:00:00

sbatch -t 0-10:00 -J archive-${year}-${month} -n 1 --mem=8G -o /sps/grand/prod_grand/archiving/archiving-${year}-${month} --begin=${nextdate} --mail-user=${mail_user} --mail-type=${mail_type} /pbs/home/p/prod_grand/softs/grand/scripts/archiving/archiver.bash
echo "/pbs/home/p/prod_grand/softs/grand/scripts/archiving/archive_grandraw.bash -d ${delay}"
/pbs/home/p/prod_grand/softs/grand/scripts/archiving/archive_grandraw.bash -d ${delay}
