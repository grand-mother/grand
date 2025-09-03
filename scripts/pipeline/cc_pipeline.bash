#!/bin/bash
# SLURM options:
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --mem=1000
#SBATCH --time=0-00:15:00
#SBATCH --mail-user=fleg@lpnhe.in2p3.fr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --licenses=sps

# Get tag and database file to use
while getopts ":t:d:s:" option ${args}; do
   case $option in
      t)
         tag=${OPTARG};;
      d)
         db=${OPTARG};;
      s)
        site=${OPTARG};;
      :)
         printf "option -${OPTARG} need an argument\n"
         exit 1;;
      ?) # Invalid option
         printf "Error: Invalid option -${OPTARG}\n"
         exit 1;;
   esac
   if [ "${OPTARG:0:1}" == "-" ]; then
           printf "option -${option} need an argument\n"
           exit 1
   fi
done

