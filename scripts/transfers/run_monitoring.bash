#!/bin/bash

# SLURM options:

#SBATCH --job-name=monitoring    # Nom du job
#S BATCH --output=serial_test_%j.log   # Standard output et error log

#SBATCH --partition=htc               # Choix de partition (htc par défaut)

#SBATCH --ntasks=10                    # Exécuter une seule tâche
#SBATCH --mem=15000                    # Mémoire en MB par défaut
#SBATCH --time=0-02:01:00             # Délai max = 7 jours

#SBATCH --mail-user=fleg@lpnhe.in2p3.fr          # Où envoyer l'e-mail
#SBATCH --mail-type=BEGIN,END,FAIL          # Événements déclencheurs (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --licenses=sps                # Déclaration des ressources de stockage et/ou logicielles

# Commandes à soumettre :


while getopts ":t:" option; do
  case $option in
    t)
      tag=${OPTARG};;
    :)
      printf "option -${OPTARG} need an argument\n"
      exit 1;;
    *) # Invalid option
      printf "Error: Invalid option -${OPTARG}\n"
      exit 1;;
  esac
done
shift $(($OPTIND - 1))

if [ -z "$tag" ]; then
  echo "Error: -t <tag> is required"
  echo "Usage: $0 -t <tag>"
  exit 1
fi



cd /pbs/home/p/prod_grand/softs/grand/
source /pbs/throng/grand/soft/miniconda3/etc/profile.d/conda.sh
conda activate /sps/grand/software/conda/grandlib_test
source env/setup.sh 
cd granddb/
python3 /pbs/home/p/prod_grand/softs/grand/granddb/monitoring.py -t ${tag}

