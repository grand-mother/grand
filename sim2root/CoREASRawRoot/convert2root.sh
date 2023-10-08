#!/bin/bash
#SBATCH --job-name=2root
#SBATCH --output=2root.out

# Request resources (e.g., CPU cores, memory, and time)
#SBATCH --nodes=1                   # Number of compute nodes
#SBATCH --ntasks=1                  # Number of CPU cores
#SBATCH --mem=4G                    # Memory per node
#SBATCH --time=1:00:00              # Maximum job duration (hours:minutes:seconds)

python3 unzip_and_convert2root.py -f /sps/grand/jelena/stshp+GP13/archive/proton_run01.tar.gz
