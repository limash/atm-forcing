#!/bin/bash
#SBATCH --account=nn9490k
#SBATCH --job-name=carra2_roms
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=carra2_roms_%j.log

set -o errexit
set -o nounset

/cluster/projects/nn9490k/sia/miniforge3/envs/nora3/bin/python \
    /cluster/home/shmiak/src/atm-forcing/app/carra2.py --use-roms
