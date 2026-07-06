#!/bin/bash
#SBATCH --account=nn9297k
#SBATCH --job-name=nora3_roms
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=nora3_roms_%j.log

set -o errexit
set -o nounset

/cluster/projects/nn9490k/sia/miniforge3/envs/nora3/bin/python \
    /cluster/home/shmiak/src/atm-forcing/app/nora3.py --use-roms
