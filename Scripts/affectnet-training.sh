#!/bin/bash
#SBATCH -N 2
#SBATCH -p GPU
#SBATCH --ntasks-per-node 28
#SBATCH -t 00:05:00
#SBATCH --gres=gpu:p100:2
set -x
cd ~/deepFAU/
module load anaconda2/5.1.0
source activate /pylon5/ir5fpcp/amogh112/faus_dl
python Scripts/affectnet-training.py
