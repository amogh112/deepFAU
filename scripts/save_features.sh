#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH --ntasks-per-node 28
#SBATCH -t 15:00:00
set -x
cd ~/deepFAU/
module load anaconda2/5.1.0
source activate /pylon5/ir5fpcp/amogh112/detect_face
python Scripts/face_scripts.py
