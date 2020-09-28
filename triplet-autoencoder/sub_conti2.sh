#!/bin/bash

#SBATCH -N 1
#SBATCH -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

module load anaconda/3.7.4
module load nvidia/cuda/10.1

source /home/hszhao/.brashrc
export PYTHONUNBUFFERED=1
source activate /dat01/hszhao/pytorch

python /dat01/hszhao/code/triplet_ae.py
