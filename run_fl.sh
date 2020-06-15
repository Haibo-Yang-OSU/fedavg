#!/bin/bash

#SBATCH --partition=singularity
#SBATCH --time=5-48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=neurips20

module load py-pip/9.0.1-py3-dpds55c
module load cuda/9.2.88-ytn5jg2
pip install torchvision==0.2.1 --user
pip install tensorboardX --user
pip install absl-py --user

srun python run.py