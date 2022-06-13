#!/bin/bash

#SBATCH -J train_unet
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amanagar@iu.edu
#SBATCH --gpus-per-node=v100:2
#SBATCH --time=04:00:00

#Load any modules that your program needs
module load deeplearning/2.6.0

#Run your program
python train.py
