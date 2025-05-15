#!/bin/bash
#SBATCH --job-name=ddpm_cifar100lt
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=swanb045@umn.edu

module load cuda
source activate env

# Run your script
python -u train_MSI.py
