#!/bin/bash
#SBATCH -J diffusion-moe
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 # this requests 1 node, 1 core.
#SBATCH --time=0-55:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=cs5824_2

module load Anaconda3/2020.11
source activate nn_reduction
python ddpm_sep_models.py