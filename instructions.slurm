#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --job-name="blackmamba-test"
#SBATCH --output=blackmamba-test3.out

cd ${SLURM_SUBMIT_DIR}/

module purge
module load Anaconda3/2023.09-0
conda activate blackmamba_env

srun python main.py
