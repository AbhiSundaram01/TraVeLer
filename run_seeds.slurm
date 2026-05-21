#!/bin/bash
#SBATCH -J vf_seeds
#SBATCH -A VENKATARAMANAN-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=1-5
#SBATCH --mail-type=NONE
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

module purge
module load rhel8/default-icl

mkdir -p logs

export SEED=$SLURM_ARRAY_TASK_ID

bash hpctest.sh
