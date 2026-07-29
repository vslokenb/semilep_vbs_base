#!/bin/bash
#SBATCH --job-name=mix_years
#SBATCH --output=logs/mix_years_%A_%a.out
#SBATCH --error=logs/mix_years_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=hammer-g
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-0

INPUT_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dirs.txt)

echo "Running mix_years.py on: $INPUT_DIR"
echo "Host: $(hostname), Time: $(date)"

python mix_years.py "$INPUT_DIR"