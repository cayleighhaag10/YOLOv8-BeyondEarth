#!/bin/bash
#SBATCH --job-name=sam2_experiments
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/sam2_experiments_%j.out
#SBATCH --error=logs/sam2_experiments_%j.err

set -e

ml py-jupyterlab/4.0.8_py39

mkdir -p "$HOME/YOLOv8-BeyondEarth/resources/slurm/logs"

OUT_DIR="$SCRATCH/bouldernet_results/sam2_experiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "Output directory: $OUT_DIR"
echo "Starting SAM2 experiments at $(date)"

cd "$OUT_DIR"

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=21600 \
    --output "$OUT_DIR/sam2_experiments_results.ipynb" \
    "$HOME/YOLOv8-BeyondEarth/resources/nb/sam2_experiments.ipynb"

echo "Done at $(date). Results in $OUT_DIR"
