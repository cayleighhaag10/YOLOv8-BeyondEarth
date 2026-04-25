#!/bin/bash
#SBATCH --job-name=orientation_bug
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM:24GB
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/orientation_bug_%j.out
#SBATCH --error=logs/orientation_bug_%j.err

set -e

ml py-jupyterlab/4.0.8_py39

# Save plots and output notebook to a dated directory in scratch
OUT_DIR="$SCRATCH/bouldernet_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "Output directory: $OUT_DIR"
echo "Starting notebook execution at $(date)"

# Run from OUT_DIR so relative plt.savefig() calls land there
cd "$OUT_DIR"

jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output "$OUT_DIR/test_orientation_bug_results.ipynb" \
    "$HOME/YOLOv8-BeyondEarth/resources/nb/test_orientation_bug.ipynb"

echo "Done at $(date). Results in $OUT_DIR"
