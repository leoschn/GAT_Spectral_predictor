#!/bin/bash
#SBATCH --job-name=estimation_pt
#SBATCH --output=logs/estimate_proteometools.out
#SBATCH --error=logs/estimate_proteometools.err
#SBATCH --time=2:00:00
#SBATCH --account=bun@v100
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --qos=qos_gpu-dev

set -euo pipefail
mkdir -p logs

module purge
module load pytorch-gpu/py3/2.8.0

python estimate_precursor_variance_bound_proteometools.py \
    --h5 /lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset/traintest_hcd.hdf5 \
    --group-by-ce --min-replicates 1 --optimize-steps 0 \
    --out-dir logs/precursor_variance_bound_proteometools_traintest
