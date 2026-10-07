#!/bin/bash
#SBATCH --job-name=ptm_prosit2019
#SBATCH --output=logs/precompute_ptms_prosit2019_%j.out
#SBATCH --error=logs/precompute_ptms_prosit2019_%j.err
#SBATCH --account=bun@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

# Rebuilds the 3 PTM datasets with the Prosit 2019 processing (FTMS HCD only,
# Andromeda score > 50, no decoys, top 3 per precursor/CE/raw file, fragment
# intensities re-annotated from the raw peak lists, nominal NCE).
#
# Everything is written under ptm_pipeline_common.OUTPUT_ROOT
# (.../ptm_datasets_prosit2019), including the annotated msms files: the old
# ptm_datasets/ folder and the raw PRIDE folders are only read.
#
# Every step is resumable (per-source CSVs and the feature stores skip what
# already exists), so resubmitting after a timeout continues where it stopped.

set -euo pipefail
cd /lustre/fswork/projects/rech/bun/ucg81ws/these/Graph_variant
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module purge
module load pytorch-gpu/py3/2.8.0

set -x

# Dataset 1 builds the 21 per-PTM CSVs and the shared feature store;
# dataset 2 re-splits the same spectra and reuses both.
python -u precompute_dataset_ptms.py
python -u precompute_dataset_ptm_holdout.py
python -u precompute_dataset_unmod_baseline.py
