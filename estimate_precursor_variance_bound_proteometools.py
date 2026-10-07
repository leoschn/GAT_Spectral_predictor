"""
Precursor-variance lower bound on the ProteomeTools (Prosit HDF5) datasets.

Same evaluation as `estimate_precursor_variance_bound.py` (in-sample,
leave-one-out and metric-optimized barycenters of the replicate spectra
of each precursor), but reading the Prosit-format HDF5 files used by
`precompute_dataset.py` instead of the PTM CSVs.

A precursor is `sequence_integer` (M(ox) is its own token) + precursor
charge, optionally + collision energy (`--group-by-ce`).

Which collision energy to group on
----------------------------------
The model is fed `collision_energy_aligned`, i.e. the nominal NCE plus a
per-raw-file calibration offset. That value is essentially unique per raw
file, so grouping on it leaves mostly singleton precursors whose in-sample
bound is trivially 0. The default `--ce-col collision_energy` (nominal
NCE: 20, 23, 25, 28, 30, 35) gives the bound for a model that sees the
nominal energy but not the per-run calibration -- slightly optimistic for
the real model; the leave-one-out row is the safer number to compare
against.

Usage
-----
    python estimate_precursor_variance_bound_proteometools.py                  # holdout
    python estimate_precursor_variance_bound_proteometools.py --group-by-ce
    python estimate_precursor_variance_bound_proteometools.py --h5 <...>/val_hcd.hdf5
"""

import argparse

import h5py
import numpy as np
import pandas as pd

from estimate_precursor_variance_bound import run_bound


RAW_DIR = "/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset"
HOLDOUT_H5 = f"{RAW_DIR}/holdout_hcd.hdf5"

# Mirrors data/graph_creation_utils.alphabet (not imported: it pulls in RDKit).
ALPHABET = ["", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y", "M(ox)"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", default=HOLDOUT_H5,
                        help="Prosit-format HDF5 file (default: holdout_hcd.hdf5, the model's test set)")
    parser.add_argument("--group-by-ce", action="store_true",
                        help="Also split precursors by collision energy")
    parser.add_argument("--ce-col", default="collision_energy",
                        choices=["collision_energy", "collision_energy_aligned",
                                 "collision_energy_aligned_normed"],
                        help="Collision energy dataset used with --group-by-ce (see module docstring)")
    parser.add_argument("--ce-decimals", type=int, default=None,
                        help="Rounding applied to the collision energy (default: exact value)")
    parser.add_argument("--min-replicates", type=int, default=2,
                        help="Only precursors with at least this many spectra are scored")
    parser.add_argument("--optimize-steps", type=int, default=200,
                        help="Adam steps refining the barycenter on the exact metric (0 to disable)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Number of HDF5 rows read at once")
    parser.add_argument("--batch-rows", type=int, default=500_000,
                        help="Spectra processed at once on the device (bounds peak memory)")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available)")
    parser.add_argument("--out-dir", default="logs/precursor_variance_bound_proteometools",
                        help="Where per-spectrum / per-precursor CSVs and summary are written")
    args = parser.parse_args()
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def load_data(args):
    with h5py.File(args.h5, "r") as f:
        n = f["intensities_raw"].shape[0]
        intensities, seq_int, charge, ce = [], [], [], []
        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)
            intensities.append(f["intensities_raw"][start:end].astype(np.float32))
            seq_int.append(f["sequence_integer"][start:end])
            charge.append(f["precursor_charge_onehot"][start:end].argmax(axis=1) + 1)
            ce.append(f[args.ce_col][start:end, 0])
            print(f"  loaded {end:,}/{n:,} spectra", end="\r")
    print()

    seq_int = np.concatenate(seq_int)
    meta = pd.DataFrame({
        "charge": np.concatenate(charge),
        "collision_energy": np.concatenate(ce),
    })
    # Decode each distinct sequence once, then broadcast back.
    uniq, inverse = np.unique(seq_int, axis=0, return_inverse=True)
    decoded = np.array(["".join(ALPHABET[t] for t in row) for row in uniq], dtype=object)
    meta["sequence"] = decoded[inverse.reshape(-1)]
    meta["length"] = (seq_int > 0).sum(axis=1)
    return np.concatenate(intensities), meta


def group_ids(meta, args):
    keys = ["sequence", "charge"]
    if args.group_by_ce:
        ce = meta["collision_energy"]
        meta["ce_key"] = ce if args.ce_decimals is None else ce.round(args.ce_decimals)
        keys.append("ce_key")
        print(f"Collision energies ({args.ce_col}): {meta['ce_key'].nunique()} distinct values")
    gid, _ = pd.MultiIndex.from_frame(meta[keys]).factorize()
    return gid, keys


def main():
    args = parse_args()
    print(f"Reading {args.h5}")
    y_np, meta = load_data(args)
    gid_np, keys = group_ids(meta, args)
    # Aligned energies are continuous: only the nominal NCE makes a readable breakdown.
    breakdown = ("charge", "collision_energy") if args.ce_col == "collision_energy" else ("charge",)
    run_bound(y_np, meta, gid_np, keys, args, breakdown_cols=breakdown)


if __name__ == "__main__":
    main()
