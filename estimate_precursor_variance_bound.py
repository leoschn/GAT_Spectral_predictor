"""
Estimate the irreducible error ("noise floor") of the PTM dataset.

A predictor only sees (sequence, charge[, collision energy]), so every
replicate spectrum of the same precursor receives the *same* prediction.
The best any model can do on a precursor is therefore to output the
barycenter of its replicate spectra, and the mean spectral distance
between that barycenter and the replicates is a lower bound on the
error of a perfect model.

For every precursor (same modified `sequence` + precursor charge) with
at least `--min-replicates` spectra, this script computes:

  * in-sample barycenter  : mean of the L2-normalized masked spectra
                            (optimal for mean cosine similarity).
  * leave-one-out (LOO)   : barycenter of the *other* replicates, scored
                            on the held-out one. Removes the optimistic
                            bias of the in-sample bound on small groups,
                            i.e. an estimate a model could actually reach.
  * optimized (optional)  : barycenter refined by gradient descent to
                            directly minimize the mean masked spectral
                            distance (the exact metric used in main.py),
                            i.e. the tightest in-sample lower bound.

Distances are computed with `model.losses.masked_spectral_distance`, the
same function used for training/evaluation.

Usage
-----
    python estimate_precursor_variance_bound.py                       # whole PTM dataset
    python estimate_precursor_variance_bound.py --csv <split>/test.csv
    python estimate_precursor_variance_bound.py --group-by-ce         # also group by collision energy
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.losses import masked_spectral_distance


# Mirrors ptm_pipeline_common (not imported: it pulls in the RDKit graph code).
ALL_PTMS_CSV = "/lustre/fsn1/projects/rech/bun/ucg81ws/ptm_datasets/csv/df_21_ptms.csv"
INTENSITIES_COL = "intensities"
CHARGE_COL = "precursor_charge_onehot"
ENERGY_COL = "collision_energy"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=ALL_PTMS_CSV,
                        help="CSV with intensities / sequence / precursor_charge_onehot columns "
                             "(default: combined 21-PTM CSV)")
    parser.add_argument("--seq-col", default="sequence",
                        help="Sequence column defining the precursor (default: modified sequence)")
    parser.add_argument("--group-by-ce", action="store_true",
                        help="Also split precursors by collision energy")
    parser.add_argument("--ce-decimals", type=int, default=None,
                        help="Rounding applied to collision energy when --group-by-ce is set "
                             "(default: exact value, as seen by the model; CE is normalized, "
                             "e.g. 0.25, so rounding to 0 decimals would merge all energies)")
    parser.add_argument("--min-replicates", type=int, default=2,
                        help="Only precursors with at least this many spectra are scored")
    parser.add_argument("--optimize-steps", type=int, default=200,
                        help="Adam steps refining the barycenter on the exact metric (0 to disable)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=200_000,
                        help="Number of CSV rows read at once")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="logs/precursor_variance_bound",
                        help="Where per-spectrum / per-precursor CSVs and summary are written")
    return parser.parse_args()


def parse_vector_column(series):
    """Fast parsing of stringified lists such as "[0.0, -1.0, 0.3]"."""
    return np.stack([
        np.fromstring(s.strip()[1:-1], sep=",", dtype=np.float32) for s in series
    ])


def load_data(args):
    usecols = [INTENSITIES_COL, args.seq_col, CHARGE_COL]
    if args.group_by_ce:
        usecols.append(ENERGY_COL)
    if "ptm_type" in pd.read_csv(args.csv, nrows=0).columns:
        usecols.append("ptm_type")

    intensities, meta = [], []
    for chunk in pd.read_csv(args.csv, usecols=usecols, chunksize=args.chunk_size):
        intensities.append(parse_vector_column(chunk[INTENSITIES_COL]))
        chunk = chunk.drop(columns=[INTENSITIES_COL])
        chunk["charge"] = parse_vector_column(chunk[CHARGE_COL]).argmax(axis=1) + 1
        meta.append(chunk.drop(columns=[CHARGE_COL]))
        print(f"  loaded {sum(len(m) for m in meta):,} spectra", end="\r")
    print()

    meta = pd.concat(meta, ignore_index=True)
    meta[args.seq_col] = meta[args.seq_col].astype(str).str.strip().str.strip("_")
    return np.concatenate(intensities), meta


def group_ids(meta, args):
    keys = [args.seq_col, "charge"]
    if args.group_by_ce:
        ce = meta[ENERGY_COL]
        meta["ce_key"] = ce if args.ce_decimals is None else ce.round(args.ce_decimals)
        keys.append("ce_key")
        print(f"Collision energies: {meta['ce_key'].nunique()} distinct values "
              f"{sorted(meta['ce_key'].unique())[:10]}")
    gid, _ = pd.MultiIndex.from_frame(meta[keys]).factorize()
    return gid, keys


def normalize_masked(y):
    """Zero the impossible (-1) ions, then L2-normalize -- exactly what
    masked_spectral_distance does to both vectors before the dot product."""
    return F.normalize(y.clamp_min(0.0), p=2, dim=-1)


def group_sum(x, gid, n_groups):
    out = torch.zeros(n_groups, x.shape[1], dtype=x.dtype, device=x.device)
    return out.index_add_(0, gid, x)


def optimize_barycenter(y, gid, init, counts, steps, lr):
    """Minimize the per-precursor mean masked spectral distance directly
    (each precursor's term is independent, so summing them is fine)."""
    param = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([param], lr=lr)
    weights = 1.0 / counts[gid].float()
    for step in range(steps):
        opt.zero_grad()
        loss = (masked_spectral_distance(y, param[gid]) * weights).sum() / len(counts)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"  [optimize] step {step:4d}  mean-per-precursor SA = {loss.item():.5f}")
    # Negative intensities can only lower the cosine with non-negative spectra.
    return param.detach().clamp_min(0.0)


def summarize(name, dist, gid, counts):
    per_precursor = group_sum(dist[:, None], gid, len(counts)).squeeze(1) / counts
    d = dist.cpu().numpy()
    p = per_precursor.cpu().numpy()
    return {
        "barycenter": name,
        "spectrum_mean": d.mean(),
        "spectrum_median": np.median(d),
        "spectrum_std": d.std(),
        "precursor_mean": p.mean(),
        "precursor_median": np.median(p),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading {args.csv}")
    y_np, meta = load_data(args)
    gid_np, keys = group_ids(meta, args)

    counts_all = np.bincount(gid_np)
    keep = counts_all[gid_np] >= args.min_replicates
    print(f"Spectra: {len(meta):,} | precursors ({' + '.join(keys)}): {len(counts_all):,}")
    print(f"Precursors with >= {args.min_replicates} replicates: "
          f"{(counts_all >= args.min_replicates).sum():,} "
          f"covering {keep.sum():,} spectra ({100 * keep.mean():.1f}%)")
    print("Replicates per kept precursor: "
          + ", ".join(f"{q}={np.percentile(counts_all[counts_all >= args.min_replicates], q):.0f}"
                      for q in (25, 50, 75, 95, 100)))

    meta = meta[keep].reset_index(drop=True)
    gid_np = pd.factorize(gid_np[keep])[0]
    y = torch.from_numpy(y_np[keep]).to(device)
    del y_np
    gid = torch.from_numpy(gid_np).to(device)
    n_groups = int(gid.max()) + 1
    counts = torch.bincount(gid, minlength=n_groups).to(y.dtype)

    # All replicates share sequence + charge, hence the same -1 mask, so the
    # barycenter is well defined position-wise.
    y_norm = normalize_masked(y)
    sums = group_sum(y_norm, gid, n_groups)

    results = {}

    # 1. In-sample barycenter (normalized mean of the normalized spectra).
    bary = sums / counts[:, None]
    results["in_sample"] = masked_spectral_distance(y, bary[gid])

    # 2. Leave-one-out barycenter: mean of the other n-1 replicates.
    loo = sums[gid] - y_norm
    results["leave_one_out"] = masked_spectral_distance(y, loo)

    # 3. Barycenter optimized on the exact metric.
    if args.optimize_steps > 0:
        opt_bary = optimize_barycenter(y, gid, bary, counts, args.optimize_steps, args.lr)
        with torch.no_grad():
            results["optimized"] = masked_spectral_distance(y, opt_bary[gid])

    summary = pd.DataFrame([summarize(k, v, gid, counts) for k, v in results.items()])
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 200):
        print("\n" + summary.to_string(index=False))

    # Per-spectrum output (useful for breaking down by PTM type, length, ...).
    per_spectrum = meta.copy()
    per_spectrum["precursor_id"] = gid_np
    per_spectrum["n_replicates"] = counts.cpu().numpy()[gid_np].astype(int)
    for k, v in results.items():
        per_spectrum[f"sa_{k}"] = v.cpu().numpy()

    if "ptm_type" in per_spectrum.columns:
        sa_cols = [f"sa_{k}" for k in results]
        by_ptm = per_spectrum.groupby("ptm_type")[sa_cols].mean()
        by_ptm.insert(0, "n_spectra", per_spectrum.groupby("ptm_type").size())
        with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 200):
            print("\nMean spectral distance per PTM type:\n" + by_ptm.to_string())
        by_ptm.to_csv(os.path.join(args.out_dir, "by_ptm_type.csv"))

    per_spectrum.to_csv(os.path.join(args.out_dir, "per_spectrum.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(f"\nResults written to {args.out_dir}")


if __name__ == "__main__":
    main()
