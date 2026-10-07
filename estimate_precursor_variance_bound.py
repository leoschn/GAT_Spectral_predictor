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
    parser.add_argument("--batch-rows", type=int, default=500_000,
                        help="Spectra processed at once on the device (bounds peak memory)")
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


def row_batches(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def batched_distance(y, pred_fn, batch_rows):
    """masked_spectral_distance(y, pred) computed `batch_rows` spectra at a
    time; `pred_fn(rows)` builds the predictions for the slice `rows`."""
    return torch.cat([masked_spectral_distance(y[rows], pred_fn(rows))
                      for rows in row_batches(len(y), batch_rows)])


def optimize_barycenter(y, gid, init, counts, steps, lr, batch_rows):
    """Minimize the per-precursor mean masked spectral distance directly
    (each precursor's term is independent, so summing them is fine).
    Gradients are accumulated over row batches to bound peak memory."""
    param = init.clone().requires_grad_(True)
    opt = torch.optim.Adam([param], lr=lr)
    weights = 1.0 / counts[gid].float()
    # Adam can overshoot on tight precursors: keep each precursor's best
    # iterate so the result is never worse than the in-sample barycenter.
    best = init.clone()
    best_loss = torch.full_like(counts, float("inf"))
    for step in range(steps + 1):
        opt.zero_grad()
        dist = []
        for rows in row_batches(len(y), batch_rows):
            d = masked_spectral_distance(y[rows], param[gid[rows]])
            if step < steps:
                ((d * weights[rows]).sum() / len(counts)).backward()
            dist.append(d.detach())
        dist = torch.cat(dist)
        with torch.no_grad():
            per_precursor = group_sum(dist[:, None], gid, len(counts)).squeeze(1) / counts
            improved = per_precursor < best_loss
            best[improved] = param[improved]
            best_loss = torch.minimum(best_loss, per_precursor)
        if step % 50 == 0 or step == steps:
            print(f"  [optimize] step {step:4d}  mean-per-precursor SA = {per_precursor.mean().item():.5f}"
                  f"  (best {best_loss.mean().item():.5f})")
        if step == steps:
            break
        opt.step()
    # Negative intensities can only lower the cosine with non-negative spectra.
    return best.clamp_min(0.0)


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


def run_bound(y_np, meta, gid_np, keys, args, breakdown_cols=("ptm_type",)):
    """Score the in-sample / leave-one-out / optimized barycenters of the
    precursor groups `gid_np` and write the per-spectrum, summary and
    per-`breakdown_cols` results to `args.out_dir`."""
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

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
    sums = torch.zeros(n_groups, y.shape[1], dtype=y.dtype, device=device)
    for rows in row_batches(len(y), args.batch_rows):
        sums.index_add_(0, gid[rows], normalize_masked(y[rows]))

    results = {}

    # 1. In-sample barycenter (normalized mean of the normalized spectra).
    bary = sums / counts[:, None]
    results["in_sample"] = batched_distance(y, lambda rows: bary[gid[rows]], args.batch_rows)

    # 2. Leave-one-out barycenter: mean of the other n-1 replicates.
    results["leave_one_out"] = batched_distance(
        y, lambda rows: sums[gid[rows]] - normalize_masked(y[rows]), args.batch_rows)

    # 3. Barycenter optimized on the exact metric.
    if args.optimize_steps > 0:
        opt_bary = optimize_barycenter(y, gid, bary, counts, args.optimize_steps, args.lr,
                                       args.batch_rows)
        with torch.no_grad():
            results["optimized"] = batched_distance(y, lambda rows: opt_bary[gid[rows]],
                                                    args.batch_rows)

    summary = pd.DataFrame([summarize(k, v, gid, counts) for k, v in results.items()])
    with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 200):
        print("\n" + summary.to_string(index=False))

    # Per-spectrum output (useful for breaking down by PTM type, length, ...).
    per_spectrum = meta.copy()
    per_spectrum["precursor_id"] = gid_np
    per_spectrum["n_replicates"] = counts.cpu().numpy()[gid_np].astype(int)
    for k, v in results.items():
        per_spectrum[f"sa_{k}"] = v.cpu().numpy()

    sa_cols = [f"sa_{k}" for k in results]
    for col in breakdown_cols:
        if col not in per_spectrum.columns:
            continue
        by_col = per_spectrum.groupby(col)[sa_cols].mean()
        by_col.insert(0, "n_spectra", per_spectrum.groupby(col).size())
        with pd.option_context("display.float_format", "{:.5f}".format, "display.width", 200):
            print(f"\nMean spectral distance per {col}:\n" + by_col.to_string())
        by_col.to_csv(os.path.join(args.out_dir, f"by_{col}.csv"))

    per_spectrum.to_csv(os.path.join(args.out_dir, "per_spectrum.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(f"\nResults written to {args.out_dir}")


def main():
    args = parse_args()
    print(f"Reading {args.csv}")
    y_np, meta = load_data(args)
    gid_np, keys = group_ids(meta, args)
    run_bound(y_np, meta, gid_np, keys, args)


if __name__ == "__main__":
    main()
