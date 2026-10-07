"""
Shared building blocks for the three PTM-impact datasets.

This module centralizes everything that is duplicated across
`precompute_dataset_ptms.py` (dataset 1: all 21 PTMs), the PTM-holdout
generalization script (dataset 2: 15 seen / 6 unseen PTMs) and the
unmodified-baseline script (dataset 3), so the raw-data metadata and the
splitting/feature-computation logic can never drift between the three.

Key idea to avoid paying for graph-feature computation twice
--------------------------------------------------------------
Dataset 1 and dataset 2 are built from the *same* 21 PTM-type search
results -- dataset 2 just partitions those same spectra differently
(by PTM type instead of by sequence only). Recomputing the (expensive)
`aa_ptm_to_pyg` graph construction for every spectrum in both scripts
would double the compute for no reason.

Instead:

1. `build_ptm_type_csv` / `build_unmod_csv` turn raw MaxQuant search
   folders into per-source CSVs (cheap: I/O + string parsing).
2. `combine_and_tag` concatenates them and stamps every row with a
   globally unique, stable `row_id`.
3. `compute_feature_store` turns every row into a PyG `Data` object
   *once* and persists it, keyed by `row_id`, in a shared "feature
   store" directory. It is resumable/idempotent: if a `row_id` is
   already present in the store's index, it is never recomputed.
4. `gather_rows_into_dataset` builds the final, split-specific chunked
   dataset directories (the `chunk_i.pt` + `meta.txt` format consumed by
   `HierarchicalStreamingAAPTMDataset`) by *copying* already-computed
   `Data` objects out of the feature store for the requested `row_id`s.
   This step is pure tensor I/O -- no chemistry/graph computation.

Dataset 1's script and dataset 2's script both point at the same
feature-store directory (see `ALL_PTMS_STORE_DIR` below), so whichever
one runs first pays the featurization cost and the other one reuses it.
Dataset 3 (unmodified baseline) is built from disjoint raw data (the
`*_Unmod` search folders), so it gets its own, separate feature store.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

from data.hierarchical_aa_ptm_dataset import aa_ptm_to_pyg
from data.sequence_preprocessing import convert_msms_to_prosit, annotate_msms_with_acquisition


# =========================
# CONFIG
# =========================
CHUNK_SIZE = 2048
BATCH_SIZE = 2048

# Column names in the per-source CSVs produced by convert_msms_to_prosit.
INTENSITIES_COL = "intensities"
CHARGE_COL = "precursor_charge_onehot"
ENERGY_COL = "collision_energy"

# Columns every per-source CSV is trimmed down to before being combined.
OUTPUT_COLUMNS = [
    "intensities",
    "sequence",
    "sequence_no_mod",
    "precursor_charge_onehot",
    "collision_energy",
    "raw_file",
    "scan_number",
    "score",
    "ptm_type",
]

# ------------------------------------------------------------------
# Where things live. All three dataset scripts import these so they
# always agree on where the shared feature store is.
#
# Set to the cluster paths: raw "SEARCH_*" MaxQuant search folders live
# under PRIDE_ROOT (locally, the same layout is checked into
# data_jz/data/pride -- switch back to that for a local dry run).
# ------------------------------------------------------------------

PRIDE_ROOT = "/lustre/fsn1/projects/rech/bun/ucg81ws/data/pride"

# Root everything this pipeline produces gets written under.
# New root for the Prosit 2019 processing (FTMS only, score > 50, top 3,
# intensities re-annotated from the raw spectra, NCE units): the CSVs and
# feature stores are resumable and keyed by row_id, so pointing at the old
# root would silently reuse graphs built from the previous selection.
OUTPUT_ROOT = "/lustre/fsn1/projects/rech/bun/ucg81ws/ptm_datasets_prosit2019"

# msms.txt annotated with RAW acquisition metadata + peak lists (~1 GB per
# source). Kept under OUTPUT_ROOT so the raw PRIDE folders, and the
# annotated files earlier datasets were built from, are never overwritten.
ANNOTATED_DIR = os.path.join(OUTPUT_ROOT, "annotated_msms")

# Combined, row_id-tagged CSV + shared feature store for the 21
# PTM-modified sources. Both dataset 1 and dataset 2 read/write here.
ALL_PTMS_CSV = os.path.join(OUTPUT_ROOT, "csv", "df_21_ptms.csv")
ALL_PTMS_STORE_DIR = os.path.join(OUTPUT_ROOT, "feature_store", "all_ptms_with_mod")

# Combined, row_id-tagged CSV + feature store for the unmodified
# baseline sources (dataset 3). Disjoint raw data -> its own store.
UNMOD_CSV = os.path.join(OUTPUT_ROOT, "csv", "df_unmod_baseline.csv")
UNMOD_STORE_DIR = os.path.join(OUTPUT_ROOT, "feature_store", "unmod_baseline")


# =========================
# PTM SOURCE METADATA (21 modification types)
# =========================
# One entry per "SEARCH_*" MaxQuant search folder under PRIDE_ROOT.
# Transcribed verbatim from the residue_list / mod_code_list /
# mod_code_list_modified lists that used to live in
# precompute_dataset_ptms.py -- do not "clean up" the mod codes, several
# of them (di/ds/da, gl/gy) were deliberately picked to avoid collisions
# between distinct PTMs that would otherwise share a residue + raw code
# (see git history: "fix mod_code duplicate issue", "fix lysine
# methylation error", "fix Hydroxyproline").
_PTM_FOLDERS = [
    "SEARCH_Kmod_Formyl", "SEARCH_Kmod_Propion", "SEARCH_Ymod_Nitrotyr",
    "SEARCH_Kmod_Acetyl", "SEARCH_Kmod_Glutaryl", "SEARCH_Kmod_Succinyl",
    "SEARCH_Rmod_Citrullin", "SEARCH_Ymod_Phospho", "SEARCH_Kmod_Biotin",
    "SEARCH_Kmod_GlyGly", "SEARCH_Kmod_Trimethyl", "SEARCH_Rmod_Dimethyl-as",
    "SEARCH_Kmod_Butyryl", "SEARCH_Kmod_Hydroxyisobut", "SEARCH_Rmod_Dimethyl-sym",
    "SEARCH_Kmod_Crotonyl", "SEARCH_Kmod_Malonyl", "SEARCH_Rmod_Methyl",
    "SEARCH_Kmod_Dimethyl", "SEARCH_Kmod_Methyl", "SEARCH_Pmod_Hydroxypro",
]

_PTM_PROB_COLS = [
    "Formyl (K) Probabilities", "Propion  (K) Probabilities", "Nitrotyrosine (Y) Probabilities",
    "Acetyl (K) Probabilities", "Glutaryl (K) Probabilities", "Succinyl (K) Probabilities",
    "Citrullin (R) Probabilities", "Phospho (Y) Probabilities", "Biotin (K) Probabilities",
    "GlyGly (K) Probabilities", "Trimethyl (K) Probabilities", "Dimethyl (R) Probabilities",
    "Butyryl (K) Probabilities", "Hydroxyisobutyryl (K) Probabilities", "Dimethyl (R) Probabilities",
    "Crotonyl (K) Probabilities", "Malonyl (K) Probabilities", "Methyl (R) Probabilities",
    "Dimethyl (K) Probabilities", "Methyl (KR) Probabilities", "Hydroxyproline manuell (M) Probabilities",
]

_PTM_RESIDUES = [
    "K", "K", "Y", "K", "K", "K", "R", "Y", "K", "K", "K", "R",
    "K", "K", "R", "K", "K", "R", "K", "K", "P",
]

_PTM_MOD_CODES = [
    "fo", "pr", "ni", "ac", "gl", "su", "ci", "ph", "bi", "gl", "tr", "di",
    "bu", "hy", "di", "cr", "ma", "me", "di", "me", "hy",
]

_PTM_MOD_CODES_MODIFIED = [
    None, None, None, None, None, None, None, None, None, "gy", None, "da",
    None, None, "ds", None, None, None, None, None, None,
]

assert len(_PTM_FOLDERS) == 21
assert len({len(_PTM_FOLDERS), len(_PTM_PROB_COLS), len(_PTM_RESIDUES),
            len(_PTM_MOD_CODES), len(_PTM_MOD_CODES_MODIFIED)}) == 1

PTM_SOURCES = [
    dict(folder=folder, prob_col=prob_col, residue=residue,
         mod_code=mod_code, mod_code_modified=mod_code_modified)
    for folder, prob_col, residue, mod_code, mod_code_modified in zip(
        _PTM_FOLDERS, _PTM_PROB_COLS, _PTM_RESIDUES,
        _PTM_MOD_CODES, _PTM_MOD_CODES_MODIFIED,
    )
]

# The 4 "no variable modification searched" control folders, one per
# residue family, used as the unmodified baseline (dataset 3).
UNMOD_SOURCES = [
    dict(folder="SEARCH_Kmod_Unmod"),
    dict(folder="SEARCH_Rmod_Unmod"),
    dict(folder="SEARCH_Ymod_Unmod"),
    dict(folder="SEARCH_Pmod_Unmod"),
]


# =========================
# RAW -> PER-SOURCE CSV
# =========================
def raw_msms_paths(folder, pride_root):
    """
    'SEARCH_Kmod_Formyl' -> (
        <pride_root>/SEARCH_Kmod_Formyl/Kmod_Formyl/combined/txt/msms,
        <pride_root>/SEARCH_Kmod_Formyl/Kmod_Formyl,
    )
    """
    subdir = folder.split("_")[1] + "_" + folder.split("_")[2]
    search_dir = os.path.join(pride_root, folder, subdir)
    full_path = os.path.join(search_dir, "combined", "txt", "msms")
    return full_path, search_dir


def build_ptm_type_csv(entry, pride_root, out_csv, force=False):
    """
    Raw MaxQuant search folder -> per-PTM-type CSV with columns
    OUTPUT_COLUMNS. Cheap step (I/O + regex), not the expensive one.
    """
    if os.path.exists(out_csv) and not force:
        print(f"[skip] {out_csv} already exists")
        return pd.read_csv(out_csv)

    full_path, raw_dir = raw_msms_paths(entry["folder"], pride_root)
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    annotated_path = os.path.join(ANNOTATED_DIR, entry["folder"] + "_msms_annotated.txt")

    print(f"Annotating {full_path}.txt ...")
    annotate_msms_with_acquisition(
        input_file=full_path + ".txt",
        raw_msms_dir=raw_dir,
        output_file=annotated_path,
    )

    tmp_csv = out_csv + ".raw.csv"
    convert_msms_to_prosit(
        msms_file=annotated_path,
        output_file=tmp_csv,
        residue=entry["residue"],
        mod_code=entry["mod_code"],
        mod_code_modified=entry["mod_code_modified"],
        prob_col_name=entry["prob_col"],
    )

    df = pd.read_csv(tmp_csv)
    df["intensities"] = df["intensities_norm"]
    df["ptm_type"] = entry["folder"]
    df = df[OUTPUT_COLUMNS]
    df.to_csv(out_csv, index=False)
    return df


def build_unmod_csv(entry, pride_root, out_csv, force=False):
    """
    Raw "no PTM variable modification searched" folder -> per-source CSV.

    `convert_msms_to_prosit` always needs a `* Probabilities` column to
    resolve localization for its target residue. These control searches
    never had one (no variable PTM was searched), so we inject an
    all-empty column: `resolve_localized_modification` treats an empty/
    NaN probability string as "nothing to localize, keep sequence as
    is", which is exactly the unmodified-baseline behavior we want (only
    M(ox) -- resolved separately via the real "Oxidation (M)
    Probabilities" column that IS present -- can still show up).
    """
    if os.path.exists(out_csv) and not force:
        print(f"[skip] {out_csv} already exists")
        return pd.read_csv(out_csv)

    full_path, raw_dir = raw_msms_paths(entry["folder"], pride_root)
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    annotated_path = os.path.join(ANNOTATED_DIR, entry["folder"] + "_msms_annotated.txt")

    print(f"Annotating {full_path}.txt ...")
    annotate_msms_with_acquisition(
        input_file=full_path + ".txt",
        raw_msms_dir=raw_dir,
        output_file=annotated_path,
    )

    dummy_col = "__no_ptm_prob__"
    df_annot = pd.read_csv(annotated_path, sep="\t", low_memory=False)
    df_annot[dummy_col] = np.nan
    dummy_annotated_path = os.path.join(ANNOTATED_DIR, entry["folder"] + "_msms_annotated_unmod.txt")
    df_annot.to_csv(dummy_annotated_path, sep="\t", index=False)

    tmp_csv = out_csv + ".raw.csv"
    convert_msms_to_prosit(
        msms_file=dummy_annotated_path,
        output_file=tmp_csv,
        residue="X",              # unused: dummy prob col is always NaN
        mod_code="__none__",      # unused: dummy prob col is always NaN
        mod_code_modified=None,
        prob_col_name=dummy_col,
    )

    df = pd.read_csv(tmp_csv)
    df["intensities"] = df["intensities_norm"]
    df["ptm_type"] = entry["folder"]

    # Sanity check: these are "unmodified" searches, nothing besides
    # M(ox) should ever show up in `sequence`.
    unexpected = (
        df["sequence"]
        .str.replace(r"M\(ox\)", "", regex=True)
        .str.contains(r"\(", regex=True)
    )
    if unexpected.any():
        print(
            f"WARNING: {int(unexpected.sum())} spectra in "
            f"{entry['folder']} carry an unexpected modification code "
            f"-- check the raw search."
        )

    df = df[OUTPUT_COLUMNS]
    df.to_csv(out_csv, index=False)
    return df


def combine_and_tag(csv_paths, out_csv):
    """Concatenate per-source CSVs and stamp every row with a globally
    unique `row_id` -- the join key used by the feature store."""
    dfs = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    df.insert(0, "row_id", np.arange(len(df)))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Combined {len(csv_paths)} sources -> {len(df):,} spectra ({out_csv})")
    return df


# =========================
# CSV FIELD PARSING (unchanged from precompute_dataset_ptms.py)
# =========================
def parse_stringified_list(value):
    import ast
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32)


def load_sequences(df, seq_col):
    return [str(s).strip().strip("_") for s in df[seq_col].tolist()]


def load_intensities(df):
    return np.stack([parse_stringified_list(v) for v in df[INTENSITIES_COL].tolist()])


def load_charge(df):
    return np.stack([parse_stringified_list(v) for v in df[CHARGE_COL].tolist()])


def load_energy(df):
    return df[ENERGY_COL].to_numpy(dtype=np.float32)


# =========================
# SEQUENCE-LEVEL SPLITTING
# =========================
def split_dataframe_by_sequence(
    df,
    group_col="sequence_no_mod",
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.10,
    seed=42,
):
    """
    Split according to unique `group_col` values (typically
    sequence_no_mod). All spectra sharing the same underlying peptide
    sequence are guaranteed to land in the same split.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), (
        "Split ratios must sum to 1.0"
    )

    unique_groups = df[group_col].astype(str).str.strip().unique()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    groups_str = df[group_col].astype(str)
    train_df = df[groups_str.isin(train_groups)].copy()
    val_df = df[groups_str.isin(val_groups)].copy()
    test_df = df[groups_str.isin(test_groups)].copy()

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    print("\n" + "=" * 60)
    print(f"SPLIT SUMMARY (grouped by '{group_col}')")
    print("=" * 60)
    print(f"Total spectra: {len(df):,} | Total unique groups: {n:,}")
    print(f"TRAIN: {len(train_df):,} spectra | {len(train_groups):,} unique groups")
    print(f"VAL:   {len(val_df):,} spectra | {len(val_groups):,} unique groups")
    print(f"TEST:  {len(test_df):,} spectra | {len(test_groups):,} unique groups")
    print("=" * 60)

    return train_df, val_df, test_df


def verify_disjoint(named_dfs, group_col):
    """
    `named_dfs`: dict[name -> DataFrame]. Raises AssertionError if any
    two of them share a `group_col` value (sequence leakage / PTM-type
    leakage between splits).
    """
    sets = {name: set(df[group_col].astype(str)) for name, df in named_dfs.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            assert sets[a].isdisjoint(sets[b]), (
                f"ERROR: {group_col} leakage between '{a}' and '{b}'"
            )
    print(f"No '{group_col}' leakage detected among: {', '.join(names)}")


# =========================
# CHUNK I/O
# =========================
def save_chunk(buffer, out_dir, chunk_id):
    path = os.path.join(out_dir, f"chunk_{chunk_id}.pt")
    torch.save(buffer, path)
    with open(os.path.join(out_dir, "meta.txt"), "a") as f:
        f.write(f"{path},{len(buffer)}\n")


# =========================
# FEATURE STORE: compute each spectrum's graph ONCE
# =========================
def compute_feature_store(df, seq_col, store_dir, with_position=False):
    """
    Compute a PyG `Data` object for every row of `df` exactly once and
    persist it under `store_dir`, keyed by the row's `row_id`.

    Resumable: rows whose `row_id` is already present in
    `store_dir/row_index.csv` are skipped, so re-running this (e.g.
    after a crash, or from a second script pointed at the same store)
    never recomputes an already-featurized spectrum.

    `df` must contain a globally unique integer `row_id` column (see
    `combine_and_tag`). Every produced Data object also carries
    `data.row_id` (and `data.ptm_type`, if that column is present in
    `df`), so a chunk can be inspected on its own.
    """
    os.makedirs(store_dir, exist_ok=True)
    meta_path = os.path.join(store_dir, "meta.txt")
    index_path = os.path.join(store_dir, "row_index.csv")

    if os.path.exists(index_path):
        done_index = pd.read_csv(index_path)
        done_ids = set(done_index["row_id"].tolist())
        chunk_id = sum(1 for _ in open(meta_path)) if os.path.exists(meta_path) else 0
    else:
        pd.DataFrame(columns=["row_id", "chunk_id", "offset"]).to_csv(index_path, index=False)
        open(meta_path, "w").close()
        done_ids = set()
        chunk_id = 0

    has_ptm_type = "ptm_type" in df.columns
    remaining = (
        df[~df["row_id"].isin(done_ids)]
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    print(
        f"Feature store '{store_dir}': {len(done_ids):,} rows already "
        f"done, {len(remaining):,} remaining (of {len(df):,} total)"
    )

    if len(remaining) == 0:
        print("Nothing to do.")
        return

    sequences = load_sequences(remaining, seq_col)
    labels = load_intensities(remaining)
    charge = load_charge(remaining)
    energy = load_energy(remaining)
    row_ids = remaining["row_id"].to_numpy()
    ptm_types = remaining["ptm_type"].to_numpy() if has_ptm_type else None

    buffer, index_rows = [], []

    def flush():
        nonlocal buffer, chunk_id, index_rows
        if not buffer:
            return
        save_chunk(buffer, store_dir, chunk_id)
        pd.DataFrame(index_rows, columns=["row_id", "chunk_id", "offset"]).to_csv(
            index_path, mode="a", header=False, index=False
        )
        chunk_id += 1
        buffer = []
        index_rows = []

    n = len(remaining)
    for i in range(n):
        if i % BATCH_SIZE == 0:
            print(f"  featurizing {i}/{n}")

        try:
            data = aa_ptm_to_pyg(
                sequences[i],
                charge_ohe=charge[i],
                energy=energy[i],
                y=labels[i],
                with_position=with_position,
            )
            data.row_id = int(row_ids[i])
            if has_ptm_type:
                data.ptm_type = str(ptm_types[i])

            index_rows.append((int(row_ids[i]), chunk_id, len(buffer)))
            buffer.append(data)

        except Exception as exc:
            print(f"[Sequence ERROR] row_id={row_ids[i]}: {exc}")

        if len(buffer) >= CHUNK_SIZE:
            flush()

    flush()
    print(f"Feature store DONE: {store_dir}")


def gather_rows_into_dataset(row_ids, store_dir, out_dir):
    """
    Build a split-specific chunked dataset directory (`chunk_i.pt` +
    `meta.txt`, same format `compute_feature_store`/`save_chunk`
    produce) by copying already-computed Data objects for `row_ids` out
    of the feature store at `store_dir`.

    This performs NO feature recomputation: only torch.load/torch.save
    of tensors that `compute_feature_store` already built once.
    """
    os.makedirs(out_dir, exist_ok=True)
    index = pd.read_csv(os.path.join(store_dir, "row_index.csv"))

    wanted = set(int(r) for r in row_ids)
    index = index[index["row_id"].isin(wanted)]

    missing = wanted - set(index["row_id"].tolist())
    if missing:
        print(
            f"WARNING: {len(missing):,} row_id(s) requested for {out_dir} "
            f"were never successfully featurized (dropped upstream, see "
            f"'[Sequence ERROR]' logs) and will be skipped."
        )

    meta_path = os.path.join(out_dir, "meta.txt")
    open(meta_path, "w").close()

    out_chunk_id = 0
    buffer = []

    for src_chunk_id, group in index.sort_values(["chunk_id", "offset"]).groupby("chunk_id"):
        chunk_path = os.path.join(store_dir, f"chunk_{src_chunk_id}.pt")
        chunk_data = torch.load(chunk_path, weights_only=False)

        for offset in group["offset"]:
            buffer.append(chunk_data[offset])
            if len(buffer) >= CHUNK_SIZE:
                save_chunk(buffer, out_dir, out_chunk_id)
                buffer = []
                out_chunk_id += 1

    if buffer:
        save_chunk(buffer, out_dir, out_chunk_id)

    n_written = len(index)
    print(f"Gathered {n_written:,} graphs (of {len(wanted):,} requested) -> {out_dir}")


def build_all_sources(sources, pride_root, csv_dir, builder_fn, force=False):
    """Run `builder_fn` (build_ptm_type_csv / build_unmod_csv) over every
    entry in `sources`, returning the list of produced per-source CSV
    paths."""
    os.makedirs(csv_dir, exist_ok=True)
    csv_paths = []
    for entry in sources:
        print(f"\n{'=' * 60}\nSOURCE: {entry['folder']}\n{'=' * 60}")
        out_csv = os.path.join(csv_dir, entry["folder"] + ".csv")
        builder_fn(entry, pride_root, out_csv, force=force)
        csv_paths.append(out_csv)
    return csv_paths
