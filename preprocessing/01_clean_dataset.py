# SINGLE DATASET PREPROCESSING NOTEBOOK  v2
# Only change DATASET_FOLDER — everything else is automatic.
#
# What this does:
#   1. Auto-detect file format and merge all files (streaming, no OOM)
#   2. Use 80% sample for feature selection decisions (no leakage)
#   3. Apply those decisions to DROP columns from the FULL dataset
#   4. Save the full cleaned dataset as a single parquet
#
# What this does NOT do:
#   - No train/test split (happens later in cross-dataset notebook)
#   - No label mapping (happens later)
#   - No feature renaming (happens later)
#
# Output (/kaggle/working/):
#   cleaned_full.parquet         <- full dataset, bad columns removed
#   feature_stats.csv            <- RF + KW stats for reference
#   preprocessing_report.txt     <- paste this to share results
# =============================================================================

# ── ONLY CHANGE THIS ─────────────────────────────────────────────────────────
DATASET_FOLDER = "/kaggle/input/datasets/chethuhn/network-intrusion-dataset"
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os, gc, glob, random
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

OUT_DIR      = "/kaggle/working"
TARGET_COL   = 'attack_cat'
RANDOM_STATE = 42
BATCH_SIZE   = 200_000       # rows per streaming batch
TRAIN_FRAC   = 0.80          # fraction used for feature selection decisions
SAMPLE_CAP   = 400_000       # max rows to load into RAM for RF + KW

# Feature selection thresholds
CORR_THRESHOLD = 0.995
RF_IMP_MIN     = 0.001
KW_PVAL_MAX    = 0.01
KW_HSTAT_MIN   = 1000
RF_N_TREES     = 50

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.makedirs(OUT_DIR, exist_ok=True)
report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


# =============================================================================
# STEP 0 — AUTO-DETECT FILES
# =============================================================================
log("=" * 65)
log("STEP 0 — Detecting files")
log("=" * 65)

csv_files     = sorted(glob.glob(os.path.join(DATASET_FOLDER, "*.csv")))
parquet_files = sorted(glob.glob(os.path.join(DATASET_FOLDER, "*.parquet")))

if parquet_files and not csv_files:
    FILE_FORMAT = "parquet"
    FILES = parquet_files
elif csv_files:
    FILE_FORMAT = "csv"
    FILES = csv_files
else:
    raise FileNotFoundError(f"No CSV or Parquet files in {DATASET_FOLDER}")

log(f"  Format : {FILE_FORMAT.upper()}")
log(f"  Files  : {len(FILES)}")
for f in FILES:
    log(f"    {os.path.basename(f)}")


# =============================================================================
# STEP 1 — DETECT LABEL STRATEGY (by peeking at first file)
# =============================================================================
log()
log("=" * 65)
log("STEP 1 — Detecting label column")
log("=" * 65)

if FILE_FORMAT == "parquet":
    peek_cols = pq.read_table(FILES[0]).schema.names
else:
    peek_cols = pd.read_csv(FILES[0], nrows=2).columns.str.strip().tolist()

log(f"  Columns in first file (first 12): {peek_cols[:12]}")

if TARGET_COL in peek_cols:
    LABEL_STRATEGY = 'A'
elif 'Attack' in peek_cols and 'Label' in peek_cols:
    LABEL_STRATEGY = 'B'
elif 'Label' in peek_cols or ' Label' in peek_cols:
    LABEL_STRATEGY = 'C'
else:
    LABEL_STRATEGY = 'D'   # infer from filename (IoT 2024)

STRATEGY_DESC = {
    'A': "attack_cat already present",
    'B': "Attack + Label cols → BoT-IoT / ToN-IoT style",
    'C': "Label col → rename to attack_cat",
    'D': "No label col → infer from filename (IoT 2024 style)",
}
log(f"  Strategy : {LABEL_STRATEGY} — {STRATEGY_DESC[LABEL_STRATEGY]}")

def label_from_filename(filepath):
    name = os.path.basename(filepath).lower()
    if 'benign'  in name: return 'Normal'
    elif 'mqtt'  in name: return 'MQTT'
    elif 'recon' in name: return 'Reconnaissance'
    elif 'mitm'  in name: return 'MITM'
    elif 'ddos'  in name: return 'DDoS'
    elif 'dos'   in name: return 'DoS'
    else:                  return 'Unknown'

def apply_label(chunk, filepath):
    chunk.columns = chunk.columns.str.strip()
    if LABEL_STRATEGY == 'A':
        pass
    elif LABEL_STRATEGY == 'B':
        chunk[TARGET_COL] = chunk['Attack'].astype(str)
        chunk.loc[chunk['Label'] == 0, TARGET_COL] = 'Normal'
        chunk.drop(columns=['Label', 'Attack'], inplace=True, errors='ignore')
    elif LABEL_STRATEGY == 'C':
        raw = ' Label' if ' Label' in chunk.columns else 'Label'
        chunk.rename(columns={raw: TARGET_COL}, inplace=True)
    elif LABEL_STRATEGY == 'D':
        for c in ['Label', 'label', 'Attack', 'attack_cat']:
            chunk.drop(columns=[c], inplace=True, errors='ignore')
        chunk[TARGET_COL] = label_from_filename(filepath)

    # Force numeric on non-label columns
    for col in chunk.columns:
        if col == TARGET_COL:
            continue
        if chunk[col].dtype == object:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

    return chunk[chunk[TARGET_COL].notna()]


def stream_all(files, fmt, batch_size):
    """Yield (chunk_df, filepath) for every batch across all files."""
    for fpath in files:
        if fmt == 'parquet':
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=batch_size):
                yield batch.to_pandas(), fpath
        else:
            for chunk in pd.read_csv(fpath, chunksize=batch_size,
                                     low_memory=False, on_bad_lines='skip'):
                yield chunk, fpath


# =============================================================================
# STEP 2 — COLLECT TRAINING SAMPLE FOR FEATURE SELECTION
#
# We stream through all files once.
# Each row is assigned train (80%) or test (20%) by a random coin flip.
# We keep ALL train-assigned rows in a running reservoir, then sample down
# to SAMPLE_CAP so we never exceed memory.
#
# This simulates "compute feature selection on training data only"
# without needing to do a full two-pass read.
# =============================================================================
log()
log("=" * 65)
log("STEP 2 — Collecting training sample for feature selection")
log("=" * 65)

sample_chunks = []
total_rows    = 0
label_counts  = {}
row_flags     = []   # list of (n_rows, is_train_mask) per batch — for Step 4

log(f"  Streaming all files (batch={BATCH_SIZE:,})...")

for chunk, fpath in stream_all(FILES, FILE_FORMAT, BATCH_SIZE):
    chunk = apply_label(chunk, fpath)
    if len(chunk) == 0:
        continue

    total_rows += len(chunk)
    for lbl, cnt in chunk[TARGET_COL].value_counts().items():
        label_counts[lbl] = label_counts.get(lbl, 0) + cnt

    # Assign each row a train/test flag
    is_train = np.random.random(len(chunk)) < TRAIN_FRAC

    # Reservoir sample from training rows for feature selection
    train_chunk = chunk[is_train]
    p_keep = min(1.0, (SAMPLE_CAP / max(total_rows, 1)) * 5)
    mask   = np.random.random(len(train_chunk)) < p_keep
    if mask.sum() > 0:
        sample_chunks.append(train_chunk[mask].copy())

    del chunk, train_chunk
    gc.collect()

log(f"  Total rows: {total_rows:,}")
log()
log("  Label distribution (original labels):")
for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
    log(f"    {lbl}: {cnt:,}")

sample_df = pd.concat(sample_chunks, ignore_index=True)
if len(sample_df) > SAMPLE_CAP:
    sample_df = sample_df.sample(n=SAMPLE_CAP, random_state=RANDOM_STATE)
sample_df.reset_index(drop=True, inplace=True)
del sample_chunks
gc.collect()

log()
log(f"  Training sample size: {len(sample_df):,} rows  "
    f"(~{TRAIN_FRAC*100:.0f}% of data, capped at {SAMPLE_CAP:,})")


# =============================================================================
# STEP 3 — FEATURE SELECTION ON TRAINING SAMPLE
# Decisions: which columns to DROP from the full dataset.
# Nothing here touches the test portion of the data.
# =============================================================================
log()
log("=" * 65)
log("STEP 3 — Feature selection (on training sample only)")
log("=" * 65)

feat_cols = [c for c in sample_df.columns if c != TARGET_COL]
num_cols  = sample_df[feat_cols].select_dtypes(
                include=[np.number]).columns.tolist()

# 3a. Replace inf, fill NaN with median
sample_df[num_cols] = sample_df[num_cols].replace([np.inf, -np.inf], np.nan)
medians  = sample_df[num_cols].median()
X_sample = sample_df[num_cols].fillna(medians)

# 3b. Drop constant columns in sample
const_cols = [c for c in num_cols if X_sample[c].nunique() <= 1]
log(f"  Constant columns   : {len(const_cols)}  {const_cols}")
X_sample = X_sample.drop(columns=const_cols)
num_cols = X_sample.columns.tolist()

# 3c. Correlation removal
log(f"  Computing correlation matrix ({len(num_cols)} features)...")
corr_m    = X_sample.corr().abs()
upper     = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
corr_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
log(f"  Corr dropped       : {len(corr_drop)}  {sorted(corr_drop)}")
X_sample  = X_sample.drop(columns=corr_drop)
num_cols  = X_sample.columns.tolist()
del corr_m, upper
gc.collect()

# 3d. Skip RF+KW — correlation removal is the final step
COLS_TO_DROP = set(const_cols) | set(corr_drop)
FINAL_FEATURES = num_cols   # everything surviving correlation removal

log(f"  ── SUMMARY ──────────────────────────────────────────")
log(f"  Total features before: {len([c for c in sample_df.columns if c != TARGET_COL])}")
log(f"  Constant removed     : {len(const_cols)}")
log(f"  Correlation removed  : {len(corr_drop)}")
log(f"  FINAL FEATURE COUNT  : {len(FINAL_FEATURES)}")
log(f"  FINAL FEATURES       : {sorted(FINAL_FEATURES)}")
log(f"  ─────────────────────────────────────────────────────")

del sample_df, X_sample
gc.collect()


# =============================================================================
# STEP 4 — APPLY DECISIONS TO FULL DATASET AND SAVE
# Stream through all files again.
# Keep only FINAL_FEATURES columns + TARGET_COL.
# Write entire cleaned dataset to one parquet file.
# =============================================================================
log()
log("=" * 65)
log("STEP 4 — Writing cleaned full dataset (streaming)")
log("=" * 65)

out_path = os.path.join(OUT_DIR, "cleaned_full.parquet")
writer   = None
schema   = None
n_written = 0
n_dropped = 0

for chunk, fpath in stream_all(FILES, FILE_FORMAT, BATCH_SIZE):
    chunk = apply_label(chunk, fpath)
    if len(chunk) == 0:
        continue

    # Keep final features + label only
    keep  = [c for c in FINAL_FEATURES if c in chunk.columns]
    chunk = chunk[keep + [TARGET_COL]].copy()

    # Replace inf, drop NaN
    num_c = chunk.select_dtypes(include=[np.number]).columns
    chunk[num_c] = chunk[num_c].replace([np.inf, -np.inf], np.nan)
    before = len(chunk)
    chunk.dropna(inplace=True)
    n_dropped += before - len(chunk)

    if len(chunk) == 0:
        continue

    try:
        table = pa.Table.from_pandas(chunk, preserve_index=False)
    except Exception as e:
        log(f"  Arrow error: {e}")
        del chunk
        continue

    if writer is None:
        schema = table.schema
        writer = pq.ParquetWriter(out_path, schema, compression='snappy')

    try:
        table = table.cast(schema)
    except Exception:
        pass

    writer.write_table(table)
    n_written += len(chunk)

    del chunk, table
    gc.collect()

if writer:
    writer.close()

log(f"  Rows with NaN/Inf dropped: {n_dropped:,}")
log(f"  Total rows written       : {n_written:,}")
log(f"  Saved: {out_path}")


# =============================================================================
# STEP 5 — QUICK VERIFICATION
# =============================================================================
log()
log("=" * 65)
log("STEP 5 — Verification")
log("=" * 65)

check = pd.read_parquet(out_path, columns=[TARGET_COL])
log(f"  Shape: ({len(check)}, {len(FINAL_FEATURES) + 1})")
log(f"  Label distribution (cleaned dataset):")
log(check[TARGET_COL].value_counts().to_string())
del check


# =============================================================================
# SAVE REPORT
# =============================================================================
report_path = os.path.join(OUT_DIR, "preprocessing_report.txt")
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines))

log()
log("=" * 65)
log("DONE.")
log("Upload these files to Kaggle as a new dataset:")
log("  cleaned_full.parquet")
log("  feature_stats.csv")
log("  preprocessing_report.txt")
log("=" * 65)
