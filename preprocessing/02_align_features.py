# FEATURE ALIGNMENT NOTEBOOK
# Datasets: IDS2017, IDS2018, BOT_IoT
#
# Inputs : 3 cleaned_full.parquet files (one per dataset, already preprocessed)
# Outputs: 3 aligned parquet files with 42 canonical features + attack_cat
#          feature_intersection.csv   — 42 canonical feature names
#
# Feature counts after dataset-specific preprocessing:
#   IDS2017  : 55 features
#   IDS2018  : 53 features
#   BOT_IoT  : 56 features
#   Intersection: 42 features  (confirmed by manual cross-check below)
# =============================================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS  — update dataset names to match your Kaggle input datasets
# =============================================================================
IDS2017_PATH = "/kaggle/input/datasets/tarunperisetti/ids2017-cleaned/cleaned_full.parquet"
IDS2018_PATH = "/kaggle/input/datasets/tarunperisetti/ids2018-cleaned/cleaned_full.parquet"
BOTIOT_PATH  = "/kaggle/input/datasets/tarunperisetti/bot-cleaned/cleaned_full.parquet"
OUT_DIR      = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = 'attack_cat'
NORMAL = 'Normal'

# =============================================================================
# COLUMN NAME MAPPINGS  (raw preprocessed column → canonical snake_case)
# Only includes columns that survive into the 42-feature intersection.
# Columns in each dataset that are NOT in the intersection are silently dropped.
# =============================================================================

# ── IDS2017 column → canonical ───────────────────────────────────────────────
MAP_2017 = {
    'ACK Flag Count'           : 'ack_flag_count',
    'Active Max'               : 'active_max',
    'Active Mean'              : 'active_mean',
    'Active Min'               : 'active_min',
    'Active Std'               : 'active_std',
    'Bwd IAT Max'              : 'bwd_iat_max',
    'Bwd IAT Mean'             : 'bwd_iat_mean',
    'Bwd IAT Min'              : 'bwd_iat_min',
    'Bwd IAT Std'              : 'bwd_iat_std',
    'Bwd IAT Total'            : 'bwd_iat_total',
    'Bwd Packet Length Max'    : 'bwd_pkt_len_max',
    'Bwd Packet Length Mean'   : 'bwd_pkt_len_mean',
    'Bwd Packet Length Min'    : 'bwd_pkt_len_min',
    'Bwd Packet Length Std'    : 'bwd_pkt_len_std',
    'Bwd Packets/s'            : 'bwd_pkts_s',
    'Down/Up Ratio'            : 'down_up_ratio',
    'FIN Flag Count'           : 'fin_flag_count',
    'Flow Bytes/s'             : 'flow_bytes_s',
    'Flow Duration'            : 'flow_duration',
    'Flow IAT Max'             : 'flow_iat_max',
    'Flow IAT Mean'            : 'flow_iat_mean',
    'Flow IAT Std'             : 'flow_iat_std',
    'Flow Packets/s'           : 'flow_pkts_s',
    'Fwd IAT Min'              : 'fwd_iat_min',
    'Fwd IAT Std'              : 'fwd_iat_std',
    'Fwd Packet Length Max'    : 'fwd_pkt_len_max',
    'Fwd Packet Length Mean'   : 'fwd_pkt_len_mean',
    'Fwd Packet Length Min'    : 'fwd_pkt_len_min',
    'Fwd Packet Length Std'    : 'fwd_pkt_len_std',
    'Fwd Packets/s'            : 'fwd_pkts_s',
    'Idle Max'                 : 'idle_max',
    'Idle Mean'                : 'idle_mean',
    'Idle Std'                 : 'idle_std',
    'Init_Win_bytes_backward'  : 'init_bwd_win_bytes',
    'Max Packet Length'        : 'pkt_len_max',
    'Packet Length Mean'       : 'pkt_len_mean',
    'Min Packet Length'        : 'pkt_len_min',
    'Packet Length Std'        : 'pkt_len_std',
    'Packet Length Variance'   : 'pkt_len_var',
    'RST Flag Count'           : 'rst_flag_count',
    'Total Fwd Packets'        : 'tot_fwd_pkts',
    'Total Length of Fwd Packets': 'totlen_fwd_pkts',
    TARGET                     : TARGET,
}

# ── IDS2018 column → canonical ───────────────────────────────────────────────
MAP_2018 = {
    'ACK Flag Cnt'    : 'ack_flag_count',
    'Active Max'      : 'active_max',
    'Active Mean'     : 'active_mean',
    'Active Min'      : 'active_min',
    'Active Std'      : 'active_std',
    'Bwd IAT Max'     : 'bwd_iat_max',
    'Bwd IAT Mean'    : 'bwd_iat_mean',
    'Bwd IAT Min'     : 'bwd_iat_min',
    'Bwd IAT Std'     : 'bwd_iat_std',
    'Bwd IAT Tot'     : 'bwd_iat_total',
    'Bwd Pkt Len Max' : 'bwd_pkt_len_max',
    'Bwd Pkt Len Mean': 'bwd_pkt_len_mean',
    'Bwd Pkt Len Min' : 'bwd_pkt_len_min',
    'Bwd Pkt Len Std' : 'bwd_pkt_len_std',
    'Bwd Pkts/s'      : 'bwd_pkts_s',
    'Down/Up Ratio'   : 'down_up_ratio',
    'FIN Flag Cnt'    : 'fin_flag_count',
    'Flow Byts/s'     : 'flow_bytes_s',
    'Flow Duration'   : 'flow_duration',
    'Flow IAT Max'    : 'flow_iat_max',
    'Flow IAT Mean'   : 'flow_iat_mean',
    'Flow IAT Std'    : 'flow_iat_std',
    'Flow Pkts/s'     : 'flow_pkts_s',
    'Fwd IAT Min'     : 'fwd_iat_min',
    'Fwd IAT Std'     : 'fwd_iat_std',
    'Fwd Pkt Len Max' : 'fwd_pkt_len_max',
    'Fwd Pkt Len Mean': 'fwd_pkt_len_mean',
    'Fwd Pkt Len Min' : 'fwd_pkt_len_min',
    'Fwd Pkt Len Std' : 'fwd_pkt_len_std',
    'Fwd Pkts/s'      : 'fwd_pkts_s',
    'Idle Max'        : 'idle_max',
    'Idle Mean'       : 'idle_mean',
    'Idle Std'        : 'idle_std',
    'Init Bwd Win Byts': 'init_bwd_win_bytes',
    'Pkt Len Max'     : 'pkt_len_max',
    'Pkt Len Mean'    : 'pkt_len_mean',
    'Pkt Len Min'     : 'pkt_len_min',
    'Pkt Len Std'     : 'pkt_len_std',
    'Pkt Len Var'     : 'pkt_len_var',
    'RST Flag Cnt'    : 'rst_flag_count',
    'Tot Fwd Pkts'    : 'tot_fwd_pkts',
    'TotLen Fwd Pkts' : 'totlen_fwd_pkts',
    TARGET            : TARGET,
}

# ── BOT_IoT column → canonical ───────────────────────────────────────────────
MAP_BOTIOT = {
    'ACK Flag Count'         : 'ack_flag_count',
    'Active Max'             : 'active_max',
    'Active Mean'            : 'active_mean',
    'Active Min'             : 'active_min',
    'Active Std'             : 'active_std',
    'Bwd IAT Max'            : 'bwd_iat_max',
    'Bwd IAT Mean'           : 'bwd_iat_mean',
    'Bwd IAT Min'            : 'bwd_iat_min',
    'Bwd IAT Std'            : 'bwd_iat_std',
    'Bwd IAT Total'          : 'bwd_iat_total',
    'Bwd Packet Length Max'  : 'bwd_pkt_len_max',
    'Bwd Packet Length Mean' : 'bwd_pkt_len_mean',
    'Bwd Packet Length Min'  : 'bwd_pkt_len_min',
    'Bwd Packet Length Std'  : 'bwd_pkt_len_std',
    'Bwd Packets/s'          : 'bwd_pkts_s',
    'Down/Up Ratio'          : 'down_up_ratio',
    'FIN Flag Count'         : 'fin_flag_count',
    'Flow Bytes/s'           : 'flow_bytes_s',
    'Flow Duration'          : 'flow_duration',
    'Flow IAT Max'           : 'flow_iat_max',
    'Flow IAT Mean'          : 'flow_iat_mean',
    'Flow IAT Std'           : 'flow_iat_std',
    'Flow Packets/s'         : 'flow_pkts_s',
    'Fwd IAT Min'            : 'fwd_iat_min',
    'Fwd IAT Std'            : 'fwd_iat_std',
    'Fwd Packet Length Max'  : 'fwd_pkt_len_max',
    'Fwd Packet Length Mean' : 'fwd_pkt_len_mean',
    'Fwd Packet Length Min'  : 'fwd_pkt_len_min',
    'Fwd Packet Length Std'  : 'fwd_pkt_len_std',
    'Fwd Packets Length Total': 'totlen_fwd_pkts',
    'Fwd Packets/s'          : 'fwd_pkts_s',
    'Idle Max'               : 'idle_max',
    'Idle Mean'              : 'idle_mean',
    'Idle Std'               : 'idle_std',
    'Init Bwd Win Bytes'     : 'init_bwd_win_bytes',
    'Packet Length Max'      : 'pkt_len_max',
    'Packet Length Mean'     : 'pkt_len_mean',
    'Packet Length Min'      : 'pkt_len_min',
    'Packet Length Std'      : 'pkt_len_std',
    'Packet Length Variance' : 'pkt_len_var',
    'RST Flag Count'         : 'rst_flag_count',
    'Total Fwd Packets'      : 'tot_fwd_pkts',
    TARGET                   : TARGET,
}

# ── 42 canonical feature names (intersection) ────────────────────────────────
FEATURES_42 = [
    'ack_flag_count', 'active_max', 'active_mean', 'active_min', 'active_std',
    'bwd_iat_max', 'bwd_iat_mean', 'bwd_iat_min', 'bwd_iat_std', 'bwd_iat_total',
    'bwd_pkt_len_max', 'bwd_pkt_len_mean', 'bwd_pkt_len_min', 'bwd_pkt_len_std',
    'bwd_pkts_s', 'down_up_ratio', 'fin_flag_count', 'flow_bytes_s',
    'flow_duration', 'flow_iat_max', 'flow_iat_mean', 'flow_iat_std',
    'flow_pkts_s', 'fwd_iat_min', 'fwd_iat_std', 'fwd_pkt_len_max',
    'fwd_pkt_len_mean', 'fwd_pkt_len_min', 'fwd_pkt_len_std', 'fwd_pkts_s',
    'idle_max', 'idle_mean', 'idle_std', 'init_bwd_win_bytes',
    'pkt_len_max', 'pkt_len_mean', 'pkt_len_min', 'pkt_len_std', 'pkt_len_var',
    'rst_flag_count', 'tot_fwd_pkts', 'totlen_fwd_pkts',
]
N_FEATURES = len(FEATURES_42)
print(f"Canonical feature count: {N_FEATURES}")
assert N_FEATURES == 42

# =============================================================================
# LABEL NORMALISATION
# =============================================================================
def normalise_labels(df):
    """Map all benign/normal variants to 'Normal'. Keep attack labels as-is."""
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df[TARGET] = df[TARGET].str.replace(r'\.0$', '', regex=True)
    benign_vals = {
        'BENIGN', 'Benign', 'benign', 'Normal', 'normal', 'NORMAL', '0'
    }
    df.loc[df[TARGET].isin(benign_vals), TARGET] = NORMAL
    return df

# =============================================================================
# ALIGNMENT FUNCTION
# =============================================================================
def align_dataset(path, col_map, dataset_name):
    """
    Load a preprocessed parquet, rename columns to canonical names,
    select 42-feature intersection + TARGET, normalise labels.
    """
    print(f"\n{'='*60}")
    print(f"Aligning: {dataset_name}")
    print(f"{'='*60}")

    df = pd.read_parquet(path)
    print(f"  Loaded: {df.shape}")
    print(f"  Columns available: {df.shape[1]}")

    # Rename using map (only mapped columns survive)
    df = df.rename(columns=col_map)

    # Verify all 42 features are present after mapping
    missing = [f for f in FEATURES_42 if f not in df.columns]
    if missing:
        print(f"  WARNING — missing after mapping: {missing}")
    else:
        print(f"  All 42 features present after mapping.")

    # Select 42 features + label
    keep = FEATURES_42 + [TARGET]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Normalise labels
    df = normalise_labels(df)

    # Report
    print(f"  Final shape: {df.shape}")
    print(f"  Label distribution:")
    vc = df[TARGET].value_counts()
    for label, count in vc.items():
        print(f"    {label:<40}: {count:>10,}")

    # Sanity: no NaN in features
    nan_count = df[FEATURES_42].isna().sum().sum()
    print(f"  NaN in features: {nan_count}")

    return df

# =============================================================================
# LOAD AND ALIGN
# =============================================================================
df_2017  = align_dataset(IDS2017_PATH, MAP_2017,  'IDS2017')
df_2018  = align_dataset(IDS2018_PATH, MAP_2018,  'IDS2018')
df_botiot = align_dataset(BOTIOT_PATH, MAP_BOTIOT, 'BOT_IoT')

# =============================================================================
# SAVE ALIGNED PARQUETS
# =============================================================================
out_2017   = os.path.join(OUT_DIR, 'aligned_IDS2017.parquet')
out_2018   = os.path.join(OUT_DIR, 'aligned_IDS2018.parquet')
out_botiot = os.path.join(OUT_DIR, 'aligned_BOT_IoT.parquet')

df_2017.to_parquet(out_2017,   index=False)
df_2018.to_parquet(out_2018,   index=False)
df_botiot.to_parquet(out_botiot, index=False)

print(f"\nSaved aligned parquets:")
print(f"  {out_2017}  ({len(df_2017):,} rows)")
print(f"  {out_2018}  ({len(df_2018):,} rows)")
print(f"  {out_botiot}  ({len(df_botiot):,} rows)")

# Save canonical feature list
pd.Series(FEATURES_42, name='feature').to_csv(
    os.path.join(OUT_DIR, 'feature_intersection.csv'), index=True)
print(f"  feature_intersection.csv saved.")
