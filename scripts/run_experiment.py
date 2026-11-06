# =============================================================================
# run_experiment.py
# Two-stage hybrid NIDS benchmark: one source dataset and one seed per run.
#   Stage-1: calibrated RF or XGBoost (binary, Platt scaling, threshold tau1).
#   Stage-2: AE / Isolation Forest / VAE fitted on source-normal flows only,
#            fixed or tuned by GA, PSO or DE (threshold tau2).
# Every trained configuration is evaluated within the source dataset and on
# the two other datasets (zero-shot, no retraining or recalibration).
#
# Usage: set SOURCE_DATASET ('IDS2017', 'IDS2018' or 'BOT_IoT') and SEED
# (42, 7 or 2026) below and run in a fresh process. The seed controls the
# sampling, the splits, network initialisation and the optimiser populations.
# Outputs (one CSV set per source and seed): results, routing, auc, hist,
# convergence, hparams, thresholds, timing, infer, size.
# Dataset paths are the Kaggle input paths listed in DATASETS; edit them if
# the CSV files are stored elsewhere (see README.md for the sources).
# TAU_FLOOR = 0.30 is a lower bound on the Stage-1 threshold tau1.
# =============================================================================

import pandas as pd
import numpy as np
import os, gc, warnings, time, ctypes
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
for _gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

# =============================================================================
# >>> SET THIS BEFORE EACH RUN <<<
# =============================================================================
SOURCE_DATASET = 'IDS2017'   # change to 'IDS2018' then 'BOT_IoT' on next runs
SEEDS = [42]                 # ONE seed per Kaggle session (fresh process); run 42, 7, 2026 in separate sessions

# =============================================================================
# PATHS
# =============================================================================
ALIGNED_DIR = "/kaggle/input/datasets/tarunperisetti/3-dataset-study"
OUT_DIR     = "/kaggle/working"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ['IDS2017', 'IDS2018', 'BOT_IoT']
assert SOURCE_DATASET in DATASETS

# Per-dataset row caps to prevent OOM when loading large datasets.
# IDS2018: 16.1M rows -> capped at 3M (still larger than IDS2017's 2.8M).
# BOT_IoT: 11.5M rows but only 38,435 normal rows -> capped at 4M, with the
#          Normal class fully protected from downsampling (see load_aligned)
#          since uniform sampling would shrink an already-tiny class further.
# IDS2017: 2.8M rows -- no cap needed, confirmed stable across multiple runs.
DATASET_ROW_CAP = {
    'IDS2017': None,
    'IDS2018': 3_000_000,
    'BOT_IoT': 4_000_000,
}
# Class label that must never be downsampled when capping (None = no protection)
DATASET_PROTECT_LABEL = {
    'IDS2017': None,
    'IDS2018': None,
    'BOT_IoT': 'Normal',
}

TARGET   = 'attack_cat'
NORMAL   = 'Normal'
SEED     = 42
TAU_FLOOR = 0.30   # minimum Stage-1 deferral threshold -- see header note

np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

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
assert N_FEATURES == 42

# =============================================================================
# HYPERPARAMETERS — STAGE-1 (fixed, not optimised)
# =============================================================================
RF_TREES    = 200; RF_DEPTH = 15; RF_MIN_LEAF = 5
AE_EPOCHS   = 50;  AE_BATCH = 2048; AE_PATIENCE = 5
SCORE_BATCH = AE_BATCH
TAU_RF_STEPS  = np.linspace(0.01, 0.99, 99)
TAU_ANOM_PCTS = np.arange(80, 99, 1)
BAL_CAP = 10_000

AE_LATENT_FIXED   = 16;  AE_NOISE_FIXED = 0.02; AE_LR_FIXED = 1e-3
ISOF_TREES_FIXED  = 300; ISOF_SAMPLES_FIXED = 50_000
VAE_LATENT_FIXED  = 16;  VAE_LR_FIXED = 1e-3

OPT_POP       = 5
OPT_ITER      = 8
OPT_EPOCHS    = 15
OPT_PATIENCE  = 2
OPT_BAL_CAP   = 5_000
OPT_NOR_CAP   = 200_000
FINAL_NOR_CAP = 500_000

LATENT_DIM_OPTIONS = [8, 12, 16, 20, 24, 32]
ISOF_N_EST_OPTIONS = [100, 200, 300, 400, 500]
ISOF_MS_OPTIONS    = [10_000, 25_000, 50_000, 75_000, 100_000]
AE_NDIM   = 3
VAE_NDIM  = 2
ISOF_NDIM = 3

def decode_latent(v):
    idx = int(v * (len(LATENT_DIM_OPTIONS) - 1 + 0.999))
    return LATENT_DIM_OPTIONS[np.clip(idx, 0, len(LATENT_DIM_OPTIONS)-1)]

def decode_ae_hparams(v):
    return (int(decode_latent(v[0])),
            float(0.005 + v[1] * 0.095),
            float(10 ** (-4.0 + v[2] * 2.0)))

def decode_vae_hparams(v):
    return (int(decode_latent(v[0])),
            float(10 ** (-4.0 + v[1] * 2.0)))

def decode_isof_hparams(v):
    n_idx = int(v[0] * (len(ISOF_N_EST_OPTIONS) - 1 + 0.999))
    n_est = ISOF_N_EST_OPTIONS[np.clip(n_idx, 0, len(ISOF_N_EST_OPTIONS)-1)]
    m_idx = int(v[1] * (len(ISOF_MS_OPTIONS) - 1 + 0.999))
    ms    = ISOF_MS_OPTIONS[np.clip(m_idx, 0, len(ISOF_MS_OPTIONS)-1)]
    mf    = float(0.5 + v[2] * 0.5)
    return (int(n_est), int(ms), mf)

# =============================================================================
# HELPERS
# =============================================================================
def normalise_labels(df):
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df[TARGET] = df[TARGET].str.replace(r'\.0$', '', regex=True)
    for val in ['0','Normal','BENIGN','Benign','benign','normal','NORMAL']:
        df.loc[df[TARGET] == val, TARGET] = NORMAL
    return df

def load_aligned(name, cap=None, protect_label=None):
    """
    Load an aligned parquet. If cap is set, stream in 200k-row batches and
    keep a proportional sample (peak RAM ~ one batch + accumulated sample,
    not the full file -- critical for IDS2018 at 2.77 GB raw).

    If protect_label is set (e.g. 'Normal' for BOT_IoT), rows of that class
    are NEVER downsampled -- only other classes are proportionally reduced
    to fit the budget. This matters when the protected class is already a
    tiny minority (BOT_IoT: 38,435 normal rows out of 11.5M); uniform
    sampling would shrink it further and starve Stage-2 AE/VAE training.
    """
    import pyarrow.parquet as pq

    path = os.path.join(ALIGNED_DIR, f"aligned_{name}.parquet")
    cols = FEATURES_42 + [TARGET]

    if cap is None:
        df = pd.read_parquet(path, columns=cols)
        df = normalise_labels(df)
        return df.dropna(subset=FEATURES_42).reset_index(drop=True)

    pf         = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows

    protected_count = 0
    if protect_label is not None:
        for batch in pf.iter_batches(batch_size=200_000, columns=[TARGET]):
            s = batch.to_pandas()[TARGET].astype(str).str.strip()
            s = s.str.replace(r'\.0$', '', regex=True)
            protected_count += int(s.isin(
                ['0','Normal','BENIGN','Benign','benign','normal','NORMAL']
            ).sum())

    other_count      = max(total_rows - protected_count, 1)
    other_budget      = max(cap - protected_count, 0)
    keep_rate_other   = min(1.0, other_budget / other_count)

    print(f"  Streaming {name}: total={total_rows:,}  "
          f"protected={protected_count:,}  other={other_count:,}  "
          f"keep_rate_other={keep_rate_other:.3f}")

    rng       = np.random.default_rng(SEED)
    batch_dfs = []
    for batch in pf.iter_batches(batch_size=200_000, columns=cols):
        bdf = normalise_labels(batch.to_pandas())
        if protect_label is not None:
            is_prot = (bdf[TARGET] == NORMAL)
            prot_part  = bdf[is_prot]
            other_part = bdf[~is_prot]
        else:
            prot_part  = bdf.iloc[0:0]
            other_part = bdf

        if keep_rate_other < 1.0 and len(other_part) > 0:
            n_keep = max(1, int(round(len(other_part) * keep_rate_other)))
            if n_keep < len(other_part):
                other_part = other_part.sample(
                    n=n_keep, random_state=int(rng.integers(1_000_000)))

        bdf2 = pd.concat([prot_part, other_part], ignore_index=True)
        batch_dfs.append(bdf2)
        del bdf, prot_part, other_part, bdf2; gc.collect()

    df = pd.concat(batch_dfs, ignore_index=True)
    del batch_dfs; gc.collect()
    print(f"  Sampled rows: {len(df):,}")
    return df.dropna(subset=FEATURES_42).reset_index(drop=True)

def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1)); fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0)); tn = np.sum((y_true==0) & (y_pred==0))
    recall = tp / (tp + fn + 1e-9);  far = fp / (fp + tn + 1e-9)
    prec   = tp / (tp + fp + 1e-9);  f1  = 2*prec*recall / (prec + recall + 1e-9)
    return round(float(recall),4), round(float(far),4), round(float(f1),4)

def compute_metrics_full(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = float(np.sum((y_true==1) & (y_pred==1))); fp = float(np.sum((y_true==0) & (y_pred==1)))
    fn = float(np.sum((y_true==1) & (y_pred==0))); tn = float(np.sum((y_true==0) & (y_pred==0)))
    recall = tp / (tp + fn + 1e-9); far = fp / (fp + tn + 1e-9)
    prec   = tp / (tp + fp + 1e-9); f1  = 2*prec*recall / (prec + recall + 1e-9)
    ba     = 0.5 * (recall + (1.0 - far))
    den    = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc    = (tp*tn - fp*fn) / den if den > 0 else 0.0
    return {'Recall': round(recall,4), 'FAR': round(far,4), 'F1': round(f1,4),
            'Precision': round(prec,4), 'BalAcc': round(ba,4), 'MCC': round(float(mcc),4),
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)}

def balanced_idx(y, cap=BAL_CAP):
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    n    = min(len(idx0), len(idx1), cap)
    if n == 0: return np.arange(len(y))
    rng  = np.random.default_rng(SEED)
    return np.concatenate([rng.choice(idx0, n, replace=False),
                           rng.choice(idx1, n, replace=False)])

def best_tau_rf(probs, y_true, candidates):
    idx = balanced_idx(y_true); pb = probs[idx]; yb = y_true[idx]
    best_f1, best_tau = 0, 0.5
    for tau in candidates:
        f1 = f1_score(yb, (pb >= tau).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_tau = f1, tau
    return best_tau

def best_tau_anomaly(calib_scores, calib_labels, normal_scores, pcts):
    idx = balanced_idx(calib_labels, cap=BAL_CAP)
    sb  = calib_scores[idx]; lb = calib_labels[idx]
    best_f1 = 0; best_tau = np.percentile(normal_scores, 95)
    for p in pcts:
        tau = np.percentile(normal_scores, p)
        f1  = f1_score(lb, (sb > tau).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_tau = f1, tau
    return best_tau

def recon_score(model, X):
    return np.mean(
        (X - model.predict(X, verbose=0, batch_size=SCORE_BATCH))**2, axis=1)

def iso_score(model, X):
    return -model.decision_function(X)

def anomaly_score(model, s2_type, X):
    if s2_type == 'isof':
        return iso_score(model, X)
    return recon_score(model, X)

def _cap(X, cap):
    if len(X) > cap:
        rng = np.random.default_rng(SEED)
        return X[rng.choice(len(X), cap, replace=False)]
    return X

def _mem(tag=""):
    """Print process RSS and free RAM (GB). Diagnostic only; never raises."""
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
        free = psutil.virtual_memory().available / 1e9
        print(f"    [mem] {tag}: process={rss:.1f} GB, free={free:.1f} GB", flush=True)
    except Exception:
        pass

def _cleanup():
    keras.backend.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# =============================================================================
# MODEL BUILDERS — BASELINE (fixed hyperparameters)
# =============================================================================
def build_ae_fixed(n):
    inp = keras.Input(shape=(n,))
    x   = layers.GaussianNoise(AE_NOISE_FIXED)(inp)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(64,  activation='relu')(x)
    x   = layers.Dense(32,  activation='relu')(x)
    x   = layers.Dense(AE_LATENT_FIXED, activation='relu')(x)
    x   = layers.Dense(32,  activation='relu')(x)
    x   = layers.Dense(64,  activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n,   activation='sigmoid')(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(AE_LR_FIXED), loss='mse')
    return m

class VAEBase(keras.Model):
    def __init__(self, n, latent_dim, lr):
        super().__init__()
        self._lr = lr
        self.enc = keras.Sequential([layers.Dense(64, activation='relu'),
                                     layers.Dense(32, activation='relu')])
        self.zm  = layers.Dense(latent_dim)
        self.zlv = layers.Dense(latent_dim)
        self.dec = keras.Sequential([layers.Dense(32, activation='relu'),
                                     layers.Dense(64, activation='relu'),
                                     layers.Dense(n,  activation='sigmoid')])
    def encode(self, x):
        h = self.enc(x); return self.zm(h), self.zlv(h)
    def call(self, x):
        zm, zlv = self.encode(x)
        z = zm + tf.exp(0.5*zlv) * tf.random.normal(shape=tf.shape(zm))
        recon = self.dec(z)
        self.add_loss(
            tf.reduce_mean(tf.reduce_sum(tf.square(x - recon), axis=1)) +
            (-0.5 * tf.reduce_mean(1 + zlv - tf.square(zm) - tf.exp(zlv))))
        return recon

def build_vae_fixed(n):
    v = VAEBase(n, VAE_LATENT_FIXED, VAE_LR_FIXED)
    v.compile(optimizer=keras.optimizers.Adam(VAE_LR_FIXED))
    return v

def build_ae_custom(n, latent_dim, noise_sigma, lr):
    inp = keras.Input(shape=(n,))
    x   = layers.GaussianNoise(noise_sigma)(inp)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dense(latent_dim, activation='relu')(x)
    x   = layers.Dense(32, activation='relu')(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(n, activation='sigmoid')(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    return m

def build_vae_custom(n, latent_dim, lr):
    v = VAEBase(n, latent_dim, lr)
    v.compile(optimizer=keras.optimizers.Adam(lr))
    return v

# =============================================================================
# FITNESS FUNCTIONS
# =============================================================================
def _opt_cb():
    return [keras.callbacks.EarlyStopping(
        patience=OPT_PATIENCE, restore_best_weights=True)]

def fitness_ae(v, X_nor_s, X_cal_s, y_cal, rf_probs, tau_rf):
    ld, ns, lr = decode_ae_hparams(v)
    Xn = _cap(X_nor_s, OPT_NOR_CAP)
    m  = build_ae_custom(N_FEATURES, ld, ns, lr)
    m.fit(Xn, Xn, epochs=OPT_EPOCHS, batch_size=AE_BATCH,
          validation_split=0.1, callbacks=_opt_cb(), verbose=0)
    norm_s = recon_score(m, Xn); cal_s = recon_score(m, X_cal_s)
    prf    = (rf_probs >= tau_rf).astype(int); mrf = (prf == 0)
    best   = 0.0
    for p in TAU_ANOM_PCTS:
        tau  = np.percentile(norm_s, p)
        pred = prf.copy(); pred[mrf] = (cal_s[mrf] > tau).astype(int)
        idx  = balanced_idx(y_cal, cap=OPT_BAL_CAP)
        best = max(best, f1_score(y_cal[idx], pred[idx], zero_division=0))
    del m; _cleanup()
    return -best

def fitness_vae(v, X_nor_s, X_cal_s, y_cal, xgb_probs, tau_xgb):
    ld, lr = decode_vae_hparams(v)
    Xn     = _cap(X_nor_s, OPT_NOR_CAP)
    m      = build_vae_custom(N_FEATURES, ld, lr)
    m.fit(Xn, epochs=OPT_EPOCHS, batch_size=AE_BATCH,
          validation_split=0.1, callbacks=_opt_cb(), verbose=0)
    norm_s = recon_score(m, Xn); cal_s = recon_score(m, X_cal_s)
    pxgb   = (xgb_probs >= tau_xgb).astype(int); mxgb = (pxgb == 0)
    best   = 0.0
    for p in TAU_ANOM_PCTS:
        tau  = np.percentile(norm_s, p)
        pred = pxgb.copy(); pred[mxgb] = (cal_s[mxgb] > tau).astype(int)
        idx  = balanced_idx(y_cal, cap=OPT_BAL_CAP)
        best = max(best, f1_score(y_cal[idx], pred[idx], zero_division=0))
    del m; _cleanup()
    return -best

def fitness_isof(v, X_nor_s, X_cal_s, y_cal, rf_probs, tau_rf):
    n_est, ms, mf = decode_isof_hparams(v)
    Xn = _cap(X_nor_s, OPT_NOR_CAP)
    m  = IsolationForest(n_estimators=n_est,
                         max_samples=min(ms, len(Xn)),
                         max_features=mf,
                         random_state=SEED, n_jobs=-1)
    m.fit(Xn)
    norm_s = iso_score(m, Xn); cal_s = iso_score(m, X_cal_s)
    prf    = (rf_probs >= tau_rf).astype(int); mrf = (prf == 0)
    best   = 0.0
    for p in TAU_ANOM_PCTS:
        tau  = np.percentile(norm_s, p)
        pred = prf.copy(); pred[mrf] = (cal_s[mrf] > tau).astype(int)
        idx  = balanced_idx(y_cal, cap=OPT_BAL_CAP)
        best = max(best, f1_score(y_cal[idx], pred[idx], zero_division=0))
    del m; gc.collect()
    return -best

# =============================================================================
# META-HEURISTIC ALGORITHMS
# =============================================================================
_TRACE = []   # best fitness after each generation of the current optimiser run

def run_ga(fn, n_dim, n_pop=OPT_POP, n_gen=OPT_ITER, seed=SEED):
    rng = np.random.RandomState(seed)
    pop = rng.rand(n_pop, n_dim); bv, bf = pop[0].copy(), np.inf
    def tourn(fv):
        i, j = rng.randint(0, n_pop, 2)
        return pop[i].copy() if fv[i] <= fv[j] else pop[j].copy()
    for gen in range(n_gen):
        fv = np.array([fn(v) for v in pop])
        bi = np.argmin(fv)
        if fv[bi] < bf: bf, bv = fv[bi], pop[bi].copy()
        print(f"    GA  gen {gen+1:2d}/{n_gen}: F1={-bf:.4f}")
        _TRACE.append(float(-bf))
        new = []
        while len(new) < n_pop:
            p1, p2 = tourn(fv), tourn(fv)
            c = np.where(rng.rand(n_dim) < 0.5, p1, p2) \
                if rng.rand() < 0.8 else p1.copy()
            if rng.rand() < 0.3:
                c = np.clip(c + rng.randn(n_dim) * 0.1, 0, 1)
            new.append(c)
        pop = np.array(new)
    return bv, -bf

def run_pso(fn, n_dim, n_particles=OPT_POP, n_iter=OPT_ITER, seed=SEED):
    rng = np.random.RandomState(seed)
    pos = rng.rand(n_particles, n_dim)
    vel = rng.randn(n_particles, n_dim) * 0.05
    pbp = pos.copy()
    pbf = np.array([fn(p) for p in pos])
    gi  = np.argmin(pbf); gp, gf = pbp[gi].copy(), pbf[gi]
    w, c1, c2 = 0.7, 1.5, 1.5
    for it in range(n_iter):
        r1, r2 = rng.rand(n_particles, n_dim), rng.rand(n_particles, n_dim)
        vel    = np.clip(w*vel + c1*r1*(pbp-pos) + c2*r2*(gp-pos), -0.5, 0.5)
        pos    = np.clip(pos + vel, 0, 1)
        for i, p in enumerate(pos):
            f = fn(p)
            if f < pbf[i]: pbf[i], pbp[i] = f, p.copy()
        gi = np.argmin(pbf); gf, gp = pbf[gi], pbp[gi].copy()
        print(f"    PSO iter {it+1:2d}/{n_iter}: F1={-gf:.4f}")
        _TRACE.append(float(-gf))
    return gp, -gf

def run_de(fn, n_dim, n_pop=OPT_POP, n_iter=OPT_ITER,
           F=0.8, CR=0.7, seed=SEED):
    rng = np.random.RandomState(seed)
    pop = rng.rand(n_pop, n_dim)
    fit = np.array([fn(v) for v in pop])
    bi  = np.argmin(fit); bv, bf = pop[bi].copy(), fit[bi]
    for gen in range(n_iter):
        for i in range(n_pop):
            donors = [j for j in range(n_pop) if j != i]
            a, b, c = rng.choice(donors, 3, replace=False)
            mut   = np.clip(pop[a] + F*(pop[b] - pop[c]), 0, 1)
            cross = rng.rand(n_dim) < CR
            cross[rng.randint(n_dim)] = True
            trial = np.where(cross, mut, pop[i])
            tf_   = fn(trial)
            if tf_ <= fit[i]:
                pop[i], fit[i] = trial, tf_
                if tf_ < bf: bf, bv = tf_, trial.copy()
        print(f"    DE  gen {gen+1:2d}/{n_iter}: F1={-bf:.4f}")
        _TRACE.append(float(-bf))
    return bv, -bf

META_CONFIGS = [
    ('RF_AE_GA',    run_ga,  AE_NDIM,   'ae',   decode_ae_hparams),
    ('RF_AE_PSO',   run_pso, AE_NDIM,   'ae',   decode_ae_hparams),
    ('RF_AE_DE',    run_de,  AE_NDIM,   'ae',   decode_ae_hparams),
    ('RF_IsoF_GA',  run_ga,  ISOF_NDIM, 'isof', decode_isof_hparams),
    ('RF_IsoF_PSO', run_pso, ISOF_NDIM, 'isof', decode_isof_hparams),
    ('RF_IsoF_DE',  run_de,  ISOF_NDIM, 'isof', decode_isof_hparams),
    ('XGB_VAE_GA',  run_ga,  VAE_NDIM,  'vae',  decode_vae_hparams),
    ('XGB_VAE_PSO', run_pso, VAE_NDIM,  'vae',  decode_vae_hparams),
    ('XGB_VAE_DE',  run_de,  VAE_NDIM,  'vae',  decode_vae_hparams),
]

# =============================================================================
# EVALUATION HELPERS
# =============================================================================
def evaluate_with_scores(y, rf_p, xgb_p, scores_dict, taus, model_types):
    results = {}
    pred_rf  = (rf_p >= taus['rf']).astype(int)
    mask_rf  = (pred_rf == 0)
    pred_xgb = (xgb_p >= taus['xgb']).astype(int) if xgb_p is not None else None
    mask_xgb = (pred_xgb == 0) if pred_xgb is not None else None

    results['RF_only']  = compute_metrics_full(y, pred_rf)
    if xgb_p is not None:
        results['XGB_only'] = compute_metrics_full(y, pred_xgb)

    for cfg_name, scores in scores_dict.items():
        s2_type = model_types[cfg_name]
        tau     = taus[cfg_name]
        if s2_type in ('ae', 'isof'):
            pred = pred_rf.copy()
            pred[mask_rf] = (scores[mask_rf] > tau).astype(int)
        else:
            if xgb_p is None:
                results[cfg_name] = None
                continue
            pred = pred_xgb.copy()
            pred[mask_xgb] = (scores[mask_xgb] > tau).astype(int)
        results[cfg_name] = compute_metrics_full(y, pred)
    return results

def score_target(tgt_name, scaler, rf, platt_rf,
                 xgb, platt_xgb, models_with_types):
    path = os.path.join(ALIGNED_DIR, f"aligned_{tgt_name}.parquet")
    import pyarrow.parquet as pq
    pf   = pq.ParquetFile(path)
    ay=[]; ar=[]; ax_=[]; abl={k:[] for k in models_with_types}

    for batch in pf.iter_batches(batch_size=200_000,
                                  columns=FEATURES_42 + [TARGET]):
        chunk = batch.to_pandas()
        chunk = normalise_labels(chunk).dropna(subset=FEATURES_42)
        if len(chunk) == 0: continue
        X  = chunk[FEATURES_42].values.astype(np.float32)
        y  = (chunk[TARGET] != NORMAL).values.astype(int)
        Xs = np.nan_to_num(scaler.transform(X),
                           nan=0.0, posinf=1.0, neginf=0.0)
        rp = platt_rf.predict_proba(
             rf.predict_proba(Xs)[:, 1].reshape(-1, 1))[:, 1]
        ay.append(y); ar.append(rp)
        if xgb is not None:
            xp = platt_xgb.predict_proba(
                 xgb.predict_proba(Xs)[:, 1].reshape(-1, 1))[:, 1]
            ax_.append(xp)
        for k, (m, s2t) in models_with_types.items():
            abl[k].append(anomaly_score(m, s2t, Xs))
        del chunk, X, Xs; gc.collect()

    return (np.concatenate(ay),
            np.concatenate(ar),
            np.concatenate(ax_) if ax_ else None,
            {k: np.concatenate(v) for k, v in abl.items()})


# =============================================================================
# REVISION DIAGNOSTICS (new: routing, threshold-free AUCs, score histograms)
# =============================================================================
HIST_BINS     = 60
HIST_CONFIGS  = ['RF_AE_fixed', 'RF_IsoF_fixed', 'XGB_VAE_fixed']

def _safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    if len(y) == 0 or y.min() == y.max():
        return (np.nan, np.nan)
    s = np.nan_to_num(s, nan=0.0, posinf=np.finfo(np.float64).max / 10,
                      neginf=-np.finfo(np.float64).max / 10)
    return (round(float(roc_auc_score(y, s)), 4),
            round(float(average_precision_score(y, s)), 4))

def _hist_rows(tag, comp, src_norm, tgt_scores, y, log_scale):
    """Histogram of one score for source-normal, target-normal, target-attack."""
    rows = []
    src_norm = np.asarray(src_norm, dtype=np.float64)
    tgt_scores = np.asarray(tgt_scores, dtype=np.float64)
    pool = np.concatenate([src_norm, tgt_scores[::max(1, len(tgt_scores)//500_000)]])
    pool = pool[np.isfinite(pool)]
    if log_scale:
        pool = pool[pool > 0]
        if len(pool) == 0:
            return rows
        lo, hi = np.percentile(pool, [0.1, 99.9])
        lo = max(lo, 1e-12); hi = max(hi, lo * 10)
        edges = np.logspace(np.log10(lo), np.log10(hi), HIST_BINS + 1)
    else:
        lo, hi = np.percentile(pool, [0.1, 99.9])
        if hi <= lo: hi = lo + 1e-6
        edges = np.linspace(lo, hi, HIST_BINS + 1)
    groups = {'source_normal': src_norm,
              'target_normal': tgt_scores[np.asarray(y) == 0],
              'target_attack': tgt_scores[np.asarray(y) == 1]}
    for g, v in groups.items():
        v = v[np.isfinite(v)]
        v = np.clip(v, edges[0], edges[-1])
        cnt, _ = np.histogram(v, bins=edges)
        for i in range(HIST_BINS):
            rows.append({'Tag': tag, 'Component': comp, 'Group': g,
                         'BinLeft': float(edges[i]), 'BinRight': float(edges[i+1]),
                         'Count': int(cnt[i]), 'GroupTotal': int(len(v))})
    return rows

def diagnostics(tag, y, rf_p, xgb_p, scores_dict, taus, model_types, src_norm):
    """Routing rates, threshold-free AUCs and score histograms for one evaluation."""
    y = np.asarray(y)
    route_rows, auc_rows, hist_rows = [], [], []
    fwd = {'RF': rf_p < taus['rf']}
    if xgb_p is not None:
        fwd['XGB'] = xgb_p < taus['xgb']
    for s1, mask in fwd.items():
        route_rows.append({
            'Tag': tag, 'Stage1': s1, 'Tau1': taus['rf' if s1 == 'RF' else 'xgb'],
            'AttackForwarded': round(float(mask[y == 1].mean()) if (y == 1).any() else np.nan, 4),
            'NormalForwarded': round(float(mask[y == 0].mean()) if (y == 0).any() else np.nan, 4),
            'N_attack': int((y == 1).sum()), 'N_normal': int((y == 0).sum())})
    for s1, p in [('RF', rf_p), ('XGB', xgb_p)]:
        if p is None: continue
        roc, pr = _safe_auc(y, p)
        auc_rows.append({'Tag': tag, 'Component': f'Stage1_{s1}', 'Subset': 'all',
                         'ROC_AUC': roc, 'PR_AUC': pr})
    for cfg, s in scores_dict.items():
        st = model_types[cfg]
        mask = fwd['RF'] if st in ('ae', 'isof') else fwd.get('XGB')
        roc, pr = _safe_auc(y, s)
        auc_rows.append({'Tag': tag, 'Component': cfg, 'Subset': 'all',
                         'ROC_AUC': roc, 'PR_AUC': pr})
        if mask is not None:
            roc, pr = _safe_auc(y[mask], s[mask])
            auc_rows.append({'Tag': tag, 'Component': cfg, 'Subset': 'forwarded_to_stage2',
                             'ROC_AUC': roc, 'PR_AUC': pr})
    hist_rows += _hist_rows(tag, 'Stage1_RF', src_norm['Stage1_RF'], rf_p, y, False)
    if xgb_p is not None:
        hist_rows += _hist_rows(tag, 'Stage1_XGB', src_norm['Stage1_XGB'], xgb_p, y, False)
    for cfg in HIST_CONFIGS:
        if cfg in scores_dict:
            hist_rows += _hist_rows(tag, cfg, src_norm[cfg], scores_dict[cfg], y,
                                    model_types[cfg] in ('ae', 'vae'))
    return route_rows, auc_rows, hist_rows

def _print_res(res):
    for cfg, m in res.items():
        if m is None: continue
        print(f"    {cfg:<25}: Recall={m['Recall']}  FAR={m['FAR']}  F1={m['F1']}  "
              f"MCC={m['MCC']}  BalAcc={m['BalAcc']}")

# =============================================================================
# SINGLE-SOURCE, SINGLE-SEED RUN (original pipeline, unchanged settings)
# =============================================================================
N_INFER_SAMPLES = 10_000
N_INFER_REPEATS = 5

def run_one_seed(seed):
    global SEED
    SEED = seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    all_results  = []
    hparams_log  = {SOURCE_DATASET: {}}
    timing_records = []
    infer_records  = []
    size_records   = []
    route_records, auc_records, hist_records, conv_records = [], [], [], []

    cb_final = [
        keras.callbacks.EarlyStopping(patience=AE_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
    ]

    src_name = SOURCE_DATASET
    print(f"\n{'='*65}")
    print(f"SOURCE: {src_name}   SEED: {seed}")
    print(f"{'='*65}")
    t0 = time.time()

    df = load_aligned(src_name,
                      cap=DATASET_ROW_CAP.get(src_name),
                      protect_label=DATASET_PROTECT_LABEL.get(src_name))
    print(f"  Loaded {src_name}: {df.shape} | "
          f"Normal:{(df[TARGET]==NORMAL).sum():,} | "
          f"Attack:{(df[TARGET]!=NORMAL).sum():,}")

    df_tv, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df[TARGET])
    del df; gc.collect()
    df_train, df_calib = train_test_split(
        df_tv, test_size=0.2, random_state=SEED, stratify=df_tv[TARGET])
    del df_tv; gc.collect()
    df_optval, df_thrcal = train_test_split(
        df_calib, test_size=0.5, random_state=SEED, stratify=df_calib[TARGET])
    del df_calib; gc.collect()

    df_normal = df_train[df_train[TARGET] == NORMAL]
    X_tr     = df_train[FEATURES_42].values.astype(np.float32)
    y_tr     = df_train[TARGET].values
    X_opt    = df_optval[FEATURES_42].values.astype(np.float32)
    y_opt    = (df_optval[TARGET] != NORMAL).values.astype(int)
    X_thr    = df_thrcal[FEATURES_42].values.astype(np.float32)
    y_thr    = (df_thrcal[TARGET] != NORMAL).values.astype(int)
    X_nor    = df_normal[FEATURES_42].values.astype(np.float32)
    X_te     = df_test[FEATURES_42].values.astype(np.float32)
    y_te     = (df_test[TARGET] != NORMAL).values.astype(int)
    print(f"  Train:{len(df_train):,}  OptVal:{len(df_optval):,}  "
          f"ThrCal:{len(df_thrcal):,}  Test:{len(df_test):,}  Normal:{len(X_nor):,}")
    del df_train, df_optval, df_thrcal, df_test, df_normal; gc.collect()

    scaler = MinMaxScaler()
    scaler.fit(X_nor)
    X_tr_s   = scaler.transform(X_tr);  del X_tr;  gc.collect()
    X_opt_s  = scaler.transform(X_opt); del X_opt; gc.collect()
    X_thr_s  = scaler.transform(X_thr); del X_thr; gc.collect()
    X_nor_s  = scaler.transform(X_nor); del X_nor; gc.collect()
    X_te_s   = scaler.transform(X_te);  del X_te;  gc.collect()

    print("  Training RF (fixed)...")
    t_rf = time.time()
    rf = RandomForestClassifier(
        n_estimators=RF_TREES, max_depth=RF_DEPTH,
        min_samples_leaf=RF_MIN_LEAF, class_weight='balanced',
        random_state=SEED, n_jobs=-1)
    y_bin    = (y_tr != NORMAL).astype(int)   # binary attack label for Stage-1
    rf.fit(X_tr_s, y_bin)
    rf_train_secs = time.time() - t_rf
    timing_records.append({'Source': src_name, 'Component': 'RF',
                           'Phase': 'train', 'Seconds': rf_train_secs})
    size_records.append({'Source': src_name, 'Component': 'RF',
                         'Metric': 'n_trees', 'Value': RF_TREES})
    print(f"  RF training: {rf_train_secs:.1f}s")
    rf_cr_opt  = rf.predict_proba(X_opt_s)[:, 1]
    platt_rf   = LogisticRegression(random_state=SEED)
    platt_rf.fit(rf_cr_opt.reshape(-1, 1), y_opt)
    rf_opt_p   = platt_rf.predict_proba(rf_cr_opt.reshape(-1, 1))[:, 1]
    rf_thr_p   = platt_rf.predict_proba(
        rf.predict_proba(X_thr_s)[:, 1].reshape(-1, 1))[:, 1]
    tau_rf_raw = best_tau_rf(rf_thr_p, y_thr, TAU_RF_STEPS)
    tau_rf     = max(tau_rf_raw, TAU_FLOOR)
    print(f"  RF threshold: {tau_rf_raw:.2f} -> floored to {tau_rf:.2f}"
          if tau_rf != tau_rf_raw else f"  RF threshold: {tau_rf:.2f}")

    HAS_XGB = False; xgb = platt_xgb = None
    tau_xgb = 0.5;   xgb_opt_p = None; xgb_thr_p = None; tau_xgb_raw = None
    try:
        from xgboost import XGBClassifier
        print("  Training XGBoost (fixed)...")
        y_bin    = (y_tr != NORMAL).astype(int)
        t_xgb    = time.time()
        xgb      = XGBClassifier(n_estimators=200, max_depth=6,
                                 learning_rate=0.1, eval_metric='logloss',
                                 random_state=SEED, n_jobs=-1, verbosity=0)
        xgb.fit(X_tr_s, y_bin)
        xgb_train_secs = time.time() - t_xgb
        timing_records.append({'Source': src_name, 'Component': 'XGB',
                               'Phase': 'train', 'Seconds': xgb_train_secs})
        size_records.append({'Source': src_name, 'Component': 'XGB',
                             'Metric': 'n_trees', 'Value': 200})
        print(f"  XGB training: {xgb_train_secs:.1f}s")
        xgb_cr_opt = xgb.predict_proba(X_opt_s)[:, 1]
        platt_xgb  = LogisticRegression(random_state=SEED)
        platt_xgb.fit(xgb_cr_opt.reshape(-1, 1), y_opt)
        xgb_opt_p  = platt_xgb.predict_proba(xgb_cr_opt.reshape(-1, 1))[:, 1]
        xgb_thr_p  = platt_xgb.predict_proba(
            xgb.predict_proba(X_thr_s)[:, 1].reshape(-1, 1))[:, 1]
        tau_xgb_raw = best_tau_rf(xgb_thr_p, y_thr, TAU_RF_STEPS)
        tau_xgb     = max(tau_xgb_raw, TAU_FLOOR)
        HAS_XGB  = True
        print(f"  XGB threshold: {tau_xgb_raw:.2f} -> floored to {tau_xgb:.2f}"
              if tau_xgb != tau_xgb_raw else f"  XGB threshold: {tau_xgb:.2f}")
    except ImportError:
        print("  XGBoost not available.")

    taus = {'rf': tau_rf, 'xgb': tau_xgb}
    thr_log = [{'Source': src_name, 'Seed': seed, 'Stage1': 'RF',
                'TauRaw': float(tau_rf_raw), 'TauUsed': float(tau_rf)}]
    if HAS_XGB:
        thr_log.append({'Source': src_name, 'Seed': seed, 'Stage1': 'XGB',
                        'TauRaw': float(tau_xgb_raw), 'TauUsed': float(tau_xgb)})

    # source-normal reference scores for histograms
    src_norm = {'Stage1_RF': rf_thr_p[y_thr == 0]}
    if HAS_XGB:
        src_norm['Stage1_XGB'] = xgb_thr_p[y_thr == 0]

    del X_tr_s; gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    print(f"  Training data freed. Starting Stage-2...")

    print("\n  Training fixed Stage-2 baselines...")
    fixed_models = {}

    print("  Training RF_AE_fixed...")
    t_ae_fixed = time.time()
    m_ae = build_ae_fixed(N_FEATURES)
    m_ae.fit(_cap(X_nor_s, FINAL_NOR_CAP), _cap(X_nor_s, FINAL_NOR_CAP),
             epochs=AE_EPOCHS, batch_size=AE_BATCH,
             validation_split=0.1, callbacks=cb_final, verbose=0)
    ae_fixed_train_secs = time.time() - t_ae_fixed
    timing_records.append({'Source': src_name, 'Component': 'AE_fixed',
                           'Phase': 'train', 'Seconds': ae_fixed_train_secs})
    size_records.append({'Source': src_name, 'Component': 'AE_fixed',
                         'Metric': 'n_params', 'Value': m_ae.count_params()})
    print(f"    AE_fixed training: {ae_fixed_train_secs:.1f}s  "
          f"params={m_ae.count_params():,}")
    ns_ae = recon_score(m_ae, _cap(X_nor_s, FINAL_NOR_CAP))
    cs_ae = recon_score(m_ae, X_thr_s)
    tau   = best_tau_anomaly(cs_ae, y_thr, ns_ae, TAU_ANOM_PCTS)
    fixed_models['RF_AE_fixed'] = (m_ae, 'ae')
    taus['RF_AE_fixed'] = tau
    src_norm['RF_AE_fixed'] = ns_ae
    print(f"    tau={tau:.5f}")
    _cleanup()

    print("  Training RF_IsoF_fixed...")
    t_isof_fixed = time.time()
    m_isof = IsolationForest(n_estimators=ISOF_TREES_FIXED,
                             max_samples=min(ISOF_SAMPLES_FIXED, len(X_nor_s)),
                             random_state=SEED, n_jobs=-1)
    m_isof.fit(X_nor_s)
    isof_fixed_train_secs = time.time() - t_isof_fixed
    timing_records.append({'Source': src_name, 'Component': 'IsoF_fixed',
                           'Phase': 'train', 'Seconds': isof_fixed_train_secs})
    size_records.append({'Source': src_name, 'Component': 'IsoF_fixed',
                         'Metric': 'n_trees', 'Value': ISOF_TREES_FIXED})
    print(f"    IsoF_fixed training: {isof_fixed_train_secs:.1f}s")
    ns_if = iso_score(m_isof, X_nor_s)
    cs_if = iso_score(m_isof, X_thr_s)
    tau   = best_tau_anomaly(cs_if, y_thr, ns_if, TAU_ANOM_PCTS)
    fixed_models['RF_IsoF_fixed'] = (m_isof, 'isof')
    taus['RF_IsoF_fixed'] = tau
    src_norm['RF_IsoF_fixed'] = _cap(ns_if, FINAL_NOR_CAP)
    print(f"    tau={tau:.5f}")

    if HAS_XGB:
        print("  Training XGB_VAE_fixed...")
        t_vae_fixed = time.time()
        m_vae = build_vae_fixed(N_FEATURES)
        m_vae.fit(_cap(X_nor_s, FINAL_NOR_CAP),
                  epochs=AE_EPOCHS, batch_size=AE_BATCH,
                  validation_split=0.1, callbacks=cb_final, verbose=0)
        vae_fixed_train_secs = time.time() - t_vae_fixed
        timing_records.append({'Source': src_name, 'Component': 'VAE_fixed',
                               'Phase': 'train', 'Seconds': vae_fixed_train_secs})
        size_records.append({'Source': src_name, 'Component': 'VAE_fixed',
                             'Metric': 'n_params', 'Value': m_vae.count_params()})
        print(f"    VAE_fixed training: {vae_fixed_train_secs:.1f}s  "
              f"params={m_vae.count_params():,}")
        ns_vae = recon_score(m_vae, _cap(X_nor_s, FINAL_NOR_CAP))
        cs_vae = recon_score(m_vae, X_thr_s)
        tau    = best_tau_anomaly(cs_vae, y_thr, ns_vae, TAU_ANOM_PCTS)
        fixed_models['XGB_VAE_fixed'] = (m_vae, 'vae')
        taus['XGB_VAE_fixed'] = tau
        src_norm['XGB_VAE_fixed'] = ns_vae
        print(f"    tau={tau:.5f}")
        _cleanup()

    meta_models = {}

    for cfg_name, opt_fn, n_dim, s2_type, decoder in META_CONFIGS:
        if s2_type == 'vae' and not HAS_XGB:
            print(f"  Skipping {cfg_name} (XGBoost unavailable)")
            continue

        print(f"\n  [{cfg_name}] Optimising (pop={OPT_POP}, iter={OPT_ITER})...")
        _mem(cfg_name)
        t_opt_start = time.time()

        if s2_type == 'ae':
            fn = lambda v, _n=X_nor_s, _c=X_opt_s, _y=y_opt, \
                         _rp=rf_opt_p, _tr=tau_rf: \
                     fitness_ae(v, _n, _c, _y, _rp, _tr)
        elif s2_type == 'isof':
            fn = lambda v, _n=X_nor_s, _c=X_opt_s, _y=y_opt, \
                         _rp=rf_opt_p, _tr=tau_rf: \
                     fitness_isof(v, _n, _c, _y, _rp, _tr)
        else:
            fn = lambda v, _n=X_nor_s, _c=X_opt_s, _y=y_opt, \
                         _xp=xgb_opt_p, _tx=tau_xgb: \
                     fitness_vae(v, _n, _c, _y, _xp, _tx)

        _TRACE.clear()
        best_v, best_calib_f1 = opt_fn(fn, n_dim, seed=SEED)
        for g, val in enumerate(_TRACE, start=1):
            conv_records.append({'Source': src_name, 'Seed': seed, 'Config': cfg_name,
                                 'Generation': g, 'BestFitnessF1': round(val, 4)})
        opt_secs = time.time() - t_opt_start
        timing_records.append({'Source': src_name, 'Component': cfg_name,
                               'Phase': 'optimise', 'Seconds': opt_secs})
        decoded = decoder(best_v)
        hparams_log[src_name][cfg_name] = decoded
        print(f"  Best hparams: {decoded}  |  calib F1={best_calib_f1:.4f}  "
              f"opt time={opt_secs/60:.1f}min")

        _cleanup()
        print(f"  Training final {cfg_name} (50 epochs)...")
        t_final_start = time.time()

        Xnf = _cap(X_nor_s, FINAL_NOR_CAP)
        if s2_type == 'ae':
            ld, ns, lr = decoded
            final_m = build_ae_custom(N_FEATURES, ld, ns, lr)
            final_m.fit(Xnf, Xnf, epochs=AE_EPOCHS, batch_size=AE_BATCH,
                        validation_split=0.1, callbacks=cb_final, verbose=0)
            norm_s = recon_score(final_m, Xnf)
        elif s2_type == 'isof':
            n_est, ms, mf = decoded
            final_m = IsolationForest(
                n_estimators=n_est, max_samples=min(ms, len(Xnf)),
                max_features=mf, random_state=SEED, n_jobs=-1)
            final_m.fit(Xnf)
            norm_s = iso_score(final_m, Xnf)
        else:
            ld, lr = decoded
            final_m = build_vae_custom(N_FEATURES, ld, lr)
            final_m.fit(Xnf, epochs=AE_EPOCHS, batch_size=AE_BATCH,
                        validation_split=0.1, callbacks=cb_final, verbose=0)
            norm_s = recon_score(final_m, Xnf)

        cal_s_thr = anomaly_score(final_m, s2_type, X_thr_s)
        tau = best_tau_anomaly(cal_s_thr, y_thr, norm_s, TAU_ANOM_PCTS)
        meta_models[cfg_name] = (final_m, s2_type)
        taus[cfg_name] = tau

        final_secs = time.time() - t_final_start
        timing_records.append({'Source': src_name, 'Component': cfg_name,
                               'Phase': 'final_train', 'Seconds': final_secs})
        if s2_type == 'isof':
            size_records.append({'Source': src_name, 'Component': cfg_name,
                                 'Metric': 'n_trees', 'Value': decoded[0]})
        else:
            size_records.append({'Source': src_name, 'Component': cfg_name,
                                 'Metric': 'n_params',
                                 'Value': final_m.count_params()})
        print(f"  Final threshold: {tau:.5f}  final_train={final_secs:.1f}s")

    all_models  = {**fixed_models, **meta_models}
    model_types = {k: v[1] for k, v in all_models.items()}

    def _store(res, tgt, typ):
        for cfg, m in res.items():
            if m is None: continue
            all_results.append({'Source': src_name, 'Target': tgt, 'Seed': seed,
                                'Type': typ, 'Config': cfg, **m})

    def _store_diag(tag, y, rp, xp, sc):
        r, a, h = diagnostics(tag, y, rp, xp, sc, taus, model_types, src_norm)
        for row in r + a + h:
            row['Source'] = src_name; row['Seed'] = seed
        route_records.extend(r); auc_records.extend(a); hist_records.extend(h)

    print(f"\n  Evaluating WITHIN {src_name}...")
    rf_te_p = platt_rf.predict_proba(
        rf.predict_proba(X_te_s)[:, 1].reshape(-1, 1))[:, 1]
    xgb_te_p = None
    if HAS_XGB:
        xgb_te_p = platt_xgb.predict_proba(
            xgb.predict_proba(X_te_s)[:, 1].reshape(-1, 1))[:, 1]

    te_scores = {k: anomaly_score(m, s2t, X_te_s)
                 for k, (m, s2t) in all_models.items()}

    w_res = evaluate_with_scores(
        y_te, rf_te_p, xgb_te_p, te_scores, taus, model_types)
    _print_res(w_res)
    _store(w_res, src_name, 'within')
    _store_diag(f'{src_name}->{src_name}', y_te, rf_te_p, xgb_te_p, te_scores)
    del X_te_s, y_te, rf_te_p, xgb_te_p, te_scores; gc.collect()

    print(f"\n  Measuring inference latency ({N_INFER_SAMPLES} flows, "
          f"{N_INFER_REPEATS} repeats)...")
    df_infer_full = load_aligned(src_name,
                                 cap=DATASET_ROW_CAP.get(src_name),
                                 protect_label=DATASET_PROTECT_LABEL.get(src_name))
    df_infer = df_infer_full.sample(
        n=min(N_INFER_SAMPLES, len(df_infer_full)), random_state=SEED)
    del df_infer_full
    X_inf  = df_infer[FEATURES_42].values.astype(np.float32)
    X_infs = np.nan_to_num(scaler.transform(X_inf), nan=0.0, posinf=1.0, neginf=0.0)
    n_inf  = len(X_infs)
    del df_infer, X_inf; gc.collect()

    t0i = time.time()
    for _ in range(N_INFER_REPEATS):
        _ = platt_rf.predict_proba(rf.predict_proba(X_infs)[:, 1].reshape(-1, 1))[:, 1]
    rf_infer_ms = (time.time() - t0i) / N_INFER_REPEATS / n_inf * 1000
    infer_records.append({'Source': src_name, 'Config': 'RF_only',
                          'Stage': 'Stage-1 (RF)', 'ms_per_flow': rf_infer_ms})
    print(f"    RF Stage-1 inference: {rf_infer_ms:.4f} ms/flow")

    xgb_infer_ms = 0.0
    if HAS_XGB:
        t0i = time.time()
        for _ in range(N_INFER_REPEATS):
            _ = platt_xgb.predict_proba(
                xgb.predict_proba(X_infs)[:, 1].reshape(-1, 1))[:, 1]
        xgb_infer_ms = (time.time() - t0i) / N_INFER_REPEATS / n_inf * 1000
        infer_records.append({'Source': src_name, 'Config': 'XGB_only',
                              'Stage': 'Stage-1 (XGB)', 'ms_per_flow': xgb_infer_ms})
        print(f"    XGB Stage-1 inference: {xgb_infer_ms:.4f} ms/flow")

    for cfg_name, (m, s2_type) in all_models.items():
        t0i = time.time()
        for _ in range(N_INFER_REPEATS):
            _ = anomaly_score(m, s2_type, X_infs)
        s2_ms = (time.time() - t0i) / N_INFER_REPEATS / n_inf * 1000
        if s2_type in ('ae', 'isof'):
            s1_ms, s1_label = rf_infer_ms, 'RF'
        else:
            s1_ms, s1_label = xgb_infer_ms, 'XGB'
        total_ms = s1_ms + s2_ms
        infer_records.append({'Source': src_name, 'Config': cfg_name,
                              'Stage': f'Stage-2 ({s2_type.upper()})',
                              'ms_per_flow': s2_ms})
        infer_records.append({'Source': src_name, 'Config': cfg_name,
                              'Stage': f'Total ({s1_label}+{s2_type.upper()})',
                              'ms_per_flow': total_ms})
        print(f"    {cfg_name:<25}: S2={s2_ms:.4f} ms/flow  total={total_ms:.4f} ms/flow")

    del X_infs; gc.collect()

    for tgt_name in DATASETS:
        if tgt_name == src_name:
            continue
        print(f"\n  Cross {src_name} -> {tgt_name}...")
        y_t, rf_p, xgb_p, abl_s = score_target(
            tgt_name, scaler, rf, platt_rf,
            xgb if HAS_XGB else None,
            platt_xgb if HAS_XGB else None,
            all_models)

        c_res = evaluate_with_scores(y_t, rf_p, xgb_p, abl_s, taus, model_types)
        _print_res(c_res)
        _store(c_res, tgt_name, 'cross')
        _store_diag(f'{src_name}->{tgt_name}', y_t, rf_p, xgb_p, abl_s)
        del y_t, rf_p, xgb_p, abl_s; gc.collect()

    print(f"\n  Best hyperparameters found for {src_name} (seed {seed}):")
    for cfg, hp in hparams_log[src_name].items():
        print(f"    {cfg:<22}: {hp}")

    print(f"\n  Source {src_name}, seed {seed} done in {(time.time()-t0)/60:.1f} min.")

    sfx = f"{src_name}_seed{seed}"
    hparams_rows = [{'Source': src_name, 'Seed': seed, 'Config': cfg, 'Hparams': str(hp)}
                    for cfg, hp in hparams_log[src_name].items()]
    for name, rows in [('results', all_results), ('timing', timing_records),
                       ('infer', infer_records), ('size', size_records),
                       ('hparams', hparams_rows), ('routing', route_records),
                       ('auc', auc_records), ('hist', hist_records),
                       ('convergence', conv_records), ('thresholds', thr_log)]:
        out = pd.DataFrame(rows)
        if 'Seed' not in out.columns and len(out):
            out['Seed'] = seed
        out.to_csv(os.path.join(OUT_DIR, f'{name}_{sfx}.csv'), index=False)
    print(f"  Saved 10 CSV files with suffix _{sfx}")

    del rf, xgb, platt_rf, platt_xgb, all_models, fixed_models, meta_models, scaler
    _cleanup()


# =============================================================================
# MAIN: one fresh pipeline per seed; each seed's files are written as soon as
# that seed finishes, so a crash in a later seed does not lose earlier ones.
# =============================================================================
for _seed in SEEDS:
    run_one_seed(_seed)
    _cleanup()

print(f"\n{'='*65}")
print(f"ALL SEEDS DONE for {SOURCE_DATASET}: {SEEDS}")
print("Download every CSV in /kaggle/working (or the whole output as a zip).")
print(f"{'='*65}")
