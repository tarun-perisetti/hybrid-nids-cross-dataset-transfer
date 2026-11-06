"""
Aggregate per-seed output CSVs of run_experiment.py into summary tables and statistics.

Usage:  python aggregate.py <results_root> <out_dir>
<results_root> is searched recursively for results_<SRC>_seed<N>.csv (and the matching
auc/routing/convergence/thresholds/timing/hparams/infer/size files).
Nothing here depends on the number of seeds; it simply uses what it finds.
"""
import sys, glob, os, json, itertools
import numpy as np, pandas as pd
from scipy import stats

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

def load(kind):
    fs = sorted(glob.glob(os.path.join(root, '**', f'{kind}_*_seed*.csv'), recursive=True))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

R = load('results'); R['Pair'] = R.Source + '->' + R.Target
SEEDS = sorted(R.Seed.unique())
ORDER = ['RF_only', 'XGB_only', 'RF_AE_fixed', 'RF_IsoF_fixed', 'XGB_VAE_fixed',
         'RF_AE_GA', 'RF_AE_PSO', 'RF_AE_DE', 'RF_IsoF_GA', 'RF_IsoF_PSO', 'RF_IsoF_DE',
         'XGB_VAE_GA', 'XGB_VAE_PSO', 'XGB_VAE_DE']
SRCS = ['IDS2017', 'IDS2018', 'BOT_IoT']
WITHIN = [f'{s}->{s}' for s in SRCS]
CROSS = [f'{a}->{b}' for a in SRCS for b in SRCS if a != b]
CROSSCIC = [p for p in CROSS if not p.endswith('BOT_IoT')]  # BoT-IoT target is 99.7% attacks: F1 is a prevalence artefact
FAMILY = {'AE': ['RF_AE_fixed', 'RF_AE_GA', 'RF_AE_PSO', 'RF_AE_DE'],
          'IsoF': ['RF_IsoF_fixed', 'RF_IsoF_GA', 'RF_IsoF_PSO', 'RF_IsoF_DE'],
          'VAE': ['XGB_VAE_fixed', 'XGB_VAE_GA', 'XGB_VAE_PSO', 'XGB_VAE_DE']}
METRICS = ['Recall', 'FAR', 'F1', 'Precision', 'BalAcc', 'MCC']

# ---------- summary of every metric -------------------------------------------------
g = R.groupby(['Pair', 'Config'])[METRICS].agg(['mean', 'std', 'count'])
g.columns = ['_'.join(c) for c in g.columns]
g = g.reset_index()
g.to_csv(os.path.join(out, 'summary_mean_sd.csv'), index=False)

def mean_of(pair, cfg, m):
    return R[(R.Pair == pair) & (R.Config == cfg)][m].mean()

# per-seed pair x config matrices (for tests)
def seed_mat(m, pairs):
    return {s: pd.DataFrame({p: R[(R.Pair == p) & (R.Seed == s)].set_index('Config')[m] for p in pairs}).loc[ORDER]
            for s in SEEDS}

num = {'seeds': [int(s) for s in SEEDS], 'n_seeds': len(SEEDS)}

# ---------- Q1: supervised-only vs best hybrid (cross-dataset, direction means) -----
def best_hybrid(pair, m, pool):
    vals = {c: mean_of(pair, c, m) for c in pool}
    c = max(vals, key=vals.get); return c, vals[c]
hyb_all = [c for c in ORDER if c not in ('RF_only', 'XGB_only')]
rows = []
for p in CROSS + WITHIN:
    row = {'Pair': p}
    for m in ['F1', 'MCC', 'BalAcc']:
        sup = max(mean_of(p, 'RF_only', m), mean_of(p, 'XGB_only', m))
        c, v = best_hybrid(p, m, hyb_all)
        row.update({f'{m}_sup': sup, f'{m}_besthyb': v, f'{m}_besthyb_cfg': c})
    rows.append(row)
Q1 = pd.DataFrame(rows); Q1.to_csv(os.path.join(out, 'q1_supervised_vs_best_hybrid.csv'), index=False)

# fixed hybrid vs supervised (not best-of-9, which is optimistic): per family fixed
def paired(a_cfgs, b_cfgs, pairs, m):
    a = np.array([np.mean([mean_of(p, c, m) for c in a_cfgs]) for p in pairs])
    b = np.array([np.mean([mean_of(p, c, m) for c in b_cfgs]) for p in pairs])
    d = a - b
    try: w = stats.wilcoxon(a, b)
    except ValueError: w = None
    return {'mean_diff': float(d.mean()), 'wins': int((d > 0).sum()), 'n': len(d),
            'wilcoxon_p': (float(w.pvalue) if w is not None else None)}
fixed_hyb = ['RF_AE_fixed', 'RF_IsoF_fixed', 'XGB_VAE_fixed']
sup = ['RF_only', 'XGB_only']
for m in ['F1', 'MCC', 'BalAcc']:
    num[f'fixedhyb_vs_sup_cross_{m}'] = paired(fixed_hyb, sup, CROSS, m)
    num[f'fixedhyb_vs_sup_within_{m}'] = paired(fixed_hyb, sup, WITHIN, m)
    tuned = [c for c in hyb_all if not c.endswith('fixed')]
    num[f'tuned_vs_sup_cross_{m}'] = paired(tuned, sup, CROSS, m)
    num[f'tuned_vs_fixed_cross_{m}'] = paired(tuned, fixed_hyb, CROSS, m)
    num[f'tuned_vs_fixed_within_{m}'] = paired(tuned, fixed_hyb, WITHIN, m)
    num[f'fixedhyb_vs_sup_crossCIC_{m}'] = paired(fixed_hyb, sup, CROSSCIC, m)
    num[f'tuned_vs_sup_crossCIC_{m}'] = paired(tuned, sup, CROSSCIC, m)
    num[f'tuned_vs_fixed_crossCIC_{m}'] = paired(tuned, fixed_hyb, CROSSCIC, m)

# ---------- Q2: tuned vs fixed per family, Q3: GA vs PSO vs DE (Friedman) -----------
fam_rows = []
for fam, cfgs in FAMILY.items():
    fixed = cfgs[0]; opt = cfgs[1:]
    for scope, pairs in (('cross', CROSS), ('crossCIC', CROSSCIC), ('within', WITHIN)):
        for m in ['F1', 'MCC']:
            fx = np.array([mean_of(p, fixed, m) for p in pairs])
            tu = np.array([[mean_of(p, c, m) for c in opt] for p in pairs])  # pairs x 3
            best = tu.max(axis=1)
            fam_rows.append({'Family': fam, 'Scope': scope, 'Metric': m,
                             'fixed_mean': fx.mean(), 'GA_mean': tu[:, 0].mean(), 'PSO_mean': tu[:, 1].mean(),
                             'DE_mean': tu[:, 2].mean(), 'tuned_mean': tu.mean(),
                             'tuned_minus_fixed': (tu.mean(axis=1) - fx).mean(),
                             'dirs_tuned_beats_fixed': int((tu.mean(axis=1) > fx).sum()), 'n_dirs': len(pairs),
                             'bestopt_minus_fixed': (best - fx).mean()})
pd.DataFrame(fam_rows).to_csv(os.path.join(out, 'q2_tuned_vs_fixed_by_family.csv'), index=False)

# Friedman on GA/PSO/DE; blocks = (family x pair) using seed-mean values
for scope, pairs in (('cross', CROSS), ('crossCIC', CROSSCIC), ('within', WITHIN), ('all', CROSS + WITHIN)):
    for m in ['F1', 'MCC']:
        blocks = [[mean_of(p, c, m) for c in (f'RF_{f}_{o}' if f != 'VAE' else f'XGB_VAE_{o}'
                                              for o in ('GA', 'PSO', 'DE'))]
                  for f in ('AE', 'IsoF', 'VAE') for p in pairs]
        # the generator above is evaluated per f/p; rebuild explicitly for clarity
        blocks = []
        for f in ('AE', 'IsoF', 'VAE'):
            pre = 'RF_AE_' if f == 'AE' else ('RF_IsoF_' if f == 'IsoF' else 'XGB_VAE_')
            for p in pairs:
                blocks.append([mean_of(p, pre + o, m) for o in ('GA', 'PSO', 'DE')])
        B = np.array(blocks)
        chi, pv = stats.friedmanchisquare(B[:, 0], B[:, 1], B[:, 2])
        ranks = stats.rankdata(-B, axis=1).mean(axis=0)  # rank 1 = best
        num[f'friedman_{scope}_{m}'] = {'chi2': float(chi), 'p': float(pv), 'n_blocks': len(B),
                                        'mean_rank_GA_PSO_DE': [float(x) for x in ranks]}
# optimiser spread vs seed spread: how much do GA/PSO/DE differ compared with seed-to-seed SD?
sd_rows = []
for f, pre in (('AE', 'RF_AE_'), ('IsoF', 'RF_IsoF_'), ('VAE', 'XGB_VAE_')):
    for p in CROSS:
        opt_means = [mean_of(p, pre + o, 'F1') for o in ('GA', 'PSO', 'DE')]
        seed_sd = np.mean([R[(R.Pair == p) & (R.Config == pre + o)].F1.std() for o in ('GA', 'PSO', 'DE')])
        sd_rows.append({'Family': f, 'Pair': p, 'optimiser_range': max(opt_means) - min(opt_means), 'mean_seed_sd': seed_sd})
SD = pd.DataFrame(sd_rows); SD.to_csv(os.path.join(out, 'q3_optimiser_range_vs_seed_sd.csv'), index=False)
num['optimiser_range_gt_seed_sd'] = int((SD.optimiser_range > SD.mean_seed_sd).sum()); num['optimiser_range_n'] = len(SD)

# ---------- prevalence artefact: F1 vs MCC toward BoT-IoT --------------------------
bt = R[R.Target == 'BOT_IoT'].groupby(['Source', 'Config'])[['F1', 'MCC', 'BalAcc']].mean()
bt.to_csv(os.path.join(out, 'bot_target_f1_vs_mcc.csv'))

# ---------- routing, AUC, thresholds, timing, convergence, hparams, infer, size ------
ro = load('routing'); ro['Pair'] = ro.Tag
ro.groupby(['Pair', 'Stage1'])[['AttackForwarded', 'NormalForwarded']].agg(['mean', 'std']).to_csv(os.path.join(out, 'routing.csv'))
au = load('auc'); au.groupby(['Tag', 'Component', 'Subset'])[['ROC_AUC', 'PR_AUC']].agg(['mean', 'std']).to_csv(os.path.join(out, 'auc.csv'))
th = load('thresholds'); th.to_csv(os.path.join(out, 'thresholds_all.csv'), index=False)
tm = load('timing'); tm.groupby(['Component', 'Phase']).Seconds.agg(['mean', 'std', 'min', 'max']).to_csv(os.path.join(out, 'timing.csv'))
cv = load('convergence'); cv.to_csv(os.path.join(out, 'convergence_all.csv'), index=False)
cvs = cv.sort_values('Generation').groupby(['Source', 'Seed', 'Config']).BestFitnessF1.agg(['first', 'last'])
cvs['gain'] = cvs['last'] - cvs['first']
num['conv_runs'] = int(len(cvs)); num['conv_no_gain'] = int((cvs.gain < 1e-6).sum())
num['conv_median_gain'] = float(cvs.gain.median()); num['conv_max_gain'] = float(cvs.gain.max())
def last_imp(d):
    d = d.sort_values('Generation'); inc = d.BestFitnessF1.diff().fillna(0) > 1e-9
    return int(d.loc[inc, 'Generation'].max()) if inc.any() else 0
li = cv.groupby(['Source', 'Seed', 'Config']).apply(last_imp, include_groups=False)
num['conv_last_improve_at_final_gen'] = int((li == cv.Generation.max()).sum())
num['conv_final_gen'] = int(cv.Generation.max())
hp = load('hparams'); hp.to_csv(os.path.join(out, 'hparams_all.csv'), index=False)
inf = load('infer'); inf.groupby(['Config', 'Stage']).ms_per_flow.agg(['mean', 'std']).to_csv(os.path.join(out, 'infer.csv'))
sz = load('size'); sz.to_csv(os.path.join(out, 'size_all.csv'), index=False)

# threshold facts
num['tau_floor_binding'] = int((th.TauUsed > th.TauRaw + 1e-9).sum()); num['tau_rows'] = int(len(th))

json.dump(num, open(os.path.join(out, 'numbers.json'), 'w'), indent=1, default=float)
print('seeds', SEEDS); print(json.dumps(num, indent=1, default=float)[:6000])
