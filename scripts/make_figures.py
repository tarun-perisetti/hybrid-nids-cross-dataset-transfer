"""Figures for the revised manuscript. Usage: python make_figures.py <results_root> <agg_dir> <out_dir>
results_root holds one folder per run (e.g. 2017_s42) with the per-seed CSVs; agg_dir holds aggregate.py output."""
import sys, glob, ast
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

root, agg, out = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8.5,
                     'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.7, 'figure.dpi': 100})
C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'pink': '#CC79A7', 'grey': '#7f7f7f', 'red': '#D55E00'}
SHORT = {'IDS2017': '17', 'IDS2018': '18', 'BOT_IoT': 'B'}
def lab(p):
    a, b = p.split('->'); return f'{SHORT[a]}$\\to${SHORT[b]}'
CROSS = ['IDS2017->IDS2018', 'IDS2018->IDS2017', 'BOT_IoT->IDS2017', 'BOT_IoT->IDS2018', 'IDS2017->BOT_IoT', 'IDS2018->BOT_IoT']
ALLP = ['IDS2017->IDS2017', 'IDS2018->IDS2018', 'BOT_IoT->BOT_IoT'] + CROSS

# ---------------- Figure A: routing, Stage-2 AUC, score-exceedance ----------------
R = pd.concat([pd.read_csv(f) for f in glob.glob(f'{root}/*/routing_*.csv')])
A = pd.concat([pd.read_csv(f) for f in glob.glob(f'{root}/*/auc_*.csv')])
H = pd.concat([pd.read_csv(f) for f in glob.glob(f'{root}/*/hist_*.csv')])

fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4), gridspec_kw={'width_ratios': [1.15, 1.15, 1.15]})
# (a) share of target attacks forwarded to Stage-2
g = R.groupby(['Tag', 'Stage1']).AttackForwarded.agg(['mean', 'std']).reset_index()
x = np.arange(len(ALLP)); w = 0.38
for k, (s1, col) in enumerate([('RF', C['blue']), ('XGB', C['orange'])]):
    d = g[g.Stage1 == s1].set_index('Tag').loc[ALLP]
    ax[0].bar(x + (k - .5) * w, d['mean'], w, yerr=d['std'], color=col, label=s1, error_kw={'lw': .7, 'capsize': 1.5})
ax[0].set_xticks(x); ax[0].set_xticklabels([lab(p) for p in ALLP], rotation=60, ha='right', rotation_mode='anchor')
ax[0].set_ylabel('Target attacks forwarded'); ax[0].set_ylim(0, 1.35); ax[0].set_yticks([0,.2,.4,.6,.8,1.0]); ax[0].set_title('(a) Stage-1 routing')
ax[0].legend(frameon=False, ncol=2, loc='upper left', handlelength=1, columnspacing=.8, bbox_to_anchor=(0.0, 1.0))
ax[0].axvline(2.5, color='#bbbbbb', lw=.6, ls=':')
# (b) Stage-2 ROC-AUC on forwarded flows
cfgs = [('RF_AE_fixed', 'AE', C['blue']), ('RF_IsoF_fixed', 'IsoF', C['green']), ('XGB_VAE_fixed', 'VAE', C['pink'])]
f = A[A.Subset == 'forwarded_to_stage2']
w = 0.27
for k, (cfg, nm, col) in enumerate(cfgs):
    d = f[f.Component == cfg].groupby('Tag').ROC_AUC.agg(['mean', 'std']).reindex(ALLP)
    ax[1].bar(x + (k - 1) * w, d['mean'], w, yerr=d['std'], color=col, label=nm, error_kw={'lw': .7, 'capsize': 1.5})
ax[1].axhline(0.5, color='k', lw=.8, ls='--'); ax[1].set_ylim(0, 1.2); ax[1].set_yticks([0,.2,.4,.6,.8,1.0])
ax[1].set_xticks(x); ax[1].set_xticklabels([lab(p) for p in ALLP], rotation=60, ha='right', rotation_mode='anchor')
ax[1].set_ylabel('Stage-2 ROC-AUC (forwarded)'); ax[1].set_title('(b) Stage-2 ranking quality')
ax[1].legend(frameon=False, ncol=3, loc='upper left', handlelength=1, columnspacing=.8, bbox_to_anchor=(0.0, 1.03))
ax[1].axvline(2.5, color='#bbbbbb', lw=.6, ls=':')
# (c) share of target flows above the 95th percentile of source-normal scores
def exceed(tag, comp, seed):
    h = H[(H.Tag == tag) & (H.Component == comp) & (H.Seed == seed)]
    s = h[h.Group == 'source_normal'].sort_values('BinLeft')
    cum = np.cumsum(s.Count.values / s.GroupTotal.iloc[0]); i = int(np.searchsorted(cum, 0.95))
    res = {}
    for grp in ['target_normal', 'target_attack']:
        t = h[h.Group == grp].sort_values('BinLeft')
        res[grp] = float(t.Count.values[i + 1:].sum() / t.GroupTotal.iloc[0])
    return res
rows = []
for tag in CROSS:
    for comp in ['RF_AE_fixed', 'XGB_VAE_fixed']:
        for seed in sorted(H.Seed.unique()):
            r = exceed(tag, comp, seed); rows.append({'Tag': tag, 'Component': comp, 'Seed': seed, **r})
E_ = pd.DataFrame(rows); E_.to_csv(f'{out}/score_exceedance_p95.csv', index=False)
Eg = E_.groupby(['Tag', 'Component'])[['target_normal', 'target_attack']].agg(['mean', 'std'])
xc = np.arange(len(CROSS)); w = 0.2
specs = [('RF_AE_fixed', 'target_normal', C['blue'], '', 'AE, normal'), ('RF_AE_fixed', 'target_attack', C['blue'], '///', 'AE, attack'),
         ('XGB_VAE_fixed', 'target_normal', C['pink'], '', 'VAE, normal'), ('XGB_VAE_fixed', 'target_attack', C['pink'], '///', 'VAE, attack')]
for k, (comp, grp, col, hat, nm) in enumerate(specs):
    m = [Eg.loc[(t, comp), (grp, 'mean')] for t in CROSS]; s = [Eg.loc[(t, comp), (grp, 'std')] for t in CROSS]
    ax[2].bar(xc + (k - 1.5) * w, m, w, yerr=s, color=col if not hat else 'white', edgecolor=col, hatch=hat, lw=.8, label=nm,
              error_kw={'lw': .7, 'capsize': 1.2})
ax[2].set_xticks(xc); ax[2].set_xticklabels([lab(p) for p in CROSS], rotation=60, ha='right', rotation_mode='anchor')
ax[2].set_ylabel('Share above source P95'); ax[2].set_ylim(0, 1.5); ax[2].set_yticks([0,.2,.4,.6,.8,1.0]); ax[2].set_title('(c) Stage-2 score shift')
ax[2].legend(frameon=False, ncol=2, loc='upper left', handlelength=1.2, columnspacing=.8, fontsize=7.8, bbox_to_anchor=(0.0, 1.03))
fig.tight_layout(w_pad=1.2); fig.savefig(f'{out}/fig_diagnostics.png', dpi=300); plt.close(fig)

# ---------------- Figure B: convergence + optimiser range vs seed SD ----------------
Cv = pd.concat([pd.read_csv(f) for f in glob.glob(f'{root}/*/convergence_*.csv')])
Cv['Opt'] = Cv.Config.str.split('_').str[-1]; Cv['Fam'] = Cv.Config.str.split('_').str[1]
first = Cv[Cv.Generation == 1].set_index(['Source', 'Seed', 'Config']).BestFitnessF1
Cv = Cv.join(first.rename('F0'), on=['Source', 'Seed', 'Config']); Cv['Gain'] = Cv.BestFitnessF1 - Cv.F0
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.05), gridspec_kw={'width_ratios': [1.1, 1]})
oc = {'GA': C['blue'], 'PSO': C['orange'], 'DE': C['green']}
for o in ['GA', 'PSO', 'DE']:
    d = Cv[Cv.Opt == o].groupby('Generation').Gain.agg(['mean', 'sem'])
    ax[0].errorbar(d.index, d['mean'] * 1000, yerr=d['sem'] * 1000, color=oc[o], marker='o', ms=3.5, lw=1.4, capsize=2, label=o)
ax[0].set_xlabel('Generation'); ax[0].set_ylabel('Gain ($10^{-3}$ F1)')
ax[0].set_title('(a) Search convergence'); ax[0].legend(frameon=False, loc='upper left'); ax[0].set_xticks(range(1, 9))
Q = pd.read_csv(f'{agg}/q3_optimiser_range_vs_seed_sd.csv')
fc = {'AE': C['blue'], 'IsoF': C['green'], 'VAE': C['pink']}; fm = {'AE': 'o', 'IsoF': 's', 'VAE': '^'}
for fam in ['AE', 'IsoF', 'VAE']:
    d = Q[Q.Family == fam]
    ax[1].scatter(d.mean_seed_sd, d.optimiser_range, color=fc[fam], marker=fm[fam], s=26, label=fam, edgecolor='white', lw=.4)
mx = max(Q.mean_seed_sd.max(), Q.optimiser_range.max()) * 1.05
ax[1].plot([0, mx], [0, mx], color='k', lw=.8, ls='--'); ax[1].set_xlim(0, mx); ax[1].set_ylim(0, mx)
ax[1].set_xlabel('Seed SD of F1 (mean over GA/PSO/DE)'); ax[1].set_ylabel('F1 range, GA/PSO/DE')
ax[1].set_title('(b) Optimiser gap vs seed noise'); ax[1].legend(frameon=False, loc='upper right')
fig.tight_layout(w_pad=1.5); fig.savefig(f'{out}/fig_convergence.png', dpi=300); plt.close(fig)
print(Cv.groupby('Opt').Gain.describe())
print(E_.groupby(['Tag', 'Component'])[['target_normal', 'target_attack']].mean().round(3))
