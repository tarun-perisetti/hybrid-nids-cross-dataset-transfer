"""LaTeX tables for the revised manuscript. Usage: python make_tables.py <agg_dir> <out_dir>"""
import sys, numpy as np, pandas as pd
agg, out = sys.argv[1], sys.argv[2]
S = pd.read_csv(f'{agg}/summary_mean_sd.csv')
PAIRS = ['IDS2017->IDS2017', 'IDS2018->IDS2018', 'BOT_IoT->BOT_IoT', 'IDS2017->IDS2018', 'IDS2018->IDS2017',
         'BOT_IoT->IDS2017', 'BOT_IoT->IDS2018', 'IDS2017->BOT_IoT', 'IDS2018->BOT_IoT']
ROWS = [('RF_only', 'RF-only'), ('XGB_only', 'XGB-only'), None,
        ('RF_AE_fixed', 'RF+AE (fixed)'), ('RF_IsoF_fixed', 'RF+IsoF (fixed)'), ('XGB_VAE_fixed', 'XGB+VAE (fixed)'), None,
        ('RF_AE_GA', 'RF+AE (GA)'), ('RF_AE_PSO', 'RF+AE (PSO)'), ('RF_AE_DE', 'RF+AE (DE)'),
        ('RF_IsoF_GA', 'RF+IsoF (GA)'), ('RF_IsoF_PSO', 'RF+IsoF (PSO)'), ('RF_IsoF_DE', 'RF+IsoF (DE)'),
        ('XGB_VAE_GA', 'XGB+VAE (GA)'), ('XGB_VAE_PSO', 'XGB+VAE (PSO)'), ('XGB_VAE_DE', 'XGB+VAE (DE)')]
def cell(m, s, bold):
    t = f'{m:.3f}'.replace('0.', '0.', 1) if m >= 0 else f'$-${abs(m):.3f}'
    sd = f'{s:.3f}'[1:]
    body = f'{t}{{\\scriptsize$\\pm${sd}}}'
    return f'\\textbf{{{t}}}{{\\scriptsize$\\pm${sd}}}' if bold else body
def main_table(metric, caption, label):
    P = S.pivot(index='Config', columns='Pair', values=f'{metric}_mean').loc[:, PAIRS]
    D = S.pivot(index='Config', columns='Pair', values=f'{metric}_std').loc[:, PAIRS]
    best = P.drop(index=[], errors='ignore').idxmax()
    L = ['\\begin{table*}[pos=t]', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\footnotesize', '\\setlength{\\tabcolsep}{3.0pt}',
         '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l|ccc|cccc|cc@{}}', '\\toprule',
         '\\textbf{Configuration} & \\multicolumn{3}{c|}{\\textbf{Within-dataset}} & \\multicolumn{4}{c|}{\\textbf{Cross: CIC and BoT-IoT sources}} & \\multicolumn{2}{c}{\\textbf{Cross: BoT-IoT target}$^\\dagger$} \\\\',
         ' & 17 & 18 & B & 17$\\to$18 & 18$\\to$17 & B$\\to$17 & B$\\to$18 & 17$\\to$B & 18$\\to$B \\\\', '\\midrule']
    for r in ROWS:
        if r is None: L.append('\\midrule'); continue
        k, nm = r
        L.append(nm + ' & ' + ' & '.join(cell(P.loc[k, p], D.loc[k, p], best[p] == k) for p in PAIRS) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular*}', '\\end{table*}']
    open(f'{out}/tab_{metric}.tex', 'w').write('\n'.join(L) + '\n')
main_table('F1', 'F1 (mean $\\pm$ SD over three seeds). 17 = CIC-IDS2017, 18 = CSE-CIC-IDS2018, B = CIC-BoT-IoT. $^\\dagger$BoT-IoT is 99.7\\% attacks, so F1 there is dominated by prevalence. \\textbf{Bold}: best mean per column.', 'tab:f1')
main_table('MCC', 'MCC (mean $\\pm$ SD over three seeds), layout as Table~\\ref{tab:f1}. Values near zero mean no better than chance. \\textbf{Bold}: best mean per column.', 'tab:mcc')

# BoT-target diagnostic table
B = S.set_index(['Pair', 'Config'])
rows = [('RF_only', 'RF-only'), ('XGB_only', 'XGB-only'), ('RF_AE_fixed', 'RF+AE (fixed)'), ('RF_IsoF_fixed', 'RF+IsoF (fixed)'), ('XGB_VAE_fixed', 'XGB+VAE (fixed)')]
L = ['\\begin{table}[pos=t]', '\\caption{BoT-IoT as target (mean over three seeds): F1 against MCC, balanced accuracy (BA) and false alarm rate (FAR).}', '\\label{tab:bot}', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
     '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{@{}l|cccc|cccc@{}}', '\\toprule',
     ' & \\multicolumn{4}{c|}{\\textbf{17$\\to$B}} & \\multicolumn{4}{c}{\\textbf{18$\\to$B}} \\\\', '\\textbf{Configuration} & F1 & MCC & BA & FAR & F1 & MCC & BA & FAR \\\\', '\\midrule']
for k, nm in rows:
    v = []
    for p in ['IDS2017->BOT_IoT', 'IDS2018->BOT_IoT']:
        r = B.loc[(p, k)]
        v += [f"{r.F1_mean:.3f}", (f"{r.MCC_mean:.3f}" if r.MCC_mean >= 0 else f"$-${abs(r.MCC_mean):.3f}"), f"{r.BalAcc_mean:.3f}", f"{r.FAR_mean:.3f}"]
    L.append(nm + ' & ' + ' & '.join(v) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table}']
open(f'{out}/tab_bot.tex', 'w').write('\n'.join(L) + '\n')
print(open(f'{out}/tab_bot.tex').read())
