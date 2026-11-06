"""Extra summary numbers used in the text. Usage: python analysis_extras.py <agg_dir>"""
import sys, pandas as pd, numpy as np
from scipy.stats import spearmanr
agg=sys.argv[1]; S=pd.read_csv(f'{agg}/summary_mean_sd.csv')
rows=[]
for m in ['F1','MCC','BalAcc']:
    P=S.pivot(index='Config',columns='Pair',values=m+'_mean')
    for src in ['IDS2017','IDS2018','BOT_IoT']:
        for tgt in ['IDS2017','IDS2018','BOT_IoT']:
            if src!=tgt:
                r,p=spearmanr(P[f'{src}->{src}'],P[f'{src}->{tgt}']); rows.append((m,f'{src}->{tgt}',r,p))
pd.DataFrame(rows,columns=['Metric','Direction','Spearman_rho','p']).to_csv(f'{agg}/rank_corr_within_vs_cross.csv',index=False)
F=S.pivot(index='Config',columns='Pair',values='FAR_mean'); M=S.pivot(index='Config',columns='Pair',values='MCC_mean'); R=S.pivot(index='Config',columns='Pair',values='Recall_mean')
pd.set_option('display.width',250,'display.max_columns',30)
print('FAR\n',F.round(3)); print('Recall\n',R.round(3))
cic=['IDS2017->IDS2018','IDS2018->IDS2017','BOT_IoT->IDS2017','BOT_IoT->IDS2018']
print('MCC cross-CIC max',M[cic].max().round(3).to_dict(),'overall max',M[cic].values.max())
print('cells with MCC>0.3 in cross-CIC:',(M[cic]>0.3).sum().sum(),'of',M[cic].size)
print('mean MCC within by config class'); 
w=['IDS2017->IDS2017','IDS2018->IDS2018','BOT_IoT->BOT_IoT']
print(M[w].mean(axis=1).round(3)); print(M[cic].mean(axis=1).round(3))
