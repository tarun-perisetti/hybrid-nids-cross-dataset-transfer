# Cross-dataset transfer of hybrid network intrusion detectors

Code, configuration and per-seed results for the paper "Cross-Dataset Transfer Behaviour of Hybrid Network Intrusion Detectors: A Benchmark with Evolutionary Hyperparameter Optimisation" (Pattern Recognition Letters, under revision).

A calibrated supervised Stage-1 classifier (Random Forest or XGBoost) is followed by an anomaly detector fitted on source-normal flows only (autoencoder, Isolation Forest or variational autoencoder), with fixed hyperparameters or tuned by GA, PSO or DE. Fourteen configurations are trained on one dataset and evaluated within it and on the other two, without retraining or recalibration.

## Datasets

Official sources (use these first):

- CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CSE-CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- CIC-BoT-IoT (University of Queensland, Layeghy, Sarhan and Portmann): https://doi.org/10.48610/C80FCCD

Cleaned copies used in the experiments (Kaggle mirrors):

- CIC-IDS2017: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
- CSE-CIC-IDS2018: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
- CIC-BoT-IoT: https://www.kaggle.com/datasets/dhoogla/cicbotiot

Raw data are not included in this repository.

### Preprocessing (run before the experiments)

1. `preprocessing/01_clean_dataset.py`: run once per dataset (set `DATASET_FOLDER` to the dataset's input folder). It streams all files, assigns the label column, draws a random sample of at most 400k rows, removes constant columns and one column of each pair with absolute Pearson correlation above 0.995 (decided on the sample, without labels), then writes `cleaned_full.parquet` for the whole dataset with rows containing missing or infinite values removed. The printed reports of the three runs are in `preprocessing/logs/` (55, 53 and 56 columns kept for CIC-IDS2017, CSE-CIC-IDS2018 and CIC-BoT-IoT).
2. `preprocessing/02_align_features.py`: maps the column names of the three cleaned files onto one snake-case vocabulary, keeps the 42 columns common to all three plus `attack_cat`, normalises benign labels to `Normal`, and writes `aligned_IDS2017.parquet`, `aligned_IDS2018.parquet`, `aligned_BOT_IoT.parquet` and `feature_intersection.csv`. Ports, protocol and addresses are not among the 42 columns. Edit the three input paths at the top of the script.

### Input format expected by `scripts/run_experiment.py`

The three `aligned_*.parquet` files from step 2 in the directory `ALIGNED_DIR`, each with the 42 shared features (`FEATURES_42` in the script) and an `attack_cat` column.

## Running

1. Install the packages in `requirements.txt` (the experiments used a Kaggle notebook with 4 CPU cores, 30 GB RAM and one NVIDIA T4).
2. Set `SOURCE_DATASET` (`IDS2017`, `IDS2018`, `BOT_IoT`) and the seed (`SEEDS` and `SEED`, one of 42, 7, 2026) in `scripts/run_experiment.py` and run it in a fresh process. This gives nine runs (3 sources x 3 seeds). Each run writes CSV files for results, routing, auc, hist, convergence, hparams, thresholds, timing, infer and size.
3. Aggregate and build tables and figures:

```
python scripts/aggregate.py results aggregated
python scripts/analysis_extras.py aggregated
python scripts/make_tables.py aggregated tables
python scripts/make_figures.py results aggregated figures
```

## Layout

- `results/<source>_s<seed>/`: per-seed outputs of the nine runs (`2017`, `2018`, `BOT` with seeds 42, 7, 2026).
- `aggregated/`: mean and SD over seeds, paired tests, Friedman tests, rank correlations, routing, AUC, thresholds (raw and used), timing, inference time, model size, convergence, hyperparameters, score-exceedance table.
- `preprocessing/`: cleaning and feature-alignment scripts and the cleaning logs.
- `scripts/`: experiment runner and the aggregation, table and figure scripts.

## Mapping from paper to files

| Paper item | File |
|---|---|
| Table 2 (F1) and Table 3 (MCC) | `scripts/make_tables.py` from `aggregated/summary_mean_sd.csv` |
| Fig. 2 (routing, AUC, score exceedance) | `scripts/make_figures.py` from `results/*/routing_*.csv`, `auc_*.csv`, `hist_*.csv`; exceedance values in `aggregated/score_exceedance_p95.csv` |
| Fig. 3 (convergence, optimiser gap vs seed noise) | `scripts/make_figures.py` from `convergence_*.csv`; `aggregated/q3_optimiser_range_vs_seed_sd.csv` |
| Supervised vs hybrid, tuned vs fixed, Wilcoxon and Friedman results | `aggregated/numbers.json`, `q1_*.csv`, `q2_*.csv` |
| Rank correlation, within vs cross | `aggregated/rank_corr_within_vs_cross.csv` |
| Routing rates, Stage-2 AUC | `aggregated/routing.csv`, `aggregated/auc.csv` |
| Thresholds (floor 0.30) | `aggregated/thresholds_all.csv` |
| Compute cost, inference time, model size | `aggregated/timing.csv`, `infer.csv`, `size_all.csv` |
| Best hyperparameters per search | `aggregated/hparams_all.csv` |
| BoT-IoT target, F1 vs MCC, BA, FAR | `aggregated/bot_target_f1_vs_mcc.csv` |

## Protocol summary

Stratified 64/8/8/20 split (train, optimisation-validation, calibration, test) per source; Platt scaling on the optimisation-validation part; tau1 and tau2 chosen on the calibration part; MinMax scaler fitted on source-normal training flows; cross-dataset evaluation on the whole cleaned target dataset. Evolutionary search: population 5, 8 generations, candidate training of at most 15 epochs, final retraining of at most 50 epochs.
