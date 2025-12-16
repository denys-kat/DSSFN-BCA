# DSSFN-BCA experiments for hyperspectral image classification

This repository contains an implementation of DSSFN (dual-stream self-attention fusion network) and extensions used to run a controlled comparison on common hyperspectral image (HSI) benchmarks.

The codebase is set up around reproducible, apples-to-apples evaluation across a small set of configurations:

- `base`: DSSFN baseline.
- `bca`: DSSFN with intermediate cross-attention between the spatial and spectral streams (BCA).
- `bca_sc_optimized`: a single-criterion (SC) Optuna-optimized BCA architecture.
- `adaptive_mc_optimized`: an adaptive DSSFN variant with early exit (ACT-style) using multicriteria (MC) selection against an SC reference.

The main entrypoints live in `scripts/` and write artifacts under `results/configurations/`.

Note: `scripts/main.py` exists but is not the recommended entrypoint for the current experimental protocol; prefer the scripts documented below.

## What is implemented

### Model variants

- dual-stream backbone (spectral 1d stream + spatial 2d stream)
- optional intermediate cross-attention at configurable stages via `intermediate_attention` and `cross_attention_heads`
- fusion mechanism (current default is `AdaptiveWeight`)
- band selection
  - `SWGMF` (used by the base baseline)
  - `E-FDPC` (used by bca/adaptive configs; band count depends on `efdpc_dc_percent`)
- adaptive early-exit model (`AdaptiveDSSFN`) with depth statistics recorded during evaluation

### Strict comparability and defensive normalization

The evaluation pipeline normalizes parameter files to avoid “silent fallbacks” (for example, a bca param file missing band-selection keys accidentally running with the base `SWGMF` defaults). Channel sizes are capped to 64/128/256 during evaluation and optimization.

## Repository layout

```
.
├─ data/
│  ├─ ip/
│  ├─ pu/
│  └─ botswana/
├─ docs/
├─ scripts/
│  ├─ evaluate_model.py
│  ├─ optimize_model.py
│  ├─ run_full_experiment.py
│  └─ generate_summary.py
├─ src/
└─ results/
   ├─ COMPREHENSIVE_RESULTS_SUMMARY.md
   └─ configurations/
      ├─ base/
      │  ├─ evaluation/
      │  ├─ logs/
      │  └─ maps/
      ├─ bca/
      │  ├─ evaluation/
      │  ├─ logs/
      │  ├─ maps/
      │  └─ optimization/
      ├─ bca_sc_optimized/
      │  ├─ evaluation/
      │  ├─ logs/
      │  └─ maps/
      ├─ adaptive/
      │  └─ optimization/
      └─ adaptive_mc_optimized/
         ├─ evaluation/
         ├─ logs/
         └─ maps/
```

## Requirements and installation

- python 3.10+ (the repo has been used with python 3.12)

Create a virtual environment and install pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `requirements.txt` pins a cuda-enabled PyTorch build. If you are on cpu-only or a different cuda version, install a compatible torch build first (see https://pytorch.org/) and then install the remaining dependencies.

## Data

Place datasets under `data/` using the expected file names:

- `data/ip/indianpinearray.npy` and `data/ip/IPgt.npy`
- `data/pu/PaviaU.mat` and `data/pu/PaviaU_gt.mat`
- `data/botswana/Botswana.mat` and `data/botswana/Botswana_gt.mat`

The loaders support `.npy` and `.mat`. For `.mat` files this repo uses fixed keys internally (see `scripts/evaluate_model.py`, `get_dataset_config`).

## How to run

All commands below are run from the repository root.

### Run a single evaluation

This trains and evaluates one configuration and writes artifacts under `results/configurations/{config_name}/`.

```bash
python scripts/evaluate_model.py --dataset IndianPines --model_type base --config_name base --train_ratio 0.05 --val_ratio 0.15 --seed 42
```

Evaluate bca with a params file (for example, an sc-optimized architecture):

```bash
python scripts/evaluate_model.py --dataset IndianPines --model_type bca --config_name bca_sc_optimized --params_file results/configurations/bca/optimization/best_params_bca_IndianPines.json --train_ratio 0.05 --val_ratio 0.15 --seed 42
```

Reproducibility note:

- evaluation is deterministic by default
- pass `--non_deterministic` to allow faster (but potentially slightly variable) cuda kernels

### Optimize a configuration

Single-criterion bca optimization (writes `best_params_bca_{dataset}.json` under `results/configurations/bca/optimization/`):

```bash
python scripts/optimize_model.py --dataset IndianPines --model_type bca --trials 50 --train_ratio 0.05 --val_ratio 0.15 --no_early_stopping
```

Multicriteria adaptive optimization (writes `best_params_adaptive_{dataset}_mc.json` under `results/configurations/adaptive/optimization/`):

```bash
python scripts/optimize_model.py --dataset IndianPines --model_type adaptive --multicriteria --trials 50 --train_ratio 0.05 --val_ratio 0.15 --no_early_stopping --sc_reference_config bca_sc_optimized
```

### Run the full pipeline

`scripts/run_full_experiment.py` orchestrates the end-to-end pipeline across datasets:

1) evaluate `base`
2) evaluate `bca`
3) optimize `bca` (sc)
4) evaluate `bca_sc_optimized`
5) optimize `adaptive` (mc) using an sc reference
6) evaluate `adaptive_mc_optimized`

Example:

```bash
python scripts/run_full_experiment.py --gpu 0 --trials 20 --seed 42 --datasets IndianPines,PaviaUniversity,Botswana --eval_train_ratios 0.05 0.10 0.70 --val_ratio 0.15 --opt_train_ratio 0.05
```

### Generate the results summary markdown

After evaluations, regenerate `results/COMPREHENSIVE_RESULTS_SUMMARY.md`:

```bash
python scripts/generate_summary.py
```

To generate a lightweight version (no embedded raw JSON / long reports):

```bash
python scripts/generate_summary.py --light
```

This writes `results/COMPREHENSIVE_RESULTS_SUMMARY_LIGHT.md`.

The generated summary:

- selects the latest evaluation JSON per (config, dataset, split)
- bolds best-in-column metrics
- bolds the best model row (by OA) per dataset/split
- embeds the full evaluation JSON and related optimization JSON in collapsible sections

## Results and artifacts

Evaluation artifacts are written per configuration under `results/configurations/{config_name}/`:

- `evaluation/`: `results_{dataset}_tr*_val*_{timestamp}.json` (metrics, params, report)
- `logs/`: `eval_{dataset}_tr*_val*_{timestamp}.log`
- `maps/`: classification maps (when enabled)

Optimization artifacts are written under `results/configurations/{model_type}/optimization/`:

- `best_params_bca_{dataset}.json`
- `best_params_adaptive_{dataset}_mc.json` (includes `mc_selection` metadata and the sc reference used)

## Notes

- if you add new datasets, update `get_dataset_config` in `scripts/evaluate_model.py`.
- most comparisons assume channel caps of 64/128/256; changing these will change compute and should be treated as a new experimental protocol.