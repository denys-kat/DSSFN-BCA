# scripts/optimize_model.py
# -*- coding: utf-8 -*-
"""
Unified Optimization Script for DSSFN Models.
Updated Folder Structure: results/configurations/{model_type}/optimization/

IMPROVEMENTS:
- Decoupled Optimization: 'adaptive' phase now FREEZES architecture from 'bca' phase.
- Non-Linear Rewards: Uses depth_penalty_scale for dynamic ponder weighting.
- Channel Constraints: Capped at 64/128/256 to prevent efficiency collapse.
"""

import os
import sys
import argparse
import logging
import json
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import time
from typing import Any, Dict, Optional, Tuple
import re

# --- Add src directory to path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.config as base_cfg
import src.data_utils as du
import src.band_selection as bs
import src.sampling as spl
import src.datasets as ds
from src.model import DSSFN, AdaptiveDSSFN
from src.engine import train_model, train_adaptive_model, evaluate_adaptive_model
from src.utils import count_parameters, FLOPsCounter


def _trial_acc(t) -> Optional[float]:
    try:
        if t is None or t.values is None:
            return None
        return float(t.values[0])
    except Exception:
        return None


def _trial_eff_score(t) -> Optional[float]:
    """Higher is better. For MC adaptive trials this is derived from avg_depth."""
    try:
        if t is None or t.values is None or len(t.values) < 2:
            return None
        return float(t.values[1])
    except Exception:
        return None


def _trial_avg_depth(t) -> Optional[float]:
    """Lower is better. Prefer recorded avg_depth; fallback to inverse from eff_score."""
    if t is None:
        return None
    try:
        d = t.user_attrs.get("avg_depth")
        if d is not None:
            return float(d)
    except Exception:
        pass
    eff = _trial_eff_score(t)
    if eff is None:
        return None
    try:
        # eff_score = sqrt((3.0 - avg_depth)/3.0)  =>  avg_depth = 3.0 * (1 - eff^2)
        reduction = float(eff) ** 2
        return 3.0 * (1.0 - reduction)
    except Exception:
        return None


def _acc_key(t) -> float:
    a = _trial_acc(t)
    return float(a) if a is not None else -1e9


def _eff_key(t) -> float:
    e = _trial_eff_score(t)
    return float(e) if e is not None else -1e9


def _depth_key(t) -> float:
    # Lower is better, but for max() tie-breaks we use negative depth.
    d = _trial_avg_depth(t)
    return float(d) if d is not None else 1e9


def _select_mc_trial_tiered(
    pareto_trials,
    acc_tol: float = 1e-3,
    eff_tol: float = 1e-4,
    depth_tol: float = 1e-3,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Select a single Pareto trial using a tiered policy against the study's SC reference.

    SC reference is defined as the maximum-accuracy trial (as a proxy for a single-criterion pick).

    Tiers (higher is better):
      1) beats SC in BOTH accuracy and efficiency (eff_score or avg_depth)
      2) beats SC in efficiency only
      3) beats SC in accuracy only
      4) beats SC in neither

    Within the highest non-empty tier, pick the most balanced solution:
      - prefer higher accuracy
      - then higher efficiency
      - then lower avg_depth
    """
    pareto = [t for t in (pareto_trials or []) if _trial_acc(t) is not None and _trial_eff_score(t) is not None]
    if not pareto:
        return None, {}, {}

    # SC reference: max accuracy among Pareto (always included if Pareto is well-formed).
    # SC reference: maximum accuracy among the (already filtered) Pareto set.
    sc_trial = max(pareto, key=_acc_key)
    sc_acc = _trial_acc(sc_trial)
    sc_eff = _trial_eff_score(sc_trial)
    sc_depth = _trial_avg_depth(sc_trial)

    def beats_acc(t):
        a = _trial_acc(t)
        return (a is not None) and (sc_acc is not None) and (a > sc_acc + acc_tol)

    def beats_eff(t):
        e = _trial_eff_score(t)
        if (e is not None) and (sc_eff is not None) and (e > sc_eff + eff_tol):
            return True
        # Depth fallback (lower is better)
        d = _trial_avg_depth(t)
        if d is not None and sc_depth is not None and d < sc_depth - depth_tol:
            return True
        return False

    tiers = {1: [], 2: [], 3: [], 4: []}
    for t in pareto:
        ba = beats_acc(t)
        be = beats_eff(t)
        if ba and be:
            tiers[1].append(t)
        elif be and (not ba):
            tiers[2].append(t)
        elif ba and (not be):
            tiers[3].append(t)
        else:
            tiers[4].append(t)

    chosen_tier = next((k for k in (1, 2, 3, 4) if tiers[k]), 4)
    candidates = tiers[chosen_tier]

    # Balanced tie-break: high acc first, then high eff_score, then low avg_depth.
    def sort_key(t):
        # Balanced: maximize acc, then eff, then minimize depth.
        return (_acc_key(t), _eff_key(t), -_depth_key(t))

    best = max(candidates, key=sort_key)
    best_acc = _trial_acc(best)
    best_eff = _trial_eff_score(best)
    best_depth = _trial_avg_depth(best)
    sc_ref = {
        "sc_acc": sc_acc,
        "sc_eff_score": sc_eff,
        "sc_avg_depth": sc_depth,
    }
    chosen_meta = {
        "selected_tier": int(chosen_tier),
        "selected_acc": best_acc,
        "selected_eff_score": best_eff,
        "selected_avg_depth": best_depth,
        "delta_acc_vs_sc": (best_acc - sc_acc) if (best_acc is not None and sc_acc is not None) else None,
        "delta_eff_vs_sc": (best_eff - sc_eff) if (best_eff is not None and sc_eff is not None) else None,
        "delta_depth_vs_sc": (best_depth - sc_depth) if (best_depth is not None and sc_depth is not None) else None,
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tolerances": {"acc_tol": acc_tol, "eff_tol": eff_tol, "depth_tol": depth_tol},
    }
    return best, sc_ref, chosen_meta


def _parse_eval_filename(filename: str):
    """Parse `results_{dataset}_tr0p05_val0p15_YYYYMMDD_HHMMSS.json`."""
    m = re.match(
        r"^results_(?P<dataset>.+?)_tr(?P<tr>[0-9p.]+)_val(?P<val>[0-9p.]+)_(?P<ts>\d{8}_\d{6})\.json$",
        filename,
    )
    if not m:
        return None
    ds = m.group("dataset")
    tr = float(m.group("tr").replace("p", "."))
    vr = float(m.group("val").replace("p", "."))
    ts = datetime.datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S")
    return ds, tr, vr, ts


def _load_latest_eval_metrics(config_id: str, dataset: str, train_ratio: float, val_ratio: float) -> Optional[Dict[str, Any]]:
    """Load latest evaluation metrics for a given config/dataset/split.

    Returns dict with at least: {"OA": float, "FLOPs": int|float, "InferenceMs": float} when available.
    """
    eval_dir = os.path.join(base_cfg.OUTPUT_DIR, 'configurations', config_id, 'evaluation')
    if not os.path.isdir(eval_dir):
        return None

    best_path = None
    best_ts = None
    for fn in os.listdir(eval_dir):
        parsed = _parse_eval_filename(fn)
        if not parsed:
            continue
        ds_name, tr, vr, ts = parsed
        if ds_name != dataset:
            continue
        # Match split exactly (these are canonical ratios like 0.05/0.15)
        if abs(tr - float(train_ratio)) > 1e-9 or abs(vr - float(val_ratio)) > 1e-9:
            continue
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best_path = os.path.join(eval_dir, fn)

    if not best_path:
        return None

    try:
        with open(best_path, 'r') as f:
            data = json.load(f)
        metrics = data.get('metrics', {}) or {}
        return {
            "source": {
                "config_id": config_id,
                "path": best_path,
                "timestamp": best_ts.strftime("%Y%m%d_%H%M%S") if best_ts else None,
            },
            "OA": metrics.get('OA'),
            "FLOPs": metrics.get('FLOPs'),
            "InferenceMs": metrics.get('InferenceMs'),
        }
    except Exception:
        return None


def _select_mc_trial_tiered_vs_external_sc(
    pareto_trials,
    sc_eval: Dict[str, Any],
    acc_tol: float = 1e-3,
    flops_rel_tol: float = 0.0,
    depth_tol: float = 1e-3,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Tiered selection of MC Pareto trials vs an external SC reference.

    External SC is loaded from an evaluation JSON, so accuracy uses OA and efficiency uses FLOPs.
    For MC trials, efficiency may be measured by trial_flops (preferred) and/or avg_depth.
    """
    pareto = [t for t in (pareto_trials or []) if _trial_acc(t) is not None]
    if not pareto:
        return None, {}, {}

    sc_acc = sc_eval.get('OA')
    sc_flops = sc_eval.get('FLOPs')
    try:
        sc_acc = float(sc_acc) if sc_acc is not None else None
    except Exception:
        sc_acc = None
    try:
        sc_flops = float(sc_flops) if sc_flops is not None else None
    except Exception:
        sc_flops = None

    def trial_flops(t) -> Optional[float]:
        try:
            f = t.user_attrs.get('flops')
            return float(f) if f is not None else None
        except Exception:
            return None

    def beats_acc(t) -> bool:
        a = _trial_acc(t)
        return (a is not None) and (sc_acc is not None) and (a > sc_acc + acc_tol)

    def beats_eff(t) -> bool:
        # Prefer FLOPs vs SC if available.
        tf = trial_flops(t)
        if tf is not None and sc_flops is not None:
            threshold = sc_flops * (1.0 - float(flops_rel_tol))
            if tf < threshold:
                return True
        # Depth fallback (lower avg_depth indicates early halting). Compare vs full depth=3.
        d = _trial_avg_depth(t)
        if d is not None and d < 3.0 - depth_tol:
            return True
        return False

    tiers = {1: [], 2: [], 3: [], 4: []}
    for t in pareto:
        ba = beats_acc(t)
        be = beats_eff(t)
        if ba and be:
            tiers[1].append(t)
        elif be and (not ba):
            tiers[2].append(t)
        elif ba and (not be):
            tiers[3].append(t)
        else:
            tiers[4].append(t)

    chosen_tier = next((k for k in (1, 2, 3, 4) if tiers[k]), 4)
    candidates = tiers[chosen_tier]

    def sort_key(t):
        # Balanced: maximize acc, then minimize FLOPs (if measured), then maximize eff_score, then minimize depth.
        a = _acc_key(t)
        tf = trial_flops(t)
        tf_key = -(tf if tf is not None else 1e18)  # lower flops => larger key
        e = _eff_key(t)
        d = -_depth_key(t)  # lower depth => larger key
        return (a, tf_key, e, d)

    best = max(candidates, key=sort_key)

    best_acc = _trial_acc(best)
    best_depth = _trial_avg_depth(best)
    best_eff = _trial_eff_score(best)
    best_flops = trial_flops(best)

    sc_ref = {
        "sc_accuracy_metric": "OA",
        "sc_efficiency_metric": "FLOPs",
        "sc_oa": sc_acc,
        "sc_flops": sc_flops,
        "sc_inference_ms": sc_eval.get('InferenceMs'),
        "sc_source": sc_eval.get('source'),
    }

    chosen_meta = {
        "policy": "tiered_vs_external_sc_eval",
        "selected_tier": int(chosen_tier),
        "tier_counts": {k: len(v) for k, v in tiers.items()},
        "tolerances": {"acc_tol": acc_tol, "flops_rel_tol": flops_rel_tol, "depth_tol": depth_tol},
        "selected_acc": best_acc,
        "selected_flops": best_flops,
        "selected_eff_score": best_eff,
        "selected_avg_depth": best_depth,
        "delta_acc_vs_sc": (best_acc - sc_acc) if (best_acc is not None and sc_acc is not None) else None,
        "delta_flops_vs_sc": (best_flops - sc_flops) if (best_flops is not None and sc_flops is not None) else None,
    }
    return best, sc_ref, chosen_meta


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Best-effort reproducibility across numpy/torch/python RNGs."""
    try:
        seed = int(seed)
    except Exception:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Fully deterministic CUDA often requires CUBLAS_WORKSPACE_CONFIG.
        # If it's not set, don't force deterministic algorithms (would error at runtime).
        if torch.cuda.is_available() and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            return

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

def setup_logging(output_dir, log_name):
    os.makedirs(output_dir, exist_ok=True)
    log_filepath = os.path.join(output_dir, log_name)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)])
    return log_filepath

def get_dataset_config(dataset_name):
    # Fixed keys to match main() usage (DATA_FILE, GT_FILE)
    configs = {
        'IndianPines': {'rel_path': 'ip/', 'DATA_FILE': 'indianpinearray.npy', 'GT_FILE': 'IPgt.npy', 'NUM_CLASSES': 16},
        'PaviaUniversity': {'rel_path': 'pu/', 'DATA_FILE': 'PaviaU.mat', 'GT_FILE': 'PaviaU_gt.mat', 'NUM_CLASSES': 9, 'DATA_MAT_KEY': 'paviaU', 'GT_MAT_KEY': 'paviaU_gt'},
        'Botswana': {'rel_path': 'botswana/', 'DATA_FILE': 'Botswana.mat', 'GT_FILE': 'Botswana_gt.mat', 'NUM_CLASSES': 14, 'DATA_MAT_KEY': 'Botswana', 'GT_MAT_KEY': 'Botswana_gt'}
    }
    cfg = configs[dataset_name]
    cfg['DATA_PATH'] = os.path.join(base_cfg.DATA_BASE_PATH, cfg['rel_path'])
    return cfg

def load_bca_best_params(dataset_name):
    """Loads fixed architecture params from the previous BCA optimization phase."""
    # Look for the BCA best params file
    path = os.path.join(base_cfg.OUTPUT_DIR, 'configurations', 'bca', 'optimization', f'best_params_bca_{dataset_name}.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            loaded = json.load(f)
        # Backfill legacy files written before we started persisting these fields.
        loaded.setdefault('band_selection', 'E-FDPC')
        loaded.setdefault('intermediate_attention', [1])
        loaded.setdefault('fusion_mechanism', 'AdaptiveWeight')
        return loaded
    print(f"WARNING: Could not load BCA params from {path}. Using defaults.")
    # Return sensible defaults if file missing (e.g. for testing)
    return {
        'efdpc_dc_percent': 5.0,
        's1_channels': 64, 'spec_c2': 128, 'spat_c2': 128, 's3_channels': 256,
        'cross_attention_heads': 4,
        'band_selection': 'E-FDPC',
        'intermediate_attention': [1],
        'fusion_mechanism': 'AdaptiveWeight',
    }

def objective(trial, args, device, data_cube, gt_map, split_coords):
    # Keep trials comparable and reduce noisy trial-to-trial variance.
    # Note: fixed seed intentionally; differences then mostly come from hyperparameters.
    set_global_seed(42, deterministic=True)
    # --- 1. Hyperparameter Suggestions ---
    
    # Defaults
    attn_stages = []

    # Initialize variables to keep static analyzers happy and ensure robust fallbacks.
    band_selection_method = 'SWGMF'
    num_bands = 30
    dc_percent = 5.0
    s1_channels = 64
    spec_c2 = 128
    spat_c2 = 128
    s3_channels = 256
    spec_c3 = s3_channels
    spat_c3 = s3_channels
    cross_attn_heads = 0
    # ACT defaults (only used for adaptive)
    halting_bias_init = 0.0
    ponder_cost_weight = 0.0
    depth_penalty_scale = 1.0
    act_epsilon = 0.20
    effective_ponder = 0.0
    
    if args.model_type == 'base':
        # Base: Optimize basic parameters only
        num_bands = trial.suggest_int('swgmf_bands', 10, 60)
        band_selection_method = 'SWGMF'
        # Keep a concrete float even if unused in SWGMF mode.
        dc_percent = 5.0
        
        # Channel search (Restricted)
        s1_channels = trial.suggest_categorical('s1_channels', [16, 32, 64])
        spec_c2 = trial.suggest_categorical('spec_c2', [32, 64, 128])
        spat_c2 = trial.suggest_categorical('spat_c2', [32, 64, 128])
        s3_channels = trial.suggest_categorical('s3_channels', [64, 128, 256])
        spec_c3 = s3_channels; spat_c3 = s3_channels
        cross_attn_heads = 0 
        
    elif args.model_type == 'bca':
        # BCA: Optimize Architecture (SC)
        band_selection_method = 'E-FDPC'
        num_bands = None
        # dc_percent controls the E-FDPC cutoff. The best range is dataset-dependent.
        # IndianPines often benefits from a larger cutoff (more bands kept), so don't over-restrict.
        if str(args.dataset).lower() == 'indianpines':
            dc_percent = trial.suggest_float('efdpc_dc_percent', 1.0, 15.0)
        else:
            # For the other datasets, large dc_percent tends to include noisy bands and hurts generalization.
            dc_percent = trial.suggest_float('efdpc_dc_percent', 1.0, 8.0)
        
        # Channel search (Restricted)
        s1_channels = trial.suggest_categorical('s1_channels', [16, 32, 64])
        c2 = trial.suggest_categorical('c2', [32, 64, 128])
        spec_c2 = c2
        spat_c2 = c2
        s3_channels = trial.suggest_categorical('s3_channels', [64, 128, 256])
        spec_c3 = s3_channels; spat_c3 = s3_channels
        
        cross_attn_heads = trial.suggest_categorical('cross_attention_heads', [4, 8])
        # BCA always has stage-1 attention; optionally enable stage-2 as well.
        attn_stages = trial.suggest_categorical('intermediate_attention', [[1], [1, 2]])
        
    elif args.model_type == 'adaptive':
        # Adaptive (MC): FIX Architecture, Optimize ACT
        # Load fixed params from BCA phase
        fixed_params = load_bca_best_params(args.dataset)
        
        band_selection_method = 'E-FDPC'
        num_bands = None
        dc_percent = fixed_params.get('efdpc_dc_percent', 5.0)
        
        # Use FIXED channels/heads
        s1_channels = fixed_params.get('s1_channels', 64)
        spec_c2 = fixed_params.get('spec_c2', 128)
        spat_c2 = fixed_params.get('spat_c2', 128)
        s3_channels = fixed_params.get('s3_channels', 256)
        spec_c3 = s3_channels; spat_c3 = s3_channels
        cross_attn_heads = fixed_params.get('cross_attention_heads', 4)
        
        attn_stages = fixed_params.get('intermediate_attention', [1])
        
        # ACT Optimization (Wide Search Space from older scripts)
        # Wider ranges to actually encourage early halting.
        halting_bias_init = trial.suggest_float('halting_bias_init', -1.5, 3.0)
        ponder_cost_weight = trial.suggest_float('ponder_cost_weight', 0.0, 0.30)
        # Non-linear efficiency scaling factor
        depth_penalty_scale = trial.suggest_float('depth_penalty_scale', 0.5, 3.0)
        # Epsilon directly sets the ACT threshold (1-epsilon). Too small => almost never halts early.
        act_epsilon = trial.suggest_float('act_epsilon', 0.05, 0.60)
        
        # Calculate effective ponder weight dynamically
        effective_ponder = ponder_cost_weight * depth_penalty_scale
    
    else:
        # Fallbacks for non-adaptive types
        # Keep concrete floats (adaptive init expects float types).
        halting_bias_init = 0.0
        effective_ponder = 0.0
        depth_penalty_scale = 1.0
        act_epsilon = 0.20

    # --- 2. Data Preparation ---
    selected_data = None
    try:
        if band_selection_method == 'SWGMF':
            selected_data, _ = bs.apply_swgmf(data_cube, window_size=5, target_bands=num_bands)
        elif band_selection_method == 'E-FDPC':
            selected_data, _ = bs.apply_efdpc(data_cube, float(dc_percent))
    except Exception:
        raise optuna.TrialPruned()

    if selected_data is None:
        raise optuna.TrialPruned()

    if selected_data.shape[-1] < 3: raise optuna.TrialPruned()

    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, base_cfg.BORDER_SIZE)
    train_patches, train_labels = du.create_patches_from_coords(padded_data, split_coords['train_coords'], base_cfg.PATCH_SIZE)
    val_patches, val_labels = du.create_patches_from_coords(padded_data, split_coords['val_coords'], base_cfg.PATCH_SIZE)
    
    loaders = ds.create_dataloaders(
        {'train_patches': train_patches, 'train_labels': train_labels, 'val_patches': val_patches, 'val_labels': val_labels},
        batch_size=64, num_workers=0
    )
    if loaders['train'] is None: raise optuna.TrialPruned()

    # --- 3. Model ---
    ds_cfg = get_dataset_config(args.dataset)
    spec_channels = [s1_channels, spec_c2, spec_c3]
    spatial_channels = [s1_channels, spat_c2, spat_c3]
    
    if args.model_type == 'adaptive':
        # Pass effective ponder weight to model if needed (or just use it in loss calculation)
        # Note: In this script structure, ponder weight is passed to train function, not model init usually, 
        # but model stores init params.
        model = AdaptiveDSSFN(
            selected_data.shape[-1],
            ds_cfg['NUM_CLASSES'],
            base_cfg.PATCH_SIZE,
            spec_channels,
            spatial_channels,
            cross_attn_heads,
            0.1,
            act_epsilon,
            halting_bias_init,
            attn_stages,
        )
    else:
        model = DSSFN(selected_data.shape[-1], ds_cfg['NUM_CLASSES'], base_cfg.PATCH_SIZE, spec_channels, spatial_channels, 'AdaptiveWeight', cross_attn_heads, 0.1, attn_stages)
    
    model = model.to(device)
    
    # --- 4. Train ---
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
    
    early_stopping = not args.no_early_stopping
    
    if args.model_type == 'adaptive':
        # Pass effective ponder weight (scaled)
        model, hist = train_adaptive_model(model, crit, opt, sched, loaders['train'], loaders['val'], device, 25, effective_ponder, early_stopping_enabled=early_stopping)
    else:
        model, hist = train_model(model, crit, opt, sched, loaders['train'], loaders['val'], device, 25, early_stopping_enabled=early_stopping)
        
    best_acc = max(hist['val_accuracy'])

    # Record trial compute proxy for tie-breaking / constraints.
    try:
        dummy_shape = (1, selected_data.shape[-1], base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE)
        trial_flops = int(FLOPsCounter(model, dummy_shape, device).count())
        trial.set_user_attr("flops", trial_flops)
        trial.set_user_attr("input_bands", int(selected_data.shape[-1]))
    except Exception:
        pass
    
    if args.multicriteria and args.model_type == 'adaptive':
        res = evaluate_adaptive_model(model, loaders['val'], device)
        # Objectives: Accuracy (Max), Efficiency Score (Max)
        # Efficiency Score logic from older scripts: sqrt((3.0 - avg_depth)/3.0)
        # This non-linear scaling rewards early gains in efficiency more
        avg_depth = res['avg_depth']
        trial.set_user_attr("avg_depth", float(avg_depth))
        efficiency_score = ((3.0 - avg_depth) / 3.0) ** 0.5 
        return best_acc, efficiency_score
    
    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model_type', required=True, choices=['base', 'bca', 'adaptive'])
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--train_ratio', type=float, default=0.05)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--multicriteria', action='store_true')
    parser.add_argument('--no_early_stopping', action='store_true')
    parser.add_argument('--output_dir', default='results/optimization')
    # When selecting an MC point, compare against an *actual* SC evaluation run.
    # Default aligns with current reporting (SC = DSSFN-BCA (Optimized)).
    parser.add_argument('--sc_reference_config', default='bca_sc_optimized')
    args = parser.parse_args()

    # Global determinism for the whole optimization run.
    set_global_seed(42, deterministic=True)
    
    if args.output_dir == 'results/optimization':
        args.output_dir = os.path.join(base_cfg.OUTPUT_DIR, 'configurations', args.model_type, 'optimization')

    mc_suffix = "_mc" if args.multicriteria else ""
    log_file = setup_logging(args.output_dir, f"opt_{args.model_type}_{args.dataset}{mc_suffix}.log")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Optimizing {args.model_type} on {args.dataset} (MC={args.multicriteria}). Out: {args.output_dir}")
    
    ds_cfg = get_dataset_config(args.dataset)
    if args.dataset == 'IndianPines':
        data = np.load(os.path.join(ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE']))
        gt = np.load(os.path.join(ds_cfg['DATA_PATH'], ds_cfg['GT_FILE']))
    else:
        import scipy.io
        data = scipy.io.loadmat(os.path.join(ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE']))[ds_cfg['DATA_MAT_KEY']]
        gt = scipy.io.loadmat(os.path.join(ds_cfg['DATA_PATH'], ds_cfg['GT_FILE']))[ds_cfg['GT_MAT_KEY']]
        
    all_coords, labels, orig_idx = spl.get_labeled_coordinates_and_indices(gt)
    split = spl.split_data_random_stratified(all_coords, labels, orig_idx, args.train_ratio, args.val_ratio, ds_cfg['NUM_CLASSES'], 42)
    
    if args.multicriteria:
        study = optuna.create_study(directions=['maximize', 'maximize'])
    else:
        study = optuna.create_study(direction='maximize')
    
    study.optimize(lambda t: objective(t, args, device, data, gt, split), n_trials=args.trials)
    
    avg_blocks = None
    # Initialized for traceability persistence when multicriteria is enabled.
    sc_ref = {}
    chosen_meta = {}
    if args.multicriteria:
        logging.info(f"Pareto Front size: {len(study.best_trials)}")
        pareto = study.best_trials

        # Prefer tiering vs an *external* SC reference from evaluation JSONs.
        sc_eval = _load_latest_eval_metrics(
            config_id=str(args.sc_reference_config),
            dataset=args.dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        if sc_eval is not None:
            best_trial, sc_ref, chosen_meta = _select_mc_trial_tiered_vs_external_sc(
                pareto,
                sc_eval=sc_eval,
                acc_tol=1e-3,
                flops_rel_tol=0.0,
                depth_tol=1e-3,
            )
        else:
            # Fallback: tier vs the study's own max-accuracy reference.
            best_trial, sc_ref, chosen_meta = _select_mc_trial_tiered(
                pareto,
                acc_tol=1e-3,
                eff_tol=1e-4,
                depth_tol=1e-3,
            )
        if best_trial is None:
            # Fallback to previous behavior if something goes wrong.
            max_acc = max(t.values[0] for t in pareto)
            candidates = [t for t in pareto if t.values[0] >= (max_acc - 0.010)]
            best_trial = max(candidates, key=lambda t: t.values[1])
            sc_ref = {"sc_acc": float(max_acc), "sc_eff_score": None, "sc_avg_depth": None}
            chosen_meta = {"selected_tier": None, "tier_counts": None}

        logging.info("MC selection policy: tiered vs SC reference")
        logging.info(f"   SC reference config: {args.sc_reference_config}")
        logging.info(f"   SC ref: {sc_ref}")

        if isinstance(chosen_meta, dict):
            logging.info(f"   Tier counts: {chosen_meta.get('tier_counts')}")
            logging.info(f"   Selected tier: {chosen_meta.get('selected_tier')}")
            sa = chosen_meta.get("selected_acc")
            se = chosen_meta.get("selected_eff_score")
            sd = chosen_meta.get("selected_avg_depth")
            if isinstance(sa, (int, float)) and isinstance(se, (int, float)):
                logging.info(f"   Selected acc={float(sa):.4f} eff={float(se):.4f} avg_depth={sd}")
            else:
                logging.info(f"   Selected meta: {chosen_meta}")
        else:
            logging.info(f"   Selected meta: {chosen_meta}")
        logging.info(f"   Params: {best_trial.params}")

        # Derive avg_blocks estimate for saving (approx == avg_depth).
        avg_blocks = _trial_avg_depth(best_trial)
        if avg_blocks is not None:
            logging.info(f"   Est. Avg Blocks Used: {avg_blocks:.2f}")

    else:
        # Single-criterion: pick best accuracy, then prefer lower FLOPs among near-ties.
        best_trial = study.best_trial
        try:
            max_acc = max(t.value for t in study.trials if t.value is not None)
            candidates = [t for t in study.trials if t.value is not None and t.value >= (max_acc - 0.001)]
            candidates_with_flops = [t for t in candidates if t.user_attrs.get("flops") is not None]
            if candidates_with_flops:
                best_trial = min(candidates_with_flops, key=lambda t: int(t.user_attrs["flops"]))
        except Exception:
            pass
        
    params_path = os.path.join(args.output_dir, f"best_params_{args.model_type}_{args.dataset}{mc_suffix}.json")
    
    params_to_save = best_trial.params.copy()
    if avg_blocks is not None:
        params_to_save['avg_blocks'] = avg_blocks
    # Persist MC selection context for traceability.
    if args.multicriteria:
        try:
            params_to_save["mc_selection"] = {
                "policy": chosen_meta.get("policy", "tiered_vs_sc_reference") if isinstance(chosen_meta, dict) else "tiered_vs_sc_reference",
                "sc_reference": sc_ref,
                "selection": chosen_meta,
                "sc_reference_config": str(args.sc_reference_config),
                "sc_reference_split": {"dataset": args.dataset, "train_ratio": args.train_ratio, "val_ratio": args.val_ratio},
            }
        except Exception:
            pass

    # Expand tied/derived params for downstream evaluation scripts.
    if "c2" in params_to_save and ("spec_c2" not in params_to_save and "spat_c2" not in params_to_save):
        params_to_save["spec_c2"] = params_to_save["c2"]
        params_to_save["spat_c2"] = params_to_save["c2"]
        del params_to_save["c2"]
        
    # If adaptive, MERGE fixed params from BCA into result so eval script has everything
    if args.model_type == 'adaptive':
        fixed = load_bca_best_params(args.dataset)
        params_to_save.update(fixed)
        # Ensure downstream evaluation is deterministic and comparable.
        params_to_save["band_selection"] = "E-FDPC"
        params_to_save.setdefault("intermediate_attention", fixed.get("intermediate_attention", [1]))
        params_to_save.setdefault("fusion_mechanism", fixed.get("fusion_mechanism", "AdaptiveWeight"))
        params_to_save.setdefault("act_epsilon", 0.20)
    elif args.model_type == 'bca':
        params_to_save["band_selection"] = "E-FDPC"
        params_to_save.setdefault("intermediate_attention", [1])
        params_to_save.setdefault("fusion_mechanism", "AdaptiveWeight")
    elif args.model_type == 'base':
        params_to_save.setdefault("band_selection", "SWGMF")
        params_to_save.setdefault("intermediate_attention", [])
        params_to_save.setdefault("fusion_mechanism", "AdaptiveWeight")

    with open(params_path, 'w') as f: json.dump(params_to_save, f, indent=4)
    logging.info(f"Saved best params to {params_path}")

if __name__ == "__main__":
    main()
