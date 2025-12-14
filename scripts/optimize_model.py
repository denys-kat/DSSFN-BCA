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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import time

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
from src.utils import count_parameters

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
            return json.load(f)
    print(f"WARNING: Could not load BCA params from {path}. Using defaults.")
    # Return sensible defaults if file missing (e.g. for testing)
    return {
        'efdpc_dc_percent': 5.0,
        's1_channels': 64, 'spec_c2': 128, 'spat_c2': 128, 's3_channels': 256,
        'cross_attention_heads': 4
    }

def objective(trial, args, device, data_cube, gt_map, split_coords):
    # --- 1. Hyperparameter Suggestions ---
    
    # Defaults
    attn_stages = []
    
    if args.model_type == 'base':
        # Base: Optimize basic parameters only
        num_bands = trial.suggest_int('swgmf_bands', 10, 60)
        band_selection_method = 'SWGMF'
        dc_percent = None
        
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
        dc_percent = trial.suggest_float('efdpc_dc_percent', 1.0, 10.0)
        
        # Channel search (Restricted)
        s1_channels = trial.suggest_categorical('s1_channels', [16, 32, 64])
        spec_c2 = trial.suggest_categorical('spec_c2', [32, 64, 128])
        spat_c2 = trial.suggest_categorical('spat_c2', [32, 64, 128])
        s3_channels = trial.suggest_categorical('s3_channels', [64, 128, 256])
        spec_c3 = s3_channels; spat_c3 = s3_channels
        
        cross_attn_heads = trial.suggest_categorical('cross_attention_heads', [4, 8])
        attn_stages = [1] # BCA always has stage 1 attn
        
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
        
        attn_stages = [1] # Keep consistent with BCA
        
        # ACT Optimization (Wide Search Space from older scripts)
        halting_bias_init = trial.suggest_float('halting_bias_init', -2.0, 2.0)
        ponder_cost_weight = trial.suggest_float('ponder_cost_weight', 0.0, 0.15)
        # Non-linear efficiency scaling factor
        depth_penalty_scale = trial.suggest_float('depth_penalty_scale', 0.5, 2.0)
        
        # Calculate effective ponder weight dynamically
        effective_ponder = ponder_cost_weight * depth_penalty_scale
    
    else:
        # Fallbacks for non-adaptive types
        halting_bias_init = None; effective_ponder = 0.0; depth_penalty_scale = 1.0

    # --- 2. Data Preparation ---
    try:
        if band_selection_method == 'SWGMF':
            selected_data, _ = bs.apply_swgmf(data_cube, window_size=5, target_bands=num_bands)
        elif band_selection_method == 'E-FDPC':
            selected_data, _ = bs.apply_efdpc(data_cube, dc_percent)
    except Exception:
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
        model = AdaptiveDSSFN(selected_data.shape[-1], ds_cfg['NUM_CLASSES'], base_cfg.PATCH_SIZE, spec_channels, spatial_channels, cross_attn_heads, 0.1, 0.01, halting_bias_init, attn_stages)
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
    
    if args.multicriteria and args.model_type == 'adaptive':
        res = evaluate_adaptive_model(model, loaders['val'], device)
        # Objectives: Accuracy (Max), Efficiency Score (Max)
        # Efficiency Score logic from older scripts: sqrt((3.0 - avg_depth)/3.0)
        # This non-linear scaling rewards early gains in efficiency more
        avg_depth = res['avg_depth']
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
    args = parser.parse_args()
    
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
    if args.multicriteria:
        logging.info(f"Pareto Front size: {len(study.best_trials)}")
        pareto = study.best_trials
        
        max_acc = max(t.values[0] for t in pareto)
        # 0.5% tolerance
        candidates = [t for t in pareto if t.values[0] >= (max_acc - 0.005)]
        best_trial = max(candidates, key=lambda t: t.values[1])
        
        logging.info(f"Absolute Max Acc in Pareto: {max_acc:.4f}")
        logging.info(f"Selected Balanced Trial (Max Efficiency within 0.5% of Max Acc):")
        logging.info(f"   Acc: {best_trial.values[0]:.4f}")
        logging.info(f"   Eff Score: {best_trial.values[1]:.4f}")
        logging.info(f"   Params: {best_trial.params}")
        
        # Reverse calc depth from score: score = (reduction)^0.5 -> reduction = score^2
        if len(best_trial.values) > 1:
             eff_score = best_trial.values[1]
             reduction = eff_score ** 2
             avg_blocks = 3.0 * (1.0 - reduction)
             logging.info(f"   Est. Avg Blocks Used: {avg_blocks:.2f}")

    else:
        best_trial = study.best_trial
        
    params_path = os.path.join(args.output_dir, f"best_params_{args.model_type}_{args.dataset}{mc_suffix}.json")
    
    params_to_save = best_trial.params.copy()
    if avg_blocks is not None:
        params_to_save['avg_blocks'] = avg_blocks
        
    # If adaptive, MERGE fixed params from BCA into result so eval script has everything
    if args.model_type == 'adaptive':
        fixed = load_bca_best_params(args.dataset)
        params_to_save.update(fixed)

    with open(params_path, 'w') as f: json.dump(params_to_save, f, indent=4)
    logging.info(f"Saved best params to {params_path}")

if __name__ == "__main__":
    main()