# -*- coding: utf-8 -*-
"""
Script to optimize the AdaptiveDSSFN architecture using Optuna.
Includes new hyperparameters for Adaptive Computation Time (ACT):
- halting_bias_init: Initial bias for halting module (controls early exit tendency)
- ponder_cost_weight: Lambda for ponder cost regularization
- act_epsilon: Threshold for halting (typically fixed at 0.01)
"""

import os
import sys
import argparse
import logging
import time
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from collections import defaultdict

# --- Add src directory to path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import Custom Modules ---
import src.config as base_cfg
import src.data_utils as du
import src.band_selection as bs
import src.sampling as spl
import src.datasets as ds
from src.model import AdaptiveDSSFN
import src.engine as engine
from src.utils import count_parameters

# --- Setup Logging ---
os.makedirs(base_cfg.OUTPUT_DIR, exist_ok=True)

log_filename = f"adaptive_optimization_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_filepath = os.path.join(base_cfg.OUTPUT_DIR, log_filename)

# Remove any existing handlers and configure fresh
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"Logging to file: {log_filepath}")


def get_dataset_config(dataset_name):
    """Returns the configuration dictionary for a specific dataset."""
    configs = {
        'IndianPines': {
            'DATASET_NAME': 'IndianPines',
            'DATA_PATH_RELATIVE': 'ip/',
            'DATA_FILE': 'indianpinearray.npy',
            'GT_FILE': 'IPgt.npy',
            'NUM_CLASSES': 16,
            'EXPECTED_DATA_SHAPE': (145, 145, 200),
            'EXPECTED_GT_SHAPE': (145, 145),
            'DATA_MAT_KEY': None, 'GT_MAT_KEY': None,
        },
        'PaviaUniversity': {
            'DATASET_NAME': 'PaviaUniversity',
            'DATA_PATH_RELATIVE': 'pu/',
            'DATA_FILE': 'PaviaU.mat',
            'GT_FILE': 'PaviaU_gt.mat',
            'NUM_CLASSES': 9,
            'EXPECTED_DATA_SHAPE': (610, 340, 103),
            'EXPECTED_GT_SHAPE': (610, 340),
            'DATA_MAT_KEY': 'paviaU', 'GT_MAT_KEY': 'paviaU_gt',
        },
        'Botswana': {
            'DATASET_NAME': 'Botswana',
            'DATA_PATH_RELATIVE': 'botswana/',
            'DATA_FILE': 'Botswana.mat',
            'GT_FILE': 'Botswana_gt.mat',
            'NUM_CLASSES': 14,
            'EXPECTED_DATA_SHAPE': (1476, 256, 145),
            'EXPECTED_GT_SHAPE': (1476, 256),
            'DATA_MAT_KEY': 'Botswana', 'GT_MAT_KEY': 'Botswana_gt',
        }
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    cfg = configs[dataset_name]
    cfg['DATA_PATH'] = os.path.join(base_cfg.DATA_BASE_PATH, cfg['DATA_PATH_RELATIVE'])
    return cfg


def get_best_architecture_params(dataset_name):
    """
    Returns the best architecture parameters from Phase 1 optimization.
    These are fixed during adaptive optimization.
    """
    params = {
        'IndianPines': {
            'efdpc_dc_percent': 7.22,
            'cross_attention_heads': 4,
            's1_channels': 128,
            'spec_c2': 128,
            'spat_c2': 64,
            's3_channels': 256,
        },
        'PaviaUniversity': {
            'efdpc_dc_percent': 4.59,
            'cross_attention_heads': 8,
            's1_channels': 128,
            'spec_c2': 128,
            'spat_c2': 256,
            's3_channels': 128,
        },
        'Botswana': {
            'efdpc_dc_percent': 7.30,
            'cross_attention_heads': 4,
            's1_channels': 256,
            'spec_c2': 32,
            'spat_c2': 256,
            's3_channels': 256,
        },
    }
    return params[dataset_name]


def objective(trial, dataset_name, device, data_cube, gt_map, split_coords_dict, 
              base_arch_params, quick_mode=False):
    """
    Optuna objective function for AdaptiveDSSFN.
    
    Optimizes:
    - halting_bias_init: [-2.0, 2.0] - controls early exit tendency
    - ponder_cost_weight: [0.0, 0.1] - regularization strength for ponder cost
    """
    
    # --- 1. Suggest ACT Hyperparameters ---
    
    # Halting bias: more positive = earlier exit tendency
    # Range: -2.0 to 2.0 (was -3.0 which prevented any early exits)
    halting_bias_init = trial.suggest_float('halting_bias_init', -2.0, 2.0)
    
    # Ponder cost weight (lambda): higher = stronger incentive for early exit
    # Range: 0.0 to 0.1 (0.01-0.05 typically works well)
    ponder_cost_weight = trial.suggest_float('ponder_cost_weight', 0.0, 0.1)
    
    # ACT epsilon: typically fixed
    act_epsilon = 0.01
    
    # --- 2. Configuration for this Run ---
    run_cfg = {
        'BAND_SELECTION_METHOD': 'E-FDPC',
        'E_FDPC_DC_PERCENT': base_arch_params['efdpc_dc_percent'],
        'INTERMEDIATE_ATTENTION_STAGES': [1],
        'PATCH_SIZE': 15,
        'BORDER_SIZE': 7,
        'BATCH_SIZE': 64,
        'EPOCHS': 15 if quick_mode else 30,  # 30 epochs for proper optimization
        'LEARNING_RATE': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'OPTIMIZER_TYPE': 'AdamW',
        'USE_SCHEDULER': True,
        'SCHEDULER_STEP_SIZE': 10,
        'SCHEDULER_GAMMA': 0.1,
        'CROSS_ATTENTION_HEADS': base_arch_params['cross_attention_heads'],
        'CROSS_ATTENTION_DROPOUT': 0.1,
        # ACT parameters
        'HALTING_BIAS_INIT': halting_bias_init,
        'PONDER_COST_WEIGHT': ponder_cost_weight,
        'ACT_EPSILON': act_epsilon,
    }
    
    # Update with dataset specifics
    ds_cfg = get_dataset_config(dataset_name)
    run_cfg.update(ds_cfg)
    
    # Channel configurations from Phase 1
    spec_channels = [
        base_arch_params['s1_channels'],
        base_arch_params['spec_c2'],
        base_arch_params['s3_channels']
    ]
    spatial_channels = [
        base_arch_params['s1_channels'],
        base_arch_params['spat_c2'],
        base_arch_params['s3_channels']
    ]
    
    # --- 3. Prepare Data (Band Selection & Splitting) ---
    try:
        selected_data, _ = bs.apply_efdpc(data_cube, run_cfg['E_FDPC_DC_PERCENT'])
    except Exception as e:
        logging.warning(f"E-FDPC failed: {e}")
        raise optuna.TrialPruned(f"E-FDPC failed: {e}")

    input_bands = selected_data.shape[-1]
    if input_bands < 3:
        raise optuna.TrialPruned(f"Too few bands selected: {input_bands}")
    
    # Preprocessing
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, run_cfg['BORDER_SIZE'])
    
    # Create Patches
    data_splits = {}
    for split in ['train', 'val']: 
        coords = split_coords_dict.get(f'{split}_coords', [])
        patches, labels = du.create_patches_from_coords(padded_data, coords, run_cfg['PATCH_SIZE'])
        data_splits[f'{split}_patches'] = patches
        data_splits[f'{split}_labels'] = labels
    
    data_splits['test_patches'] = []
    data_splits['test_labels'] = []
        
    # DataLoaders
    loaders = ds.create_dataloaders(data_splits, run_cfg['BATCH_SIZE'], 0, True)
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    if not train_loader or not val_loader:
        raise optuna.TrialPruned("Empty dataloaders")

    # --- 4. Model Initialization ---
    try:
        model = AdaptiveDSSFN(
            input_bands=input_bands,
            num_classes=run_cfg['NUM_CLASSES'],
            patch_size=run_cfg['PATCH_SIZE'],
            spec_channels=spec_channels,
            spatial_channels=spatial_channels,
            cross_attention_heads=run_cfg['CROSS_ATTENTION_HEADS'],
            cross_attention_dropout=run_cfg['CROSS_ATTENTION_DROPOUT'],
            halting_bias_init=run_cfg['HALTING_BIAS_INIT'],
            act_epsilon=run_cfg['ACT_EPSILON'],
        ).to(device)
    except ValueError as e:
        raise optuna.TrialPruned(f"Invalid config: {e}")

    # --- 5. Training Loop ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=run_cfg['LEARNING_RATE'], 
                           weight_decay=run_cfg['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=run_cfg['SCHEDULER_STEP_SIZE'], 
                                          gamma=run_cfg['SCHEDULER_GAMMA'])
    
    best_val_acc = 0.0
    best_avg_depth = 3.0
    ponder_weight = run_cfg['PONDER_COST_WEIGHT']
    
    for epoch in range(run_cfg['EPOCHS']):
        model.train()
        epoch_depth_sum = 0.0
        epoch_samples = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # Forward pass returns (logits, ponder_cost, halting_step)
            logits, ponder_cost, halting_step = model(data)
            
            # Classification loss
            cls_loss = criterion(logits, target)
            
            # Ponder cost regularization (mean across batch to get scalar)
            ponder_loss = ponder_weight * ponder_cost.mean()
            
            # Total loss
            loss = cls_loss + ponder_loss
            
            loss.backward()
            optimizer.step()
            
            # Track depth
            epoch_depth_sum += halting_step.sum().item()
            epoch_samples += data.size(0)
            
        if scheduler:
            scheduler.step()
        
        avg_train_depth = epoch_depth_sum / epoch_samples if epoch_samples > 0 else 3.0
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_depth_sum = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, _, halting_step = model(data)
                
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                val_depth_sum += halting_step.sum().item()
        
        val_acc = correct / total
        val_avg_depth = val_depth_sum / total if total > 0 else 3.0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_avg_depth = val_avg_depth
        
        # Report to Optuna
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Log trial result
    flops_reduction = (3.0 - best_avg_depth) / 3.0 * 100
    logging.info(f"Trial {trial.number}: bias={halting_bias_init:.2f}, lambda={ponder_weight:.4f} "
                f"-> OA={best_val_acc*100:.2f}%, Depth={best_avg_depth:.2f}, FLOPs↓={flops_reduction:.1f}%")
    
    # Return validation accuracy as primary metric
    # Could also use a composite metric: val_acc - alpha * avg_depth
    return best_val_acc


def run_optimization(dataset_name, n_trials=30, train_ratio=0.1, quick_mode=False):
    """Run Optuna optimization for a single dataset."""
    logging.info(f"{'='*60}")
    logging.info(f"Starting AdaptiveDSSFN optimization for {dataset_name}")
    logging.info(f"Trials: {n_trials}, Train ratio: {train_ratio}, Quick mode: {quick_mode}")
    logging.info(f"{'='*60}")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    ds_cfg = get_dataset_config(dataset_name)
    base_arch_params = get_best_architecture_params(dataset_name)
    
    logging.info(f"Base architecture params (from Phase 1): {base_arch_params}")
    
    # Load Data Once
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Create a fixed split for optimization
    all_labeled_coords, labels_np_array, original_idx_array = spl.get_labeled_coordinates_and_indices(gt_map)
    split_coords_dict = spl.split_data_random_stratified(
        all_labeled_coords, labels_np_array, original_idx_array,
        train_ratio=train_ratio, val_ratio=0.1, num_classes=ds_cfg['NUM_CLASSES'], random_seed=42
    )
    
    logging.info(f"Train samples: {len(split_coords_dict['train_coords'])}, "
                f"Val samples: {len(split_coords_dict['val_coords'])}")
    
    # Define study
    study_name = f"adaptive_dssfn_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_url = f"sqlite:///{os.path.join(base_cfg.OUTPUT_DIR, 'optuna_adaptive_studies.db')}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, dataset_name, device, data_cube, gt_map, 
                               split_coords_dict, base_arch_params, quick_mode),
        n_trials=n_trials
    )
    
    # Results
    logging.info(f"\n{'='*60}")
    logging.info(f"Optimization complete for {dataset_name}")
    logging.info(f"{'='*60}")
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best validation accuracy: {study.best_value * 100:.2f}%")
    logging.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logging.info(f"  {key}: {value}")
    
    # Save best params
    output_dir = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results')
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        'dataset': dataset_name,
        'best_val_accuracy': study.best_value,
        'best_params': study.best_params,
        'base_arch_params': base_arch_params,
        'n_trials': n_trials,
        'train_ratio': train_ratio,
    }
    
    result_path = os.path.join(output_dir, f"adaptive_best_params_{dataset_name}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
    logging.info(f"Saved best params to: {result_path}")
    
    # Also save all trials summary
    trials_summary = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_summary.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params
            })
    
    summary_path = os.path.join(output_dir, f"adaptive_all_trials_{dataset_name}.json")
    with open(summary_path, 'w') as f:
        json.dump(trials_summary, f, indent=4)
        
    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize AdaptiveDSSFN architecture (ACT hyperparameters).")
    parser.add_argument('--dataset', type=str, default='IndianPines', 
                       choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    parser.add_argument('--trials', type=int, default=30, help='Number of trials per dataset')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='Ratio of training data')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer epochs)')
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset != 'ALL' else ['IndianPines', 'PaviaUniversity', 'Botswana']
    
    all_results = {}
    for ds_name in datasets:
        best_params = run_optimization(ds_name, args.trials, args.train_ratio, args.quick)
        all_results[ds_name] = best_params
    
    # Final summary
    logging.info("\n" + "="*60)
    logging.info("FINAL SUMMARY - All Datasets")
    logging.info("="*60)
    for ds_name, params in all_results.items():
        logging.info(f"\n{ds_name}:")
        for key, value in params.items():
            logging.info(f"  {key}: {value}")
