# -*- coding: utf-8 -*-
"""
Script to optimize the DSSFN architecture (channel dimensions, heads, E-FDPC) using Optuna.
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
import importlib
import copy

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
from src.model import DSSFN
import src.engine as engine

# --- Setup Logging ---
# Ensure output directory exists
os.makedirs(base_cfg.OUTPUT_DIR, exist_ok=True)

log_filename = f"optimization_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_filepath = os.path.join(base_cfg.OUTPUT_DIR, log_filename)

# Remove any existing handlers and configure fresh (force=True for Python 3.8+)
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

def objective(trial, dataset_name, device, data_cube, gt_map, split_coords_dict, train_ratio):
    """
    Optuna objective function.
    """
    # --- 1. Suggest Hyperparameters ---
    
    # A. E-FDPC DC Percent
    # Range: 1.0% to 10.0%
    efdpc_dc = trial.suggest_float('efdpc_dc_percent', 1.0, 10.0)
    
    # B. Cross Attention Heads
    # Choices: 4, 8 (Must divide channel dimensions)
    # We will ensure channel dimensions are multiples of 8 to be safe for both 4 and 8.
    ca_heads = trial.suggest_categorical('cross_attention_heads', [4, 8])
    
    # C. Channel Dimensions
    # Choices: [16, 32, 64, 128, 256] - all divisible by 8.
    channel_choices = [16, 32, 64, 128, 256]
    
    # Stage 1: Must be equal for Spectral and Spatial if Intermediate Attention is used at Stage 1
    # We assume the "Term Paper" config which uses IA at Stage 1.
    s1_channels = trial.suggest_categorical('s1_channels', channel_choices)
    spec_c1 = s1_channels
    spat_c1 = s1_channels 
    
    # Stage 2: Can be independent (IA not used at Stage 2 in Term Paper config, or if used, we'd need to constrain)
    # Term Paper config: INTERMEDIATE_ATTENTION_STAGES = [1]
    # So Stage 2 is free.
    spec_c2 = trial.suggest_categorical('spec_c2', channel_choices)
    spat_c2 = trial.suggest_categorical('spat_c2', channel_choices)
    
    # Stage 3: Must be equal for Final Fusion (AdaptiveWeight or CrossAttention)
    s3_channels = trial.suggest_categorical('s3_channels', channel_choices)
    spec_c3 = s3_channels
    spat_c3 = s3_channels
    
    spec_channels = [spec_c1, spec_c2, spec_c3]
    spatial_channels = [spat_c1, spat_c2, spat_c3]
    
    # --- 2. Configuration for this Run ---
    run_cfg = {
        'BAND_SELECTION_METHOD': 'E-FDPC',
        'E_FDPC_DC_PERCENT': efdpc_dc,
        'INTERMEDIATE_ATTENTION_STAGES': [1], # Fixed to Term Paper config
        'FUSION_MECHANISM': 'AdaptiveWeight', # Fixed to Term Paper config
        'PATCH_SIZE': 15,
        'BORDER_SIZE': 7,
        'BATCH_SIZE': 64,
        'EPOCHS': 25, # Reduced epochs for optimization speed (was 30)
        'LEARNING_RATE': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'OPTIMIZER_TYPE': 'AdamW',
        'USE_SCHEDULER': True,
        'SCHEDULER_STEP_SIZE': 10,
        'SCHEDULER_GAMMA': 0.1,
        'LOSS_EPSILON': 1e-7,
        'NUM_WORKERS': 0,
        'PIN_MEMORY': True,
        'CROSS_ATTENTION_HEADS': ca_heads,
        'CROSS_ATTENTION_DROPOUT': 0.1
    }
    
    # Update with dataset specifics
    ds_cfg = get_dataset_config(dataset_name)
    run_cfg.update(ds_cfg)
    
    # --- 3. Prepare Data (Band Selection & Splitting) ---
    
    # Band Selection (E-FDPC)
    # This might take a few seconds, but necessary as dc_percent changes
    try:
        selected_data, _ = bs.apply_efdpc(data_cube, run_cfg['E_FDPC_DC_PERCENT'])
    except Exception as e:
        # If E-FDPC fails (e.g. too few bands selected), prune trial
        logging.warning(f"E-FDPC failed with dc={run_cfg['E_FDPC_DC_PERCENT']}: {e}")
        raise optuna.TrialPruned(f"E-FDPC failed: {e}")

    input_bands = selected_data.shape[-1]
    if input_bands < 3:
         raise optuna.TrialPruned(f"Too few bands selected: {input_bands}")
    
    # Preprocessing
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, run_cfg['BORDER_SIZE'])
    
    # Create Patches (using pre-calculated splits)
    data_splits = {}
    for split in ['train', 'val']: 
        coords = split_coords_dict.get(f'{split}_coords', [])
        patches, labels = du.create_patches_from_coords(padded_data, coords, run_cfg['PATCH_SIZE'])
        data_splits[f'{split}_patches'] = patches
        data_splits[f'{split}_labels'] = labels
    
    # Add empty test splits
    data_splits['test_patches'] = []
    data_splits['test_labels'] = []
        
    # DataLoaders
    loaders = ds.create_dataloaders(data_splits, run_cfg['BATCH_SIZE'], run_cfg['NUM_WORKERS'], run_cfg['PIN_MEMORY'])
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    if not train_loader or not val_loader:
        raise optuna.TrialPruned("Empty dataloaders")

    # --- 4. Model Initialization ---
    # Temporarily set config module variables for model init
    base_cfg.INTERMEDIATE_ATTENTION_STAGES = run_cfg['INTERMEDIATE_ATTENTION_STAGES']
    base_cfg.FUSION_MECHANISM = run_cfg['FUSION_MECHANISM']
    
    try:
        model = DSSFN(
            input_bands=input_bands,
            num_classes=run_cfg['NUM_CLASSES'],
            patch_size=run_cfg['PATCH_SIZE'],
            spec_channels=spec_channels,
            spatial_channels=spatial_channels,
            fusion_mechanism=run_cfg['FUSION_MECHANISM'],
            cross_attention_heads=run_cfg['CROSS_ATTENTION_HEADS'],
            cross_attention_dropout=run_cfg['CROSS_ATTENTION_DROPOUT']
        ).to(device)
    except ValueError as e:
        # Invalid configuration (e.g. heads not divisible)
        raise optuna.TrialPruned(f"Invalid config: {e}")

    # --- 5. Training Loop ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=run_cfg['LEARNING_RATE'], weight_decay=run_cfg['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=run_cfg['SCHEDULER_STEP_SIZE'], gamma=run_cfg['SCHEDULER_GAMMA'])
    
    best_val_acc = 0.0
    
    for epoch in range(run_cfg['EPOCHS']):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            if run_cfg['FUSION_MECHANISM'] == 'AdaptiveWeight':
                out1, out2 = model(data)
                loss1 = criterion(out1, target)
                loss2 = criterion(out2, target)
                loss = loss1 + loss2 
            else:
                out = model(data)
                loss = criterion(out, target)
                
            loss.backward()
            optimizer.step()
            
        if scheduler:
            scheduler.step()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                if run_cfg['FUSION_MECHANISM'] == 'AdaptiveWeight':
                    out1, out2 = model(data)
                    output = (out1 + out2) / 2
                else:
                    output = model(data)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)
        
        # Report to Optuna
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return best_val_acc

def run_optimization(dataset_name, n_trials=20, train_ratio=0.1):
    logging.info(f"Starting optimization for {dataset_name} with {n_trials} trials and train_ratio={train_ratio}...")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = get_dataset_config(dataset_name)
    
    # Load Data Once
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Create a fixed split for optimization to ensure comparability
    all_labeled_coords, labels_np_array, original_idx_array = spl.get_labeled_coordinates_and_indices(gt_map)
    split_coords_dict = spl.split_data_random_stratified(
        all_labeled_coords, labels_np_array, original_idx_array,
        train_ratio=train_ratio, val_ratio=0.1, num_classes=ds_cfg['NUM_CLASSES'], random_seed=42
    )
    
    # Define study
    study_name = f"dssfn_arch_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage_url = f"sqlite:///{os.path.join(base_cfg.OUTPUT_DIR, 'optuna_studies.db')}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, dataset_name, device, data_cube, gt_map, split_coords_dict, train_ratio),
        n_trials=n_trials
    )
    
    # Results
    logging.info(f"Best trial for {dataset_name}:")
    logging.info(f"  Value: {study.best_value}")
    logging.info(f"  Params: {study.best_params}")
    
    # Save best params
    output_dir = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"best_params_{dataset_name}.json"), 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    return study.best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize DSSFN architecture.")
    parser.add_argument('--dataset', type=str, default='IndianPines', choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    parser.add_argument('--trials', type=int, default=20, help='Number of trials per dataset')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='Ratio of training data (default: 0.1)')
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset != 'ALL' else ['IndianPines', 'PaviaUniversity', 'Botswana']
    
    for ds_name in datasets:
        run_optimization(ds_name, args.trials, args.train_ratio)
