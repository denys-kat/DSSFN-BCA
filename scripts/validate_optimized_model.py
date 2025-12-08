# -*- coding: utf-8 -*-
"""
Script to validate the optimized DSSFN architecture against the baseline.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def run_validation(dataset_name, optimized_params, baseline_params, train_ratio=0.1):
    logging.info(f"Starting validation for {dataset_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_cfg = get_dataset_config(dataset_name)
    
    # Load Data
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Band Selection (E-FDPC)
    selected_data, _ = bs.apply_efdpc(data_cube, 2.5)
    input_bands = selected_data.shape[-1]
    
    # Preprocessing
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, base_cfg.BORDER_SIZE)
    
    # Splitting
    all_labeled_coords, labels_np_array, original_idx_array = spl.get_labeled_coordinates_and_indices(gt_map)
    split_coords_dict = spl.split_data_random_stratified(
        all_labeled_coords, labels_np_array, original_idx_array,
        train_ratio=train_ratio, val_ratio=0.1, num_classes=ds_cfg['NUM_CLASSES'], random_seed=42
    )
    
    data_splits = {}
    for split in ['train', 'val', 'test']:
        coords = split_coords_dict.get(f'{split}_coords', [])
        patches, labels = du.create_patches_from_coords(padded_data, coords, base_cfg.PATCH_SIZE)
        data_splits[f'{split}_patches'] = patches
        data_splits[f'{split}_labels'] = labels
        
    loaders = ds.create_dataloaders(data_splits, base_cfg.BATCH_SIZE, base_cfg.NUM_WORKERS, base_cfg.PIN_MEMORY)
    train_loader, val_loader, test_loader = loaders['train'], loaders['val'], loaders['test']
    
    results = []
    
    # Define configurations to test
    configs = [
        ('Baseline', baseline_params),
        ('Optimized', optimized_params)
    ]
    
    for name, params in configs:
        logging.info(f"Training {name} model for {dataset_name}...")
        
        # Update global config for model init
        base_cfg.INTERMEDIATE_ATTENTION_STAGES = params['intermediate_attention']
        base_cfg.FUSION_MECHANISM = params['fusion_mechanism']
        
        model = DSSFN(
            input_bands=input_bands,
            num_classes=ds_cfg['NUM_CLASSES'],
            patch_size=base_cfg.PATCH_SIZE,
            spec_channels=params['spec_channels'],
            spatial_channels=params['spatial_channels'],
            fusion_mechanism=params['fusion_mechanism'],
            cross_attention_heads=base_cfg.CROSS_ATTENTION_HEADS,
            cross_attention_dropout=base_cfg.CROSS_ATTENTION_DROPOUT
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=base_cfg.LEARNING_RATE, weight_decay=base_cfg.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=base_cfg.SCHEDULER_STEP_SIZE, gamma=base_cfg.SCHEDULER_GAMMA)
        
        start_time = time.time()
        trained_model, history = engine.train_model(
            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=base_cfg.EPOCHS,
            loss_epsilon=base_cfg.LOSS_EPSILON,
            use_scheduler=base_cfg.USE_SCHEDULER,
            save_best_model=True,
            early_stopping_enabled=base_cfg.EARLY_STOPPING_ENABLED,
            early_stopping_patience=base_cfg.EARLY_STOPPING_PATIENCE,
            early_stopping_metric=base_cfg.EARLY_STOPPING_METRIC,
            early_stopping_min_delta=base_cfg.EARLY_STOPPING_MIN_DELTA
        )
        train_time = time.time() - start_time
        
        # Evaluation
        oa, aa, kappa, report, _, _ = engine.evaluate_model(
            model=trained_model, test_loader=test_loader, device=device,
            criterion=criterion, loss_epsilon=base_cfg.LOSS_EPSILON
        )
        
        results.append({
            'Dataset': dataset_name,
            'Model': name,
            'OA': oa,
            'AA': aa,
            'Kappa': kappa,
            'Train Time (s)': train_time,
            'Params': str(params)
        })
        logging.info(f"{name} Results: OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}, Time={train_time:.2f}s")
        
    return results

if __name__ == "__main__":
    # Load optimized parameters
    opt_results_dir = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results')
    
    datasets = ['IndianPines', 'PaviaUniversity', 'Botswana']
    all_results = []
    
    for ds_name in datasets:
        # Load best params
        with open(os.path.join(opt_results_dir, f"best_params_{ds_name}.json"), 'r') as f:
            best_params_raw = json.load(f)
            
        # Construct optimized config
        optimized_params = {
            'spec_channels': [best_params_raw['s1_channels'], best_params_raw['spec_c2'], best_params_raw['s3_channels']],
            'spatial_channels': [best_params_raw['s1_channels'], best_params_raw['spat_c2'], best_params_raw['s3_channels']],
            'intermediate_attention': [1], # Assuming IA=[1] was fixed during optimization or is the desired config
            'fusion_mechanism': 'AdaptiveWeight'
        }
        
        # Construct baseline config (Term Paper)
        baseline_params = {
            'spec_channels': [64, 128, 256],
            'spatial_channels': [64, 128, 256],
            'intermediate_attention': [1],
            'fusion_mechanism': 'AdaptiveWeight'
        }
        
        ds_results = run_validation(ds_name, optimized_params, baseline_params, train_ratio=0.1) # Use 10% for final validation
        all_results.extend(ds_results)
        
    # Save results
    df = pd.DataFrame(all_results)
    print("\nFinal Validation Results:")
    print(df[['Dataset', 'Model', 'OA', 'AA', 'Kappa', 'Train Time (s)']])
    df.to_csv(os.path.join(base_cfg.OUTPUT_DIR, 'validation_results_comparison.csv'), index=False)
