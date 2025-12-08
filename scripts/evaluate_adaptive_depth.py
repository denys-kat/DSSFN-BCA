# -*- coding: utf-8 -*-
"""
Evaluate Adaptive Depth DSSFN vs Fixed Depth DSSFN.

This script compares:
1. AdaptiveDSSFN with ACT (Adaptive Computation Time)
2. AdaptiveDSSFN with fixed full depth (3 stages)
3. Original DSSFN baseline (for reference)

Metrics: OA, AA, Kappa, Average Depth, FLOPs, Inference Time
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
import time

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
from src.model import DSSFN, AdaptiveDSSFN
from src.engine import train_adaptive_model, evaluate_adaptive_model, train_model, evaluate_model
from src.utils import FLOPsCounter, count_parameters, measure_inference_time

# --- Setup Logging ---
os.makedirs(base_cfg.OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"adaptive_depth_evaluation_{timestamp}.log"
log_filepath = os.path.join(base_cfg.OUTPUT_DIR, log_filename)

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

logging.info(f"Logging to: {log_filepath}")


def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset."""
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
            'CLASS_NAMES': [
                "Alfalfa", "Corn-notill", "Corn-min", "Corn", "Grass-pasture",
                "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                "Soybean-notill", "Soybean-min", "Soybean-clean", "Wheat",
                "Woods", "Bldg-Grass-Tree-Drives", "Stone-Steel-Towers"
            ]
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
            'CLASS_NAMES': [
                "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets",
                "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"
            ]
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
            'CLASS_NAMES': [
                "Water", "Hippo grass", "Floodplain grasses 1", "Floodplain grasses 2",
                "Reeds 1", "Riparian", "Firescar 2", "Island interior",
                "Acacia woodlands", "Acacia shrublands", "Acacia grasslands",
                "Short mopane", "Mixed mopane", "Exposed soils"
            ]
        }
    }
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    cfg = configs[dataset_name]
    cfg['DATA_PATH'] = os.path.join(base_cfg.DATA_BASE_PATH, cfg['DATA_PATH_RELATIVE'])
    return cfg


def load_optimized_params(dataset_name):
    """Load optimized hyperparameters from Phase 1 optimization."""
    params_path = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results', f'best_params_{dataset_name}.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        logging.info(f"Loaded optimized params for {dataset_name}: {params}")
        return params
    else:
        logging.warning(f"No optimized params found for {dataset_name}, using defaults")
        return {
            'efdpc_dc_percent': 5.0,
            'cross_attention_heads': 4,
            's1_channels': 128,
            'spec_c2': 128,
            'spat_c2': 128,
            's3_channels': 256
        }


def run_experiment(
    dataset_name: str,
    device: torch.device,
    ponder_cost_weights: list = [0.0, 0.001, 0.01, 0.1],
    num_runs: int = 3,
    epochs: int = 100,
    early_stopping_patience: int = 25
):
    """
    Run adaptive depth experiment on a single dataset.
    
    Args:
        dataset_name: Name of dataset
        device: PyTorch device
        ponder_cost_weights: List of lambda values to test
        num_runs: Number of runs per configuration
        epochs: Max training epochs
        early_stopping_patience: Early stopping patience
        
    Returns:
        Dict with results for all configurations
    """
    logging.info("=" * 60)
    logging.info(f"EXPERIMENT: {dataset_name}")
    logging.info("=" * 60)
    
    # Load dataset config and optimized params
    ds_cfg = get_dataset_config(dataset_name)
    opt_params = load_optimized_params(dataset_name)
    
    # For AdaptiveDSSFN, we need symmetric channels (spec == spat at each stage)
    # Use the average of spec/spat if they differ, or just use s1/s3 which are already symmetric
    s1_channels = opt_params.get('s1_channels', 128)
    # For Stage 2, average spec_c2 and spat_c2, then round to nearest power of 2
    spec_c2 = opt_params.get('spec_c2', 128)
    spat_c2 = opt_params.get('spat_c2', 128)
    s2_channels = max(32, min(256, 2 ** round(np.log2((spec_c2 + spat_c2) / 2))))
    s3_channels = opt_params.get('s3_channels', 256)
    
    channels = [s1_channels, s2_channels, s3_channels]
    logging.info(f"Using symmetric channels: {channels}")
    
    # Load and preprocess data
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Band selection
    dc_percent = opt_params.get('efdpc_dc_percent', 5.0)
    selected_data, selected_bands = bs.apply_efdpc(data_cube, dc_percent)
    input_bands = selected_data.shape[-1]
    logging.info(f"E-FDPC selected {input_bands} bands: {selected_bands}")
    
    # Preprocess
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, base_cfg.BORDER_SIZE)
    
    results = {
        'dataset': dataset_name,
        'input_bands': input_bands,
        'channels': channels,
        'configurations': {}
    }
    
    # Test each ponder cost weight
    for ponder_weight in ponder_cost_weights:
        config_name = f"lambda_{ponder_weight}"
        if ponder_weight == 0.0:
            config_name = "fixed_depth"  # lambda=0 means no penalty for depth
        
        logging.info(f"\n--- Configuration: {config_name} (λ={ponder_weight}) ---")
        
        run_results = []
        
        for run_idx in range(num_runs):
            logging.info(f"\nRun {run_idx + 1}/{num_runs}")
            
            # Create fresh data splits for each run
            all_coords, labels, orig_idx = spl.get_labeled_coordinates_and_indices(gt_map)
            split_coords = spl.split_data_random_stratified(
                all_coords, labels, orig_idx,
                train_ratio=0.1, val_ratio=0.1,
                num_classes=ds_cfg['NUM_CLASSES'],
                random_seed=42 + run_idx
            )
            
            # Create patches
            train_patches, train_labels = du.create_patches_from_coords(
                padded_data, split_coords['train_coords'], base_cfg.PATCH_SIZE
            )
            val_patches, val_labels = du.create_patches_from_coords(
                padded_data, split_coords['val_coords'], base_cfg.PATCH_SIZE
            )
            test_patches, test_labels = du.create_patches_from_coords(
                padded_data, split_coords['test_coords'], base_cfg.PATCH_SIZE
            )
            
            data_splits = {
                'train_patches': train_patches, 'train_labels': train_labels,
                'val_patches': val_patches, 'val_labels': val_labels,
                'test_patches': test_patches, 'test_labels': test_labels
            }
            
            loaders = ds.create_dataloaders(
                data_splits, batch_size=64, num_workers=0, pin_memory=True
            )
            
            # Set intermediate attention stages from config
            base_cfg.INTERMEDIATE_ATTENTION_STAGES = [1]
            
            # Create model
            model = AdaptiveDSSFN(
                input_bands=input_bands,
                num_classes=ds_cfg['NUM_CLASSES'],
                patch_size=base_cfg.PATCH_SIZE,
                spec_channels=channels,
                spatial_channels=channels,
                cross_attention_heads=opt_params.get('cross_attention_heads', 4),
                cross_attention_dropout=0.1,
                act_epsilon=0.01,
                halting_bias_init=-3.0
            ).to(device)
            
            # Count parameters
            params = count_parameters(model)
            logging.info(f"Model parameters: {params['trainable']:,}")
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            
            # Train
            train_start = time.time()
            model, history = train_adaptive_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                device=device,
                epochs=epochs,
                ponder_cost_weight=ponder_weight,
                use_scheduler=True,
                save_best_model=True,
                early_stopping_enabled=True,
                early_stopping_patience=early_stopping_patience,
                early_stopping_metric='val_loss',
                early_stopping_min_delta=0.0001
            )
            train_time = time.time() - train_start
            
            # Evaluate
            eval_results = evaluate_adaptive_model(
                model=model,
                test_loader=loaders['test'],
                device=device,
                criterion=criterion
            )
            
            # Measure inference time
            sample_input = torch.randn(1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE).to(device)
            model.eval()
            timing = measure_inference_time(
                model, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE),
                device='cuda' if device.type == 'cuda' else 'cpu',
                num_warmup=10, num_runs=100
            )
            
            # Count FLOPs
            flops_counter = FLOPsCounter(
                model, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE),
                device='cpu'
            )
            flops = flops_counter.count()
            
            run_result = {
                'OA': eval_results['OA'],
                'AA': eval_results['AA'],
                'Kappa': eval_results['Kappa'],
                'avg_depth': eval_results['avg_depth'],
                'stage_distribution': eval_results['stage_distribution'],
                'flops_reduction': eval_results['flops_reduction'],
                'train_time': train_time,
                'inference_ms': timing['mean_ms'],
                'flops': flops,
                'epochs_trained': len(history['train_loss'])
            }
            run_results.append(run_result)
            
            logging.info(f"  OA: {run_result['OA']:.4f}, Avg Depth: {run_result['avg_depth']:.2f}")
        
        # Aggregate results
        results['configurations'][config_name] = {
            'ponder_weight': ponder_weight,
            'runs': run_results,
            'mean_OA': np.mean([r['OA'] for r in run_results]),
            'std_OA': np.std([r['OA'] for r in run_results]),
            'mean_AA': np.mean([r['AA'] for r in run_results]),
            'std_AA': np.std([r['AA'] for r in run_results]),
            'mean_Kappa': np.mean([r['Kappa'] for r in run_results]),
            'mean_depth': np.mean([r['avg_depth'] for r in run_results]),
            'std_depth': np.std([r['avg_depth'] for r in run_results]),
            'mean_flops_reduction': np.mean([r['flops_reduction'] for r in run_results]),
            'mean_inference_ms': np.mean([r['inference_ms'] for r in run_results])
        }
        
        agg = results['configurations'][config_name]
        logging.info(f"\n{config_name} Summary:")
        logging.info(f"  OA: {agg['mean_OA']:.2%} ± {agg['std_OA']:.2%}")
        logging.info(f"  Avg Depth: {agg['mean_depth']:.2f} ± {agg['std_depth']:.2f}")
        logging.info(f"  FLOPs Reduction: {agg['mean_flops_reduction']:.1%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Adaptive Depth DSSFN")
    parser.add_argument('--dataset', type=str, default='IndianPines',
                        choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    parser.add_argument('--runs', type=int, default=3, help='Runs per configuration')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--patience', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--ponder-weights', type=str, default='0.0,0.001,0.01,0.1',
                        help='Comma-separated ponder cost weights to test')
    args = parser.parse_args()
    
    # Parse ponder weights
    ponder_weights = [float(w) for w in args.ponder_weights.split(',')]
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Datasets to evaluate
    datasets = [args.dataset] if args.dataset != 'ALL' else ['IndianPines', 'PaviaUniversity', 'Botswana']
    
    all_results = {}
    
    for dataset_name in datasets:
        results = run_experiment(
            dataset_name=dataset_name,
            device=device,
            ponder_cost_weights=ponder_weights,
            num_runs=args.runs,
            epochs=args.epochs,
            early_stopping_patience=args.patience
        )
        all_results[dataset_name] = results
    
    # Save results
    output_dir = os.path.join(base_cfg.OUTPUT_DIR, 'adaptive_depth_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f'adaptive_depth_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logging.info(f"\nResults saved to: {results_path}")
    
    # Print final summary
    logging.info("\n" + "=" * 60)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 60)
    
    for dataset_name, results in all_results.items():
        logging.info(f"\n{dataset_name}:")
        for config_name, config_results in results['configurations'].items():
            logging.info(
                f"  {config_name}: OA={config_results['mean_OA']:.2%}, "
                f"Depth={config_results['mean_depth']:.2f}/3, "
                f"FLOPs↓={config_results['mean_flops_reduction']:.1%}"
            )


if __name__ == "__main__":
    main()
