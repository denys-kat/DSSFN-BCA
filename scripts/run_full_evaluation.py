# -*- coding: utf-8 -*-
"""
Full Evaluation: Adaptive vs Non-Adaptive DSSFN

This script runs comprehensive evaluation comparing:
1. Original DSSFN (non-adaptive, fixed architecture)
2. AdaptiveDSSFN with optimized ACT parameters

Each model is evaluated 3 times per dataset with different random seeds.
Metrics: OA, AA, Kappa, FLOPs, Inference Time, Average Depth (for adaptive)
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
from src.model import DSSFN, AdaptiveDSSFN
from src.engine import train_model, evaluate_model, train_adaptive_model, evaluate_adaptive_model
from src.utils import FLOPsCounter, count_parameters, measure_inference_time

# --- Setup Logging ---
os.makedirs(base_cfg.OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"full_evaluation_{timestamp}.log"
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


def load_phase1_params(dataset_name):
    """Load Phase 1 optimized architecture params (non-adaptive baseline)."""
    params_path = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results', f'best_params_{dataset_name}.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        logging.info(f"Loaded Phase 1 params for {dataset_name}")
        return params
    else:
        logging.warning(f"No Phase 1 params found for {dataset_name}, using defaults")
        return {
            'efdpc_dc_percent': 5.0,
            'cross_attention_heads': 4,
            's1_channels': 128,
            'spec_c2': 128,
            'spat_c2': 128,
            's3_channels': 256
        }


def load_adaptive_params(dataset_name):
    """Load Phase 2 optimized adaptive params (halting bias, ponder cost weight)."""
    params_path = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results', f'adaptive_best_params_{dataset_name}.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded adaptive params for {dataset_name}")
        return data['best_params'], data.get('base_arch_params', {})
    else:
        logging.warning(f"No adaptive params found for {dataset_name}, using defaults")
        return {
            'halting_bias_init': -1.5,
            'ponder_cost_weight': 0.03
        }, {}


def count_flops_for_model(model, input_shape, device):
    """Count FLOPs for a model."""
    try:
        flops_counter = FLOPsCounter(model, input_shape, device='cpu')
        return flops_counter.count()
    except Exception as e:
        logging.warning(f"FLOPs counting failed: {e}")
        return 0


def run_non_adaptive_evaluation(
    dataset_name: str,
    ds_cfg: dict,
    phase1_params: dict,
    data_cube: np.ndarray,
    gt_map: np.ndarray,
    device: torch.device,
    num_runs: int = 3,
    epochs: int = 100,
    early_stopping_patience: int = 25
):
    """Run non-adaptive (original DSSFN) evaluation."""
    logging.info("\n" + "=" * 50)
    logging.info("NON-ADAPTIVE DSSFN (Phase 1 Baseline)")
    logging.info("=" * 50)
    
    # Band selection
    dc_percent = phase1_params.get('efdpc_dc_percent', 5.0)
    selected_data, selected_bands = bs.apply_efdpc(data_cube, dc_percent)
    input_bands = selected_data.shape[-1]
    logging.info(f"E-FDPC selected {input_bands} bands")
    
    # Preprocess
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, base_cfg.BORDER_SIZE)
    
    # Model architecture params - convert to channel lists for DSSFN
    s1_channels = phase1_params.get('s1_channels', 128)
    spec_c2 = phase1_params.get('spec_c2', 128)
    spat_c2 = phase1_params.get('spat_c2', 128)
    s3_channels = phase1_params.get('s3_channels', 256)
    cross_attn_heads = phase1_params.get('cross_attention_heads', 4)
    
    # DSSFN requires spec_channels and spatial_channels as lists
    spec_channels = [s1_channels, spec_c2, s3_channels]
    spatial_channels = [s1_channels, spat_c2, s3_channels]
    logging.info(f"Spec channels: {spec_channels}, Spatial channels: {spatial_channels}")
    
    run_results = []
    
    for run_idx in range(num_runs):
        logging.info(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        
        # Create fresh data splits
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
        
        loaders = ds.create_dataloaders(data_splits, batch_size=64, num_workers=0, pin_memory=True)
        
        # Set intermediate attention stages
        base_cfg.INTERMEDIATE_ATTENTION_STAGES = [1]
        
        # Create non-adaptive DSSFN model
        model = DSSFN(
            input_bands=input_bands,
            num_classes=ds_cfg['NUM_CLASSES'],
            patch_size=base_cfg.PATCH_SIZE,
            spec_channels=spec_channels,
            spatial_channels=spatial_channels,
            cross_attention_heads=cross_attn_heads,
            cross_attention_dropout=0.1
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
        model, history = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            device=device,
            epochs=epochs,
            use_scheduler=True,
            save_best_model=True,
            early_stopping_enabled=True,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric='val_loss',
            early_stopping_min_delta=0.0001
        )
        train_time = time.time() - train_start
        
        # Evaluate
        oa, aa, kappa, report, all_preds, all_labels = evaluate_model(
            model, loaders['test'], device, criterion
        )
        
        # Measure inference time
        model.eval()
        timing = measure_inference_time(
            model, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE),
            device='cuda' if device.type == 'cuda' else 'cpu',
            num_warmup=10, num_runs=100
        )
        
        # Count FLOPs - ensure model is on CPU for FLOPs counting
        model_cpu = model.cpu()
        flops = count_flops_for_model(
            model_cpu, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE), 'cpu'
        )
        model.to(device)  # Move back to GPU
        
        run_result = {
            'OA': oa,
            'AA': aa,
            'Kappa': kappa,
            'train_time': train_time,
            'inference_ms': timing['mean_ms'],
            'inference_std_ms': timing['std_ms'],
            'flops': flops,
            'epochs_trained': len(history['train_loss']),
            'parameters': params['trainable']
        }
        run_results.append(run_result)
        
        logging.info(f"  OA: {run_result['OA']:.4f}, AA: {run_result['AA']:.4f}, Kappa: {run_result['Kappa']:.4f}")
        logging.info(f"  Inference: {run_result['inference_ms']:.3f}ms, FLOPs: {run_result['flops']:,}")
    
    # Aggregate results
    aggregated = {
        'model_type': 'DSSFN (non-adaptive)',
        'runs': run_results,
        'mean_OA': np.mean([r['OA'] for r in run_results]),
        'std_OA': np.std([r['OA'] for r in run_results]),
        'mean_AA': np.mean([r['AA'] for r in run_results]),
        'std_AA': np.std([r['AA'] for r in run_results]),
        'mean_Kappa': np.mean([r['Kappa'] for r in run_results]),
        'std_Kappa': np.std([r['Kappa'] for r in run_results]),
        'mean_inference_ms': np.mean([r['inference_ms'] for r in run_results]),
        'std_inference_ms': np.std([r['inference_ms'] for r in run_results]),
        'mean_flops': np.mean([r['flops'] for r in run_results]),
        'parameters': run_results[0]['parameters'] if run_results else 0
    }
    
    logging.info(f"\nNon-Adaptive Summary: OA={aggregated['mean_OA']:.2%} ± {aggregated['std_OA']:.2%}")
    
    return aggregated, input_bands, padded_data


def run_adaptive_evaluation(
    dataset_name: str,
    ds_cfg: dict,
    phase1_params: dict,
    adaptive_params: dict,
    input_bands: int,
    padded_data: np.ndarray,
    gt_map: np.ndarray,
    device: torch.device,
    num_runs: int = 3,
    epochs: int = 100,
    early_stopping_patience: int = 25
):
    """Run adaptive DSSFN evaluation with optimized ACT parameters."""
    logging.info("\n" + "=" * 50)
    logging.info("ADAPTIVE DSSFN (Phase 2 with ACT)")
    logging.info("=" * 50)
    
    # Adaptive hyperparameters
    halting_bias = adaptive_params.get('halting_bias_init', -1.5)
    ponder_weight = adaptive_params.get('ponder_cost_weight', 0.03)
    logging.info(f"Halting bias: {halting_bias:.4f}, Ponder weight: {ponder_weight:.4f}")
    
    # For AdaptiveDSSFN, use symmetric channels
    s1_channels = phase1_params.get('s1_channels', 128)
    spec_c2 = phase1_params.get('spec_c2', 128)
    spat_c2 = phase1_params.get('spat_c2', 128)
    s2_channels = max(32, min(256, 2 ** round(np.log2((spec_c2 + spat_c2) / 2))))
    s3_channels = phase1_params.get('s3_channels', 256)
    channels = [s1_channels, s2_channels, s3_channels]
    cross_attn_heads = phase1_params.get('cross_attention_heads', 4)
    
    logging.info(f"Symmetric channels: {channels}")
    
    run_results = []
    
    for run_idx in range(num_runs):
        logging.info(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        
        # Create fresh data splits
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
        
        loaders = ds.create_dataloaders(data_splits, batch_size=64, num_workers=0, pin_memory=True)
        
        # Set intermediate attention stages
        base_cfg.INTERMEDIATE_ATTENTION_STAGES = [1]
        
        # Create AdaptiveDSSFN model
        model = AdaptiveDSSFN(
            input_bands=input_bands,
            num_classes=ds_cfg['NUM_CLASSES'],
            patch_size=base_cfg.PATCH_SIZE,
            spec_channels=channels,
            spatial_channels=channels,
            cross_attention_heads=cross_attn_heads,
            cross_attention_dropout=0.1,
            act_epsilon=0.01,
            halting_bias_init=halting_bias
        ).to(device)
        
        # Count parameters
        params = count_parameters(model)
        logging.info(f"Model parameters: {params['trainable']:,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Train with ponder cost
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
        model.eval()
        timing = measure_inference_time(
            model, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE),
            device='cuda' if device.type == 'cuda' else 'cpu',
            num_warmup=10, num_runs=100
        )
        
        # Count FLOPs (approximate - uses full network since ACT is data-dependent)
        flops = count_flops_for_model(
            model, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE), device
        )
        
        run_result = {
            'OA': eval_results['OA'],
            'AA': eval_results['AA'],
            'Kappa': eval_results['Kappa'],
            'avg_depth': eval_results['avg_depth'],
            'stage_distribution': eval_results['stage_distribution'],
            'flops_reduction': eval_results['flops_reduction'],
            'train_time': train_time,
            'inference_ms': timing['mean_ms'],
            'inference_std_ms': timing['std_ms'],
            'flops': flops,
            'epochs_trained': len(history['train_loss']),
            'parameters': params['trainable']
        }
        run_results.append(run_result)
        
        logging.info(f"  OA: {run_result['OA']:.4f}, AA: {run_result['AA']:.4f}, Kappa: {run_result['Kappa']:.4f}")
        logging.info(f"  Avg Depth: {run_result['avg_depth']:.2f}/3, FLOPs↓: {run_result['flops_reduction']:.1%}")
        logging.info(f"  Stage distribution: {run_result['stage_distribution']}")
    
    # Aggregate results
    aggregated = {
        'model_type': 'AdaptiveDSSFN',
        'halting_bias_init': halting_bias,
        'ponder_cost_weight': ponder_weight,
        'runs': run_results,
        'mean_OA': np.mean([r['OA'] for r in run_results]),
        'std_OA': np.std([r['OA'] for r in run_results]),
        'mean_AA': np.mean([r['AA'] for r in run_results]),
        'std_AA': np.std([r['AA'] for r in run_results]),
        'mean_Kappa': np.mean([r['Kappa'] for r in run_results]),
        'std_Kappa': np.std([r['Kappa'] for r in run_results]),
        'mean_depth': np.mean([r['avg_depth'] for r in run_results]),
        'std_depth': np.std([r['avg_depth'] for r in run_results]),
        'mean_flops_reduction': np.mean([r['flops_reduction'] for r in run_results]),
        'std_flops_reduction': np.std([r['flops_reduction'] for r in run_results]),
        'mean_inference_ms': np.mean([r['inference_ms'] for r in run_results]),
        'std_inference_ms': np.std([r['inference_ms'] for r in run_results]),
        'mean_flops': np.mean([r['flops'] for r in run_results]),
        'parameters': run_results[0]['parameters'] if run_results else 0
    }
    
    logging.info(f"\nAdaptive Summary: OA={aggregated['mean_OA']:.2%} ± {aggregated['std_OA']:.2%}")
    logging.info(f"  Depth: {aggregated['mean_depth']:.2f}/3, FLOPs↓: {aggregated['mean_flops_reduction']:.1%}")
    
    return aggregated


def run_full_evaluation(dataset_name: str, device: torch.device, num_runs: int = 3, epochs: int = 100):
    """Run full evaluation for a single dataset."""
    logging.info("\n" + "=" * 60)
    logging.info(f"FULL EVALUATION: {dataset_name}")
    logging.info("=" * 60)
    
    # Load configs and params
    ds_cfg = get_dataset_config(dataset_name)
    phase1_params = load_phase1_params(dataset_name)
    adaptive_params, base_arch_params = load_adaptive_params(dataset_name)
    
    # Load data
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Run non-adaptive evaluation
    non_adaptive_results, input_bands, padded_data = run_non_adaptive_evaluation(
        dataset_name=dataset_name,
        ds_cfg=ds_cfg,
        phase1_params=phase1_params,
        data_cube=data_cube,
        gt_map=gt_map,
        device=device,
        num_runs=num_runs,
        epochs=epochs
    )
    
    # Run adaptive evaluation
    adaptive_results = run_adaptive_evaluation(
        dataset_name=dataset_name,
        ds_cfg=ds_cfg,
        phase1_params=phase1_params,
        adaptive_params=adaptive_params,
        input_bands=input_bands,
        padded_data=padded_data,
        gt_map=gt_map,
        device=device,
        num_runs=num_runs,
        epochs=epochs
    )
    
    return {
        'dataset': dataset_name,
        'non_adaptive': non_adaptive_results,
        'adaptive': adaptive_results
    }


def print_comparison_table(all_results):
    """Print a comparison table of results."""
    logging.info("\n" + "=" * 80)
    logging.info("COMPARISON TABLE: Adaptive vs Non-Adaptive")
    logging.info("=" * 80)
    
    header = f"{'Dataset':<20} {'Model':<15} {'OA (%)':<15} {'AA (%)':<15} {'Kappa':<12} {'Depth':<10} {'Inf(ms)':<10}"
    logging.info(header)
    logging.info("-" * 80)
    
    for dataset_name, results in all_results.items():
        non_adap = results['non_adaptive']
        adap = results['adaptive']
        
        # Non-adaptive row
        logging.info(
            f"{dataset_name:<20} {'DSSFN':<15} "
            f"{non_adap['mean_OA']*100:>5.2f}±{non_adap['std_OA']*100:>4.2f}   "
            f"{non_adap['mean_AA']*100:>5.2f}±{non_adap['std_AA']*100:>4.2f}   "
            f"{non_adap['mean_Kappa']:>5.4f}      "
            f"{'3.00':<10} "
            f"{non_adap['mean_inference_ms']:>6.3f}"
        )
        
        # Adaptive row
        logging.info(
            f"{'':<20} {'AdaptDSSFN':<15} "
            f"{adap['mean_OA']*100:>5.2f}±{adap['std_OA']*100:>4.2f}   "
            f"{adap['mean_AA']*100:>5.2f}±{adap['std_AA']*100:>4.2f}   "
            f"{adap['mean_Kappa']:>5.4f}      "
            f"{adap['mean_depth']:>4.2f}±{adap['std_depth']:>3.2f} "
            f"{adap['mean_inference_ms']:>6.3f}"
        )
        
        # Difference
        oa_diff = (adap['mean_OA'] - non_adap['mean_OA']) * 100
        depth_reduction = (3.0 - adap['mean_depth']) / 3.0 * 100
        logging.info(
            f"{'':<20} {'Δ':<15} "
            f"{oa_diff:>+5.2f}          "
            f"{'':<15} {'':<12} "
            f"{f'-{depth_reduction:.1f}%':<10}"
        )
        logging.info("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Full Evaluation: Adaptive vs Non-Adaptive DSSFN")
    parser.add_argument('--dataset', type=str, default='ALL',
                        choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    parser.add_argument('--runs', type=int, default=3, help='Runs per model per dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Datasets to evaluate
    datasets = [args.dataset] if args.dataset != 'ALL' else ['IndianPines', 'PaviaUniversity', 'Botswana']
    
    all_results = {}
    
    for dataset_name in datasets:
        results = run_full_evaluation(
            dataset_name=dataset_name,
            device=device,
            num_runs=args.runs,
            epochs=args.epochs
        )
        all_results[dataset_name] = results
    
    # Save results
    output_dir = os.path.join(base_cfg.OUTPUT_DIR, 'full_evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f'full_evaluation_{timestamp}.json')
    
    # Convert results to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"\nResults saved to: {results_path}")
    
    # Print comparison table
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
