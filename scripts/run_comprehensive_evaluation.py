# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation: All DSSFN Variants

This script evaluates 4 model variants:
1. DSSFN (Base) - SWGMF fusion, no intermediate cross-attention
2. DSSFN-BCA - Bidirectional Cross-Attention + E-FDPC band selection
3. Adaptive DSSFN-BCA - With ACT (single-objective: accuracy)
4. Adaptive DSSFN-BCA (Multi-Criteria) - Pareto-optimized for accuracy + efficiency

Metrics: OA, AA, Kappa, FLOPs, Inference Time, Parameters, Average Depth
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
log_filename = f"comprehensive_evaluation_{timestamp}.log"
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


def load_phase1_params(dataset_name):
    """Load Phase 1 optimized architecture params."""
    params_path = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results', f'best_params_{dataset_name}.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            return json.load(f)
    return {
        'efdpc_dc_percent': 5.0,
        'cross_attention_heads': 4,
        's1_channels': 128,
        'spec_c2': 128,
        'spat_c2': 128,
        's3_channels': 256
    }


def load_adaptive_params(dataset_name):
    """Load Phase 2 optimized adaptive params."""
    params_path = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results', f'adaptive_best_params_{dataset_name}.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data = json.load(f)
        return data['best_params'], data.get('base_arch_params', {})
    return {'halting_bias_init': -1.5, 'ponder_cost_weight': 0.03}, {}


def load_multicriteria_params(dataset_name):
    """Load multi-criteria optimized params (if available)."""
    params_path = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results', f'multicriteria_best_params_{dataset_name}.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data = json.load(f)
        return data['best_params'], data.get('base_arch_params', {})
    return None, None


def count_flops_safely(model, input_shape, device):
    """Count FLOPs safely, moving model to CPU."""
    try:
        model_cpu = model.cpu()
        model_cpu.eval()
        flops_counter = FLOPsCounter(model_cpu, input_shape, device='cpu')
        flops = flops_counter.count()
        model.to(device)
        return flops
    except Exception as e:
        logging.warning(f"FLOPs counting failed: {e}")
        model.to(device)
        return 0


def run_single_model_evaluation(
    model_name: str,
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    input_bands: int,
    epochs: int = 100,
    patience: int = 25,
    is_adaptive: bool = False,
    ponder_weight: float = 0.0
):
    """Run training and evaluation for a single model."""
    
    # Count parameters
    params = count_parameters(model)
    logging.info(f"  Parameters: {params['trainable']:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Train
    train_start = time.time()
    
    if is_adaptive:
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
            early_stopping_patience=patience,
            early_stopping_metric='val_loss',
            early_stopping_min_delta=0.0001
        )
    else:
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
            early_stopping_patience=patience,
            early_stopping_metric='val_loss',
            early_stopping_min_delta=0.0001
        )
    
    train_time = time.time() - train_start
    
    # Evaluate
    if is_adaptive:
        eval_results = evaluate_adaptive_model(
            model=model,
            test_loader=loaders['test'],
            device=device,
            criterion=criterion
        )
        oa = eval_results['OA']
        aa = eval_results['AA']
        kappa = eval_results['Kappa']
        avg_depth = eval_results['avg_depth']
        stage_dist = eval_results['stage_distribution']
        flops_reduction = eval_results['flops_reduction']
    else:
        oa, aa, kappa, report, all_preds, all_labels = evaluate_model(
            model, loaders['test'], device, criterion
        )
        avg_depth = 3.0
        stage_dist = [0.0, 0.0, 1.0]
        flops_reduction = 0.0
    
    # Measure inference time
    model.eval()
    timing = measure_inference_time(
        model, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE),
        device='cuda' if device.type == 'cuda' else 'cpu',
        num_warmup=10, num_runs=100
    )
    
    # Count FLOPs
    flops = count_flops_safely(
        model, (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE), device
    )
    
    return {
        'OA': oa,
        'AA': aa,
        'Kappa': kappa,
        'avg_depth': avg_depth,
        'stage_distribution': stage_dist,
        'flops_reduction_from_act': flops_reduction,
        'train_time': train_time,
        'inference_ms': timing['mean_ms'],
        'inference_std_ms': timing['std_ms'],
        'flops': flops,
        'epochs_trained': len(history['train_loss']),
        'parameters': params['trainable']
    }


def run_dataset_evaluation(
    dataset_name: str,
    device: torch.device,
    num_runs: int = 3,
    epochs: int = 100,
    include_multicriteria: bool = True
):
    """Run evaluation for all model variants on a single dataset."""
    
    logging.info("\n" + "=" * 70)
    logging.info(f"COMPREHENSIVE EVALUATION: {dataset_name}")
    logging.info("=" * 70)
    
    # Load configs and params
    ds_cfg = get_dataset_config(dataset_name)
    phase1_params = load_phase1_params(dataset_name)
    adaptive_params, _ = load_adaptive_params(dataset_name)
    multicriteria_params, _ = load_multicriteria_params(dataset_name)
    
    # Load data
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Band selection for BCA variants
    dc_percent = phase1_params.get('efdpc_dc_percent', 5.0)
    selected_data, selected_bands = bs.apply_efdpc(data_cube, dc_percent)
    input_bands_bca = selected_data.shape[-1]
    logging.info(f"E-FDPC selected {input_bands_bca} bands for BCA variants")
    
    # Full bands for base DSSFN
    input_bands_base = data_cube.shape[-1]
    logging.info(f"Base DSSFN uses all {input_bands_base} bands")
    
    # Preprocess both
    normalized_data_bca = du.normalize_data(selected_data)
    padded_data_bca = du.pad_data(normalized_data_bca, base_cfg.BORDER_SIZE)
    
    normalized_data_base = du.normalize_data(data_cube)
    padded_data_base = du.pad_data(normalized_data_base, base_cfg.BORDER_SIZE)
    
    # Architecture params
    s1_channels = phase1_params.get('s1_channels', 128)
    spec_c2 = phase1_params.get('spec_c2', 128)
    spat_c2 = phase1_params.get('spat_c2', 128)
    s3_channels = phase1_params.get('s3_channels', 256)
    cross_attn_heads = phase1_params.get('cross_attention_heads', 4)
    
    spec_channels = [s1_channels, spec_c2, s3_channels]
    spatial_channels = [s1_channels, spat_c2, s3_channels]
    
    # For adaptive, use symmetric channels
    s2_channels = max(32, min(256, 2 ** round(np.log2((spec_c2 + spat_c2) / 2))))
    adaptive_channels = [s1_channels, s2_channels, s3_channels]
    
    # Define model variants
    model_variants = [
        {
            'name': 'DSSFN (Base)',
            'short_name': 'Base',
            'use_bca': False,  # No intermediate cross-attention
            'is_adaptive': False,
            'input_bands': input_bands_base,
            'padded_data': padded_data_base,
        },
        {
            'name': 'DSSFN-BCA',
            'short_name': 'BCA',
            'use_bca': True,  # With intermediate cross-attention
            'is_adaptive': False,
            'input_bands': input_bands_bca,
            'padded_data': padded_data_bca,
        },
        {
            'name': 'Adaptive DSSFN-BCA',
            'short_name': 'Adapt',
            'use_bca': True,
            'is_adaptive': True,
            'ponder_weight': adaptive_params.get('ponder_cost_weight', 0.03),
            'halting_bias': adaptive_params.get('halting_bias_init', -1.5),
            'input_bands': input_bands_bca,
            'padded_data': padded_data_bca,
        },
    ]
    
    # Add multi-criteria variant if params available
    if include_multicriteria and multicriteria_params is not None:
        model_variants.append({
            'name': 'Adaptive DSSFN-BCA (MC)',
            'short_name': 'MC',
            'use_bca': True,
            'is_adaptive': True,
            'ponder_weight': multicriteria_params.get('ponder_cost_weight', 0.05),
            'halting_bias': multicriteria_params.get('halting_bias_init', 0.0),
            'input_bands': input_bands_bca,
            'padded_data': padded_data_bca,
        })
    
    results = {
        'dataset': dataset_name,
        'variants': {}
    }
    
    for variant in model_variants:
        logging.info(f"\n{'='*50}")
        logging.info(f"Evaluating: {variant['name']}")
        logging.info(f"{'='*50}")
        
        run_results = []
        
        for run_idx in range(num_runs):
            logging.info(f"\n--- Run {run_idx + 1}/{num_runs} ---")
            
            # Create data splits
            all_coords, labels, orig_idx = spl.get_labeled_coordinates_and_indices(gt_map)
            split_coords = spl.split_data_random_stratified(
                all_coords, labels, orig_idx,
                train_ratio=0.1, val_ratio=0.1,
                num_classes=ds_cfg['NUM_CLASSES'],
                random_seed=42 + run_idx
            )
            
            # Create patches
            padded_data = variant['padded_data']
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
            if variant['use_bca']:
                base_cfg.INTERMEDIATE_ATTENTION_STAGES = [1]  # BCA after stage 1
            else:
                base_cfg.INTERMEDIATE_ATTENTION_STAGES = []  # No intermediate attention
            
            # Create model
            input_bands = variant['input_bands']
            
            if variant['is_adaptive']:
                model = AdaptiveDSSFN(
                    input_bands=input_bands,
                    num_classes=ds_cfg['NUM_CLASSES'],
                    patch_size=base_cfg.PATCH_SIZE,
                    spec_channels=adaptive_channels,
                    spatial_channels=adaptive_channels,
                    cross_attention_heads=cross_attn_heads,
                    cross_attention_dropout=0.1,
                    act_epsilon=0.01,
                    halting_bias_init=variant.get('halting_bias', -1.5)
                ).to(device)
            else:
                model = DSSFN(
                    input_bands=input_bands,
                    num_classes=ds_cfg['NUM_CLASSES'],
                    patch_size=base_cfg.PATCH_SIZE,
                    spec_channels=spec_channels,
                    spatial_channels=spatial_channels,
                    cross_attention_heads=cross_attn_heads,
                    cross_attention_dropout=0.1
                ).to(device)
            
            # Run evaluation
            run_result = run_single_model_evaluation(
                model_name=variant['name'],
                model=model,
                loaders=loaders,
                device=device,
                input_bands=input_bands,
                epochs=epochs,
                patience=25,
                is_adaptive=variant['is_adaptive'],
                ponder_weight=variant.get('ponder_weight', 0.0)
            )
            
            run_results.append(run_result)
            logging.info(f"  OA: {run_result['OA']:.4f}, Depth: {run_result['avg_depth']:.2f}/3, FLOPs: {run_result['flops']:,}")
        
        # Aggregate results
        results['variants'][variant['short_name']] = {
            'full_name': variant['name'],
            'runs': run_results,
            'mean_OA': np.mean([r['OA'] for r in run_results]),
            'std_OA': np.std([r['OA'] for r in run_results]),
            'mean_AA': np.mean([r['AA'] for r in run_results]),
            'std_AA': np.std([r['AA'] for r in run_results]),
            'mean_Kappa': np.mean([r['Kappa'] for r in run_results]),
            'std_Kappa': np.std([r['Kappa'] for r in run_results]),
            'mean_depth': np.mean([r['avg_depth'] for r in run_results]),
            'std_depth': np.std([r['avg_depth'] for r in run_results]),
            'mean_flops': np.mean([r['flops'] for r in run_results]),
            'std_flops': np.std([r['flops'] for r in run_results]),
            'mean_inference_ms': np.mean([r['inference_ms'] for r in run_results]),
            'std_inference_ms': np.std([r['inference_ms'] for r in run_results]),
            'parameters': run_results[0]['parameters'] if run_results else 0,
            'input_bands': variant['input_bands']
        }
    
    return results


def print_comprehensive_table(all_results):
    """Print a comprehensive comparison table with FLOPs."""
    logging.info("\n" + "=" * 120)
    logging.info("COMPREHENSIVE COMPARISON TABLE")
    logging.info("=" * 120)
    
    # Header
    header = (
        f"{'Dataset':<15} {'Model':<25} {'OA (%)':<12} {'AA (%)':<12} "
        f"{'Kappa':<10} {'Depth':<8} {'FLOPs (M)':<12} {'Inf(ms)':<10} {'Bands':<6}"
    )
    logging.info(header)
    logging.info("-" * 120)
    
    for dataset_name, results in all_results.items():
        first_row = True
        
        # Find base FLOPs for relative comparison
        base_flops = None
        for short_name, variant in results['variants'].items():
            if short_name == 'BCA':  # Use BCA as reference for FLOPs comparison
                base_flops = variant['mean_flops']
                break
        
        for short_name, variant in results['variants'].items():
            ds_col = dataset_name if first_row else ''
            first_row = False
            
            flops_m = variant['mean_flops'] / 1e6  # Convert to millions
            flops_rel = ""
            if base_flops and base_flops > 0 and short_name != 'BCA':
                rel = (variant['mean_flops'] - base_flops) / base_flops * 100
                flops_rel = f" ({rel:+.1f}%)"
            
            logging.info(
                f"{ds_col:<15} {variant['full_name']:<25} "
                f"{variant['mean_OA']*100:>5.2f}±{variant['std_OA']*100:<4.2f}  "
                f"{variant['mean_AA']*100:>5.2f}±{variant['std_AA']*100:<4.2f}  "
                f"{variant['mean_Kappa']:>6.4f}   "
                f"{variant['mean_depth']:>4.2f}    "
                f"{flops_m:>6.1f}{flops_rel:<6} "
                f"{variant['mean_inference_ms']:>6.2f}    "
                f"{variant['input_bands']:>4}"
            )
        
        logging.info("-" * 120)
    
    # Print summary insights
    logging.info("\n" + "=" * 80)
    logging.info("KEY INSIGHTS")
    logging.info("=" * 80)
    
    for dataset_name, results in all_results.items():
        logging.info(f"\n{dataset_name}:")
        
        variants = results['variants']
        if 'Base' in variants and 'BCA' in variants:
            base_oa = variants['Base']['mean_OA']
            bca_oa = variants['BCA']['mean_OA']
            oa_improve = (bca_oa - base_oa) * 100
            logging.info(f"  BCA vs Base: OA {oa_improve:+.2f}%")
        
        if 'BCA' in variants and 'Adapt' in variants:
            bca_oa = variants['BCA']['mean_OA']
            adapt_oa = variants['Adapt']['mean_OA']
            adapt_depth = variants['Adapt']['mean_depth']
            adapt_flops = variants['Adapt']['mean_flops']
            bca_flops = variants['BCA']['mean_flops']
            
            oa_diff = (adapt_oa - bca_oa) * 100
            flops_reduction = (bca_flops - adapt_flops) / bca_flops * 100 if bca_flops > 0 else 0
            depth_reduction = (3.0 - adapt_depth) / 3.0 * 100
            
            logging.info(f"  Adaptive vs BCA: OA {oa_diff:+.2f}%, Depth↓ {depth_reduction:.1f}%, FLOPs↓ {flops_reduction:.1f}%")
        
        if 'MC' in variants and 'BCA' in variants:
            mc_oa = variants['MC']['mean_OA']
            mc_depth = variants['MC']['mean_depth']
            mc_flops = variants['MC']['mean_flops']
            bca_flops = variants['BCA']['mean_flops']
            bca_oa = variants['BCA']['mean_OA']
            
            oa_diff = (mc_oa - bca_oa) * 100
            flops_reduction = (bca_flops - mc_flops) / bca_flops * 100 if bca_flops > 0 else 0
            depth_reduction = (3.0 - mc_depth) / 3.0 * 100
            
            logging.info(f"  Multi-Criteria vs BCA: OA {oa_diff:+.2f}%, Depth↓ {depth_reduction:.1f}%, FLOPs↓ {flops_reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation: All DSSFN Variants")
    parser.add_argument('--dataset', type=str, default='ALL',
                        choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    parser.add_argument('--runs', type=int, default=3, help='Runs per model per dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Max training epochs')
    parser.add_argument('--skip-multicriteria', action='store_true', 
                        help='Skip multi-criteria variant if params not available')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    datasets = [args.dataset] if args.dataset != 'ALL' else ['IndianPines', 'PaviaUniversity', 'Botswana']
    
    all_results = {}
    
    for dataset_name in datasets:
        results = run_dataset_evaluation(
            dataset_name=dataset_name,
            device=device,
            num_runs=args.runs,
            epochs=args.epochs,
            include_multicriteria=not args.skip_multicriteria
        )
        all_results[dataset_name] = results
    
    # Save results
    output_dir = os.path.join(base_cfg.OUTPUT_DIR, 'comprehensive_evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f'comprehensive_evaluation_{timestamp}.json')
    
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
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    logging.info(f"\nResults saved to: {results_path}")
    
    # Print comprehensive table
    print_comprehensive_table(all_results)


if __name__ == "__main__":
    main()
