# -*- coding: utf-8 -*-
"""
Multi-Criteria Optimization for Adaptive DSSFN-BCA

This script uses Optuna's multi-objective optimization to find Pareto-optimal
solutions that balance:
1. Classification accuracy (maximize)
2. Computational efficiency / FLOPs reduction (maximize)

The result is a set of Pareto-optimal configurations, from which we select
the best trade-off solution (e.g., knee point of Pareto front).

Key hyperparameters being optimized:
- halting_bias_init: Controls initial tendency to halt early (-3.0 to 3.0)
- ponder_cost_weight (λ): Weight of computational cost in loss (0 to 0.2)
- depth_penalty_mode: How to penalize depth ('linear', 'quadratic', 'exponential')
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
from optuna.samplers import NSGAIISampler
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
from src.model import AdaptiveDSSFN
from src.engine import train_adaptive_model, evaluate_adaptive_model
from src.utils import count_parameters

# --- Setup Logging ---
os.makedirs(base_cfg.OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"multicriteria_optimization_{timestamp}.log"
log_filepath = os.path.join(base_cfg.OUTPUT_DIR, log_filename)

# Setup logging
logger = logging.getLogger(__name__)
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


def compute_efficiency_score(avg_depth, max_depth=3.0):
    """
    Compute efficiency score based on average depth.
    Higher score = more efficient (less computation).
    
    Score ranges from 0 (always full depth) to 1 (always early exit at stage 1).
    """
    # Depth reduction ratio
    depth_reduction = (max_depth - avg_depth) / max_depth
    
    # Apply non-linear scaling to emphasize significant reductions
    # Using quadratic scaling: small reductions are worth less, large reductions worth more
    efficiency_score = depth_reduction ** 0.5  # Square root to be more forgiving
    
    return efficiency_score


def objective(
    trial: optuna.Trial,
    dataset_name: str,
    device: torch.device,
    data_cube: np.ndarray,
    gt_map: np.ndarray,
    ds_cfg: dict,
    phase1_params: dict,
    train_ratio: float = 0.1,
    epochs: int = 50,
    quick_mode: bool = False
):
    """
    Multi-objective function optimizing both accuracy and efficiency.
    
    Returns:
        Tuple of (accuracy, efficiency_score) - both to be maximized
    """
    
    # Sample hyperparameters
    halting_bias_init = trial.suggest_float('halting_bias_init', -2.0, 2.0)
    ponder_cost_weight = trial.suggest_float('ponder_cost_weight', 0.0, 0.15)
    
    # Additional hyperparameters for multi-criteria optimization
    depth_penalty_scale = trial.suggest_float('depth_penalty_scale', 0.5, 2.0)
    
    # Band selection
    dc_percent = phase1_params.get('efdpc_dc_percent', 5.0)
    selected_data, selected_bands = bs.apply_efdpc(data_cube, dc_percent)
    input_bands = selected_data.shape[-1]
    
    # Preprocess
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, base_cfg.BORDER_SIZE)
    
    # Create data splits
    all_coords, labels, orig_idx = spl.get_labeled_coordinates_and_indices(gt_map)
    split_coords = spl.split_data_random_stratified(
        all_coords, labels, orig_idx,
        train_ratio=train_ratio, val_ratio=train_ratio,
        num_classes=ds_cfg['NUM_CLASSES'],
        random_seed=42
    )
    
    # Create patches
    train_patches, train_labels = du.create_patches_from_coords(
        padded_data, split_coords['train_coords'], base_cfg.PATCH_SIZE
    )
    val_patches, val_labels = du.create_patches_from_coords(
        padded_data, split_coords['val_coords'], base_cfg.PATCH_SIZE
    )
    
    data_splits = {
        'train_patches': train_patches, 'train_labels': train_labels,
        'val_patches': val_patches, 'val_labels': val_labels,
        'test_patches': None, 'test_labels': None
    }
    
    loaders = ds.create_dataloaders(data_splits, batch_size=64, num_workers=0, pin_memory=True)
    
    # Architecture params - symmetric channels for adaptive model
    s1_channels = phase1_params.get('s1_channels', 128)
    spec_c2 = phase1_params.get('spec_c2', 128)
    spat_c2 = phase1_params.get('spat_c2', 128)
    s3_channels = phase1_params.get('s3_channels', 256)
    s2_channels = max(32, min(256, 2 ** round(np.log2((spec_c2 + spat_c2) / 2))))
    channels = [s1_channels, s2_channels, s3_channels]
    cross_attn_heads = phase1_params.get('cross_attention_heads', 4)
    
    # Set intermediate attention
    base_cfg.INTERMEDIATE_ATTENTION_STAGES = [1]
    
    # Create model
    model = AdaptiveDSSFN(
        input_bands=input_bands,
        num_classes=ds_cfg['NUM_CLASSES'],
        patch_size=base_cfg.PATCH_SIZE,
        spec_channels=channels,
        spatial_channels=channels,
        cross_attention_heads=cross_attn_heads,
        cross_attention_dropout=0.1,
        act_epsilon=0.01,
        halting_bias_init=halting_bias_init
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Effective ponder cost weight with depth penalty scaling
    effective_ponder_weight = ponder_cost_weight * depth_penalty_scale
    
    # Train
    training_epochs = 30 if quick_mode else epochs
    
    try:
        model, history = train_adaptive_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            device=device,
            epochs=training_epochs,
            ponder_cost_weight=effective_ponder_weight,
            use_scheduler=True,
            save_best_model=True,
            early_stopping_enabled=True,
            early_stopping_patience=15,
            early_stopping_metric='val_loss',
            early_stopping_min_delta=0.0001
        )
    except Exception as e:
        logging.warning(f"Trial {trial.number} failed during training: {e}")
        return 0.0, 0.0  # Return worst objectives
    
    # Evaluate on validation set
    model.eval()
    all_preds = []
    all_labels = []
    total_depth = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for inputs, labels_batch in loaders['val']:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            
            logits, ponder_cost, halting_steps = model(inputs)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            total_depth += halting_steps.sum().item()
            n_samples += inputs.size(0)
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_depth = total_depth / n_samples if n_samples > 0 else 3.0
    efficiency_score = compute_efficiency_score(avg_depth)
    
    logging.info(
        f"Trial {trial.number}: bias={halting_bias_init:.2f}, λ={ponder_cost_weight:.4f}, "
        f"scale={depth_penalty_scale:.2f} -> OA={accuracy:.2%}, Depth={avg_depth:.2f}, Eff={efficiency_score:.3f}"
    )
    
    # Note: trial.report() is not supported for multi-objective optimization
    # Pruning must be handled differently if needed
    
    return accuracy, efficiency_score


def select_best_solution(study: optuna.Study, alpha: float = 0.7):
    """
    Select the best solution from Pareto front using weighted combination.
    
    Args:
        study: Optuna study with completed trials
        alpha: Weight for accuracy (1-alpha for efficiency). Default 0.7 prioritizes accuracy.
    
    Returns:
        Best trial based on weighted score
    """
    pareto_trials = study.best_trials
    
    if not pareto_trials:
        logging.warning("No Pareto-optimal trials found!")
        return None
    
    # Normalize objectives to [0, 1] range
    accuracies = [t.values[0] for t in pareto_trials]
    efficiencies = [t.values[1] for t in pareto_trials]
    
    min_acc, max_acc = min(accuracies), max(accuracies)
    min_eff, max_eff = min(efficiencies), max(efficiencies)
    
    best_trial = None
    best_score = -float('inf')
    
    for trial in pareto_trials:
        # Normalize
        if max_acc > min_acc:
            norm_acc = (trial.values[0] - min_acc) / (max_acc - min_acc)
        else:
            norm_acc = 1.0
        
        if max_eff > min_eff:
            norm_eff = (trial.values[1] - min_eff) / (max_eff - min_eff)
        else:
            norm_eff = 1.0
        
        # Weighted score
        score = alpha * norm_acc + (1 - alpha) * norm_eff
        
        if score > best_score:
            best_score = score
            best_trial = trial
    
    return best_trial


def run_multicriteria_optimization(
    dataset_name: str,
    n_trials: int = 50,
    train_ratio: float = 0.1,
    quick_mode: bool = False,
    alpha: float = 0.7  # Weight for accuracy in final selection
):
    """
    Run multi-criteria optimization for a dataset.
    
    Args:
        dataset_name: Name of dataset
        n_trials: Number of optimization trials
        train_ratio: Train/val ratio
        quick_mode: Use fewer epochs for quick testing
        alpha: Weight for accuracy in final solution selection (0.7 = prioritize accuracy)
    """
    logging.info("=" * 70)
    logging.info(f"Multi-Criteria Optimization for {dataset_name}")
    logging.info(f"Trials: {n_trials}, Alpha (accuracy weight): {alpha}")
    logging.info("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load dataset config and phase 1 params
    ds_cfg = get_dataset_config(dataset_name)
    phase1_params = load_phase1_params(dataset_name)
    logging.info(f"Phase 1 params: {phase1_params}")
    
    # Load data
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Create Optuna study with NSGA-II for multi-objective optimization
    sampler = NSGAIISampler(seed=42)
    study = optuna.create_study(
        study_name=f"multicriteria_{dataset_name}_{timestamp}",
        directions=['maximize', 'maximize'],  # Maximize both accuracy and efficiency
        sampler=sampler,
        storage=f"sqlite:///{base_cfg.OUTPUT_DIR}/optuna_multicriteria.db",
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, dataset_name, device, data_cube, gt_map,
            ds_cfg, phase1_params, train_ratio,
            epochs=50, quick_mode=quick_mode
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get Pareto front
    pareto_trials = study.best_trials
    logging.info(f"\n{'='*50}")
    logging.info(f"Pareto Front ({len(pareto_trials)} solutions)")
    logging.info("=" * 50)
    
    for i, trial in enumerate(pareto_trials):
        logging.info(
            f"  Solution {i+1}: OA={trial.values[0]:.2%}, Eff={trial.values[1]:.3f}, "
            f"bias={trial.params['halting_bias_init']:.2f}, λ={trial.params['ponder_cost_weight']:.4f}"
        )
    
    # Select best solution based on weighted criteria
    best_trial = select_best_solution(study, alpha=alpha)
    
    if best_trial:
        logging.info(f"\n{'='*50}")
        logging.info("SELECTED SOLUTION (Weighted Score)")
        logging.info("=" * 50)
        logging.info(f"  Accuracy: {best_trial.values[0]:.2%}")
        logging.info(f"  Efficiency Score: {best_trial.values[1]:.3f}")
        logging.info(f"  halting_bias_init: {best_trial.params['halting_bias_init']:.4f}")
        logging.info(f"  ponder_cost_weight: {best_trial.params['ponder_cost_weight']:.4f}")
        logging.info(f"  depth_penalty_scale: {best_trial.params['depth_penalty_scale']:.4f}")
        
        # Save best params
        output_dir = os.path.join(base_cfg.OUTPUT_DIR, 'optimization_results')
        os.makedirs(output_dir, exist_ok=True)
        
        best_params = {
            'dataset': dataset_name,
            'best_accuracy': best_trial.values[0],
            'best_efficiency': best_trial.values[1],
            'best_params': {
                'halting_bias_init': best_trial.params['halting_bias_init'],
                'ponder_cost_weight': best_trial.params['ponder_cost_weight'],
                'depth_penalty_scale': best_trial.params['depth_penalty_scale'],
            },
            'base_arch_params': phase1_params,
            'n_trials': n_trials,
            'alpha': alpha,
            'pareto_front_size': len(pareto_trials),
            'all_pareto_solutions': [
                {
                    'accuracy': t.values[0],
                    'efficiency': t.values[1],
                    'params': dict(t.params)
                }
                for t in pareto_trials
            ]
        }
        
        params_path = os.path.join(output_dir, f'multicriteria_best_params_{dataset_name}.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        logging.info(f"\nSaved to: {params_path}")
        
        return best_params
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Multi-Criteria Optimization for Adaptive DSSFN-BCA")
    parser.add_argument('--dataset', type=str, default='IndianPines',
                        choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for accuracy in final selection (0-1). Higher = prioritize accuracy.')
    parser.add_argument('--quick', action='store_true', help='Quick mode with fewer epochs')
    parser.add_argument('--train-ratio', type=float, default=0.1, help='Training data ratio')
    args = parser.parse_args()
    
    datasets = [args.dataset] if args.dataset != 'ALL' else ['IndianPines', 'PaviaUniversity', 'Botswana']
    
    all_results = {}
    
    for dataset_name in datasets:
        result = run_multicriteria_optimization(
            dataset_name=dataset_name,
            n_trials=args.trials,
            train_ratio=args.train_ratio,
            quick_mode=args.quick,
            alpha=args.alpha
        )
        if result:
            all_results[dataset_name] = result
    
    # Print final summary
    logging.info("\n" + "=" * 70)
    logging.info("FINAL SUMMARY - Multi-Criteria Optimization")
    logging.info("=" * 70)
    
    for dataset_name, result in all_results.items():
        logging.info(f"\n{dataset_name}:")
        logging.info(f"  Accuracy: {result['best_accuracy']:.2%}")
        logging.info(f"  Efficiency: {result['best_efficiency']:.3f}")
        logging.info(f"  halting_bias_init: {result['best_params']['halting_bias_init']:.4f}")
        logging.info(f"  ponder_cost_weight: {result['best_params']['ponder_cost_weight']:.4f}")
        logging.info(f"  Pareto front size: {result['pareto_front_size']}")


if __name__ == "__main__":
    main()
