# -*- coding: utf-8 -*-
"""
Unified Multi-Criteria Optimization and Evaluation Script

This script runs multi-criteria optimization with configurable alpha weight
and then evaluates the resulting model against other variants.

Results are saved in the organized results directory structure.
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
from tqdm import tqdm
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
from src.engine import train_adaptive_model, evaluate_adaptive_model


# === RESULTS DIRECTORY STRUCTURE ===
RESULTS_BASE = os.path.join(project_root, 'results')
OPTIMIZATION_DIR = os.path.join(RESULTS_BASE, 'phase3_multicriteria')
EVALUATION_DIR = os.path.join(RESULTS_BASE, 'evaluations', 'multicriteria')
LOGS_DIR = os.path.join(RESULTS_BASE, 'logs', 'multicriteria')
DATABASE_DIR = os.path.join(RESULTS_BASE, 'databases')

for d in [OPTIMIZATION_DIR, EVALUATION_DIR, LOGS_DIR, DATABASE_DIR]:
    os.makedirs(d, exist_ok=True)


def setup_logging(dataset_name: str, alpha: float):
    """Setup logging with organized file output."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"mc_alpha{alpha:.2f}_{dataset_name}_{timestamp}.log"
    log_filepath = os.path.join(LOGS_DIR, log_filename)
    
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
    return timestamp


def get_dataset_config(dataset_name: str):
    """Get configuration for a specific dataset."""
    configs = {
        'IndianPines': {
            'DATASET_NAME': 'IndianPines',
            'DATA_PATH_RELATIVE': 'ip/',
            'DATA_FILE': 'indianpinearray.npy',
            'GT_FILE': 'IPgt.npy',
            'NUM_CLASSES': 16,
            'DATA_MAT_KEY': None, 'GT_MAT_KEY': None,
        },
        'PaviaUniversity': {
            'DATASET_NAME': 'PaviaUniversity',
            'DATA_PATH_RELATIVE': 'pu/',
            'DATA_FILE': 'PaviaU.mat',
            'GT_FILE': 'PaviaU_gt.mat',
            'NUM_CLASSES': 9,
            'DATA_MAT_KEY': 'paviaU', 'GT_MAT_KEY': 'paviaU_gt',
        },
        'Botswana': {
            'DATASET_NAME': 'Botswana',
            'DATA_PATH_RELATIVE': 'botswana/',
            'DATA_FILE': 'Botswana.mat',
            'GT_FILE': 'Botswana_gt.mat',
            'NUM_CLASSES': 14,
            'DATA_MAT_KEY': 'Botswana', 'GT_MAT_KEY': 'Botswana_gt',
        }
    }
    return configs[dataset_name]


def load_phase1_params(dataset_name: str):
    """Load Phase 1 optimized architecture parameters."""
    params_path = os.path.join(RESULTS_BASE, 'phase1_optimization', 'optimization_results', f'best_params_{dataset_name}.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            data = json.load(f)
        return data.get('best_params', data)
    
    # Fallback defaults
    return {
        'efdpc_dc_percent': 5.0,
        'cross_attention_heads': 4,
        's1_channels': 128,
        'spec_c2': 128,
        'spat_c2': 128,
        's3_channels': 256
    }


def compute_efficiency_score(avg_depth: float, max_depth: int = 3) -> float:
    """Compute efficiency score from average depth."""
    return 1.0 - (avg_depth / max_depth)


def prepare_data(dataset_name: str, phase1_params: dict, train_ratio: float = 0.1):
    """Prepare data with band selection."""
    cfg = get_dataset_config(dataset_name)
    
    # Update base_cfg
    for key, value in cfg.items():
        setattr(base_cfg, key, value)
    base_cfg.DATA_PATH = os.path.join(base_cfg.PROJECT_ROOT, 'data', cfg['DATA_PATH_RELATIVE'])
    base_cfg.TRAIN_RATIO = train_ratio
    base_cfg.VAL_RATIO = train_ratio
    
    # Load data using proper function
    data_cube, gt_map = du.load_hyperspectral_data(
        base_cfg.DATA_PATH,
        cfg['DATA_FILE'],
        cfg['GT_FILE'],
        expected_data_shape=None,
        expected_gt_shape=None,
        data_mat_key=cfg.get('DATA_MAT_KEY'),
        gt_mat_key=cfg.get('GT_MAT_KEY')
    )
    
    # Apply E-FDPC band selection
    dc_percent = phase1_params.get('efdpc_dc_percent', 5.0)
    data_selected, selected_bands = bs.apply_efdpc(data_cube, dc_percent=dc_percent)
    
    # Normalize and prepare patches
    data_normalized = du.normalize_data(data_selected)
    data_padded = du.pad_data(data_normalized, base_cfg.BORDER_SIZE)
    
    # Get labeled pixel coordinates using correct function
    all_labeled_coords, labels_np_array, original_idx_array = spl.get_labeled_coordinates_and_indices(gt_map)
    if not all_labeled_coords:
        raise ValueError("No labeled pixels found in ground truth map")
    
    # Split data using stratified sampling
    split_coords_dict = spl.split_data_random_stratified(
        all_labeled_coords, labels_np_array, original_idx_array,
        base_cfg.TRAIN_RATIO, base_cfg.VAL_RATIO, base_cfg.NUM_CLASSES, base_cfg.RANDOM_SEED
    )
    
    train_patches, train_labels = du.create_patches_from_coords(
        data_padded, split_coords_dict['train_coords'], base_cfg.PATCH_SIZE
    )
    val_patches, val_labels = du.create_patches_from_coords(
        data_padded, split_coords_dict['val_coords'], base_cfg.PATCH_SIZE
    )
    test_patches, test_labels = du.create_patches_from_coords(
        data_padded, split_coords_dict['test_coords'], base_cfg.PATCH_SIZE
    )
    
    train_dataset = ds.HyperspectralDataset(train_patches, train_labels)
    val_dataset = ds.HyperspectralDataset(val_patches, val_labels)
    test_dataset = ds.HyperspectralDataset(test_patches, test_labels)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=base_cfg.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=base_cfg.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=base_cfg.BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, len(selected_bands), cfg['NUM_CLASSES']


def run_optimization(
    dataset_name: str,
    n_trials: int = 50,
    alpha: float = 0.5,
    epochs: int = 50,
    train_ratio: float = 0.1
):
    """Run multi-criteria optimization for a dataset."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load Phase 1 params
    phase1_params = load_phase1_params(dataset_name)
    logging.info(f"Phase 1 params: {phase1_params}")
    
    # Prepare data once
    loaders, input_bands, num_classes = prepare_data(dataset_name, phase1_params, train_ratio)
    
    # Create Optuna study
    db_path = os.path.join(DATABASE_DIR, f'optuna_mc_alpha{alpha:.2f}.db')
    study_name = f"mc_alpha{alpha:.2f}_{dataset_name}"
    
    study = optuna.create_study(
        study_name=study_name,
        directions=['maximize', 'maximize'],  # accuracy, efficiency
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    
    def objective(trial):
        # Hyperparameters to optimize
        halting_bias_init = trial.suggest_float('halting_bias_init', -3.0, 3.0)
        ponder_cost_weight = trial.suggest_float('ponder_cost_weight', 0.0, 0.3)
        depth_penalty_scale = trial.suggest_float('depth_penalty_scale', 0.5, 3.0)
        
        # Scale ponder cost by alpha (lower alpha = more aggressive efficiency push)
        effective_ponder_weight = ponder_cost_weight * (1 + (1 - alpha) * depth_penalty_scale)
        
        # Create model
        model = AdaptiveDSSFN(
            input_bands=input_bands,
            num_classes=num_classes,
            patch_size=base_cfg.PATCH_SIZE,
            spec_channels=[phase1_params.get('s1_channels', 128), phase1_params.get('spec_c2', 128), phase1_params.get('s3_channels', 256)],
            spatial_channels=[phase1_params.get('s1_channels', 128), phase1_params.get('spat_c2', 128), phase1_params.get('s3_channels', 256)],
            cross_attention_heads=phase1_params.get('cross_attention_heads', 4),
            halting_bias_init=halting_bias_init,
            act_epsilon=0.01
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
        
        # Training
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            for inputs, labels in loaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs, ponder_cost, halting_steps = model(inputs)
                ce_loss = criterion(outputs, labels)
                loss = ce_loss + effective_ponder_weight * ponder_cost.mean()
                
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            total_depth = 0
            
            with torch.no_grad():
                for inputs, labels in loaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, ponder_cost, halting_steps = model(inputs)
                    
                    ce_loss = criterion(outputs, labels)
                    val_loss += ce_loss.item()
                    
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
                    total_depth += halting_steps.sum().item()
            
            val_loss /= len(loaders['val'])
            val_acc = val_correct / val_total
            avg_depth = total_depth / val_total
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break
        
        # Load best model and evaluate on test set
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        model.eval()
        all_preds = []
        all_labels = []
        total_depth = 0
        n_samples = 0
        
        with torch.no_grad():
            for inputs, labels in loaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, halting_steps, _ = model(inputs)
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_depth += halting_steps.sum().item()
                n_samples += inputs.size(0)
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_depth = total_depth / n_samples
        efficiency_score = compute_efficiency_score(avg_depth)
        
        logging.info(
            f"Trial {trial.number}: bias={halting_bias_init:.2f}, λ={ponder_cost_weight:.4f}, "
            f"scale={depth_penalty_scale:.2f} -> OA={accuracy:.2%}, Depth={avg_depth:.2f}, Eff={efficiency_score:.3f}"
        )
        
        return accuracy, efficiency_score
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get Pareto front
    pareto_trials = study.best_trials
    
    if pareto_trials:
        # Select best using weighted score
        best_trial = max(
            pareto_trials,
            key=lambda t: alpha * t.values[0] + (1 - alpha) * t.values[1]
        )
        
        logging.info(f"\n{'='*50}")
        logging.info(f"SELECTED SOLUTION (alpha={alpha})")
        logging.info(f"{'='*50}")
        logging.info(f"  Accuracy: {best_trial.values[0]:.2%}")
        logging.info(f"  Efficiency Score: {best_trial.values[1]:.3f}")
        logging.info(f"  halting_bias_init: {best_trial.params['halting_bias_init']:.4f}")
        logging.info(f"  ponder_cost_weight: {best_trial.params['ponder_cost_weight']:.4f}")
        logging.info(f"  depth_penalty_scale: {best_trial.params['depth_penalty_scale']:.4f}")
        
        # Save results
        result = {
            'dataset': dataset_name,
            'alpha': alpha,
            'n_trials': n_trials,
            'best_accuracy': best_trial.values[0],
            'best_efficiency': best_trial.values[1],
            'best_depth': 3.0 * (1 - best_trial.values[1]),  # Convert efficiency back to depth
            'best_params': dict(best_trial.params),
            'base_arch_params': phase1_params,
            'pareto_front_size': len(pareto_trials),
            'all_pareto_solutions': [
                {
                    'accuracy': t.values[0],
                    'efficiency': t.values[1],
                    'depth': 3.0 * (1 - t.values[1]),
                    'params': dict(t.params)
                }
                for t in pareto_trials
            ]
        }
        
        # Save to organized location
        result_path = os.path.join(
            OPTIMIZATION_DIR, 
            f"mc_alpha{alpha:.2f}_{dataset_name}_results.json"
        )
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        logging.info(f"\nResults saved to: {result_path}")
        
        return result
    
    return None


def run_evaluation(
    dataset_name: str,
    mc_params: dict,
    n_runs: int = 3
):
    """Evaluate the MC-optimized model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    phase1_params = mc_params['base_arch_params']
    loaders, input_bands, num_classes = prepare_data(dataset_name, phase1_params, train_ratio=0.1)
    
    results = []
    
    for run in range(n_runs):
        logging.info(f"\n--- Evaluation Run {run+1}/{n_runs} ---")
        
        model = AdaptiveDSSFN(
            input_bands=input_bands,
            num_classes=num_classes,
            patch_size=base_cfg.PATCH_SIZE,
            spec_channels=[phase1_params.get('s1_channels', 128), phase1_params.get('spec_c2', 128), phase1_params.get('s3_channels', 256)],
            spatial_channels=[phase1_params.get('s1_channels', 128), phase1_params.get('spat_c2', 128), phase1_params.get('s3_channels', 256)],
            cross_attention_heads=phase1_params.get('cross_attention_heads', 4),
            halting_bias_init=mc_params['best_params']['halting_bias_init'],
            act_epsilon=0.01
        ).to(device)
        
        ponder_weight = mc_params['best_params']['ponder_cost_weight']
        depth_scale = mc_params['best_params']['depth_penalty_scale']
        effective_ponder = ponder_weight * (1 + (1 - mc_params['alpha']) * depth_scale)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
        
        # Training
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(100):
            model.train()
            for inputs, labels in loaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs, ponder_cost, halting_steps = model(inputs)
                loss = criterion(outputs, labels) + effective_ponder * ponder_cost.mean()
                
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels in loaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs, _, _ = model(inputs)
                    val_loss += criterion(outputs, labels).item()
            
            val_loss /= len(loaders['val'])
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Test evaluation
        model.eval()
        all_preds = []
        all_labels = []
        total_depth = 0
        n_samples = 0
        stage_counts = [0, 0, 0]
        
        with torch.no_grad():
            for inputs, labels in loaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, halting_steps, _ = model(inputs)
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_depth += halting_steps.sum().item()
                n_samples += inputs.size(0)
                
                for step in halting_steps.cpu().numpy():
                    stage_counts[int(step) - 1] += 1
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_depth = total_depth / n_samples
        
        results.append({
            'OA': accuracy,
            'avg_depth': avg_depth,
            'stage_distribution': [c / n_samples for c in stage_counts]
        })
        
        logging.info(f"  OA: {accuracy:.2%}, Depth: {avg_depth:.2f}/3")
        logging.info(f"  Stage distribution: S1={stage_counts[0]/n_samples:.1%}, S2={stage_counts[1]/n_samples:.1%}, S3={stage_counts[2]/n_samples:.1%}")
    
    # Aggregate results
    mean_oa = np.mean([r['OA'] for r in results])
    std_oa = np.std([r['OA'] for r in results])
    mean_depth = np.mean([r['avg_depth'] for r in results])
    std_depth = np.std([r['avg_depth'] for r in results])
    
    logging.info(f"\n{'='*50}")
    logging.info(f"FINAL RESULTS ({n_runs} runs)")
    logging.info(f"{'='*50}")
    logging.info(f"  OA: {mean_oa:.2%} ± {std_oa:.2%}")
    logging.info(f"  Depth: {mean_depth:.2f} ± {std_depth:.2f}")
    logging.info(f"  Depth Reduction: {(1 - mean_depth/3)*100:.1f}%")
    
    return {
        'mean_OA': mean_oa,
        'std_OA': std_oa,
        'mean_depth': mean_depth,
        'std_depth': std_depth,
        'depth_reduction_pct': (1 - mean_depth/3) * 100,
        'runs': results
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Criteria Optimization with configurable alpha")
    parser.add_argument('--dataset', type=str, default='IndianPines',
                        choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for accuracy (0-1). Lower = more efficiency-focused.')
    parser.add_argument('--eval-runs', type=int, default=3, help='Number of evaluation runs')
    parser.add_argument('--skip-opt', action='store_true', help='Skip optimization, just evaluate')
    args = parser.parse_args()
    
    datasets = ['IndianPines', 'PaviaUniversity', 'Botswana'] if args.dataset == 'ALL' else [args.dataset]
    
    all_results = {}
    
    for dataset in datasets:
        timestamp = setup_logging(dataset, args.alpha)
        
        logging.info(f"\n{'='*70}")
        logging.info(f"MULTI-CRITERIA OPTIMIZATION: {dataset}")
        logging.info(f"Alpha={args.alpha} (Accuracy weight: {args.alpha:.0%}, Efficiency weight: {1-args.alpha:.0%})")
        logging.info(f"{'='*70}")
        
        if not args.skip_opt:
            mc_result = run_optimization(
                dataset,
                n_trials=args.trials,
                alpha=args.alpha,
                epochs=50
            )
        else:
            # Load existing results
            result_path = os.path.join(OPTIMIZATION_DIR, f"mc_alpha{args.alpha:.2f}_{dataset}_results.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    mc_result = json.load(f)
            else:
                logging.error(f"No existing results found at {result_path}")
                continue
        
        if mc_result:
            eval_result = run_evaluation(dataset, mc_result, n_runs=args.eval_runs)
            
            all_results[dataset] = {
                'optimization': mc_result,
                'evaluation': eval_result
            }
    
    # Final summary
    logging.info(f"\n{'='*70}")
    logging.info(f"FINAL SUMMARY (Alpha={args.alpha})")
    logging.info(f"{'='*70}")
    
    print(f"\n{'Dataset':<18} {'OA':<16} {'Depth':<12} {'Reduction':<12}")
    print("-" * 60)
    
    for ds, res in all_results.items():
        ev = res['evaluation']
        print(f"{ds:<18} {ev['mean_OA']*100:.2f}±{ev['std_OA']*100:.2f}%  {ev['mean_depth']:.2f}±{ev['std_depth']:.2f}  {ev['depth_reduction_pct']:.1f}%")
    
    # Save final summary
    summary_path = os.path.join(EVALUATION_DIR, f"mc_alpha{args.alpha:.2f}_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4, default=float)
    
    logging.info(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
