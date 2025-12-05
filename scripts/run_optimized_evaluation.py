# -*- coding: utf-8 -*-
"""
Script to run train + test evaluation on all datasets using optimized hyperparameters.
"""

import os
import sys
import json
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

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

# --- Setup Logging ---
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"optimized_evaluation_{timestamp}.log"
log_filepath = os.path.join(base_cfg.OUTPUT_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Optimized Parameters for Each Dataset ---
OPTIMIZED_PARAMS = {
    'IndianPines': {
        'efdpc_dc_percent': 7.218499781259096,
        'cross_attention_heads': 4,
        's1_channels': 128,
        'spec_c2': 128,
        'spat_c2': 64,
        's3_channels': 256
    },
    'PaviaUniversity': {
        'efdpc_dc_percent': 4.59013537896503,
        'cross_attention_heads': 8,
        's1_channels': 128,
        'spec_c2': 128,
        'spat_c2': 256,
        's3_channels': 128
    },
    'Botswana': {
        'efdpc_dc_percent': 7.295101924264044,
        'cross_attention_heads': 4,
        's1_channels': 256,
        'spec_c2': 32,
        'spat_c2': 256,
        's3_channels': 256
    }
}

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
            'CLASS_NAMES': [
                "Alfalfa", "Corn-notill", "Corn-min", "Corn",
                "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
                "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-min",
                "Soybean-clean", "Wheat", "Woods", "Bldg-Grass-Tree-Drives",
                "Stone-Steel-Towers"
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
                "Asphalt", "Meadows", "Gravel", "Trees",
                "Painted metal sheets", "Bare Soil", "Bitumen",
                "Self-Blocking Bricks", "Shadows"
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


def run_evaluation(dataset_name, device, num_runs=3):
    """
    Run full train + test evaluation for a dataset using optimized parameters.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING: {dataset_name}")
    logger.info(f"{'='*60}")
    
    # Get dataset config and optimized params
    ds_cfg = get_dataset_config(dataset_name)
    opt_params = OPTIMIZED_PARAMS[dataset_name]
    
    logger.info(f"Optimized Parameters: {json.dumps(opt_params, indent=2)}")
    
    # Build channel lists from optimized params
    spec_channels = [opt_params['s1_channels'], opt_params['spec_c2'], opt_params['s3_channels']]
    spatial_channels = [opt_params['s1_channels'], opt_params['spat_c2'], opt_params['s3_channels']]
    
    # Fixed training config
    run_cfg = {
        'BAND_SELECTION_METHOD': 'E-FDPC',
        'E_FDPC_DC_PERCENT': opt_params['efdpc_dc_percent'],
        'INTERMEDIATE_ATTENTION_STAGES': [1],
        'FUSION_MECHANISM': 'AdaptiveWeight',
        'PATCH_SIZE': 15,
        'BORDER_SIZE': 7,
        'BATCH_SIZE': 64,
        'EPOCHS': 100,
        'LEARNING_RATE': 1e-4,
        'WEIGHT_DECAY': 1e-4,
        'CROSS_ATTENTION_HEADS': opt_params['cross_attention_heads'],
        'CROSS_ATTENTION_DROPOUT': 0.1,
        'NUM_WORKERS': 0,
        'PIN_MEMORY': True,
        'TRAIN_RATIO': 0.10,
        'VAL_RATIO': 0.10,
        'EARLY_STOPPING_PATIENCE': 25
    }
    run_cfg.update(ds_cfg)
    
    # Load Data Once
    data_cube, gt_map = du.load_hyperspectral_data(
        ds_cfg['DATA_PATH'], ds_cfg['DATA_FILE'], ds_cfg['GT_FILE'],
        ds_cfg['EXPECTED_DATA_SHAPE'], ds_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=ds_cfg['DATA_MAT_KEY'], gt_mat_key=ds_cfg['GT_MAT_KEY']
    )
    
    # Band Selection (E-FDPC)
    logger.info(f"Applying E-FDPC with dc_percent={run_cfg['E_FDPC_DC_PERCENT']:.4f}...")
    selected_data, selected_indices = bs.apply_efdpc(data_cube, run_cfg['E_FDPC_DC_PERCENT'])
    input_bands = selected_data.shape[-1]
    logger.info(f"Selected {input_bands} bands: {selected_indices}")
    
    # Preprocessing
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, run_cfg['BORDER_SIZE'])
    
    # Store results across runs
    all_oa = []
    all_aa = []
    all_kappa = []
    all_class_acc = []
    
    for run_idx in range(num_runs):
        logger.info(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        
        # Create splits with different seeds for each run
        seed = 42 + run_idx
        all_labeled_coords, labels_np_array, original_idx_array = spl.get_labeled_coordinates_and_indices(gt_map)
        split_coords_dict = spl.split_data_random_stratified(
            all_labeled_coords, labels_np_array, original_idx_array,
            train_ratio=run_cfg['TRAIN_RATIO'], val_ratio=run_cfg['VAL_RATIO'], 
            num_classes=run_cfg['NUM_CLASSES'], random_seed=seed
        )
        
        # Create Patches
        data_splits = {}
        for split in ['train', 'val', 'test']:
            coords = split_coords_dict.get(f'{split}_coords', [])
            patches, labels = du.create_patches_from_coords(padded_data, coords, run_cfg['PATCH_SIZE'])
            data_splits[f'{split}_patches'] = patches
            data_splits[f'{split}_labels'] = labels
        
        # DataLoaders
        loaders = ds.create_dataloaders(data_splits, run_cfg['BATCH_SIZE'], run_cfg['NUM_WORKERS'], run_cfg['PIN_MEMORY'])
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']
        
        if not train_loader or not test_loader:
            logger.error("Failed to create dataloaders!")
            continue
        
        # Model Initialization
        base_cfg.INTERMEDIATE_ATTENTION_STAGES = run_cfg['INTERMEDIATE_ATTENTION_STAGES']
        base_cfg.FUSION_MECHANISM = run_cfg['FUSION_MECHANISM']
        
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
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=run_cfg['LEARNING_RATE'], weight_decay=run_cfg['WEIGHT_DECAY'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(run_cfg['EPOCHS']):
            # Training phase
            model.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                out1, out2 = model(data)
                loss1 = criterion(out1, target)
                loss2 = criterion(out2, target)
                loss = loss1 + loss2
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    out1, out2 = model(data)
                    output = (out1 + out2) / 2
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            val_acc = correct / total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
            
            if patience_counter >= run_cfg['EARLY_STOPPING_PATIENCE']:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and evaluate on test set
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                out1, out2 = model(data)
                output = (out1 + out2) / 2
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        oa = accuracy_score(all_targets, all_preds)
        kappa = cohen_kappa_score(all_targets, all_preds)
        
        # Per-class accuracy
        cm = confusion_matrix(all_targets, all_preds)
        class_acc = cm.diagonal() / cm.sum(axis=1)
        aa = np.mean(class_acc)
        
        all_oa.append(oa)
        all_aa.append(aa)
        all_kappa.append(kappa)
        all_class_acc.append(class_acc)
        
        logger.info(f"  Run {run_idx + 1} Results: OA={oa*100:.2f}%, AA={aa*100:.2f}%, Kappa={kappa:.4f}")
    
    # Aggregate results
    mean_oa = np.mean(all_oa) * 100
    std_oa = np.std(all_oa) * 100
    mean_aa = np.mean(all_aa) * 100
    std_aa = np.std(all_aa) * 100
    mean_kappa = np.mean(all_kappa)
    std_kappa = np.std(all_kappa)
    
    mean_class_acc = np.mean(all_class_acc, axis=0) * 100
    std_class_acc = np.std(all_class_acc, axis=0) * 100
    
    logger.info(f"\n{'='*40}")
    logger.info(f"FINAL RESULTS for {dataset_name} ({num_runs} runs)")
    logger.info(f"{'='*40}")
    logger.info(f"Overall Accuracy (OA): {mean_oa:.2f}% ± {std_oa:.2f}%")
    logger.info(f"Average Accuracy (AA): {mean_aa:.2f}% ± {std_aa:.2f}%")
    logger.info(f"Kappa Coefficient:     {mean_kappa:.4f} ± {std_kappa:.4f}")
    logger.info(f"\nPer-Class Accuracy:")
    for i, class_name in enumerate(ds_cfg['CLASS_NAMES']):
        logger.info(f"  {class_name}: {mean_class_acc[i]:.2f}% ± {std_class_acc[i]:.2f}%")
    
    # Save results to JSON
    results = {
        'dataset': dataset_name,
        'optimized_params': opt_params,
        'spec_channels': spec_channels,
        'spatial_channels': spatial_channels,
        'num_runs': num_runs,
        'overall_accuracy': {'mean': mean_oa, 'std': std_oa},
        'average_accuracy': {'mean': mean_aa, 'std': std_aa},
        'kappa': {'mean': mean_kappa, 'std': std_kappa},
        'per_class_accuracy': {
            ds_cfg['CLASS_NAMES'][i]: {'mean': float(mean_class_acc[i]), 'std': float(std_class_acc[i])}
            for i in range(len(ds_cfg['CLASS_NAMES']))
        }
    }
    
    results_dir = os.path.join(base_cfg.OUTPUT_DIR, 'optimized_evaluation')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{dataset_name}_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"\nResults saved to: {results_file}")
    
    return results


def main():
    logger.info("="*60)
    logger.info("DSSFN-BCA Optimized Evaluation")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    datasets = ['IndianPines', 'PaviaUniversity', 'Botswana']
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_evaluation(dataset, device, num_runs=3)
            all_results[dataset] = results
        except Exception as e:
            logger.error(f"Error evaluating {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY OF ALL DATASETS")
    logger.info("="*60)
    for dataset, results in all_results.items():
        logger.info(f"{dataset}: OA={results['overall_accuracy']['mean']:.2f}% ± {results['overall_accuracy']['std']:.2f}%, "
                   f"AA={results['average_accuracy']['mean']:.2f}% ± {results['average_accuracy']['std']:.2f}%, "
                   f"Kappa={results['kappa']['mean']:.4f}")
    
    # Save summary
    summary_file = os.path.join(base_cfg.OUTPUT_DIR, 'optimized_evaluation', f"summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
