# -*- coding: utf-8 -*-
"""
Universal Evaluation Script.
Supports evaluation on specific split ratios.
Uses src.visualization for consistent, professional plotting.
Updated Folder Structure: results/configurations/{config_name}/evaluation/
"""

import os
import sys
import argparse
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
# Use Agg backend to prevent display errors on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Add src directory ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.config as base_cfg
import src.data_utils as du
import src.band_selection as bs
import src.sampling as spl
import src.datasets as ds
import src.visualization as vis
from src.model import DSSFN, AdaptiveDSSFN
from src.engine import train_model, train_adaptive_model, evaluate_adaptive_model, evaluate_model as engine_evaluate
from src.utils import FLOPsCounter, measure_inference_time, count_parameters

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

def get_dataset_config(dataset_name):
    configs = {
        'IndianPines': {'rel_path': 'ip/', 'data': 'indianpinearray.npy', 'gt': 'IPgt.npy', 'classes': 16, 'mat_key': None, 'gt_key': None, 'names': ["Background", "Alfalfa", "Corn-notill", "Corn-min", "Corn", "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-min", "Soybean-clean", "Wheat", "Woods", "Bldg-Grass-Tree-Drives", "Stone-Steel-Towers"]},
        'PaviaUniversity': {'rel_path': 'pu/', 'data': 'PaviaU.mat', 'gt': 'PaviaU_gt.mat', 'classes': 9, 'mat_key': 'paviaU', 'gt_key': 'paviaU_gt', 'names': ["Background", "Asphalt", "Meadows", "Gravel", "Trees", "Painted metal sheets", "Bare Soil", "Bitumen", "Self-Blocking Bricks", "Shadows"]},
        'Botswana': {'rel_path': 'botswana/', 'data': 'Botswana.mat', 'gt': 'Botswana_gt.mat', 'classes': 14, 'mat_key': 'Botswana', 'gt_key': 'Botswana_gt', 'names': ["Background", "Water", "Hippo grass", "Floodplain grasses 1", "Floodplain grasses 2", "Reeds 1", "Riparian", "Firescar 2", "Island interior", "Acacia woodlands", "Acacia shrublands", "Acacia grasslands", "Short mopane", "Mixed mopane", "Exposed soils"]}
    }
    return configs[dataset_name]

def run_evaluation(args, explicit_params=None):
    """
    Runs the train+eval pipeline.
    """
    # --- Folder Structure Update ---
    # User wants: results/configurations/{config_name}/evaluation/
    # args.config_name should be 'base', 'bca', etc. passed from main.py
    if args.config_name:
         base_out = os.path.join(base_cfg.OUTPUT_DIR, 'configurations', args.config_name)
    else:
         base_out = os.path.join(base_cfg.OUTPUT_DIR, 'evaluations', args.model_type)

    eval_dir = os.path.join(base_out, 'evaluation')
    map_dir = os.path.join(base_out, 'maps')
    log_dir = os.path.join(base_out, 'logs') 
    
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create unique run ID with timestamp
    # Filename format: results_{dataset}_tr{...}_val{...}_{timestamp}.json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ratio_str = f"tr{args.train_ratio:.2f}_val{args.val_ratio:.2f}".replace('.', 'p')
    
    # run_id suffix used for filenames
    run_suffix = f"{ratio_str}_{timestamp}"
    
    logger = setup_logging(os.path.join(log_dir, f"eval_{args.dataset}_{run_suffix}.log"))
    logger.info(f"Evaluating {args.dataset} (Train: {args.train_ratio}, Val: {args.val_ratio}, Suffix: {run_suffix})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Load Parameters ---
    params = {}
    
    if explicit_params:
        params = explicit_params
        logger.info("Using EXPLICIT parameters provided from main script.")
    elif args.params_file and os.path.exists(args.params_file):
        with open(args.params_file, 'r') as f: 
            params = json.load(f)
        logger.info(f"Loaded parameters from {args.params_file}")
    else:
        logger.warning("No params provided. Using MANUAL DEFAULTS.")
        params = {
            "s1_channels": 64, "spec_c2": 128, "spat_c2": 128, "s3_channels": 256,
            "swgmf_bands": 30, "band_selection": "SWGMF", "intermediate_attention": [],
            "fusion_mechanism": "AdaptiveWeight", "cross_attention_heads": 0 
        }

    ds_conf = get_dataset_config(args.dataset)
    data_path = os.path.join(base_cfg.DATA_BASE_PATH, ds_conf['rel_path'])
    data_cube, gt_map = du.load_hyperspectral_data(data_path, ds_conf['data'], ds_conf['gt'], data_mat_key=ds_conf['mat_key'], gt_mat_key=ds_conf['gt_key'])
    
    method = params.get('band_selection', 'SWGMF')
    if method == 'SWGMF': selected_data, _ = bs.apply_swgmf(data_cube, 5, params.get('swgmf_bands', 30))
    elif method == 'E-FDPC': selected_data, _ = bs.apply_efdpc(data_cube, params.get('efdpc_dc_percent', 2.0))
    else: selected_data = data_cube
    
    input_bands = selected_data.shape[-1]
    
    norm_data = du.normalize_data(selected_data)
    pad_data = du.pad_data(norm_data, base_cfg.BORDER_SIZE)
    all_coords, labels, orig_idx = spl.get_labeled_coordinates_and_indices(gt_map)
    
    # Stratified Split with Explicit Ratios
    split = spl.split_data_random_stratified(
        all_coords, labels, orig_idx, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        num_classes=ds_conf['classes'], 
        random_seed=42
    )
    
    # Loaders
    patches = {}
    for k in ['train', 'val', 'test']:
        p, l = du.create_patches_from_coords(pad_data, split[f'{k}_coords'], base_cfg.PATCH_SIZE)
        patches[f'{k}_p'] = p
        patches[f'{k}_l'] = l
        
    loaders = ds.create_dataloaders(
        {'train_patches': patches['train_p'], 'train_labels': patches['train_l'],
         'val_patches': patches['val_p'], 'val_labels': patches['val_l'],
         'test_patches': patches['test_p'], 'test_labels': patches['test_l']},
        batch_size=64, num_workers=0
    )

    spec_c = [params.get('s1_channels', 64), params.get('spec_c2', 128), params.get('s3_channels', 256)]
    spat_c = [params.get('s1_channels', 64), params.get('spat_c2', 128), params.get('s3_channels', 256)]
    base_cfg.INTERMEDIATE_ATTENTION_STAGES = params.get('intermediate_attention', [])
    
    fusion_mech = params.get('fusion_mechanism', 'AdaptiveWeight')
    base_cfg.FUSION_MECHANISM = fusion_mech
    
    if args.model_type == 'adaptive':
        model = AdaptiveDSSFN(input_bands, ds_conf['classes'], base_cfg.PATCH_SIZE, spec_c, spat_c, 
                              cross_attention_heads=params.get('cross_attention_heads', 4),
                              halting_bias_init=params.get('halting_bias_init', -1.0)).to(device)
    else:
        model = DSSFN(input_bands, ds_conf['classes'], base_cfg.PATCH_SIZE, spec_c, spat_c, 
                      fusion_mechanism=base_cfg.FUSION_MECHANISM,
                      cross_attention_heads=params.get('cross_attention_heads', 0)).to(device)

    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    
    start_time = time.time()
    epochs = args.epochs if hasattr(args, 'epochs') else 100
    
    if args.model_type == 'adaptive':
        trained_model, history = train_adaptive_model(model, crit, opt, sched, loaders['train'], loaders['val'], device, epochs=epochs, save_best_model=True, early_stopping_enabled=False)
    else:
        trained_model, history = train_model(model, crit, opt, sched, loaders['train'], loaders['val'], device, epochs=epochs, save_best_model=True, early_stopping_enabled=False)
    
    train_time = time.time() - start_time
    best_acc = max(history['val_accuracy'])
    logger.info(f"Training finished in {train_time:.2f}s. Best Val Acc: {best_acc:.4f}")

    # FLOPs
    dummy_shape = (1, input_bands, base_cfg.PATCH_SIZE, base_cfg.PATCH_SIZE)
    try:
        flops_counter = FLOPsCounter(trained_model, dummy_shape, device)
        flops_count = flops_counter.count()
    except Exception:
        flops_count = 0
    
    inf_stats = measure_inference_time(trained_model, dummy_shape, device)
    
    # --- EVALUATION LOGIC CHANGE ---
    # Use evaluate_adaptive_model if adaptive, else standard
    depth_stats = None
    if args.model_type == 'adaptive':
        eval_res = evaluate_adaptive_model(trained_model, loaders['test'], device, crit)
        oa, aa, kappa = eval_res['OA'], eval_res['AA'], eval_res['Kappa']
        report = eval_res['report']
        test_preds = eval_res['all_preds']
        # Extract depth stats
        depth_stats = {
            'avg_depth': eval_res['avg_depth'],
            'stage_distribution': eval_res['stage_distribution'],
            'flops_reduction': eval_res['flops_reduction']
        }
        logger.info(f"Test Results: OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}")
        logger.info(f"Depth Stats: Avg Depth={eval_res['avg_depth']:.2f}, Reduction={eval_res['flops_reduction']:.2f}")
    else:
        oa, aa, kappa, report, test_preds, _ = engine_evaluate(trained_model, loaders['test'], device, crit)
        logger.info(f"Test Results: OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}")
    
    # Generate Map
    full_patches, _ = du.create_patches_from_coords(pad_data, all_coords, base_cfg.PATCH_SIZE)
    full_dataset = ds.HyperspectralDataset(full_patches, labels)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=256, shuffle=False)
    
    trained_model.eval()
    all_preds_full = []
    with torch.no_grad():
        for x, _ in full_loader:
            x = x.to(device)
            if args.model_type == 'adaptive':
                out, _, _ = trained_model(x)
            else:
                out = trained_model(x)
                if isinstance(out, tuple): out = (out[0] + out[1]) / 2.0
            all_preds_full.extend(out.argmax(1).cpu().numpy())
            
    fig, _ = vis.plot_predictions(
        gt_map=gt_map,
        test_predictions=all_preds_full,
        test_coords=all_coords,
        class_names=ds_conf['names'],
        dataset_name=args.dataset,
        oa=oa,
        subtitle=f"Predicted (Train {args.train_ratio*100:.0f}%)"
    )
    
    if fig:
        map_filename = f"{args.dataset}_map_{run_suffix}.png"
        fig.savefig(os.path.join(map_dir, map_filename), dpi=300, bbox_inches='tight')
        logger.info(f"Classification map saved to {map_filename}")
        plt.close(fig)
    
    results = {
        'metrics': {
            'OA': oa, 'AA': aa, 'Kappa': kappa, 
            'TrainTime': train_time, 'FLOPs': flops_count, 
            'InferenceMs': inf_stats['mean_ms']
        }, 
        'params': params, 
        'report': report
    }
    
    if depth_stats:
        results['metrics']['depth_stats'] = depth_stats
    
    json_filename = f"results_{args.dataset}_{run_suffix}.json"
    with open(os.path.join(eval_dir, json_filename), 'w') as f: 
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {json_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model_type', default='base', help="Model type (base, bca, adaptive)")
    parser.add_argument('--config_name', required=False, help="Configuration name for output folder structure")
    parser.add_argument('--params_file', required=False, help="Optional params file.")
    parser.add_argument('--train_ratio', type=float, default=0.05)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--output_dir', default=base_cfg.OUTPUT_DIR)
    args = parser.parse_args()
    run_evaluation(args)