# -*- coding: utf-8 -*-
"""
Main entry point for DSSFN workflows.
Supports 'evaluate' (Train + Eval) and 'optimize' modes.
Uses Presets from src.config to avoid manual JSON file creation.
"""

import argparse
import os
import sys
import glob
from types import SimpleNamespace

# --- Add project root to sys.path to allow importing src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import src.config as base_cfg
# Import run_evaluation directly instead of using os.system
from evaluate_model import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="DSSFN Main Launcher")
    
    # Mode selection
    parser.add_argument('--mode', choices=['optimize', 'evaluate'], default='evaluate', 
                        help="Mode: 'evaluate' runs Train+Eval on one config. 'optimize' searches params.")
    
    # Dataset
    parser.add_argument('--dataset', required=True, 
                        choices=['IndianPines', 'PaviaUniversity', 'Botswana', 'ALL'])
    
    # Config Preset (The Key Improvement)
    parser.add_argument('--preset', choices=['base', 'bca', 'none'], default='base',
                        help="Load a predefined configuration (Base or BCA).")
    
    # Training details
    parser.add_argument('--train_ratio', type=float, default=0.05, 
                        help="Ratio of training data (0.05 = 5%)")
    parser.add_argument('--val_ratio', type=float, default=0.15, 
                        help="Ratio of validation data (0.15 = 15%)")
    
    # Arguments specific to Optimization mode
    parser.add_argument('--trials', type=int, default=20, help="Trials for optimization mode")
    parser.add_argument('--multicriteria', action='store_true', help="Enable accuracy+efficiency opt")

    args = parser.parse_args()

    # Expand dataset list
    datasets = ['IndianPines', 'PaviaUniversity', 'Botswana'] if args.dataset == 'ALL' else [args.dataset]

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f" Processing Dataset: {ds} | Mode: {args.mode} | Preset: {args.preset}")
        print(f"{'='*60}")

        if args.mode == 'optimize':
            # For optimization, we still call the optimize script via system call 
            # because it uses Optuna and is a distinct process.
            # We map 'preset' concept to 'model_type' for optimization
            model_type_map = {'base': 'base', 'bca': 'bca'}
            model_type = model_type_map.get(args.preset, 'base')
            
            cmd = f"python3 {os.path.join(current_dir, 'optimize_model.py')} --dataset {ds} --model_type {model_type} --trials {args.trials} --train_ratio {args.train_ratio} --val_ratio {args.val_ratio}"
            if args.multicriteria:
                cmd += " --multicriteria"
            
            print(f"Running Optimization Command: {cmd}")
            os.system(cmd)
            
        else:
            # === IMPROVED FLOW: In-Memory execution for Evaluation ===
            
            # 1. Load the preset dictionary
            explicit_params = None
            if args.preset != 'none':
                if args.preset in base_cfg.MODEL_PRESETS:
                    explicit_params = base_cfg.MODEL_PRESETS[args.preset]
                    print(f"Loaded preset '{args.preset}': {explicit_params}")
                else:
                    print(f"Warning: Preset '{args.preset}' not found in config.")

            # 2. Construct arguments object for run_evaluation
            
            if 'adaptive' in (explicit_params or {}).get('fusion_mechanism', '').lower() and args.preset == 'bca':
                model_type_for_eval = 'bca' 
            elif args.preset == 'base':
                model_type_for_eval = 'base'
            else:
                model_type_for_eval = 'base'

            # Update config name to simply be the preset name, creating clean folder: configurations/base/
            config_name = args.preset 

            eval_args = SimpleNamespace(
                dataset=ds,
                model_type=model_type_for_eval,
                config_name=config_name,
                params_file=None, # We are passing explicit_params instead
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                output_dir=base_cfg.OUTPUT_DIR
            )

            # 3. Call the evaluation function directly
            try:
                run_evaluation(eval_args, explicit_params=explicit_params)
            except Exception as e:
                print(f"Error evaluating {ds}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()