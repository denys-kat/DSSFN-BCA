# scripts/run_full_experiment.py
# -*- coding: utf-8 -*-
"""
Orchestrator script to run the full DSSFN experimental pipeline.

Updated Flow: Phase-oriented.
It runs a specific configuration/optimization step for ALL datasets before moving to the next configuration.

Sequence:
1. Base Preset: Eval on all datasets (Train 5% & 70%)
2. BCA Preset: Eval on all datasets (Train 5% & 70%)
3. Optimize BCA (Single Criterion): Optimize Architecture on all datasets (Train 5%)
4. Eval Optimized BCA: Eval on all datasets (Train 5% & 70%)
5. Optimize Adaptive (Multicriteria): Optimize ACT parameters using FIXED architecture from step 3 (Train 5%)
6. Eval Multicriteria Adaptive: Eval on all datasets (Train 5% & 70%)
"""

import subprocess
import argparse
import sys
import os


def _csv_list(value: str):
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _float_list(values):
    # argparse with nargs='+' already gives a list, but we normalize anyway.
    out = []
    for v in (values or []):
        try:
            out.append(float(v))
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid float value: {v}")
    return out

def run_command(cmd, step_name):
    print(f"\n{'='*80}")
    print(f"STEP: {step_name}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*80}\n")
    
    try:
        # Stream output to console
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end='')
        process.wait()
        
        if process.returncode != 0:
            print(f"\n!!! ERROR in step '{step_name}' !!!")
            sys.exit(1)
            
    except Exception as e:
        print(f"Exception running command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Full Experiment Orchestrator")
    parser.add_argument('--trials', type=int, default=20, help="Number of optimization trials")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use")

    # Reproducibility / split protocol
    parser.add_argument('--seed', type=int, default=42, help="Random seed forwarded to evaluate_model.py")
    parser.add_argument('--non_deterministic', action='store_true', help="Allow non-deterministic CUDA kernels in evaluation")

    # Dataset / split control
    parser.add_argument(
        '--datasets',
        type=_csv_list,
        default=['IndianPines', 'PaviaUniversity', 'Botswana'],
        help="Comma-separated list of datasets to run (default: IndianPines,PaviaUniversity,Botswana).",
    )
    parser.add_argument(
        '--eval_train_ratios',
        nargs='+',
        default=[0.05, 0.10, 0.70],
        help="Train ratios to evaluate (space-separated), e.g. --eval_train_ratios 0.05 0.10 0.70",
    )
    parser.add_argument('--val_ratio', type=float, default=0.15, help="Validation ratio")
    parser.add_argument('--opt_train_ratio', type=float, default=0.05, help="Train ratio for optimization phases")

    # MC selection reference
    parser.add_argument(
        '--sc_reference_config',
        type=str,
        default='bca_sc_optimized',
        help="Config ID used as SC reference when selecting MC adaptive params (forwarded to optimize_model.py).",
    )
    
    args = parser.parse_args()
    
    # Ensure scripts are called correctly relative to project root
    python_exec = sys.executable
    base_cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} {python_exec}"
    
    datasets = args.datasets

    # Ratios for Evaluation phases
    eval_ratios = _float_list(args.eval_train_ratios)
    val_ratio = float(args.val_ratio)

    # Ratio for Optimization phases
    opt_train_ratio = float(args.opt_train_ratio)

    eval_seed_args = f"--seed {args.seed}" + (" --non_deterministic" if args.non_deterministic else "")
    
    # =========================================================================
    # PHASE 1: Base Preset (Eval)
    # =========================================================================
    print("\n\n" + "#"*40 + "\nPHASE 1: Base Preset Evaluation\n" + "#"*40)
    for dataset in datasets:
        for tr in eval_ratios:
            run_command(
                f"{base_cmd} scripts/evaluate_model.py --dataset {dataset} --model_type base --config_name base --train_ratio {tr} --val_ratio {val_ratio} {eval_seed_args}",
                f"1. Train + Eval Base ({dataset}, Train={tr})"
            )

    # =========================================================================
    # PHASE 2: BCA Preset (Eval)
    # =========================================================================
    print("\n\n" + "#"*40 + "\nPHASE 2: BCA Preset Evaluation\n" + "#"*40)
    for dataset in datasets:
        for tr in eval_ratios:
            run_command(
                f"{base_cmd} scripts/evaluate_model.py --dataset {dataset} --model_type bca --config_name bca --train_ratio {tr} --val_ratio {val_ratio} {eval_seed_args}",
                f"2. Train + Eval BCA ({dataset}, Train={tr})"
            )

    # =========================================================================
    # PHASE 3: Optimize BCA (Single Criterion - Architecture)
    # =========================================================================
    print("\n\n" + "#"*40 + "\nPHASE 3: Optimize BCA (SC - Architecture)\n" + "#"*40)
    for dataset in datasets:
        run_command(
            f"{base_cmd} scripts/optimize_model.py --dataset {dataset} --model_type bca --trials {args.trials} --train_ratio {opt_train_ratio} --val_ratio {val_ratio} --no_early_stopping",
            f"3. Optimization (SC) - BCA ({dataset})"
        )

    # =========================================================================
    # PHASE 4: Eval Optimized BCA
    # =========================================================================
    print("\n\n" + "#"*40 + "\nPHASE 4: Evaluate Optimized BCA\n" + "#"*40)
    for dataset in datasets:
        # Construct path to params generated in Phase 3
        bca_params = f"results/configurations/bca/optimization/best_params_bca_{dataset}.json"
        for tr in eval_ratios:
            run_command(
                f"{base_cmd} scripts/evaluate_model.py --dataset {dataset} --model_type bca --params_file {bca_params} --config_name bca_sc_optimized --train_ratio {tr} --val_ratio {val_ratio} {eval_seed_args}",
                f"4. Evaluation - Optimized BCA (SC) ({dataset}, Train={tr})"
            )

    # =========================================================================
    # PHASE 5: Optimize Adaptive (Multicriteria - ACT Parameters Only)
    # =========================================================================
    print("\n\n" + "#"*40 + "\nPHASE 5: Optimize Adaptive (MC - ACT Parameters)\n" + "#"*40)
    for dataset in datasets:
        # Note: optimize_model.py 'adaptive' mode automatically loads BCA architecture from step 3
        run_command(
            f"{base_cmd} scripts/optimize_model.py --dataset {dataset} --model_type adaptive --trials {args.trials} --train_ratio {opt_train_ratio} --val_ratio {val_ratio} --no_early_stopping --multicriteria --sc_reference_config {args.sc_reference_config}",
            f"5. Optimization (MC) - Adaptive ({dataset})"
        )

    # =========================================================================
    # PHASE 6: Eval Multicriteria Adaptive
    # =========================================================================
    print("\n\n" + "#"*40 + "\nPHASE 6: Evaluate Optimized Adaptive (MC)\n" + "#"*40)
    for dataset in datasets:
        # Note: optimize_model.py saves combined params (arch + ACT) to this file
        adaptive_mc_params = f"results/configurations/adaptive/optimization/best_params_adaptive_{dataset}_mc.json"
        for tr in eval_ratios:
            run_command(
                f"{base_cmd} scripts/evaluate_model.py --dataset {dataset} --model_type adaptive --params_file {adaptive_mc_params} --config_name adaptive_mc_optimized --train_ratio {tr} --val_ratio {val_ratio} {eval_seed_args}",
                f"6. Evaluation - Optimized Adaptive (MC) ({dataset}, Train={tr})"
            )
    
    print("\n\nAll experiments on all datasets completed successfully.")

if __name__ == "__main__":
    main()