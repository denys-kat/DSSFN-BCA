# -*- coding: utf-8 -*-
"""
Master standalone Python script for comprehensive sweep across multiple datasets
and configurations.

This script systematically tests the DSSFN model performance by iterating through:
- Selected Datasets (via command-line: IndianPines, PaviaUniversity, Botswana, or ALL)
- Band Selection Methods (SWGMF(30), E-FDPC(2.5), None)
- Intermediate Attention (After Stage 1 vs. None)
- Final Fusion Mechanism (AdaptiveWeight only)

It performs K-fold cross-validation for each dataset-configuration pair.
Results are logged per run and aggregated (mean/std) across folds for each dataset.
This script aims to replace individual sweep_combinations_*.py scripts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import warnings
import sys
import datetime
import json
import random
import importlib
import time
import logging
import traceback
import pandas as pd
import itertools # For generating combinations
import copy    # For deep copying config/model states if needed
import types   # To check for module type
import argparse # For command-line arguments

# --- Add src directory to path ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
project_root = os.path.abspath(os.path.join(script_dir, '..'))
module_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    print(f"Adding {project_root} to sys.path")
    sys.path.insert(0, project_root)
else:
    print(f"{project_root} already in sys.path")

# --- Import Custom Modules ---
try:
    import src.config as base_cfg_module
    importlib.reload(base_cfg_module) # Reload to ensure fresh state

    import src.data_utils as du
    import src.band_selection as bs
    import src.sampling as spl
    import src.datasets as ds
    from src.model import DSSFN
    import src.engine as engine
    print("Custom modules imported successfully using 'src.' prefix.")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Traceback: {traceback.format_exc()}")
    print(f"Please ensure the 'src' directory exists at: {module_path}")
    print(f"Current sys.path: {sys.path}")
    raise

# --- Ignore specific warnings ---
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==================================
#        Dataset Definitions
# ==================================
# These are the dataset-specific parameters that would typically be in config.py
DATASET_SPECIFIC_CONFIGS = {
    'IndianPines': {
        'DATASET_NAME': 'IndianPines', # Self-reference for consistency
        'DATA_PATH_RELATIVE': 'ip/',
        'DATA_FILE': 'indianpinearray.npy',
        'GT_FILE': 'IPgt.npy',
        'NUM_CLASSES': 16,
        'EXPECTED_DATA_SHAPE': (145, 145, 200),
        'EXPECTED_GT_SHAPE': (145, 145),
        'CLASS_NAMES': [
            "Background/Untested", "Alfalfa", "Corn-notill", "Corn-min", "Corn",
            "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
            "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-min",
            "Soybean-clean", "Wheat", "Woods", "Bldg-Grass-Tree-Drives",
            "Stone-Steel-Towers"
        ],
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
        'CLASS_NAMES': [
            "Background/Untested", "Asphalt", "Meadows", "Gravel", "Trees",
            "Painted metal sheets", "Bare Soil", "Bitumen",
            "Self-Blocking Bricks", "Shadows"
        ],
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
        'CLASS_NAMES': [
            "Background/Untested",  "Water", "Hippo grass", "Floodplain grasses 1",
            "Floodplain grasses 2", "Reeds 1", "Riparian", "Firescar 2",
            "Island interior", "Acacia woodlands", "Acacia shrublands",
            "Acacia grasslands", "Short mopane", "Mixed mopane", "Exposed soils"
        ],
        'DATA_MAT_KEY': 'Botswana', 'GT_MAT_KEY': 'Botswana_gt',
    }
}

# ==================================
#        Sweep Parameters (Fixed for this script)
# ==================================
band_selection_options = [
    {'method': 'SWGMF', 'param': 30},
    {'method': 'E-FDPC', 'param': 2.5},
    {'method': 'None', 'param': None}
]
intermediate_attention_options = [
    [],    # No intermediate attention
    [1]    # Intermediate attention after stage 1
]
fusion_mechanism_options = [
    'AdaptiveWeight' # Exclude 'CrossAttention'
]

parameter_combinations = list(itertools.product(
    band_selection_options,
    intermediate_attention_options,
    fusion_mechanism_options
))
num_combinations_per_dataset = len(parameter_combinations)

K_FOLDS = 10 # Number of K-Fold cross-validation runs
# BASE_RANDOM_SEED will be taken from base_cfg_module
FOLD_TRAIN_RATIO = 0.10 # Training ratio for each fold
FOLD_VAL_RATIO = 0.10   # Validation ratio for each fold

# ==================================
#        Logger Setup
# ==================================
sweep_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
overall_sweep_output_dir_name = f"master_sweep_{sweep_timestamp}"
overall_sweep_output_dir = os.path.join(project_root, base_cfg_module.OUTPUT_DIR, overall_sweep_output_dir_name)
os.makedirs(overall_sweep_output_dir, exist_ok=True)
log_file_path = os.path.join(overall_sweep_output_dir, "master_sweep_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==================================
# Helper Function to Extract Config Values
# ==================================
def config_to_dict(cfg_module):
    """Extracts relevant non-callable, non-module attributes from a config module."""
    cfg_dict = {}
    for key in dir(cfg_module):
        if not key.startswith("__") and \
           not callable(getattr(cfg_module, key)) and \
           not isinstance(getattr(cfg_module, key), types.ModuleType):
            value = getattr(cfg_module, key)
            # Ensure deepcopy for mutable types like list, dict
            if isinstance(value, (list, dict)):
                cfg_dict[key] = copy.deepcopy(value)
            elif isinstance(value, np.ndarray):
                cfg_dict[key] = value.copy()
            else: # For immutable types like int, float, str, bool, None, tuple
                cfg_dict[key] = value
    return cfg_dict

# ==================================
# Pipeline Functions (Adapted for dynamic dataset config)
# ==================================
def setup_environment(seed):
    """Sets up device and deterministic settings for reproducibility."""
    logging.info(f"Setting up environment with seed: {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Ensure CUBLAS_WORKSPACE_CONFIG is set for deterministic behavior on CUDA
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # Or ':16:8'
        logging.info("CUDA environment set for deterministic run.")
    return device

def load_data_for_dataset(run_cfg): # Takes the full run_cfg dictionary
    """Loads data based on the provided comprehensive run configuration."""
    dataset_name = run_cfg['DATASET_NAME']
    logging.info(f"Loading data for {dataset_name} from {run_cfg['DATA_PATH']}")
    data_cube, gt_map = du.load_hyperspectral_data(
        run_cfg['DATA_PATH'], run_cfg['DATA_FILE'], run_cfg['GT_FILE'],
        run_cfg['EXPECTED_DATA_SHAPE'], run_cfg['EXPECTED_GT_SHAPE'],
        data_mat_key=run_cfg.get('DATA_MAT_KEY'), gt_mat_key=run_cfg.get('GT_MAT_KEY')
    )
    H, W, B_original = data_cube.shape
    logging.info(f"{dataset_name} - Original data: H={H}, W={W}, Bands={B_original}. Classes: {run_cfg['NUM_CLASSES']}")
    return data_cube, gt_map, B_original

def select_bands_for_run(data_cube, band_option, run_cfg_dict):
    """Performs band selection based on the band_option and run_cfg_dict."""
    method = band_option['method']
    param = band_option['param']
    B_original = data_cube.shape[-1]
    logging.info(f"Band Selection - Method: {method}, Param: {param}")
    selected_data, input_bands_run, band_selection_method_run_log = data_cube, B_original, f"{method}"

    if method == 'SWGMF':
        target_bands = param
        band_selection_method_run_log += f"(target={target_bands})"
        if target_bands and 0 < target_bands < B_original:
            try:
                selected_data, _ = bs.apply_swgmf(data_cube, run_cfg_dict['SWGMF_WINDOW_SIZE'], target_bands)
                input_bands_run = selected_data.shape[-1]
                if input_bands_run != target_bands:
                    logging.warning(f"SWGMF: Expected {target_bands} bands, got {input_bands_run}. Using actual count.")
            except Exception as e:
                logging.error(f"Error in SWGMF: {e}. Using original bands.", exc_info=True)
                selected_data, input_bands_run = data_cube, B_original
        else:
            logging.warning(f"Invalid SWGMF target_bands: {target_bands}. Using original bands.")
            selected_data, input_bands_run = data_cube, B_original
    elif method == 'E-FDPC':
        dc_percent = param
        band_selection_method_run_log += f"(dc%={dc_percent})"
        try:
            selected_data, _ = bs.apply_efdpc(data_cube, dc_percent)
            input_bands_run = selected_data.shape[-1]
            if input_bands_run <= 0:
                logging.error("E-FDPC selected 0 bands. Reverting to original.")
                selected_data, input_bands_run = data_cube, B_original
        except Exception as e:
            logging.error(f"Error in E-FDPC: {e}. Using original bands.", exc_info=True)
            selected_data, input_bands_run = data_cube, B_original
    elif method == 'None':
        band_selection_method_run_log = "None (Original Bands)"
    else:
        logging.warning(f"Unknown band selection method: {method}. Using original bands.")
        band_selection_method_run_log = f"Unknown ({method}) - Used Original"


    logging.info(f"Band Selection Summary: {band_selection_method_run_log}. Input bands for model: {input_bands_run}")
    return selected_data, input_bands_run, band_selection_method_run_log

def preprocess_and_split(selected_data, gt_map, split_coords_dict, run_cfg_dict):
    """Normalizes, pads, and creates patches using pre-split coordinates and run_cfg_dict."""
    normalized_data = du.normalize_data(selected_data)
    padded_data = du.pad_data(normalized_data, run_cfg_dict['BORDER_SIZE'])
    data_splits = {}
    for split_name in ['train', 'val', 'test']:
        coords = split_coords_dict.get(f'{split_name}_coords', [])
        if coords:
            patches, labels = du.create_patches_from_coords(padded_data, coords, run_cfg_dict['PATCH_SIZE'])
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = patches, labels
            logging.debug(f"Created {len(patches)} {split_name} patches for {run_cfg_dict['DATASET_NAME']}.")
        else:
            data_splits[f'{split_name}_patches'], data_splits[f'{split_name}_labels'] = np.array([]), np.array([])
            logging.warning(f"No coordinates for {split_name} set in {run_cfg_dict['DATASET_NAME']}.")
    data_splits['test_coords'] = split_coords_dict.get('test_coords', [])
    return data_splits

def prepare_dataloaders(data_splits, run_cfg_dict):
    """Creates DataLoaders based on run_cfg_dict."""
    loaders = ds.create_dataloaders(
        data_splits, run_cfg_dict['BATCH_SIZE'], run_cfg_dict['NUM_WORKERS'], run_cfg_dict['PIN_MEMORY'])
    if not loaders.get('train'): logging.warning(f"Train loader not created for {run_cfg_dict['DATASET_NAME']}.")
    if not loaders.get('val'): logging.debug(f"Validation loader not created for {run_cfg_dict['DATASET_NAME']}.") # Val is optional
    if not loaders.get('test'): logging.warning(f"Test loader not created for {run_cfg_dict['DATASET_NAME']}.")
    return loaders.get('train'), loaders.get('val'), loaders.get('test')

def setup_training(model, run_cfg_dict):
    """Sets up criterion, optimizer, and scheduler based on run_cfg_dict."""
    criterion = nn.CrossEntropyLoss()
    opt_type = run_cfg_dict['OPTIMIZER_TYPE'].lower()
    optimizer_map = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD
    }
    optimizer_class = optimizer_map.get(opt_type, optim.AdamW) # Default to AdamW

    optimizer_params = {
        'lr': run_cfg_dict['LEARNING_RATE'],
        'weight_decay': run_cfg_dict['WEIGHT_DECAY']
    }
    if opt_type == 'sgd':
        optimizer_params['momentum'] = run_cfg_dict.get('SGD_MOMENTUM', 0.9) # Add if SGD needs momentum

    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    logging.info(f"Optimizer: {optimizer_class.__name__} with params: {optimizer_params}")

    scheduler = None
    if run_cfg_dict['USE_SCHEDULER']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=run_cfg_dict['SCHEDULER_STEP_SIZE'], gamma=run_cfg_dict['SCHEDULER_GAMMA'])
        logging.info(f"Scheduler: StepLR (step_size={run_cfg_dict['SCHEDULER_STEP_SIZE']}, gamma={run_cfg_dict['SCHEDULER_GAMMA']})")
    return criterion, optimizer, scheduler

def train_evaluate_model(model, criterion, optimizer, scheduler, loaders, device, run_cfg_dict):
    """Trains and evaluates the model using settings from run_cfg_dict."""
    train_loader, val_loader, test_loader = loaders

    # Pass all necessary parameters from run_cfg_dict to engine.train_model
    trained_model, history = engine.train_model(
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=run_cfg_dict['EPOCHS'],
        loss_epsilon=run_cfg_dict['LOSS_EPSILON'],
        use_scheduler=run_cfg_dict['USE_SCHEDULER'],
        save_best_model=run_cfg_dict.get('SAVE_BEST_MODEL', True),
        early_stopping_enabled=run_cfg_dict.get('EARLY_STOPPING_ENABLED', True),
        early_stopping_patience=run_cfg_dict.get('EARLY_STOPPING_PATIENCE', 10),
        early_stopping_metric=run_cfg_dict.get('EARLY_STOPPING_METRIC', 'val_loss'),
        early_stopping_min_delta=run_cfg_dict.get('EARLY_STOPPING_MIN_DELTA', 0.0001)
    )
    training_successful = history is not None and len(history.get('train_loss', [])) > 0
    if training_successful:
        logging.info(f"Training completed for {run_cfg_dict['DATASET_NAME']}. Ran for {len(history['train_loss'])} epochs.")
    else:
        logging.warning(f"Training may have failed or was skipped for {run_cfg_dict['DATASET_NAME']}.")


    oa, aa, kappa, report = None, None, None, None
    evaluation_successful = False
    if trained_model and test_loader and len(test_loader.dataset) > 0:
        trained_model.to(device) # Ensure model is on correct device for eval
        oa, aa, kappa, report, _, _ = engine.evaluate_model(
            model=trained_model, test_loader=test_loader, device=device,
            criterion=criterion, loss_epsilon=run_cfg_dict['LOSS_EPSILON']
        )
        evaluation_successful = True
        logging.info(f"Evaluation for {run_cfg_dict['DATASET_NAME']}: OA={oa:.4f}, AA={aa:.4f}, Kappa={kappa:.4f}")
    elif not trained_model:
        logging.warning(f"Evaluation skipped for {run_cfg_dict['DATASET_NAME']}: Model not trained.")
    else:
        logging.warning(f"Evaluation skipped for {run_cfg_dict['DATASET_NAME']}: Test loader empty or unavailable.")

    return trained_model, history, training_successful, evaluation_successful, oa, aa, kappa, report

def format_results_table(df_agg):
    """Formats the aggregated results DataFrame into a string table for display."""
    float_cols = ['OA_mean', 'OA_std', 'AA_mean', 'AA_std', 'Kappa_mean', 'Kappa_std', 'Time_mean', 'Time_std']
    for col in float_cols:
        if col in df_agg.columns:
            df_agg[col] = df_agg[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    # Define the desired order and names of columns for the output table
    display_cols_ordered = [
        ('Band Selection', 'Band Selection'),
        ('Band Param', 'Band Param'),
        ('Int. Attention', 'Int. Attention'), # Renamed from 'Int. Attention Stages'
        ('Fusion', 'Fusion'),                 # Renamed from 'Fusion Mechanism'
        ('Successful Folds', 'Successful Folds'),
        ('OA_mean', 'OA Mean'), ('OA_std', 'OA Std'),
        ('AA_mean', 'AA Mean'), ('AA_std', 'AA Std'),
        ('Kappa_mean', 'Kappa Mean'), ('Kappa_std', 'Kappa Std'),
        ('Time_mean', 'Time Mean (s)'), ('Time_std', 'Time Std (s)')
    ]
    # Filter out columns not present in df_agg and get the final list of column names for display
    final_display_cols = [orig_col for orig_col, disp_col in display_cols_ordered if orig_col in df_agg.columns]
    df_display = df_agg[final_display_cols].copy()
    # Rename columns for better readability in the text file
    df_display.columns = [disp_col for orig_col, disp_col in display_cols_ordered if orig_col in df_agg.columns]

    return df_display.to_string(index=False)

# ==================================
# Main Execution
# ==================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSSFN sweep for specified datasets and configurations.")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['ALL'],
        choices=list(DATASET_SPECIFIC_CONFIGS.keys()) + ['ALL'],
        help='Specify one or more dataset names to run, or ALL. (default: ALL)'
    )
    args = parser.parse_args()

    datasets_to_process_names = []
    if 'ALL' in args.datasets:
        datasets_to_process_names = list(DATASET_SPECIFIC_CONFIGS.keys())
    else:
        datasets_to_process_names = [name for name in args.datasets if name in DATASET_SPECIFIC_CONFIGS]
        if not datasets_to_process_names:
            logging.error(f"No valid datasets selected from choices: {list(DATASET_SPECIFIC_CONFIGS.keys())}. Exiting.")
            sys.exit(1)

    logging.info(f"========== Starting Master Sweep: Datasets={datasets_to_process_names} ==========")
    logging.info(f"Script: {__file__ if '__file__' in locals() else 'interactive'}")
    logging.info(f"Full command: {' '.join(sys.argv)}")
    logging.info(f"Master Sweep Timestamp: {sweep_timestamp}")
    logging.info(f"Output Directory: {overall_sweep_output_dir}")
    logging.info(f"Parameter Combinations per dataset: {num_combinations_per_dataset}")
    logging.info(f"K-Folds per combination: {K_FOLDS}")

    overall_start_time = time.time()
    master_results_list_all_datasets = [] # Stores detailed results from all runs across all datasets

    try:
        # --- Get Global Default Settings from src.config.py (Once) ---
        base_cfg_dict_global_settings = config_to_dict(base_cfg_module)
        BASE_RANDOM_SEED = base_cfg_dict_global_settings.get('RANDOM_SEED', 42) # Get from config or default
        logging.info(f"Base random seed for folds: {BASE_RANDOM_SEED}")
        if base_cfg_dict_global_settings.get('EARLY_STOPPING_ENABLED', True):
             logging.info(f"Global Early Stopping is ENABLED: Patience={base_cfg_dict_global_settings.get('EARLY_STOPPING_PATIENCE',10)}, Metric='{base_cfg_dict_global_settings.get('EARLY_STOPPING_METRIC','val_loss')}'")
        else:
             logging.info("Global Early Stopping is DISABLED.")


        # --- Loop Through Selected Datasets ---
        for dataset_name in datasets_to_process_names:
            current_dataset_params = DATASET_SPECIFIC_CONFIGS[dataset_name]
            logging.info(f"\n\n\n================ Processing Dataset: {dataset_name} ================")
            dataset_specific_run_results = [] # Results for the current dataset

            # Create a base run configuration for this dataset by merging global defaults with dataset specifics
            # This run_cfg_for_dataset will be further copied and updated for each combo and fold.
            run_cfg_for_dataset = base_cfg_dict_global_settings.copy()
            run_cfg_for_dataset.update(current_dataset_params)
            # Ensure DATA_PATH is absolute
            run_cfg_for_dataset['DATA_PATH'] = os.path.join(base_cfg_module.DATA_BASE_PATH, run_cfg_for_dataset['DATA_PATH_RELATIVE'])


            # --- Load Data for the Current Dataset (Once per dataset) ---
            data_cube, gt_map, B_original = load_data_for_dataset(run_cfg_for_dataset)
            all_labeled_coords, labels_np_array, original_idx_array = \
                spl.get_labeled_coordinates_and_indices(gt_map)

            if not all_labeled_coords:
                logging.error(f"No labeled pixels found for dataset {dataset_name}. Skipping this dataset.")
                continue

            # --- K-Fold Cross-Validation Loop for the Current Dataset ---
            for fold in range(K_FOLDS):
                fold_start_time = time.time()
                current_seed_for_fold = BASE_RANDOM_SEED + fold
                logging.info(f"\n\n======= Dataset: {dataset_name} - Starting Fold {fold+1}/{K_FOLDS} (Seed: {current_seed_for_fold}) =======")
                device = setup_environment(current_seed_for_fold)

                split_coords_dict = spl.split_data_random_stratified(
                    all_labeled_coords, labels_np_array, original_idx_array,
                    FOLD_TRAIN_RATIO, FOLD_VAL_RATIO,
                    run_cfg_for_dataset['NUM_CLASSES'], current_seed_for_fold
                )

                # --- Parameter Combination Loop for the Current Dataset and Fold ---
                for combo_idx, (band_option, intermediate_attention, fusion_mechanism) in enumerate(parameter_combinations):
                    run_start_time = time.time()
                    combo_str = f"BS={band_option['method']}({band_option['param']}), IA={intermediate_attention}, Fusion={fusion_mechanism}"
                    logging.info(f"\n--- Dataset: {dataset_name}, Fold {fold+1}, Combo {combo_idx+1}/{num_combinations_per_dataset}: {combo_str} ---")

                    # Create a specific config dict for this combination and fold
                    run_cfg_dict_for_combo = run_cfg_for_dataset.copy()
                    run_cfg_dict_for_combo['RANDOM_SEED'] = current_seed_for_fold # Ensure fold's seed is used
                    run_cfg_dict_for_combo['INTERMEDIATE_ATTENTION_STAGES'] = intermediate_attention
                    run_cfg_dict_for_combo['FUSION_MECHANISM'] = fusion_mechanism
                    
                    # --- Temporarily override global config state for model.py if it reads directly ---
                    # This is a workaround. Ideally, model.py's DSSFN would take these as __init__ args.
                    _original_cfg_int_attn = base_cfg_module.INTERMEDIATE_ATTENTION_STAGES
                    _original_cfg_fusion = base_cfg_module.FUSION_MECHANISM
                    base_cfg_module.INTERMEDIATE_ATTENTION_STAGES = run_cfg_dict_for_combo['INTERMEDIATE_ATTENTION_STAGES']
                    base_cfg_module.FUSION_MECHANISM = run_cfg_dict_for_combo['FUSION_MECHANISM']
                    # --- End of temporary override ---

                    run_status = "Success"
                    oa_res, aa_res, kappa_res = None, None, None
                    actual_input_bands = B_original # Initialize

                    try:
                        selected_data_combo, actual_input_bands, _ = select_bands_for_run(
                            data_cube, band_option, run_cfg_dict_for_combo
                        )
                        if actual_input_bands <= 0:
                            run_status = "Skipped (0 Bands after BS)"
                        else:
                            data_splits_combo = preprocess_and_split(selected_data_combo, gt_map, split_coords_dict, run_cfg_dict_for_combo)
                            train_loader_combo, val_loader_combo, test_loader_combo = prepare_dataloaders(data_splits_combo, run_cfg_dict_for_combo)

                            if not train_loader_combo or len(train_loader_combo.dataset) == 0 or \
                               not test_loader_combo or len(test_loader_combo.dataset) == 0:
                                run_status = "Skipped (DataLoader Error)"
                            else:
                                model_combo = DSSFN(
                                    input_bands=actual_input_bands,
                                    num_classes=run_cfg_dict_for_combo['NUM_CLASSES'],
                                    patch_size=run_cfg_dict_for_combo['PATCH_SIZE'],
                                    spec_channels=run_cfg_dict_for_combo['SPEC_CHANNELS'],
                                    spatial_channels=run_cfg_dict_for_combo['SPATIAL_CHANNELS'],
                                    fusion_mechanism=run_cfg_dict_for_combo['FUSION_MECHANISM'], # Passed to model
                                    cross_attention_heads=run_cfg_dict_for_combo['CROSS_ATTENTION_HEADS'],
                                    cross_attention_dropout=run_cfg_dict_for_combo['CROSS_ATTENTION_DROPOUT']
                                ).to(device)

                                criterion_combo, optimizer_combo, scheduler_combo = setup_training(model_combo, run_cfg_dict_for_combo)
                                
                                _, _, training_ok, eval_ok, oa_res, aa_res, kappa_res, _ = \
                                    train_evaluate_model(model_combo, criterion_combo, optimizer_combo, scheduler_combo,
                                                         (train_loader_combo, val_loader_combo, test_loader_combo),
                                                         device, run_cfg_dict_for_combo)
                                if not training_ok or not eval_ok:
                                    run_status = f"Failed (Train:{training_ok}, Eval:{eval_ok})"
                    except Exception as e_run:
                        logging.error(f"Error in run ({dataset_name}, Fold {fold+1}, Combo {combo_idx+1}): {e_run}", exc_info=True)
                        run_status = f"Error ({type(e_run).__name__})"
                    finally:
                        # --- Restore original global config state ---
                        base_cfg_module.INTERMEDIATE_ATTENTION_STAGES = _original_cfg_int_attn
                        base_cfg_module.FUSION_MECHANISM = _original_cfg_fusion
                        # --- End of restore ---

                        run_duration_s = time.time() - run_start_time
                        result_entry = {
                            'Dataset': dataset_name, 'Fold': fold + 1, 'Seed': current_seed_for_fold,
                            'Band Selection': band_option['method'], 'Band Param': band_option['param'],
                            'Int. Attention Stages': str(intermediate_attention), # Stored as string for grouping
                            'Fusion Mechanism': fusion_mechanism,
                            'Input Bands': actual_input_bands,
                            'OA': oa_res, 'AA': aa_res, 'Kappa': kappa_res,
                            'Run Time (s)': run_duration_s, 'Status': run_status
                        }
                        dataset_specific_run_results.append(result_entry)
                        master_results_list_all_datasets.append(result_entry)

                        logging.info(f"Run Time: {run_duration_s:.2f}s. Result for {dataset_name}: OA={oa_res if oa_res else 'N/A'}, Status: {run_status}")
                        if 'cuda' in str(device): torch.cuda.empty_cache()
                # --- End of Combination Loop ---
                logging.info(f"======= Dataset: {dataset_name} - Fold {fold+1} finished in {time.time() - fold_start_time:.2f} sec =======")
            # --- End of K-Fold Loop for Current Dataset ---

            # --- Aggregate and Save Results for the Current Dataset ---
            if dataset_specific_run_results:
                df_detailed_current_dataset = pd.DataFrame(dataset_specific_run_results)
                df_detailed_current_dataset['Band Param'] = df_detailed_current_dataset['Band Param'].fillna('N/A')
                
                # Create a subdirectory for this dataset's results within the main sweep folder
                current_dataset_output_dir = os.path.join(overall_sweep_output_dir, dataset_name)
                os.makedirs(current_dataset_output_dir, exist_ok=True)

                detailed_csv_path_curr_ds = os.path.join(current_dataset_output_dir, f"sweep_detailed_results_{dataset_name}_{sweep_timestamp}.csv")
                df_detailed_current_dataset.to_csv(detailed_csv_path_curr_ds, index=False)
                logging.info(f"Detailed results for {dataset_name} saved to: {detailed_csv_path_curr_ds}")

                df_valid_eval_curr_ds = df_detailed_current_dataset.dropna(subset=['OA', 'AA', 'Kappa'])
                if not df_valid_eval_curr_ds.empty:
                    agg_funcs_ds = {'OA': ['mean', 'std'], 'AA': ['mean', 'std'], 'Kappa': ['mean', 'std'],
                                 'Run Time (s)': ['mean', 'std'], 'Fold': ['count']} # Count successful folds
                    df_aggregated_curr_ds = df_valid_eval_curr_ds.groupby(
                        ['Band Selection', 'Band Param', 'Int. Attention Stages', 'Fusion Mechanism']
                    ).agg(agg_funcs_ds)
                    df_aggregated_curr_ds.columns = ['_'.join(col).strip() for col in df_aggregated_curr_ds.columns.values]
                    df_aggregated_curr_ds = df_aggregated_curr_ds.rename(columns={'Fold_count': 'Successful Folds'}).reset_index()
                    # Rename for display
                    df_aggregated_curr_ds = df_aggregated_curr_ds.rename(columns={
                        'Int. Attention Stages': 'Int. Attention', 'Fusion Mechanism': 'Fusion',
                        'Run Time (s)_mean': 'Time_mean', 'Run Time (s)_std': 'Time_std'})
                    
                    logging.info(f"Aggregated Results for {dataset_name}:\n" + df_aggregated_curr_ds.to_string(index=False))
                    
                    agg_csv_path_curr_ds = os.path.join(current_dataset_output_dir, f"sweep_aggregated_results_{dataset_name}_{sweep_timestamp}.csv")
                    df_aggregated_curr_ds.to_csv(agg_csv_path_curr_ds, index=False)
                    
                    agg_txt_path_curr_ds = os.path.join(current_dataset_output_dir, f"sweep_aggregated_results_{dataset_name}_{sweep_timestamp}.txt")
                    with open(agg_txt_path_curr_ds, 'w') as f:
                        f.write(f"Aggregated Results ({dataset_name}) - Timestamp: {sweep_timestamp}\n")
                        f.write(f"K-Folds: {K_FOLDS}, Train Ratio: {FOLD_TRAIN_RATIO}, Val Ratio: {FOLD_VAL_RATIO}\n")
                        f.write(f"Base Random Seed: {BASE_RANDOM_SEED}\n\n")
                        f.write(format_results_table(df_aggregated_curr_ds.copy())) # Pass a copy
                    logging.info(f"Aggregated results for {dataset_name} saved to CSV and TXT in {current_dataset_output_dir}")
                else:
                    logging.warning(f"No runs completed successfully for aggregation for dataset {dataset_name}.")
            else:
                logging.warning(f"No results collected for dataset {dataset_name}.")
            # --- End of Current Dataset Processing ---

        # --- Save All Detailed Results from All Processed Datasets (Master File) ---
        if master_results_list_all_datasets:
            df_master_detailed_all = pd.DataFrame(master_results_list_all_datasets)
            df_master_detailed_all['Band Param'] = df_master_detailed_all['Band Param'].fillna('N/A') # Ensure NaNs are handled
            master_detailed_csv_path_all = os.path.join(overall_sweep_output_dir, f"MASTER_sweep_all_datasets_detailed_results_{sweep_timestamp}.csv")
            df_master_detailed_all.to_csv(master_detailed_csv_path_all, index=False)
            logging.info(f"MASTER detailed results for ALL processed datasets saved to: {master_detailed_csv_path_all}")
        else:
            logging.warning("No results collected across any dataset during the master sweep.")

        logging.info("\n\n========== Master Sweep Finished ==========")

    except Exception as e_master:
        logging.error(f"Critical error in master sweep: {e_master}", exc_info=True)
        if master_results_list_all_datasets: # Attempt to save any partial results
            try:
                pd.DataFrame(master_results_list_all_datasets).to_csv(os.path.join(overall_sweep_output_dir, f"MASTER_sweep_all_datasets_DETAILED_ERROR_{sweep_timestamp}.csv"), index=False)
                logging.info(f"Saved partial detailed results for ALL datasets after error.")
            except Exception as save_e:
                logging.error(f"Could not save partial results after error: {save_e}")
        sys.exit(1) # Exit with error code
    finally:
        total_duration_seconds = time.time() - overall_start_time
        total_duration_hours = total_duration_seconds / 3600
        logging.info(f"\nTotal master sweep execution time: {total_duration_seconds:.2f} seconds ({total_duration_hours:.2f} hours)")
        logging.info(f"========== Master Sweep Script Terminated. Output in: {overall_sweep_output_dir} ==========")
