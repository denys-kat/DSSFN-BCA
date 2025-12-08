import sys
import os
import torch
import pandas as pd
import time
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config
from src.datasets import create_dataloaders
from src.model import DSSFN
from src.engine import train_model, evaluate_model
from src.band_selection import apply_swgmf, apply_efdpc
import src.data_utils as data_utils
import src.sampling as sampling

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_experiment(dataset_name, config_name):
    logging.info(f"--- Starting Experiment: {dataset_name} | {config_name} ---")
    
    # 1. Configure Config Module
    config.DATASET_NAME = dataset_name
    # Re-load dataset specific params
    if dataset_name == 'IndianPines':
        config.DATA_PATH_RELATIVE = 'ip/'
        config.DATA_FILE = 'indianpinearray.npy'
        config.GT_FILE = 'IPgt.npy'
        config.NUM_CLASSES = 16
        config.DATA_MAT_KEY = None
        config.GT_MAT_KEY = None
    elif dataset_name == 'PaviaUniversity':
        config.DATA_PATH_RELATIVE = 'pu/'
        config.DATA_FILE = 'PaviaU.mat'
        config.GT_FILE = 'PaviaU_gt.mat'
        config.NUM_CLASSES = 9
        config.DATA_MAT_KEY = 'paviaU'
        config.GT_MAT_KEY = 'paviaU_gt'
    elif dataset_name == 'Botswana':
        config.DATA_PATH_RELATIVE = 'botswana/'
        config.DATA_FILE = 'Botswana.mat'
        config.GT_FILE = 'Botswana_gt.mat'
        config.NUM_CLASSES = 14
        config.DATA_MAT_KEY = 'Botswana'
        config.GT_MAT_KEY = 'Botswana_gt'
    
    config.DATA_PATH = os.path.join(config.DATA_BASE_PATH, config.DATA_PATH_RELATIVE)
    
    # Set Architecture/Method Params
    if config_name == 'Baseline':
        config.BAND_SELECTION_METHOD = 'SWGMF'
        config.SWGMF_TARGET_BANDS = 30
        config.INTERMEDIATE_ATTENTION_STAGES = [] # No BCA
        config.FUSION_MECHANISM = 'AdaptiveWeight'
    elif config_name == 'TermPaper':
        config.BAND_SELECTION_METHOD = 'E-FDPC'
        config.INTERMEDIATE_ATTENTION_STAGES = [1, 2] # BCA Enabled
        config.FUSION_MECHANISM = 'CrossAttention' # BCA Fusion
    
    # Ensure symmetric channels for BCA compatibility
    config.SPEC_CHANNELS = [64, 128, 256]
    config.SPATIAL_CHANNELS = [64, 128, 256]

    # 2. Load Data
    data, gt = data_utils.load_hyperspectral_data(
        config.DATA_PATH, 
        config.DATA_FILE, 
        config.GT_FILE,
        expected_data_shape=config.EXPECTED_DATA_SHAPE,
        expected_gt_shape=config.EXPECTED_GT_SHAPE,
        data_mat_key=config.DATA_MAT_KEY,
        gt_mat_key=config.GT_MAT_KEY
    )
    
    # 3. Band Selection
    start_time = time.time()
    if config.BAND_SELECTION_METHOD == 'SWGMF':
        data, selected_bands = apply_swgmf(data, config.SWGMF_WINDOW_SIZE, config.SWGMF_TARGET_BANDS)
    elif config.BAND_SELECTION_METHOD == 'E-FDPC':
        data, selected_bands = apply_efdpc(data, config.E_FDPC_DC_PERCENT)
    else:
        selected_bands = list(range(data.shape[2]))
    
    input_bands = data.shape[2]
    logging.info(f"Selected {input_bands} bands using {config.BAND_SELECTION_METHOD}")

    # 4. Preprocessing & Dataloaders
    data = data_utils.normalize_data(data)
    data = data_utils.pad_data(data, config.BORDER_SIZE)
    
    # Get labeled coordinates
    labeled_coords, labels_array, original_indices = sampling.get_labeled_coordinates_and_indices(gt)
    
    # Split data
    split_coords = sampling.split_data_random_stratified(
        labeled_coords, labels_array, original_indices,
        train_ratio=0.1, # Fixed 10%
        val_ratio=0.1,
        num_classes=config.NUM_CLASSES,
        random_seed=config.RANDOM_SEED
    )
    
    # Create patches
    data_splits = {}
    for split_name in ['train', 'val', 'test']:
        coords = split_coords[f'{split_name}_coords']
        patches, labels = data_utils.create_patches_from_coords(data, coords, config.PATCH_SIZE)
        data_splits[f'{split_name}_patches'] = patches
        data_splits[f'{split_name}_labels'] = labels

    loaders = create_dataloaders(
        data_splits, 
        batch_size=config.BATCH_SIZE, 
        num_workers=0 # Avoid multiprocessing issues in script
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # 5. Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DSSFN(
        input_bands=input_bands,
        num_classes=config.NUM_CLASSES,
        patch_size=config.PATCH_SIZE,
        spec_channels=config.SPEC_CHANNELS,
        spatial_channels=config.SPATIAL_CHANNELS,
        fusion_mechanism=config.FUSION_MECHANISM,
        cross_attention_heads=config.CROSS_ATTENTION_HEADS,
        cross_attention_dropout=config.CROSS_ATTENTION_DROPOUT
    ).to(device)

    # 6. Training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)
    
    # Train
    model, _ = train_model(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler,
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device,
        epochs=50, # Reduced epochs for speed
        use_scheduler=True
    )
    
    # 7. Evaluation
    # evaluate_model returns: overall_accuracy, average_accuracy, kappa, report_str, all_preds, all_labels
    oa, aa, kappa, _, _, _ = evaluate_model(model, test_loader, device, criterion)
    duration = time.time() - start_time
    
    return {
        'Dataset': dataset_name,
        'Config': config_name,
        'OA': oa,
        'AA': aa,
        'Kappa': kappa,
        'Bands': input_bands,
        'Time': duration
    }

if __name__ == "__main__":
    datasets = ['IndianPines', 'PaviaUniversity', 'Botswana']
    configs = ['Baseline', 'TermPaper']
    results = []

    for dataset in datasets:
        for conf in configs:
            try:
                res = run_experiment(dataset, conf)
                results.append(res)
            except Exception as e:
                logging.error(f"Failed {dataset} {conf}: {e}")
                import traceback
                traceback.print_exc()

    df = pd.DataFrame(results)
    print("\nFinal Comparative Results:")
    print(df)
    df.to_csv("results/comparative_study_results.csv", index=False)
