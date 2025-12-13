# src/config.py
# Configuration settings for the DSSFN model and experiments
# Uses absolute paths derived from project root.

import os
import numpy as np

# --- Calculate Project Root ---
# Corrected path calculation to ensure it finds the parent of 'src'
# Handles running from 'scripts' or 'src' or project root.
if os.path.basename(os.getcwd()) == 'scripts':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
elif os.path.basename(os.getcwd()) == 'src':
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
elif os.path.basename(os.getcwd()) == os.path.basename(os.path.dirname(os.path.abspath(__file__))): # if __file__ is .../src/config.py
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else: # Assume current working directory is the project root
    PROJECT_ROOT = os.getcwd()

print(f"Project Root calculated in config.py: {PROJECT_ROOT}") # For verification

# --- Base Paths (Absolute) ---
DATA_BASE_PATH = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results') # Base directory for all outputs

# --- Dataset Configuration ---
# Change this to switch datasets for scripts like main.py
DATASET_NAME = 'IndianPines' # 'IndianPines', 'PaviaUniversity', 'Salinas', 'KennedySpaceCenter', 'Botswana'

# --- Data Paths and Dataset-specific Parameters ---
if DATASET_NAME == 'IndianPines':
    DATA_PATH_RELATIVE = 'ip/'
    DATA_FILE = 'indianpinearray.npy'
    GT_FILE = 'IPgt.npy'
    NUM_CLASSES = 16
    EXPECTED_DATA_SHAPE = (145, 145, 200)
    EXPECTED_GT_SHAPE = (145, 145)
    CLASS_NAMES = [
        "Background/Untested", "Alfalfa", "Corn-notill", "Corn-min", "Corn",
        "Grass-pasture", "Grass-trees", "Grass-pasture-mowed",
        "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-min",
        "Soybean-clean", "Wheat", "Woods", "Bldg-Grass-Tree-Drives",
        "Stone-Steel-Towers"
    ]
elif DATASET_NAME == 'PaviaUniversity':
    DATA_PATH_RELATIVE = 'pu/'
    DATA_FILE = 'PaviaU.mat'
    GT_FILE = 'PaviaU_gt.mat'
    NUM_CLASSES = 9
    EXPECTED_DATA_SHAPE = (610, 340, 103)
    EXPECTED_GT_SHAPE = (610, 340)
    CLASS_NAMES = [
        "Background/Untested", "Asphalt", "Meadows", "Gravel", "Trees",
        "Painted metal sheets", "Bare Soil", "Bitumen",
        "Self-Blocking Bricks", "Shadows"
    ]
    DATA_MAT_KEY = 'paviaU'
    GT_MAT_KEY = 'paviaU_gt'

elif DATASET_NAME == 'Salinas':
    DATA_PATH_RELATIVE = 'sa/'
    DATA_FILE = 'Salinas_corrected.npy'
    GT_FILE = 'Salinas_gt.npy'
    NUM_CLASSES = 16
    EXPECTED_DATA_SHAPE = (512, 217, 204)
    EXPECTED_GT_SHAPE = (512, 217)
    CLASS_NAMES = [
        "Background/Untested", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2",
        "Fallow", "Fallow_rough_plow", "Fallow_smooth",
        "Stubble", "Celery", "Grapes_untrained",
        "Soil_vinyard_develop", "Corn_senesced_green_weeds",
        "Lettuce_romaine_4wk", "Lettuce_romaine_5wk",
        "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
        "Vinyard_untrained", "Vinyard_vertical_trellis"
    ]
    DATA_MAT_KEY = 'salinas_corrected'
    GT_MAT_KEY = 'salinas_gt'

elif DATASET_NAME == 'KennedySpaceCenter':
    DATA_PATH_RELATIVE = 'ksc/'
    DATA_FILE = 'KSC.npy'
    GT_FILE = 'KSC_gt.npy'
    NUM_CLASSES = 13
    EXPECTED_DATA_SHAPE = (512, 614, 176)
    EXPECTED_GT_SHAPE = (512, 614)
    CLASS_NAMES = [
        "Background/Untested", "Scrub", "Willow swamp",
        "Cabbage palm hammock", "Cabbage palm/oak hammock",
        "Slash pine", "Oak/broadleaf hammock", "Hardwood swamp",
        "Graminoid marsh", "Spartina marsh", "Cattail marsh",
        "Salt marsh", "Mud flats", "Water"
    ]
    DATA_MAT_KEY = 'KSC'
    GT_MAT_KEY = 'KSC_gt'

elif DATASET_NAME == 'Botswana':
    DATA_PATH_RELATIVE = 'botswana/' # Assuming data is in 'data/botswana/'
    DATA_FILE = 'Botswana.mat'
    GT_FILE = 'Botswana_gt.mat'
    NUM_CLASSES = 14 # From inspection script: 14 actual classes (0 is background)
    EXPECTED_DATA_SHAPE = (1476, 256, 145) # From inspection script
    EXPECTED_GT_SHAPE = (1476, 256)       # From inspection script
    CLASS_NAMES = [
        "Background/Untested",  # Class 0
        "Water",                # Class 1
        "Hippo grass",          # Class 2
        "Floodplain grasses 1", # Class 3
        "Floodplain grasses 2", # Class 4
        "Reeds 1",              # Class 5
        "Riparian",             # Class 6
        "Firescar 2",           # Class 7
        "Island interior",      # Class 8
        "Acacia woodlands",     # Class 9
        "Acacia shrublands",    # Class 10
        "Acacia grasslands",    # Class 11
        "Short mopane",         # Class 12
        "Mixed mopane",         # Class 13
        "Exposed soils"         # Class 14
    ]
    DATA_MAT_KEY = 'Botswana'    # From inspection script
    GT_MAT_KEY = 'Botswana_gt' # From inspection script
else:
    raise ValueError(f"Unsupported DATASET_NAME: {DATASET_NAME}")

DATA_PATH = os.path.join(DATA_BASE_PATH, DATA_PATH_RELATIVE)

# --- Band Selection Parameters ---
BAND_SELECTION_METHOD = 'E-FDPC' # 'SWGMF', 'E-FDPC', or 'None'
SWGMF_TARGET_BANDS = 30 # Adjust as needed if using SWGMF
SWGMF_WINDOW_SIZE = 5
E_FDPC_DC_PERCENT = 2.5 # Adjust as needed if using E-FDPC

# --- Patch Extraction Parameters ---
PATCH_SIZE = 15 # Consider if this is appropriate for Botswana's spatial resolution
BORDER_SIZE = PATCH_SIZE // 2

# --- Data Splitting Parameters ---
TRAIN_RATIO = 0.10
VAL_RATIO = 0.10
RANDOM_SEED = 42

# --- DataLoader Parameters ---
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = True

# --- Model Architecture Parameters ---
SPEC_CHANNELS = [64, 128, 256]
SPATIAL_CHANNELS = [64, 128, 256]
INTERMEDIATE_ATTENTION_STAGES = [] # Default: No intermediate attention
FUSION_MECHANISM = 'AdaptiveWeight' # 'AdaptiveWeight' or 'CrossAttention'
CROSS_ATTENTION_HEADS = 8
CROSS_ATTENTION_DROPOUT = 0.1

# --- Training Hyperparameters ---
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER_TYPE = 'AdamW'
LOSS_EPSILON = 1e-7

# Scheduler settings
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.1

# --- Early Stopping Parameters ---
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_METRIC = 'val_loss'
EARLY_STOPPING_MIN_DELTA = 0.0001

# --- Output & Logging ---
SAVE_BEST_MODEL = True
VERBOSE_EVALUATION = True

# --- Sanity Checks ---
if 1 in INTERMEDIATE_ATTENTION_STAGES and SPEC_CHANNELS[0] != SPATIAL_CHANNELS[0]:
    raise ValueError("Intermediate attention after Stage 1 requires spec_channels[0] == spatial_channels[0].")
if 2 in INTERMEDIATE_ATTENTION_STAGES and SPEC_CHANNELS[1] != SPATIAL_CHANNELS[1]:
    raise ValueError("Intermediate attention after Stage 2 requires spec_channels[1] == spatial_channels[1].")
if SPEC_CHANNELS[2] != SPATIAL_CHANNELS[2]:
    raise ValueError("Final stage (Stage 3) requires spec_channels[2] == spatial_channels[2].")

print(f"Configuration loaded for dataset: {DATASET_NAME}")
print(f"Data will be loaded from: {DATA_PATH}")
print(f"Number of classes: {NUM_CLASSES}")
if EARLY_STOPPING_ENABLED:
    print(f"Early stopping enabled: Patience={EARLY_STOPPING_PATIENCE}, Metric='{EARLY_STOPPING_METRIC}', MinDelta={EARLY_STOPPING_MIN_DELTA}")

# --- MODEL PRESETS ---
MODEL_PRESETS = {
    "base": {
        "s1_channels": 64,
        "spec_c2": 128,
        "spat_c2": 128,
        "s3_channels": 256,
        "swgmf_bands": 30,
        "band_selection": "SWGMF",
        "intermediate_attention": [],
        "fusion_mechanism": "AdaptiveWeight",
        "cross_attention_heads": 8
    },
    "bca": {
        "s1_channels": 64,
        "spec_c2": 128,
        "spat_c2": 128,
        "s3_channels": 256,
        "efdpc_dc_percent": 2.5,
        "band_selection": "E-FDPC",
        "intermediate_attention": [1],
        "fusion_mechanism": "AdaptiveWeight",
        "cross_attention_heads": 8
    }
}