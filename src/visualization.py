# src/visualization.py
# Contains functions for plotting training history and classification maps.
# RESTORED TO "OLD" STYLE (Pastel colors, Colorbar) per user request.

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
import numpy as np

# Need config for dataset name in plot_predictions title fallback
try:
    from . import config as cfg # Use relative import
except ImportError:
    cfg = None # Handle case where config might not be importable directly
    # print("Warning: Could not import config via relative import in visualization.py")

def plot_history(history):
    """
    Plots the training and validation loss and accuracy curves.

    Args:
        history (dict): Dictionary containing lists/arrays for 'train_loss', 'train_accuracy',
                        'val_loss', 'val_accuracy'. Should handle None values if validation skipped.

    Returns:
        tuple: (fig, axes) The matplotlib figure and axes objects. Returns (None, None) if no history.
    """
    if not history or not history.get('train_loss'):
         print("No training history data provided to plot.")
         return None, None # Return None if no history

    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Check for presence and validity of validation loss data
    has_val_loss_data = 'val_loss' in history and \
                        history['val_loss'] is not None and \
                        len(history['val_loss']) > 0 and \
                        all(v is not None for v in history['val_loss'])
    
    # Check for presence and validity of validation accuracy data
    has_val_accuracy_data = 'val_accuracy' in history and \
                            history['val_accuracy'] is not None and \
                            len(history['val_accuracy']) > 0 and \
                            all(v is not None for v in history['val_accuracy'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # axes is now a numpy array [ax1, ax2]
    ax1 = axes[0]
    ax2 = axes[1]
    fig.suptitle('Training History') # Add a main title

    # --- Plot Loss ---
    if history.get('train_loss'):
        ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='.')
    if has_val_loss_data:
        ax1.plot(epochs, history['val_loss'], label='Validation Loss', marker='.')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)

    # --- Plot Accuracy ---
    if history.get('train_accuracy'): # train_accuracy key used in engine.py
        ax2.plot(epochs, history['train_accuracy'], label='Train Accuracy', marker='.')
    if has_val_accuracy_data:
        # Corrected key from 'val_acc' to 'val_accuracy'
        ax2.plot(epochs, history['val_accuracy'], label='Validation Accuracy', marker='.')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    return fig, axes


def plot_predictions(gt_map, test_predictions, test_coords, class_names, dataset_name="Dataset", oa=None, subtitle="Predicted Classification"):
    """
    Generates and displays the ground truth map and the predicted classification map.
    Uses 'tab20' or 'gist_ncar' for professional, distinct colors and a standard colorbar.

    Args:
        gt_map (np.ndarray): The original ground truth map (H, W). Labels > 0.
        test_predictions (np.ndarray): 1D array of predicted labels (0-indexed) for test pixels.
        test_coords (list): List of coordinate dicts [{'r': r, 'c': c}, ...] for test pixels,
                              corresponding to test_predictions. 'r', 'c' are original indices.
        class_names (list): List of class names. Index 0 is Background.
        dataset_name (str): Name of the dataset for the title.
        oa (float, optional): Overall accuracy to display in the title.
        subtitle (str): Subtitle for the prediction plot (e.g., "Full Map" or "Test Only").

    Returns:
        tuple: (fig, axes) The matplotlib figure and axes objects. Returns (None, None) if plotting fails.
    """
    if class_names is None or len(class_names) == 0:
         print("Error: class_names not provided or empty. Cannot plot map.")
         return None, None

    num_vis_entries = len(class_names) # Includes background/untested
    num_classes = num_vis_entries - 1 # Actual number of classes predicted

    if test_predictions is None or test_coords is None:
        print("Error: test_predictions or test_coords is None. Cannot plot map.")
        return None, None

    if len(test_predictions) != len(test_coords):
        print(f"Error: Mismatch between predictions ({len(test_predictions)}) and test coordinates ({len(test_coords)}). Cannot plot map.")
        return None, None # Return None if error

    # --- Create Display Maps ---
    H, W = gt_map.shape

    # Ground Truth Display Map: Uses original labels (0=Bg, 1-N=Classes)
    gt_map_display = gt_map.copy().astype(np.int32) # Values 0 to N

    # Prediction Display Map: Initialize with 0 (represents Background/Not Tested)
    test_prediction_map_display = np.zeros((H, W), dtype=np.int32)

    # Populate map using the coordinates and the 0-indexed predictions
    for i, coord in enumerate(test_coords):
        r, c = coord['r'], coord['c']
        # Assign evaluated prediction + 1 (Map model output 0-(N-1) to display values 1-N)
        predicted_class_index = test_predictions[i] # 0 to N-1
        
        if 0 <= predicted_class_index < num_classes:
            display_value = predicted_class_index + 1   # 1 to N
        else:
            display_value = 0 # Assign to background/unknown

        if 0 <= r < H and 0 <= c < W:
             test_prediction_map_display[r, c] = display_value

    # --- Visualization Setup ---
    # Define Colors: 1 for Background/Not Tested (Black), N for classes
    try:
        # Attempt to get enough distinct colors from a colormap like 'tab20' or 'gist_ncar'
        if num_classes <= 0:
             print("Error: Number of classes is zero or negative. Cannot create colormap.")
             return None, None
        elif num_classes <= 20:
             # This is the "Nice" pastel palette you liked
             cmap_base = matplotlib.colormaps['tab20']
             class_colors_rgba = cmap_base(np.linspace(0, 1, num_classes))
        else: 
             # Use a map with more colors for > 20 classes
             cmap_base = matplotlib.colormaps['gist_ncar']
             class_colors_rgba = cmap_base(np.linspace(0, 1, num_classes))
    except Exception as e: # Fallback for different matplotlib versions/names
        print(f"Note: Using fallback for colormap access due to error: {e}")
        try:
            cmap_base = plt.get_cmap('tab20', num_classes) # Request N colors
        except ValueError: # If tab20 doesn't support requesting N colors
            cmap_base = plt.get_cmap('gist_ncar', num_classes)
        class_colors_rgba = [cmap_base(i) for i in range(num_classes)]

    # Build the final color list: Index 0 = Black, Indices 1 to N = class colors
    colors = [(0, 0, 0, 1.0)] # Black for value 0 (Background/Not Tested)
    colors.extend([tuple(c) for c in class_colors_rgba]) # Add N class colors for values 1 to N

    if len(colors) != num_vis_entries:
        # Pad with default if too short
        colors.extend([(0.5, 0.5, 0.5, 1.0)] * (num_vis_entries - len(colors)))

    custom_cmap = mcolors.ListedColormap(colors)
    # Bounds map values [0, 1, ..., N] to indices [0, 1, ..., N] for the colormap
    bounds = np.arange(num_vis_entries + 1) - 0.5 # Creates boundaries like [-0.5, 0.5, 1.5, ..., N+0.5]
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N) # N should be num_vis_entries

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6)) # axes is now a numpy array [ax1, ax2]

    # Ground Truth Map (values 0-N)
    axes[0].imshow(gt_map_display, cmap=custom_cmap, norm=norm, interpolation='none')
    axes[0].set_title("Ground Truth (Full)")
    axes[0].axis('off')

    # Predicted Map (values 0 for Untested/Bg, 1-N for tested classes)
    im = axes[1].imshow(test_prediction_map_display, cmap=custom_cmap, norm=norm, interpolation='none')
    axes[1].set_title(subtitle)
    axes[1].axis('off')

    # Add Colorbar Legend
    # Ticks should correspond to the *values* being mapped (0 to N)
    tick_locs = np.arange(num_vis_entries) # Locations 0, 1, ..., N

    # Formatter maps tick value (0-N) directly to class name index (0-N)
    formatter = plt.FuncFormatter(lambda val, loc: class_names[int(val)] if 0 <= int(val) < len(class_names) else "")

    # Add colorbar common to both axes
    try:
        fig.colorbar(im, ax=axes.ravel().tolist(), cmap=custom_cmap, norm=norm,
                     boundaries=bounds, ticks=tick_locs, format=formatter,
                     spacing='proportional', fraction=0.046, pad=0.04)
    except Exception as e:
        print(f"Warning: Could not create colorbar - {e}")


    # Add overall title
    title = f"{dataset_name} Classification" # Use dataset name passed as argument
    if oa is not None:
        title += f" (OA: {oa:.4f})"
    plt.suptitle(title, fontsize=14)

    return fig, axes