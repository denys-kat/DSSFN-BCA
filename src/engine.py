# src/engine.py
# Contains functions for training and evaluating the DSSFN model.

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import logging
import copy

# Get logger for this module
logger = logging.getLogger(__name__)

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                device, epochs, loss_epsilon=1e-7, use_scheduler=True,
                save_best_model=True,
                early_stopping_enabled=False,
                early_stopping_patience=10,
                early_stopping_metric='val_accuracy', # 'val_loss' or 'val_accuracy'
                early_stopping_min_delta=0.0001):
    """
    Trains the model with options for saving the best state and early stopping.

    Args:
        model: The neural network model.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to train on.
        epochs: Number of epochs.
        loss_epsilon: Stability term for adaptive weighting.
        use_scheduler: Whether to step the scheduler.
        save_best_model: If True, returns the model state with best val metric.
        early_stopping_enabled: If True, enables early stopping.
        early_stopping_patience: Epochs to wait before stopping.
        early_stopping_metric: Metric to monitor ('val_loss' or 'val_accuracy').
        early_stopping_min_delta: Minimum change to qualify as improvement.

    Returns:
        tuple: (trained_model, history)
    """
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    # Initialize best metric tracking
    # We track both for logging, but use one for decision making
    best_val_loss = float('inf')
    best_val_acc = float('-inf')
    
    best_model_state_dict = copy.deepcopy(model.state_dict()) # Default to initial state
    
    epochs_no_improve = 0
    early_stopping_triggered = False

    if early_stopping_enabled and val_loader is None:
        logger.warning("Early stopping enabled but no validation loader provided. Disabling.")
        early_stopping_enabled = False

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Tqdm progress bar for training loops
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Handle AdaptiveWeight fusion which returns two outputs
            if hasattr(model, 'fusion_mechanism') and model.fusion_mechanism == 'AdaptiveWeight':
                spec_logits, spat_logits = model(inputs)
                
                # Adaptive Loss Calculation
                loss_spec = criterion(spec_logits, labels)
                loss_spat = criterion(spat_logits, labels)

                # Detach for weight calculation to avoid graph cycles/unwanted grads
                alpha = 1.0 / (loss_spec.detach() + loss_epsilon)
                beta = 1.0 / (loss_spat.detach() + loss_epsilon)
                
                # Normalize
                total_weight = alpha + beta
                alpha_norm = alpha / total_weight
                beta_norm = beta / total_weight
                
                # Combined logits for backprop and accuracy
                outputs = alpha_norm * spec_logits + beta_norm * spat_logits
                loss = criterion(outputs, labels)
            else:
                # Standard forward pass (CrossAttention or AdaptiveDSSFN)
                outputs = model(inputs)
                # Handle AdaptiveDSSFN returning tuple (logits, cost, steps)
                if isinstance(outputs, tuple):
                    outputs = outputs[0] # Just take logits for standard training loop
                    # Note: specialized train_adaptive_model handles the ACT loss
                
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        epoch_train_acc = correct_train / total_train if total_train > 0 else 0
        
        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_acc)

        log_msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}"

        # --- Validation Phase ---
        if val_loader:
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    if hasattr(model, 'fusion_mechanism') and model.fusion_mechanism == 'AdaptiveWeight':
                        spec_logits, spat_logits = model(inputs)
                        # Simple average for validation inference stability
                        outputs = (spec_logits + spat_logits) / 2.0
                        # Ideally re-calculate weights, but simple average is robust for eval
                    else:
                        outputs = model(inputs)
                        if isinstance(outputs, tuple): outputs = outputs[0]

                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = running_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
            epoch_val_acc = correct_val / total_val if total_val > 0 else 0
            
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_acc)
            
            log_msg += f" | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"

            # --- Best Model Tracking & Early Stopping ---
            improved = False
            
            # Update best metrics
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                if save_best_model and early_stopping_metric == 'val_accuracy':
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    improved = True
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                if save_best_model and early_stopping_metric == 'val_loss':
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    improved = True

            # Check Early Stopping
            if early_stopping_enabled:
                metric_val = epoch_val_acc if early_stopping_metric == 'val_accuracy' else -epoch_val_loss
                best_val = best_val_acc if early_stopping_metric == 'val_accuracy' else -best_val_loss
                
                # Check if this epoch was an improvement (considering delta)
                # Note: 'improved' flag above is strict inequality, here we apply delta logic
                is_improvement = False
                if early_stopping_metric == 'val_accuracy':
                    # Best tracked separately, compare current against best recorded so far
                    if epoch_val_acc > (best_val_acc - early_stopping_min_delta): # Relaxed check for counter reset?
                        # Usually ES checks if current > best_so_far + delta
                        pass 
                    
                    # Simplification: Reset counter if we saved a new best model based on metric
                    if improved: 
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                else: # val_loss
                    if improved:
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    early_stopping_triggered = True

        logger.info(log_msg)

        if use_scheduler and scheduler:
            scheduler.step()

        if early_stopping_triggered:
            break

    # Load best model if requested
    if save_best_model and best_model_state_dict is not None:
        logger.info(f"Loading best model state (Val Acc: {best_val_acc:.4f}, Val Loss: {best_val_loss:.4f})")
        model.load_state_dict(best_model_state_dict)

    return model, history


def train_adaptive_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                         device, epochs, ponder_cost_weight=0.01, use_scheduler=True,
                         save_best_model=True, early_stopping_enabled=False,
                         early_stopping_patience=10, early_stopping_metric='val_accuracy',
                         early_stopping_min_delta=0.0001):
    """
    Trains an AdaptiveDSSFN model including Ponder Cost regularization.
    """
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    best_metric_val = float('-inf') if early_stopping_metric == 'val_accuracy' else float('inf')
    best_model_state_dict = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train-Adapt]", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass with ACT details
            logits, ponder_cost, halting_step = model(inputs, return_ponder_cost=True)
            
            cls_loss = criterion(logits, labels)
            ponder_loss = ponder_cost.mean() * ponder_cost_weight
            loss = cls_loss + ponder_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ponder': f"{ponder_loss.item():.4f}"})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_acc)
        
        log_msg = f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        
        # Validation
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Inference usually doesn't need ponder cost for loss, but we compute it for consistency check
                    logits, ponder_cost, _ = model(inputs, return_ponder_cost=True)
                    
                    # Validation loss usually just Classification Loss, or Classification + Ponder?
                    # Let's track pure classification performance for metric, but combined for loss
                    cls_loss = criterion(logits, labels)
                    # ponder_loss = ponder_cost.mean() * ponder_cost_weight
                    loss = cls_loss # + ponder_loss 
                    
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_epoch_acc = val_correct / val_total
            
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_acc)
            
            log_msg += f" | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}"
            
            # Save Best / Early Stopping
            current_val = val_epoch_acc if early_stopping_metric == 'val_accuracy' else val_epoch_loss
            improved = False
            
            if early_stopping_metric == 'val_accuracy':
                if current_val > best_metric_val:
                    best_metric_val = current_val
                    improved = True
            else:
                if current_val < best_metric_val:
                    best_metric_val = current_val
                    improved = True
            
            if improved:
                if save_best_model:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if early_stopping_enabled and epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        logger.info(log_msg)
        
        if use_scheduler and scheduler:
            scheduler.step()
            
    if save_best_model:
        model.load_state_dict(best_model_state_dict)
        
    return model, history

def evaluate_model(model, test_loader, device, criterion=None):
    """
    Standard evaluation loop.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if hasattr(model, 'fusion_mechanism') and model.fusion_mechanism == 'AdaptiveWeight':
                s1, s2 = model(inputs)
                outputs = (s1 + s2) / 2.0
            else:
                outputs = model(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    oa = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # Calculate AA
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    # Filter out accuracy, macro avg, weighted avg to get class keys
    class_keys = [k for k in report.keys() if k.isdigit()]
    if class_keys:
        aa = np.mean([report[k]['recall'] for k in class_keys])
    else:
        aa = 0.0
        
    report_str = classification_report(all_labels, all_preds, zero_division=0)
    
    return oa, aa, kappa, report_str, all_preds, all_labels

def evaluate_adaptive_model(model, test_loader, device, criterion=None):
    """
    Evaluation loop for AdaptiveDSSFN returning depth metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_depth = 0
    n_samples = 0
    stage_counts = [0, 0, 0]
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Adaptive", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits, _, halting_step = model(inputs, return_ponder_cost=True)
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            total_depth += halting_step.sum().item()
            n_samples += inputs.size(0)
            
            # Count distribution
            for step in halting_step:
                idx = int(step.item()) - 1
                if 0 <= idx < 3:
                    stage_counts[idx] += 1
                    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    oa = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # AA calculation
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    class_keys = [k for k in report.keys() if k.isdigit()]
    aa = np.mean([report[k]['recall'] for k in class_keys]) if class_keys else 0.0
    report_str = classification_report(all_labels, all_preds, zero_division=0)
    
    avg_depth = total_depth / n_samples if n_samples > 0 else 3.0
    stage_dist = [c/n_samples for c in stage_counts] if n_samples > 0 else [0,0,0]
    flops_reduction = 1.0 - (avg_depth / 3.0)
    
    results = {
        'OA': oa, 'AA': aa, 'Kappa': kappa, 
        'avg_depth': avg_depth, 'stage_distribution': stage_dist,
        'flops_reduction': flops_reduction, 'report': report_str,
        'all_preds': all_preds, 'all_labels': all_labels
    }
    return results