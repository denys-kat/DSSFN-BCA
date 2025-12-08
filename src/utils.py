# src/utils.py
# Utility functions for DSSFN-BCA project

import torch
import torch.nn as nn
import logging
from typing import Dict, Tuple, Optional, List
import time

logger = logging.getLogger(__name__)


def count_conv_flops(module: nn.Module, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> int:
    """
    Count FLOPs for a convolutional layer.
    
    FLOPs = 2 * K * C_in * H_out * W_out * C_out (for 2D)
    FLOPs = 2 * K * C_in * L_out * C_out (for 1D)
    
    where K is kernel size (K_h * K_w for 2D, K for 1D)
    Factor of 2 accounts for multiply-add operations.
    """
    if isinstance(module, nn.Conv2d):
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        # output_shape: (B, C_out, H_out, W_out)
        output_elements = output_shape[2] * output_shape[3]
        flops = 2 * kernel_ops * module.in_channels * output_elements * module.out_channels
        if module.bias is not None:
            flops += output_elements * module.out_channels
        return flops
    elif isinstance(module, nn.Conv1d):
        kernel_ops = module.kernel_size[0]
        # output_shape: (B, C_out, L_out)
        output_elements = output_shape[2]
        flops = 2 * kernel_ops * module.in_channels * output_elements * module.out_channels
        if module.bias is not None:
            flops += output_elements * module.out_channels
        return flops
    return 0


def count_linear_flops(module: nn.Linear, input_shape: Tuple[int, ...]) -> int:
    """
    Count FLOPs for a linear layer.
    
    FLOPs = 2 * in_features * out_features (per sample)
    """
    batch_size = input_shape[0]
    flops = 2 * module.in_features * module.out_features * batch_size
    if module.bias is not None:
        flops += module.out_features * batch_size
    return flops


def count_attention_flops(seq_len: int, dim: int, num_heads: int = 1) -> int:
    """
    Count FLOPs for a self-attention or cross-attention operation.
    
    For self-attention with sequence length N and dimension D:
    - Q, K, V projections: 3 * 2 * N * D * D = 6 * N * D^2
    - QK^T: 2 * N * N * D
    - Softmax: ~5 * N * N (approximation)
    - Attention @ V: 2 * N * N * D
    - Output projection: 2 * N * D * D
    
    Total ≈ 8 * N * D^2 + 4 * N^2 * D + 5 * N^2
    """
    # Q, K, V projections (if part of attention layer)
    proj_flops = 6 * seq_len * dim * dim
    
    # QK^T matmul
    qk_flops = 2 * seq_len * seq_len * dim
    
    # Softmax (approximate)
    softmax_flops = 5 * seq_len * seq_len
    
    # Attention @ V
    attn_v_flops = 2 * seq_len * seq_len * dim
    
    # Output projection
    out_proj_flops = 2 * seq_len * dim * dim
    
    total = proj_flops + qk_flops + softmax_flops + attn_v_flops + out_proj_flops
    return total


def count_batchnorm_flops(num_features: int, num_elements: int) -> int:
    """
    Count FLOPs for batch normalization.
    
    BN requires: (x - mean) / sqrt(var + eps) * gamma + beta
    Approximately 4 operations per element.
    """
    return 4 * num_elements


class FLOPsCounter:
    """
    A context manager / utility class to count FLOPs during model forward pass.
    
    Usage:
        counter = FLOPsCounter(model, input_shape=(1, 13, 15, 15))
        flops = counter.count()
        print(f"Total FLOPs: {flops:,}")
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...], device: Optional[str] = None):
        """
        Args:
            model: PyTorch model to analyze
            input_shape: Shape of input tensor (B, C, H, W) for spatial or similar
            device: Device to run on (auto-detected from model if None)
        """
        self.model = model
        self.input_shape = input_shape
        # Auto-detect device from model parameters
        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.flops = 0
        self._hooks = []
        
    def _register_hooks(self):
        """Register forward hooks on all relevant layers."""
        
        def conv_hook(module, input, output):
            self.flops += count_conv_flops(module, input[0].shape, output.shape)
        
        def linear_hook(module, input, output):
            self.flops += count_linear_flops(module, input[0].shape)
        
        def bn_hook(module, input, output):
            num_elements = input[0].numel()
            num_features = module.num_features
            self.flops += count_batchnorm_flops(num_features, num_elements)
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                self._hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                self._hooks.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self._hooks.append(module.register_forward_hook(bn_hook))
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def count(self) -> int:
        """
        Count total FLOPs for one forward pass.
        
        Returns:
            Total FLOPs (multiply-accumulate operations counted as 2 FLOPs)
        """
        self.flops = 0
        self._register_hooks()
        
        # Create dummy input
        x = torch.randn(*self.input_shape, device=self.device)
        
        # Run forward pass
        self.model.eval()
        with torch.no_grad():
            try:
                output = self.model(x)
            except Exception as e:
                logger.warning(f"FLOPs counting failed: {e}")
                self._remove_hooks()
                return 0
        
        self._remove_hooks()
        return self.flops
    
    def count_formatted(self) -> str:
        """Return FLOPs count in human-readable format (M/G FLOPs)."""
        flops = self.count()
        if flops >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        elif flops >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        else:
            return f"{flops} FLOPs"


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Returns:
        Dict with 'total', 'trainable', 'non_trainable' keys
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Measure inference time for a model.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor
        device: Device to run on
        num_warmup: Number of warmup iterations
        num_runs: Number of timed iterations
        
    Returns:
        Dict with 'mean_ms', 'std_ms', 'throughput_samples_per_sec' keys
    """
    model = model.to(device)
    model.eval()
    
    x = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
    
    # Synchronize if CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    import numpy as np
    times = np.array(times)
    batch_size = input_shape[0]
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'throughput_samples_per_sec': float(batch_size * 1000 / np.mean(times))
    }


class AdaptiveDepthStats:
    """
    Track statistics for adaptive depth computation.
    
    Used to monitor how many stages/blocks are used on average
    during training and inference.
    """
    
    def __init__(self, num_stages: int = 3):
        self.num_stages = num_stages
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.total_samples = 0
        self.total_stages_used = 0
        self.stage_counts = [0] * self.num_stages  # How many samples stopped at each stage
        self.halting_probs_sum = [0.0] * self.num_stages
    
    def update(self, halting_step: int, halting_probs: Optional[List[float]] = None):
        """
        Update statistics with a single sample's result.
        
        Args:
            halting_step: Which stage the sample halted at (1-indexed)
            halting_probs: List of halting probabilities at each stage
        """
        self.total_samples += 1
        self.total_stages_used += halting_step
        
        if 1 <= halting_step <= self.num_stages:
            self.stage_counts[halting_step - 1] += 1
        
        if halting_probs is not None:
            for i, p in enumerate(halting_probs):
                if i < self.num_stages:
                    self.halting_probs_sum[i] += p
    
    def update_batch(self, halting_steps: torch.Tensor, halting_probs: Optional[torch.Tensor] = None):
        """
        Update statistics with a batch of results.
        
        Args:
            halting_steps: Tensor of shape (B,) with halting step per sample
            halting_probs: Optional tensor of shape (B, num_stages) with halting probs
        """
        batch_size = halting_steps.shape[0]
        self.total_samples += batch_size
        self.total_stages_used += int(halting_steps.sum().item())
        
        for stage in range(1, self.num_stages + 1):
            self.stage_counts[stage - 1] += int((halting_steps == stage).sum().item())
        
        if halting_probs is not None:
            for i in range(min(halting_probs.shape[1], self.num_stages)):
                self.halting_probs_sum[i] += float(halting_probs[:, i].sum().item())
    
    @property
    def average_depth(self) -> float:
        """Average number of stages used per sample."""
        if self.total_samples == 0:
            return 0.0
        return self.total_stages_used / self.total_samples
    
    @property
    def stage_distribution(self) -> List[float]:
        """Fraction of samples that halted at each stage."""
        if self.total_samples == 0:
            return [0.0] * self.num_stages
        return [c / self.total_samples for c in self.stage_counts]
    
    @property
    def average_halting_probs(self) -> List[float]:
        """Average halting probability at each stage."""
        if self.total_samples == 0:
            return [0.0] * self.num_stages
        return [p / self.total_samples for p in self.halting_probs_sum]
    
    @property
    def flops_reduction_estimate(self) -> float:
        """
        Estimate FLOPs reduction compared to full depth.
        
        Assumes each stage has roughly equal FLOPs.
        Returns fraction of FLOPs saved (0 = no savings, 1 = all saved).
        """
        avg_depth = self.average_depth
        if avg_depth == 0:
            return 0.0
        return 1.0 - (avg_depth / self.num_stages)
    
    def get_summary(self) -> Dict:
        """Get a summary dictionary of all statistics."""
        return {
            'total_samples': self.total_samples,
            'average_depth': self.average_depth,
            'stage_distribution': self.stage_distribution,
            'average_halting_probs': self.average_halting_probs,
            'flops_reduction_estimate': self.flops_reduction_estimate
        }
    
    def __str__(self) -> str:
        if self.total_samples == 0:
            return "AdaptiveDepthStats: No samples recorded"
        
        dist_str = ", ".join([f"S{i+1}:{p:.1%}" for i, p in enumerate(self.stage_distribution)])
        return (
            f"AdaptiveDepthStats:\n"
            f"  Samples: {self.total_samples}\n"
            f"  Avg Depth: {self.average_depth:.2f}/{self.num_stages}\n"
            f"  Distribution: [{dist_str}]\n"
            f"  Est. FLOPs Reduction: {self.flops_reduction_estimate:.1%}"
        )
