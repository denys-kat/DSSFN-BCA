# src/modules.py
# Contains core building blocks for the DSSFN model:
# SelfAttention, CrossAttention, and PyramidalResidualBlock.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For sqrt in attention
import logging

class SelfAttention(nn.Module):
    """
    Self-attention Layer based on Section 3.3 and Figure 7 of the paper.
    Adapts for spectral (1D) or spatial (2D) input.
    """
    def __init__(self, in_dim):
        """
        Args:
            in_dim (int): Number of input channels.
        """
        super(SelfAttention, self).__init__()
        self.in_dim = in_dim
        # Use a smaller intermediate dimension for query/key
        inter_dim = max(1, in_dim // 8)

        # --- Layers for Spatial Attention (4D input: B, C, H, W) ---
        self.query_conv_2d = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.key_conv_2d = nn.Conv2d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.value_conv_2d = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # --- Layers for Spectral Attention (3D input: B, C, L) ---
        # Using Q,K,V derived from input x, similar to spatial version but with Conv1d
        self.query_conv_1d = nn.Conv1d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.key_conv_1d = nn.Conv1d(in_channels=in_dim, out_channels=inter_dim, kernel_size=1)
        self.value_conv_1d = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # Learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1) # Softmax applied over the last dimension (N or L)

    def forward(self, x):
        """
        Forward pass for Self-Attention.

        Args:
            x (torch.Tensor): Input feature map.
                              Expected shape: (B, C, L) for spectral or (B, C, H, W) for spatial.

        Returns:
            torch.Tensor: Output feature map (same shape as input) with attention applied.
                          out = gamma * attention_output + x
        """
        if x.dim() == 3: # Spectral Attention (B, C, L) -> Attends over L dimension
            m_batchsize, C, length = x.size()
            proj_query = self.query_conv_1d(x).permute(0, 2, 1) # B, L, C'
            proj_key = self.key_conv_1d(x) # B, C', L
            energy = torch.bmm(proj_query, proj_key) # B, L, L (Attention map over length)

            attention = self.softmax(energy) # B, L, L
            proj_value = self.value_conv_1d(x) # B, C, L
            # Apply attention: B,L,L @ B,C,L -> needs proj_value as (B, L, C)
            proj_value_permuted = proj_value.permute(0, 2, 1) # B, L, C
            attn_output = torch.bmm(attention, proj_value_permuted) # B, L, L @ B, L, C -> B, L, C
            attn_output = attn_output.permute(0, 2, 1) # B, C, L (Back to original format)

            out = self.gamma * attn_output + x # Residual connection

        elif x.dim() == 4: # Spatial Attention (B, C, H, W) -> Attends over N=H*W dimension
            m_batchsize, C, height, width = x.size()
            N = height * width # Number of spatial locations (pixels)

            proj_query = self.query_conv_2d(x).view(m_batchsize, -1, N).permute(0, 2, 1) # B, N, C'
            proj_key = self.key_conv_2d(x).view(m_batchsize, -1, N) # B, C', N
            energy = torch.bmm(proj_query, proj_key) # B, N, N (Spatial attention map)
            attention = self.softmax(energy) # Softmax over spatial dimension N

            proj_value = self.value_conv_2d(x).view(m_batchsize, -1, N) # B, C, N

            # Apply spatial attention B,C,N @ B,N,N.T -> B,C,N
            attn_output = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, N
            attn_output = attn_output.view(m_batchsize, C, height, width) # Reshape back B, C, H, W

            out = self.gamma * attn_output + x # Residual connection

        else:
            raise ValueError("Input tensor must be 3D (B, C, L) or 4D (B, C, H, W)")

        return out

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention Layer.
    Allows one feature stream (query) to attend to another (context) using multiple heads.
    """
    def __init__(self, in_dim, num_heads=8, dropout=0.1):
        """
        Args:
            in_dim (int): Feature dimension of query and context streams. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(MultiHeadCrossAttention, self).__init__()
        if in_dim % num_heads != 0:
            raise ValueError(f"in_dim ({in_dim}) must be divisible by num_heads ({num_heads})")

        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5 # Scaling factor for attention

        # Linear layers to project query, key, value for all heads at once
        self.to_q = nn.Linear(in_dim, in_dim, bias=False) # Projects query to Dim = num_heads * head_dim
        self.to_k = nn.Linear(in_dim, in_dim, bias=False) # Projects context to Dim
        self.to_v = nn.Linear(in_dim, in_dim, bias=False) # Projects context to Dim

        # Final linear layer after concatenating heads
        self.to_out = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Dropout(dropout)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context):
        """
        Forward pass for Multi-Head Cross-Attention.

        Args:
            x (torch.Tensor): Query feature map. Shape: (B, Seq_len_Q, Dim).
            context (torch.Tensor): Context feature map. Shape: (B, Seq_len_KV, Dim).

        Returns:
            torch.Tensor: Output feature map, shape: (B, Seq_len_Q, Dim).
        """
        B, N_Q, C = x.shape
        B, N_KV, C_ctx = context.shape
        if C != self.in_dim or C_ctx != self.in_dim:
             raise ValueError(f"Feature dimension mismatch: x({C}) or context({C_ctx}) != in_dim({self.in_dim})")

        # 1. Linear projections for Q, K, V
        q = self.to_q(x)  # (B, N_Q, Dim)
        k = self.to_k(context) # (B, N_KV, Dim)
        v = self.to_v(context) # (B, N_KV, Dim)

        # 2. Reshape and transpose for multi-head calculation
        # (B, Seq_len, Dim) -> (B, Seq_len, num_heads, head_dim) -> (B, num_heads, Seq_len, head_dim)
        q = q.view(B, N_Q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N_KV, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N_KV, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Scaled dot-product attention per head
        # (B, H, N_Q, D_h) @ (B, H, D_h, N_KV) -> (B, H, N_Q, N_KV)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = self.softmax(attn_scores) # Softmax over N_KV dimension

        # 4. Apply attention to Value per head
        # (B, H, N_Q, N_KV) @ (B, H, N_KV, D_h) -> (B, H, N_Q, D_h)
        attn_output = torch.matmul(attn_probs, v)

        # 5. Concatenate heads and reshape back
        # (B, H, N_Q, D_h) -> (B, N_Q, H, D_h) -> (B, N_Q, Dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(B, N_Q, self.in_dim)

        # 6. Final linear layer and residual connection
        out = self.to_out(attn_output) + x # Add residual connection to the original query 'x'

        return out


class PyramidalResidualBlock(nn.Module):
    """
    Implements the Pyramidal Residual Block with Self-Attention integration.
    """
    def __init__(self, in_channels, out_channels, stride=1, is_1d=False):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolution (used for downsampling). Defaults to 1.
            is_1d (bool): True for spectral (1D conv), False for spatial (2D conv). Defaults to False.
        """
        super(PyramidalResidualBlock, self).__init__()
        self.is_1d = is_1d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        Conv = nn.Conv1d if is_1d else nn.Conv2d
        BN = nn.BatchNorm1d if is_1d else nn.BatchNorm2d
        kernel_size = 3
        padding = 1

        self.conv1 = Conv(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False)
        self.bn1 = BN(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.sa = SelfAttention(out_channels)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=kernel_size,
                          stride=1, padding=padding, bias=False)
        self.bn2 = BN(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                BN(out_channels)
            )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """ Forward pass through the Pyramidal Residual Block. """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.sa(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.shortcut(identity)
        out += identity
        out = self.relu2(out)
        return out


class HaltingModule(nn.Module):
    """
    Halting Module for Adaptive Computation Time (ACT).
    Predicts a halting probability for the current state.
    
    Based on Graves (2016) "Adaptive Computation Time for Recurrent Neural Networks".
    """
    def __init__(self, in_channels, is_1d=False, init_bias=-3.0):
        """
        Args:
            in_channels (int): Number of input channels.
            is_1d (bool): True for spectral (1D), False for spatial (2D).
            init_bias (float): Initial bias for halting FC layer. Negative values
                              encourage low initial halting probability (more computation).
        """
        super(HaltingModule, self).__init__()
        self.is_1d = is_1d
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool1d(1) if is_1d else nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize bias to a negative value to encourage starting with low halting probability
        nn.init.constant_(self.fc.bias, init_bias)
        # Xavier init for weights
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map. Shape: (B, C, L) for 1D or (B, C, H, W) for 2D.
        Returns:
            torch.Tensor: Halting probability (B, 1), values in [0, 1].
        """
        # Global Average Pooling
        y = self.pool(x)  # (B, C, 1) or (B, C, 1, 1)
        y = y.flatten(1)  # (B, C)
        
        # Predict probability
        p = self.sigmoid(self.fc(y))  # (B, 1)
        return p


class ACTController:
    """
    Adaptive Computation Time Controller.
    
    Manages the halting logic across multiple stages, computing weighted
    outputs and tracking ponder cost for the loss function.
    
    The ACT mechanism allows early exit when cumulative halting probability
    exceeds (1 - epsilon), distributing computation adaptively per sample.
    """
    
    def __init__(self, num_stages: int = 3, epsilon: float = 0.01):
        """
        Args:
            num_stages: Maximum number of stages/blocks.
            epsilon: Halting threshold. Model halts when cumulative_p >= 1 - epsilon.
        """
        self.num_stages = num_stages
        self.epsilon = epsilon
        self.threshold = 1.0 - epsilon
    
    def compute_act_output(
        self,
        stage_outputs: list,
        halting_probs: list,
    ):
        """
        Compute the ACT-weighted output from stage outputs and halting probabilities.
        
        This implements the core ACT algorithm:
        1. Accumulate halting probabilities until threshold is reached
        2. Compute remainder for final stage
        3. Weight each stage's output by its contribution
        
        Args:
            stage_outputs: List of tensors, one per stage. Each: (B, ...) - feature maps or logits
            halting_probs: List of tensors (B, 1), halting probability at each stage
            
        Returns:
            Tuple of:
                - weighted_output: (B, ...) - ACT-weighted combination of stage outputs
                - ponder_cost: (B,) - Number of stages used per sample (for loss)
                - halting_step: (B,) - Which stage each sample halted at
                - stage_weights: List of (B, 1) - Weight given to each stage
        """
        batch_size = stage_outputs[0].shape[0]
        device = stage_outputs[0].device
        output_shape = stage_outputs[0].shape[1:]
        
        # Initialize accumulators
        cumulative_prob = torch.zeros(batch_size, 1, device=device)
        weighted_output = torch.zeros(batch_size, *output_shape, device=device)
        ponder_cost = torch.zeros(batch_size, device=device)
        halting_step = torch.zeros(batch_size, dtype=torch.long, device=device)
        stage_weights = []
        
        # Track which samples are still active (haven't halted yet)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for t, (output_t, p_t) in enumerate(zip(stage_outputs, halting_probs)):
            stage_idx = t + 1  # 1-indexed stage number
            
            # Check if this is the last stage
            is_last_stage = (stage_idx == self.num_stages)

            # Any active sample has "used" this stage (processed its compute path).
            # This yields a true stage-count ponder cost: halt at stage 2 -> cost 2, etc.
            ponder_cost = ponder_cost + active.float()
            
            # For active samples, check if we should halt
            should_halt = active & ((cumulative_prob.squeeze(-1) + p_t.squeeze(-1)) >= self.threshold)
            
            # Compute weights for this stage
            # If halting: remainder = 1 - cumulative_prob
            # If continuing: weight = p_t
            # If last stage: remainder for all still active
            
            weight_t = torch.zeros(batch_size, 1, device=device)
            
            if is_last_stage:
                # Last stage gets all remaining probability for active samples
                remainder = (1.0 - cumulative_prob) * active.unsqueeze(-1).float()
                weight_t = remainder
                halting_step[active] = stage_idx
            else:
                # Samples that halt here get remainder as weight
                remainder = (1.0 - cumulative_prob) * should_halt.unsqueeze(-1).float()
                weight_t = weight_t + remainder
                halting_step[should_halt] = stage_idx
                
                # Samples that continue get p_t as weight
                continuing = active & ~should_halt
                weight_t = weight_t + p_t * continuing.unsqueeze(-1).float()
            
            # Accumulate weighted output
            # Expand weight to match output dimensions
            weight_expanded = weight_t
            for _ in range(len(output_shape) - 1):
                weight_expanded = weight_expanded.unsqueeze(-1)
            if len(output_shape) > 0:
                weight_expanded = weight_expanded.expand_as(output_t)
            
            weighted_output = weighted_output + weight_expanded * output_t
            
            # Update cumulative probability for continuing samples
            if not is_last_stage:
                continuing = active & ~should_halt
                cumulative_prob = cumulative_prob + p_t * continuing.unsqueeze(-1).float()
                # Update active mask
                active = active & ~should_halt
            
            stage_weights.append(weight_t)
        
        # Ponder cost is the sum of "did we process this stage" for each sample
        # Minimum is 1 (always process at least one stage)
        ponder_cost = ponder_cost.clamp(min=1.0)
        
        return weighted_output, ponder_cost, halting_step, stage_weights
    
    def compute_ponder_loss(self, ponder_cost: torch.Tensor) -> torch.Tensor:
        """
        Compute ponder loss (regularization term encouraging early halting).
        
        Args:
            ponder_cost: (B,) tensor of stages used per sample
            
        Returns:
            Scalar tensor - mean ponder cost across batch
        """
        return ponder_cost.mean()
