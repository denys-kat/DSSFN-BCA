# src/model.py
# Defines the main DSSFN model architecture, supporting different fusion mechanisms.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Tuple, Optional, List, Dict

# Get logger for this module
logger = logging.getLogger(__name__)

# Import the building blocks from modules.py using relative import
from .modules import PyramidalResidualBlock, MultiHeadCrossAttention, HaltingModule, ACTController

def _calculate_conv_output_size(input_size, kernel_size, stride, padding):
    """ Calculates the output size of a Conv1D or Conv2D dimension. """
    # Ensure input_size is an integer
    input_size = int(input_size)
    return math.floor((input_size + 2 * padding - kernel_size) / stride) + 1

class DSSFN(nn.Module):
    """
    Dual-Stream Self-Attention Fusion Network (DSSFN).
    Supports Adaptive Weighted Fusion or a new Bidirectional Cross-Attention Fusion for final output.
    Includes optional, configurable BIDIRECTIONAL intermediate cross-attention.
    """
    def __init__(self, input_bands, num_classes, patch_size,
                 spec_channels=[64, 128, 256], spatial_channels=[64, 128, 256],
                 fusion_mechanism='AdaptiveWeight', 
                 cross_attention_heads=8,
                 cross_attention_dropout=0.1,
                 intermediate_attention_stages: List[int] = []):
        """
        Initializes the DSSFN model.
        Args:
            input_bands (int): Number of input spectral bands (after band selection).
            num_classes (int): Number of output classes.
            patch_size (int): Spatial size of the input patch (e.g., 15 for 15x15).
            spec_channels (list): List of channel counts for the three stages of the spectral stream.
            spatial_channels (list): List of channel counts for the three stages of the spatial stream.
            fusion_mechanism (str): Type of fusion for final outputs. Defaults to 'AdaptiveWeight'.
            cross_attention_heads (int): Number of heads for MultiHeadCrossAttention modules.
            cross_attention_dropout (float): Dropout rate for MultiHeadCrossAttention modules.
            intermediate_attention_stages (list): List of stages (1 or 2) after which to apply cross-attention.
        """
        super(DSSFN, self).__init__()

        # --- Basic Setup & Checks ---
        if len(spec_channels) != 3 or len(spatial_channels) != 3:
            raise ValueError("Channel lists for spectral and spatial streams must have length 3.")

        self.input_bands = input_bands # AFTER band selection
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.spec_channels = spec_channels
        self.spatial_channels = spatial_channels
        # Normalize stages defensively: allow callers to pass strings like "1" from JSON.
        stages_in = intermediate_attention_stages
        if stages_in is None:
            stages_in = []
        if isinstance(stages_in, (int, float, str)):
            stages_in = [stages_in]
        stages_norm: List[int] = []
        try:
            for s in list(stages_in):
                try:
                    si = int(s)
                except Exception:
                    continue
                if si in (1, 2):
                    stages_norm.append(si)
        except Exception:
            stages_norm = []
        self.intermediate_stages = sorted(set(stages_norm))
        
        # Enforce AdaptiveWeight default if None or invalid is passed
        if fusion_mechanism != 'AdaptiveWeight':
             logger.warning(f"Fusion mechanism '{fusion_mechanism}' requested. 'AdaptiveWeight' is strongly recommended.")
        
        self.fusion_mechanism = fusion_mechanism

        # --- Check Channel Compatibility ---
        if 1 in self.intermediate_stages and spec_channels[0] != spatial_channels[0]:
            raise ValueError("Intermediate attention after Stage 1 requires spec_channels[0] == spatial_channels[0].")
        if 2 in self.intermediate_stages and spec_channels[1] != spatial_channels[1]:
            raise ValueError("Intermediate attention after Stage 2 requires spec_channels[1] == spatial_channels[1].")
        if spec_channels[2] != spatial_channels[2]:
             raise ValueError(f"Final stage channel dimensions must match: Spec={spec_channels[2]}, Spat={spatial_channels[2]}")
        self.final_fusion_dim = spatial_channels[2] 

        # --- Define Convolutional Layers ---
        self.spec_conv_in = nn.Conv1d(1, spec_channels[0], kernel_size=3, padding=1, bias=False)
        self.spatial_conv_in = nn.Conv2d(input_bands, spatial_channels[0], kernel_size=3, padding=1, bias=False)
        
        self.spec_conv1 = nn.Conv1d(spec_channels[0], spec_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.spec_conv2 = nn.Conv1d(spec_channels[1], spec_channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.spatial_conv1 = nn.Conv2d(spatial_channels[0], spatial_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.spatial_conv2 = nn.Conv2d(spatial_channels[1], spatial_channels[2], kernel_size=3, stride=2, padding=1, bias=False)

        # --- Calculate Sequence Lengths ---
        self.spec_len_s1 = _calculate_conv_output_size(self.input_bands, kernel_size=3, stride=1, padding=1)
        self.spat_h_s1 = _calculate_conv_output_size(self.patch_size, kernel_size=3, stride=1, padding=1)
        self.spat_w_s1 = self.spat_h_s1
        self.spat_seq_len_s1 = self.spat_h_s1 * self.spat_w_s1

        self.spec_len_s2 = _calculate_conv_output_size(self.spec_len_s1, kernel_size=3, stride=2, padding=1)
        self.spat_h_s2 = _calculate_conv_output_size(self.spat_h_s1, kernel_size=3, stride=2, padding=1)
        self.spat_w_s2 = self.spat_h_s2
        self.spat_seq_len_s2 = self.spat_h_s2 * self.spat_w_s2
        
        self.spec_len_s3 = _calculate_conv_output_size(self.spec_len_s2, kernel_size=3, stride=2, padding=1)
        self.spat_h_s3 = _calculate_conv_output_size(self.spat_h_s2, kernel_size=3, stride=2, padding=1)
        self.spat_w_s3 = self.spat_h_s3
        self.spat_seq_len_s3 = self.spat_h_s3 * self.spat_w_s3

        # --- Define Remaining Layers ---
        self.spec_bn_in = nn.BatchNorm1d(spec_channels[0])
        self.spec_relu_in = nn.ReLU(inplace=True)
        self.spec_stage1 = PyramidalResidualBlock(spec_channels[0], spec_channels[0], is_1d=True)
        
        self.spec_bn1 = nn.BatchNorm1d(spec_channels[1])
        self.spec_stage2 = PyramidalResidualBlock(spec_channels[1], spec_channels[1], is_1d=True)
        
        self.spec_bn2 = nn.BatchNorm1d(spec_channels[2])
        self.spec_stage3 = PyramidalResidualBlock(spec_channels[2], spec_channels[2], is_1d=True)

        self.spatial_bn_in = nn.BatchNorm2d(spatial_channels[0])
        self.spatial_relu_in = nn.ReLU(inplace=True)
        self.spatial_stage1 = PyramidalResidualBlock(spatial_channels[0], spatial_channels[0], is_1d=False)

        self.spatial_bn1 = nn.BatchNorm2d(spatial_channels[1])
        self.spatial_stage2 = PyramidalResidualBlock(spatial_channels[1], spatial_channels[1], is_1d=False)

        self.spatial_bn2 = nn.BatchNorm2d(spatial_channels[2])
        self.spatial_stage3 = PyramidalResidualBlock(spatial_channels[2], spatial_channels[2], is_1d=False)

        # --- Intermediate Cross-Attention Modules & Positional Embeddings ---
        self.intermediate_spec_enhancer_s1, self.intermediate_spat_enhancer_s1 = None, None
        self.spec_pos_embedding_s1, self.spat_pos_embedding_s1 = None, None
        self.intermediate_spec_enhancer_s2, self.intermediate_spat_enhancer_s2 = None, None
        self.spec_pos_embedding_s2, self.spat_pos_embedding_s2 = None, None

        if 1 in self.intermediate_stages:
            dim1 = spatial_channels[0]
            self.intermediate_spec_enhancer_s1 = MultiHeadCrossAttention(dim1, cross_attention_heads, cross_attention_dropout)
            self.intermediate_spat_enhancer_s1 = MultiHeadCrossAttention(dim1, cross_attention_heads, cross_attention_dropout)
            self.spec_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spec_len_s1, dim1) * 0.02)
            self.spat_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spat_seq_len_s1, dim1) * 0.02)
            logger.info(f"DSSFN Intermediate Attention ACTIVE after Stage 1.")

        if 2 in self.intermediate_stages:
            dim2 = spatial_channels[1]
            self.intermediate_spec_enhancer_s2 = MultiHeadCrossAttention(dim2, cross_attention_heads, cross_attention_dropout)
            self.intermediate_spat_enhancer_s2 = MultiHeadCrossAttention(dim2, cross_attention_heads, cross_attention_dropout)
            self.spec_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spec_len_s2, dim2) * 0.02)
            self.spat_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spat_seq_len_s2, dim2) * 0.02)
            logger.info(f"DSSFN Intermediate Attention ACTIVE after Stage 2.")
            
        if not self.intermediate_stages:
             logger.info("DSSFN Intermediate Attention DISABLED.")

        # --- Final Fusion and Classification Layers ---
        if self.fusion_mechanism == 'AdaptiveWeight':
            self.spec_global_pool = nn.AdaptiveAvgPool1d(1)
            self.spec_fc = nn.Linear(self.final_fusion_dim, num_classes)
            
            self.spatial_global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.spatial_fc = nn.Linear(self.final_fusion_dim, num_classes)

        elif self.fusion_mechanism == 'CrossAttention':
            self.final_spat_enhancer = MultiHeadCrossAttention(self.final_fusion_dim, cross_attention_heads, cross_attention_dropout)
            self.final_spec_enhancer = MultiHeadCrossAttention(self.final_fusion_dim, cross_attention_heads, cross_attention_dropout)
            
            self.spec_pos_embedding_s3 = nn.Parameter(torch.randn(1, self.spec_len_s3, self.final_fusion_dim) * 0.02)
            self.spat_pos_embedding_s3 = nn.Parameter(torch.randn(1, self.spat_seq_len_s3, self.final_fusion_dim) * 0.02)
            
            self.fusion_global_pool = nn.AdaptiveAvgPool1d(1)
            self.fusion_fc = nn.Linear(self.final_fusion_dim * 2, num_classes) 
        else:
            raise ValueError(f"Unsupported final fusion_mechanism: {self.fusion_mechanism}")

    def _apply_intermediate_attention(self, spc_in, spt_in, stage_num):
        """ Applies bidirectional cross-attention between spectral and spatial streams. """
        if stage_num == 1:
            spec_enhancer, spat_enhancer = self.intermediate_spec_enhancer_s1, self.intermediate_spat_enhancer_s1
            spec_pos_emb, spat_pos_emb = self.spec_pos_embedding_s1, self.spat_pos_embedding_s1
        elif stage_num == 2:
            spec_enhancer, spat_enhancer = self.intermediate_spec_enhancer_s2, self.intermediate_spat_enhancer_s2
            spec_pos_emb, spat_pos_emb = self.spec_pos_embedding_s2, self.spat_pos_embedding_s2
        else:
            raise ValueError(f"Invalid stage_num for intermediate attention: {stage_num}")

        B, C, H, W = spt_in.shape
        N_spt = H * W
        spt_reshaped = spt_in.view(B, C, N_spt).permute(0, 2, 1).contiguous()
        
        B_spc, C_spc, L_spc = spc_in.shape
        spc_reshaped = spc_in.permute(0, 2, 1).contiguous()

        L_slice = min(L_spc, spec_pos_emb.shape[1])
        spc_with_pos = spc_reshaped[:, :L_slice, :] + spec_pos_emb[:, :L_slice, :]
        
        N_slice = min(N_spt, spat_pos_emb.shape[1])
        spt_with_pos = spt_reshaped[:, :N_slice, :] + spat_pos_emb[:, :N_slice, :]

        spc_enhanced_reshaped = spec_enhancer(spc_with_pos, spt_with_pos)
        spc_enhanced = spc_enhanced_reshaped.permute(0, 2, 1)
        
        if L_slice < L_spc:
            padding_needed = L_spc - L_slice
            spc_enhanced = F.pad(spc_enhanced, (0, padding_needed))

        spt_enhanced_reshaped = spat_enhancer(spt_with_pos, spc_with_pos)
        spt_enhanced_permuted = spt_enhanced_reshaped.permute(0, 2, 1)

        if N_slice < N_spt:
            padding_needed = N_spt - N_slice
            spt_enhanced_permuted = F.pad(spt_enhanced_permuted, (0, padding_needed))
            
        spt_enhanced = spt_enhanced_permuted.view(B, C, H, W)

        return spc_enhanced, spt_enhanced

    def forward(self, x_spatial):
        spt = self.spatial_relu_in(self.spatial_bn_in(self.spatial_conv_in(x_spatial)))
        
        center_pixel_r, center_pixel_c = self.patch_size // 2, self.patch_size // 2
        x_spectral = x_spatial[:, :, center_pixel_r, center_pixel_c].unsqueeze(1)
        spc = self.spec_relu_in(self.spec_bn_in(self.spec_conv_in(x_spectral)))

        # --- Stage 1 ---
        spt_s1_block_out = self.spatial_stage1(spt)
        spc_s1_block_out = self.spec_stage1(spc)

        if 1 in self.intermediate_stages:
            spc_s1_out, spt_s1_out = self._apply_intermediate_attention(spc_s1_block_out, spt_s1_block_out, 1)
        else:
            spc_s1_out, spt_s1_out = spc_s1_block_out, spt_s1_block_out
        
        # --- Stage 2 ---
        spt_s2_in = self.spatial_relu_in(self.spatial_bn1(self.spatial_conv1(spt_s1_out)))
        spc_s2_in = self.spec_relu_in(self.spec_bn1(self.spec_conv1(spc_s1_out)))
        
        spt_s2_block_out = self.spatial_stage2(spt_s2_in)
        spc_s2_block_out = self.spec_stage2(spc_s2_in)

        if 2 in self.intermediate_stages:
             spc_s2_out, spt_s2_out = self._apply_intermediate_attention(spc_s2_block_out, spt_s2_block_out, 2)
        else:
             spc_s2_out, spt_s2_out = spc_s2_block_out, spt_s2_block_out

        # --- Stage 3 ---
        spt_s3_in = self.spatial_relu_in(self.spatial_bn2(self.spatial_conv2(spt_s2_out)))
        spc_s3_in = self.spec_relu_in(self.spec_bn2(self.spec_conv2(spc_s2_out)))

        spt_features = self.spatial_stage3(spt_s3_in)
        spc_features = self.spec_stage3(spc_s3_in)

        # --- Apply FINAL Fusion Mechanism ---
        if self.fusion_mechanism == 'AdaptiveWeight':
            spt_pooled = self.spatial_global_pool(spt_features).flatten(start_dim=1)
            spatial_logits = self.spatial_fc(spt_pooled)
            
            spc_pooled = self.spec_global_pool(spc_features).flatten(start_dim=1)
            spec_logits = self.spec_fc(spc_pooled)
            
            return spec_logits, spatial_logits

        elif self.fusion_mechanism == 'CrossAttention':
            B, C3_spt, H3, W3 = spt_features.shape
            N3_spt = H3 * W3
            spt_final_reshaped = spt_features.view(B, C3_spt, N3_spt).permute(0, 2, 1).contiguous()
            
            B_spc, C3_spc, L3_spc = spc_features.shape
            spc_final_reshaped = spc_features.permute(0, 2, 1).contiguous()

            L3_slice = min(L3_spc, self.spec_pos_embedding_s3.shape[1])
            spc_final_pos = spc_final_reshaped[:, :L3_slice, :] + self.spec_pos_embedding_s3[:, :L3_slice, :]
            
            N3_slice = min(N3_spt, self.spat_pos_embedding_s3.shape[1])
            spt_final_pos = spt_final_reshaped[:, :N3_slice, :] + self.spat_pos_embedding_s3[:, :N3_slice, :]
            
            if L3_slice < spc_final_reshaped.shape[1]:
                 spc_final_pos = F.pad(spc_final_pos, (0,0, 0, spc_final_reshaped.shape[1]-L3_slice ))
            if N3_slice < spt_final_reshaped.shape[1]:
                 spt_final_pos = F.pad(spt_final_pos, (0,0, 0, spt_final_reshaped.shape[1]-N3_slice ))

            fused_spec_q = self.final_spec_enhancer(spc_final_pos, spt_final_pos)
            fused_spat_q = self.final_spat_enhancer(spt_final_pos, spc_final_pos)

            pooled_spec_q = self.fusion_global_pool(fused_spec_q.permute(0, 2, 1)).flatten(start_dim=1)
            pooled_spat_q = self.fusion_global_pool(fused_spat_q.permute(0, 2, 1)).flatten(start_dim=1)

            final_fused_pooled = torch.cat((pooled_spec_q, pooled_spat_q), dim=1)
            fused_logits = self.fusion_fc(final_fused_pooled)
            
            return fused_logits
        else:
             raise ValueError(f"Unsupported final fusion_mechanism: {self.fusion_mechanism}")


class AdaptiveDSSFN(nn.Module):
    """
    Adaptive Depth DSSFN with Adaptive Computation Time (ACT).
    """
    
    def __init__(
        self,
        input_bands: int,
        num_classes: int,
        patch_size: int,
        spec_channels: List[int] = [64, 128, 256],
        spatial_channels: List[int] = [64, 128, 256],
        cross_attention_heads: int = 8,
        cross_attention_dropout: float = 0.1,
        act_epsilon: float = 0.01,
        halting_bias_init: float = -3.0,
        intermediate_attention_stages: List[int] = []
    ):
        super(AdaptiveDSSFN, self).__init__()
        
        # Validate inputs
        if len(spec_channels) != 3 or len(spatial_channels) != 3:
            raise ValueError("Channel lists must have length 3.")
        
        self.input_bands = input_bands
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.spec_channels = spec_channels
        self.spatial_channels = spatial_channels
        self.num_stages = 3
        
        # Normalize stages defensively (same rationale as DSSFN).
        stages_in = intermediate_attention_stages
        if stages_in is None:
            stages_in = []
        if isinstance(stages_in, (int, float, str)):
            stages_in = [stages_in]
        stages_norm: List[int] = []
        try:
            for s in list(stages_in):
                try:
                    si = int(s)
                except Exception:
                    continue
                if si in (1, 2):
                    stages_norm.append(si)
        except Exception:
            stages_norm = []
        self.intermediate_stages = sorted(set(stages_norm))
        
        # ACT controller
        self.act_controller = ACTController(num_stages=3, epsilon=act_epsilon)
        
        # ===== Initial Convolutions =====
        self.spec_conv_in = nn.Conv1d(1, spec_channels[0], kernel_size=3, padding=1, bias=False)
        self.spec_bn_in = nn.BatchNorm1d(spec_channels[0])
        self.spec_relu_in = nn.ReLU(inplace=True)
        
        self.spatial_conv_in = nn.Conv2d(input_bands, spatial_channels[0], kernel_size=3, padding=1, bias=False)
        self.spatial_bn_in = nn.BatchNorm2d(spatial_channels[0])
        self.spatial_relu_in = nn.ReLU(inplace=True)
        
        # ===== Stage 1 =====
        self.spec_stage1 = PyramidalResidualBlock(spec_channels[0], spec_channels[0], is_1d=True)
        self.spatial_stage1 = PyramidalResidualBlock(spatial_channels[0], spatial_channels[0], is_1d=False)
        
        self.halting_unit_s1 = HaltingModule(spec_channels[0], is_1d=True, init_bias=halting_bias_init)
        
        self.classifier_s1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spec_channels[0], num_classes)
        )
        
        # ===== Inter-stage convolutions 1->2 =====
        self.spec_conv1 = nn.Conv1d(spec_channels[0], spec_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.spec_bn1 = nn.BatchNorm1d(spec_channels[1])
        self.spatial_conv1 = nn.Conv2d(spatial_channels[0], spatial_channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.spatial_bn1 = nn.BatchNorm2d(spatial_channels[1])
        
        # ===== Stage 2 =====
        self.spec_stage2 = PyramidalResidualBlock(spec_channels[1], spec_channels[1], is_1d=True)
        self.spatial_stage2 = PyramidalResidualBlock(spatial_channels[1], spatial_channels[1], is_1d=False)
        
        self.halting_unit_s2 = HaltingModule(spec_channels[1], is_1d=True, init_bias=halting_bias_init)
        
        self.classifier_s2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spec_channels[1], num_classes)
        )
        
        # ===== Inter-stage convolutions 2->3 =====
        self.spec_conv2 = nn.Conv1d(spec_channels[1], spec_channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.spec_bn2 = nn.BatchNorm1d(spec_channels[2])
        self.spatial_conv2 = nn.Conv2d(spatial_channels[1], spatial_channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.spatial_bn2 = nn.BatchNorm2d(spatial_channels[2])
        
        # ===== Stage 3 =====
        self.spec_stage3 = PyramidalResidualBlock(spec_channels[2], spec_channels[2], is_1d=True)
        self.spatial_stage3 = PyramidalResidualBlock(spatial_channels[2], spatial_channels[2], is_1d=False)
        
        self.classifier_s3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(spec_channels[2], num_classes)
        )
        
        # ===== Intermediate Cross-Attention =====
        self._setup_intermediate_attention(cross_attention_heads, cross_attention_dropout)
        
        logger.info(f"AdaptiveDSSFN initialized: {num_classes} classes, ACT epsilon={act_epsilon}")
    
    def _setup_intermediate_attention(self, heads: int, dropout: float):
        self.spec_len_s1 = self.input_bands
        self.spat_seq_len_s1 = self.patch_size * self.patch_size
        
        self.spec_len_s2 = (self.spec_len_s1 + 1) // 2
        self.spat_h_s2 = (self.patch_size + 1) // 2
        self.spat_seq_len_s2 = self.spat_h_s2 * self.spat_h_s2
        
        self.intermediate_spec_enhancer_s1 = None
        self.intermediate_spat_enhancer_s1 = None
        self.spec_pos_embedding_s1 = None
        self.spat_pos_embedding_s1 = None
        
        if 1 in self.intermediate_stages:
            dim1 = self.spec_channels[0]
            self.intermediate_spec_enhancer_s1 = MultiHeadCrossAttention(dim1, heads, dropout)
            self.intermediate_spat_enhancer_s1 = MultiHeadCrossAttention(dim1, heads, dropout)
            self.spec_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spec_len_s1, dim1) * 0.02)
            self.spat_pos_embedding_s1 = nn.Parameter(torch.randn(1, self.spat_seq_len_s1, dim1) * 0.02)
        
        self.intermediate_spec_enhancer_s2 = None
        self.intermediate_spat_enhancer_s2 = None
        self.spec_pos_embedding_s2 = None
        self.spat_pos_embedding_s2 = None
        
        if 2 in self.intermediate_stages:
            if self.spec_channels[1] != self.spatial_channels[1]:
                logger.warning(f"  Skipping Stage 2 intermediate attention: dimensions don't match")
            else:
                dim2 = self.spec_channels[1]
                self.intermediate_spec_enhancer_s2 = MultiHeadCrossAttention(dim2, heads, dropout)
                self.intermediate_spat_enhancer_s2 = MultiHeadCrossAttention(dim2, heads, dropout)
                self.spec_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spec_len_s2, dim2) * 0.02)
                self.spat_pos_embedding_s2 = nn.Parameter(torch.randn(1, self.spat_seq_len_s2, dim2) * 0.02)
    
    def _apply_intermediate_attention(self, spc: torch.Tensor, spt: torch.Tensor, stage: int):
        if stage == 1:
            if self.intermediate_spec_enhancer_s1 is None: return spc, spt
            spec_enhancer = self.intermediate_spec_enhancer_s1
            spat_enhancer = self.intermediate_spat_enhancer_s1
            spec_pos = self.spec_pos_embedding_s1
            spat_pos = self.spat_pos_embedding_s1
        elif stage == 2:
            if self.intermediate_spec_enhancer_s2 is None: return spc, spt
            spec_enhancer = self.intermediate_spec_enhancer_s2
            spat_enhancer = self.intermediate_spat_enhancer_s2
            spec_pos = self.spec_pos_embedding_s2
            spat_pos = self.spat_pos_embedding_s2
        else:
            return spc, spt
        
        B, C, L = spc.shape
        spc_reshaped = spc.permute(0, 2, 1)
        
        B, C, H, W = spt.shape
        spt_reshaped = spt.view(B, C, H * W).permute(0, 2, 1)
        
        L_slice = min(L, spec_pos.shape[1])
        N_slice = min(H * W, spat_pos.shape[1])
        
        spc_with_pos = spc_reshaped[:, :L_slice, :] + spec_pos[:, :L_slice, :]
        spt_with_pos = spt_reshaped[:, :N_slice, :] + spat_pos[:, :N_slice, :]
        
        spc_enhanced = spec_enhancer(spc_with_pos, spt_with_pos)
        spt_enhanced = spat_enhancer(spt_with_pos, spc_with_pos)
        
        spc_out = spc_enhanced.permute(0, 2, 1)
        if L_slice < L:
            spc_out = F.pad(spc_out, (0, L - L_slice))
        
        spt_out = spt_enhanced.permute(0, 2, 1)
        if N_slice < H * W:
            spt_out = F.pad(spt_out, (0, H * W - N_slice))
        spt_out = spt_out.view(B, C, H, W)
        
        return spc_out, spt_out
    
    def _fuse_features_for_classification(self, spc: torch.Tensor, spt: torch.Tensor, stage: int = 0) -> torch.Tensor:
        spt_pooled = F.adaptive_avg_pool2d(spt, 1).flatten(2)
        if spc.shape[1] == spt_pooled.shape[1]:
            combined = (spc + spt_pooled) / 2.0
        else:
            combined = spc
        return combined

    def forward_stage_logits(self, x_spatial: torch.Tensor, stage: int) -> torch.Tensor:
        """
        Returns the logits produced at a specific stage without ACT weighting.
        This is used for deterministic per-stage FLOPs measurement.
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2, or 3 (got {stage})")

        spt = self.spatial_relu_in(self.spatial_bn_in(self.spatial_conv_in(x_spatial)))
        center = self.patch_size // 2
        x_spectral = x_spatial[:, :, center, center].unsqueeze(1)
        spc = self.spec_relu_in(self.spec_bn_in(self.spec_conv_in(x_spectral)))

        # Stage 1
        spc_s1 = self.spec_stage1(spc)
        spt_s1 = self.spatial_stage1(spt)
        if 1 in self.intermediate_stages:
            spc_s1, spt_s1 = self._apply_intermediate_attention(spc_s1, spt_s1, 1)
        logits_s1 = self.classifier_s1(self._fuse_features_for_classification(spc_s1, spt_s1))
        if stage == 1:
            return logits_s1

        # Stage 2
        spc_s2_in = F.relu(self.spec_bn1(self.spec_conv1(spc_s1)))
        spt_s2_in = F.relu(self.spatial_bn1(self.spatial_conv1(spt_s1)))
        spc_s2 = self.spec_stage2(spc_s2_in)
        spt_s2 = self.spatial_stage2(spt_s2_in)
        if 2 in self.intermediate_stages:
            spc_s2, spt_s2 = self._apply_intermediate_attention(spc_s2, spt_s2, 2)
        logits_s2 = self.classifier_s2(self._fuse_features_for_classification(spc_s2, spt_s2))
        if stage == 2:
            return logits_s2

        # Stage 3
        spc_s3_in = F.relu(self.spec_bn2(self.spec_conv2(spc_s2)))
        spt_s3_in = F.relu(self.spatial_bn2(self.spatial_conv2(spt_s2)))
        spc_s3 = self.spec_stage3(spc_s3_in)
        spt_s3 = self.spatial_stage3(spt_s3_in)
        logits_s3 = self.classifier_s3(self._fuse_features_for_classification(spc_s3, spt_s3))
        return logits_s3

    def _forward_act_soft(self, x_spatial: torch.Tensor, return_ponder_cost: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Differentiable ACT path (computes all stages, used for training).
        """
        B = x_spatial.shape[0]
        device = x_spatial.device

        spt = self.spatial_relu_in(self.spatial_bn_in(self.spatial_conv_in(x_spatial)))
        center = self.patch_size // 2
        x_spectral = x_spatial[:, :, center, center].unsqueeze(1)
        spc = self.spec_relu_in(self.spec_bn_in(self.spec_conv_in(x_spectral)))

        stage_logits = []
        halting_probs = []

        # Stage 1
        spc_s1 = self.spec_stage1(spc)
        spt_s1 = self.spatial_stage1(spt)
        if 1 in self.intermediate_stages:
            spc_s1, spt_s1 = self._apply_intermediate_attention(spc_s1, spt_s1, 1)
        logits_s1 = self.classifier_s1(self._fuse_features_for_classification(spc_s1, spt_s1))
        halt_prob_s1 = self.halting_unit_s1(spc_s1)
        stage_logits.append(logits_s1)
        halting_probs.append(halt_prob_s1)

        # Stage 2
        spc_s2_in = F.relu(self.spec_bn1(self.spec_conv1(spc_s1)))
        spt_s2_in = F.relu(self.spatial_bn1(self.spatial_conv1(spt_s1)))
        spc_s2 = self.spec_stage2(spc_s2_in)
        spt_s2 = self.spatial_stage2(spt_s2_in)
        if 2 in self.intermediate_stages:
            spc_s2, spt_s2 = self._apply_intermediate_attention(spc_s2, spt_s2, 2)
        logits_s2 = self.classifier_s2(self._fuse_features_for_classification(spc_s2, spt_s2))
        halt_prob_s2 = self.halting_unit_s2(spc_s2)
        stage_logits.append(logits_s2)
        halting_probs.append(halt_prob_s2)

        # Stage 3
        spc_s3_in = F.relu(self.spec_bn2(self.spec_conv2(spc_s2)))
        spt_s3_in = F.relu(self.spatial_bn2(self.spatial_conv2(spt_s2)))
        spc_s3 = self.spec_stage3(spc_s3_in)
        spt_s3 = self.spatial_stage3(spt_s3_in)
        logits_s3 = self.classifier_s3(self._fuse_features_for_classification(spc_s3, spt_s3))
        halt_prob_s3 = torch.ones(B, 1, device=device)
        stage_logits.append(logits_s3)
        halting_probs.append(halt_prob_s3)

        weighted_logits, ponder_cost, halting_step, _ = self.act_controller.compute_act_output(stage_logits, halting_probs)
        if return_ponder_cost:
            return weighted_logits, ponder_cost, halting_step
        return weighted_logits, None, halting_step

    def _forward_act_hard(self, x_spatial: torch.Tensor, return_ponder_cost: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Hard ACT path (skips deeper stages for halted samples, used for inference efficiency).
        """
        B = x_spatial.shape[0]
        device = x_spatial.device
        threshold = self.act_controller.threshold

        spt = self.spatial_relu_in(self.spatial_bn_in(self.spatial_conv_in(x_spatial)))
        center = self.patch_size // 2
        x_spectral = x_spatial[:, :, center, center].unsqueeze(1)
        spc = self.spec_relu_in(self.spec_bn_in(self.spec_conv_in(x_spectral)))

        weighted_logits = torch.zeros(B, self.num_classes, device=device)
        cumulative_prob = torch.zeros(B, 1, device=device)
        halting_step = torch.zeros(B, dtype=torch.long, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)

        # Stage 1 (all samples)
        spc_s1 = self.spec_stage1(spc)
        spt_s1 = self.spatial_stage1(spt)
        if 1 in self.intermediate_stages:
            spc_s1, spt_s1 = self._apply_intermediate_attention(spc_s1, spt_s1, 1)
        logits_s1 = self.classifier_s1(self._fuse_features_for_classification(spc_s1, spt_s1))
        p1 = self.halting_unit_s1(spc_s1)  # (B, 1)

        should_halt_1 = active & (p1.squeeze(-1) >= threshold)
        continuing_1 = active & ~should_halt_1

        w1 = torch.zeros(B, 1, device=device)
        w1[should_halt_1] = 1.0
        w1[continuing_1] = p1[continuing_1]
        weighted_logits = weighted_logits + logits_s1 * w1
        halting_step[should_halt_1] = 1

        cumulative_prob[continuing_1] = p1[continuing_1]
        active = continuing_1

        # Stage 2 (active subset)
        if active.any():
            idx2 = active.nonzero(as_tuple=False).squeeze(-1)
            spc_s1_a = spc_s1[idx2]
            spt_s1_a = spt_s1[idx2]
            cum_a = cumulative_prob[idx2]

            spc_s2_in = F.relu(self.spec_bn1(self.spec_conv1(spc_s1_a)))
            spt_s2_in = F.relu(self.spatial_bn1(self.spatial_conv1(spt_s1_a)))
            spc_s2 = self.spec_stage2(spc_s2_in)
            spt_s2 = self.spatial_stage2(spt_s2_in)
            if 2 in self.intermediate_stages:
                spc_s2, spt_s2 = self._apply_intermediate_attention(spc_s2, spt_s2, 2)

            logits_s2 = self.classifier_s2(self._fuse_features_for_classification(spc_s2, spt_s2))
            p2 = self.halting_unit_s2(spc_s2)  # (B_active, 1)

            should_halt_2 = (cum_a.squeeze(-1) + p2.squeeze(-1)) >= threshold
            continuing_2 = ~should_halt_2

            w2 = torch.zeros(idx2.numel(), 1, device=device)
            w2[should_halt_2] = (1.0 - cum_a[should_halt_2]).clamp(min=0.0)
            w2[continuing_2] = p2[continuing_2]
            weighted_logits[idx2] = weighted_logits[idx2] + logits_s2 * w2

            halting_step[idx2[should_halt_2]] = 2
            cumulative_prob[idx2[continuing_2]] = cum_a[continuing_2] + p2[continuing_2]
            active[idx2[should_halt_2]] = False

            # Stage 3 (remaining active subset)
            if continuing_2.any():
                idx3 = idx2[continuing_2]
                spc_s2_a = spc_s2[continuing_2]
                spt_s2_a = spt_s2[continuing_2]
                cum3 = cumulative_prob[idx3]

                spc_s3_in = F.relu(self.spec_bn2(self.spec_conv2(spc_s2_a)))
                spt_s3_in = F.relu(self.spatial_bn2(self.spatial_conv2(spt_s2_a)))
                spc_s3 = self.spec_stage3(spc_s3_in)
                spt_s3 = self.spatial_stage3(spt_s3_in)
                logits_s3 = self.classifier_s3(self._fuse_features_for_classification(spc_s3, spt_s3))

                w3 = (1.0 - cum3).clamp(min=0.0)
                weighted_logits[idx3] = weighted_logits[idx3] + logits_s3 * w3
                halting_step[idx3] = 3

        # Any remaining unset samples (numerical edge cases) default to stage 3.
        halting_step = torch.where(halting_step == 0, torch.full_like(halting_step, 3), halting_step)

        ponder_cost = halting_step.float() if return_ponder_cost else None
        return weighted_logits, ponder_cost, halting_step

    def forward(self, x_spatial: torch.Tensor, return_ponder_cost: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.training:
            return self._forward_act_soft(x_spatial, return_ponder_cost=return_ponder_cost)
        return self._forward_act_hard(x_spatial, return_ponder_cost=return_ponder_cost)

    def forward_fixed_depth(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """ Forward pass using fixed full depth (no early exit). """
        B = x_spatial.shape[0]
        
        spt = self.spatial_relu_in(self.spatial_bn_in(self.spatial_conv_in(x_spatial)))
        center = self.patch_size // 2
        x_spectral = x_spatial[:, :, center, center].unsqueeze(1)
        spc = self.spec_relu_in(self.spec_bn_in(self.spec_conv_in(x_spectral)))
        
        # Stage 1
        spc = self.spec_stage1(spc)
        spt = self.spatial_stage1(spt)
        if 1 in self.intermediate_stages:
            spc, spt = self._apply_intermediate_attention(spc, spt, 1)
        
        # Stage 2
        spc = F.relu(self.spec_bn1(self.spec_conv1(spc)))
        spt = F.relu(self.spatial_bn1(self.spatial_conv1(spt)))
        spc = self.spec_stage2(spc)
        spt = self.spatial_stage2(spt)
        if 2 in self.intermediate_stages:
            spc, spt = self._apply_intermediate_attention(spc, spt, 2)
        
        # Stage 3
        spc = F.relu(self.spec_bn2(self.spec_conv2(spc)))
        spt = F.relu(self.spatial_bn2(self.spatial_conv2(spt)))
        spc = self.spec_stage3(spc)
        spt = self.spatial_stage3(spt)
        
        fused = self._fuse_features_for_classification(spc, spt)
        logits = self.classifier_s3(fused)
        return logits
