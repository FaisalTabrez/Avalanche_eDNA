"""
Dummy Flash Attention Module for CPU-only systems
This file sets flash_attn_qkvpacked_func to None to trigger standard attention fallback.
"""

import warnings

warnings.warn(
    "Triton flash attention not available on this system. "
    "Using standard PyTorch attention instead. "
    "For faster inference, use a system with NVIDIA GPU and install triton.",
    UserWarning
)

# Set to None to trigger standard attention fallback in bert_layers.py
flash_attn_qkvpacked_func = None

__all__ = ['flash_attn_qkvpacked_func']
