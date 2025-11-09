"""
Blackwell RTX PRO 6000 GPU Configuration
Auto-configures PyTorch optimizations on import

Hardware: 2x RTX PRO 6000 Workstation Edition
- Compute Capability: SM_12.0
- VRAM: 102GB per GPU (204GB total)
- Interconnect: PCIe (no NVLink)
- Power: 325W per GPU

Usage:
    from Services.GPU import get_gpu_config, trading_inference
    
    # Configuration applied automatically on import
    config = get_gpu_config()
    
    # Use production wrapper for inference
    compiled_model = trading_inference.compile_for_trading(model)
"""
import os
import sys
import warnings
from typing import Optional
from pathlib import Path

# CRITICAL: Set TF32 using BOTH APIs to work around PyTorch 2.9 bug
# torch.compile internals still read the old API (pad_mm.py:283)
# We must set the old API FIRST, then new API
import torch
torch.backends.cuda.matmul.allow_tf32 = True  # Old API (torch.compile still reads this)
torch.backends.cudnn.allow_tf32 = True         # Old API for cuDNN
# Then set new API
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration module
from blackwell_sm12_optimized_config import configure_sm12_optimizations

# Global configuration instance
_config: Optional[dict] = None

def get_gpu_config() -> dict:
    """
    Get or create GPU configuration
    
    Returns:
        dict: Current GPU configuration settings
    """
    global _config
    if _config is None:
        _config = setup_trading_gpu()
    return _config

def setup_trading_gpu(
    memory_fraction: float = 0.95,
    deterministic: bool = True,
    verbose: bool = False
) -> dict:
    """
    Initialize Blackwell GPUs for trading with validated settings
    
    Based on precision tests (2025-10-17):
    - TF32: Optimal (0.053ms, 1.61% variance, 0.000018 accuracy loss)
    - BF16: Slower on small models (0.104ms due to dimension misalignment)
    - FP8: Not implemented in PyTorch 2.9
    
    Args:
        memory_fraction: Fraction of GPU memory to use (default: 0.95 = 96.9GB per GPU)
        deterministic: Enable deterministic algorithms (default: True for trading)
        verbose: Print configuration details (default: False)
    
    Returns:
        dict: Applied configuration
    """
    global _config
    
    # Check if already configured
    if _config is not None and not verbose:
        return _config
    
    # Apply SM_12.0 optimized configuration with validated settings
    # Note: TF32 is enabled by default in the config module
    config = configure_sm12_optimizations(
        memory_fraction=memory_fraction,
        enable_fp8=False,           # Not functional in PyTorch 2.9
        enable_bf16=False,          # Slower on small models (tested)
        enable_p2p=False,           # No NVLink hardware
        deterministic=deterministic,
        seed=42,
        verbose=verbose
    )
    
    _config = {
        'memory_fraction': memory_fraction,
        'enable_tf32': True,  # Always enabled (optimal)
        'enable_bf16': False,
        'enable_fp8': False,
        'deterministic': deterministic,
        'gpu_count': config.get('gpu_count', 0),
        'memory_allocated_gb': config.get('memory_allocated_gb', []),
        'compute_capability': config.get('compute_capability', 'unknown'),
        'interconnect': 'PCIe (no NVLink)',
        'status': 'configured'
    }
    
    if verbose:
        import torch
        print(f"✅ Blackwell GPU Configuration Applied")
        print(f"   GPUs: {_config['gpu_count']}")
        print(f"   Memory: {_config['memory_allocated_gb']} GB")
        print(f"   Compute: {_config['compute_capability']}")
        print(f"   TF32: {_config['enable_tf32']} (optimal)")
        print(f"   Deterministic: {_config['deterministic']}")
        print(f"   Interconnect: {_config['interconnect']}")
        print(f"   PyTorch: {torch.__version__}")
    
    return _config

# Auto-configure on import (unless disabled)
if not os.environ.get('PYTORCH_NO_AUTO_CONFIG'):
    try:
        setup_trading_gpu(verbose=False)
    except Exception as e:
        warnings.warn(f"Failed to auto-configure GPUs: {e}")

# Export public API
__all__ = [
    'setup_trading_gpu',
    'get_gpu_config'
]
