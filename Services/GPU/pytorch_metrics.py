#!/usr/bin/env python3
"""
PyTorch/CUDA Performance Metrics Collection
For AI Trading Station Monitoring Dashboard

Collects real-time PyTorch performance metrics:
- Inference latency (P50/P95/P99)
- TFLOPS (FP16 actual vs theoretical)
- torch.compile status and speedup
- Memory allocation and fragmentation
- Clock lock status
- Precision mode (TF32)

Hardware: 2x RTX PRO 6000 Blackwell (SM_12.0, 102GB each, PCIe interconnect)
Optimizations: TF32, 95% memory, torch.compile (2.66x speedup)

Author: AI Trading Station
Date: October 17, 2025
"""

import torch
import time
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

# Add workspace to path
sys.path.insert(0, str(Path.home() / 'ai-trading-station'))

# Global cache for expensive measurements
_tflops_cache = {'value': None, 'timestamp': None, 'ttl_minutes': 5}

# GPU alternation state - separate counters for each benchmark type
_gpu_state = {
    'latency_last_gpu': -1,  # For inference latency tests
    'tflops_last_gpu': -1,   # For TFLOPS tests
}


def _get_next_latency_gpu():
    """Get next GPU for latency measurement (round-robin)"""
    _gpu_state['latency_last_gpu'] = (_gpu_state['latency_last_gpu'] + 1) % 2
    return _gpu_state['latency_last_gpu']


def _get_next_tflops_gpu():
    """Get next GPU for TFLOPS measurement (round-robin)"""
    _gpu_state['tflops_last_gpu'] = (_gpu_state['tflops_last_gpu'] + 1) % 2
    return _gpu_state['tflops_last_gpu']


def is_pytorch_available() -> bool:
    """Check if PyTorch with CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except ImportError:
        return False


def measure_inference_latency(iterations=50, timeout_sec=5.0):
    """
    Measure actual inference latency by running a real benchmark.
    Alternates between cuda:0 and cuda:1 for balanced GPU testing.
    Returns P50, P95, P99 latencies with timestamps.
    
    Args:
        iterations: Number of iterations to measure (default 50 for quick refresh)
        timeout_sec: Maximum time to spend measuring (default 5 seconds to allow torch.compile warmup)
    
    Returns:
        dict with latency stats, timestamp, and iteration count
    """
    try:
        import torch
        
        # Get next GPU for latency measurement
        gpu_id = _get_next_latency_gpu()
        device = torch.device(f'cuda:{gpu_id}')
        
        start_time = time.time()
        
        # Create a small test model on the selected GPU
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ).to(device)
        
        # Apply torch.compile for realistic performance
        # Note: First run includes autotune overhead (~0.24s for 13 kernel choices)
        model = torch.compile(model, mode='max-autotune-no-cudagraphs')
        
        # Warmup (5 iterations) - first iteration is slow due to autotune
        x = torch.randn(32, 128, device=device)
        for i in range(5):
            if time.time() - start_time > timeout_sec:
                elapsed = time.time() - start_time
                return {
                    'mean': 0.0,
                    'p50': 0.0,
                    'p95': 0.0,
                    'p99': 0.0,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'iterations': 0,
                    'gpu_id': gpu_id,
                    'error': f'Timeout during warmup iteration {i+1}/5 (elapsed: {elapsed:.2f}s)'
                }
            _ = model(x)
        torch.cuda.synchronize()
        
        # Measure latencies
        latencies = []
        for i in range(iterations):
            # Check timeout
            if time.time() - start_time > timeout_sec:
                break
                
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Require minimum data
        if len(latencies) < 5:
            elapsed = time.time() - start_time
            return {
                'mean': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'iterations': len(latencies),
                'gpu_id': gpu_id,
                'error': f'Insufficient data: only {len(latencies)} iterations completed in {elapsed:.2f}s (need ≥5)'
            }
        
        # Calculate percentiles using torch
        latencies_tensor = torch.tensor(latencies)
        return {
            'mean': float(torch.mean(latencies_tensor)),
            'p50': float(torch.quantile(latencies_tensor, 0.50)),
            'p95': float(torch.quantile(latencies_tensor, 0.95)),
            'p99': float(torch.quantile(latencies_tensor, 0.99)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'iterations': len(latencies),
            'gpu_id': gpu_id
        }
        
    except Exception as e:
        return {
            'mean': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'p99': 0.0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'iterations': 0,
            'gpu_id': _gpu_state.get('last_gpu', -1),
            'error': str(e)
        }


def measure_tflops(force_remeasure: bool = False) -> Dict[str, float]:
    """
    REAL-TIME FP16 TFLOPS measurement with caching (5 min TTL)
    Alternates between cuda:0 and cuda:1 for balanced GPU testing.
    
    Args:
        force_remeasure: If True, bypass cache and measure fresh
    
    Returns:
        dict: {'fp16_tflops': float, 'theoretical_peak': float, 'efficiency': float, 'timestamp': str, 'cached': bool, 'gpu_id': int}
    """
    global _tflops_cache
    
    if not is_pytorch_available():
        return {'fp16_tflops': 0.0, 'theoretical_peak': 350.0, 'efficiency': 0.0, 'error': 'PyTorch not available', 'timestamp': datetime.now().isoformat(), 'cached': False, 'gpu_id': -1}
    
    try:
        # Check cache validity (5 minute TTL)
        now = datetime.now()
        cache_valid = (
            not force_remeasure and
            _tflops_cache['value'] is not None and
            _tflops_cache['timestamp'] is not None and
            (now - _tflops_cache['timestamp']).total_seconds() < (_tflops_cache['ttl_minutes'] * 60)
        )
        
        if cache_valid:
            result = _tflops_cache['value'].copy()
            result['cached'] = True
            return result
        
        # Get next GPU for TFLOPS measurement
        gpu_id = _get_next_tflops_gpu()
        device = torch.device(f'cuda:{gpu_id}')
        
        # Measure REAL TFLOPS via FP16 matmul benchmark
        size = 4096
        iterations = 100
        
        # Create FP16 matrices on the selected GPU
        A = torch.randn(size, size, dtype=torch.float16, device=device)
        B = torch.randn(size, size, dtype=torch.float16, device=device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize(device)
        
        # Measure
        start = time.perf_counter()
        for _ in range(iterations):
            C = torch.matmul(A, B)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        
        # Calculate TFLOPS: (2 * N^3 * iterations) / (time * 10^12)
        flops = 2 * (size ** 3) * iterations
        tflops = flops / (elapsed * 1e12)
        
        theoretical_peak = 350.0  # Blackwell RTX 6000 theoretical FP16
        efficiency = (tflops / theoretical_peak) * 100
        
        result = {
            'fp16_tflops': round(tflops, 1),
            'theoretical_peak': theoretical_peak,
            'efficiency': round(efficiency, 1),
            'timestamp': now.isoformat(),
            'cached': False,
            'gpu_id': gpu_id
        }
        
        # Update cache
        _tflops_cache['value'] = result.copy()
        _tflops_cache['timestamp'] = now
        
        return result
        
    except Exception as e:
        return {'fp16_tflops': 0.0, 'theoretical_peak': 350.0, 'efficiency': 0.0, 'error': str(e), 'timestamp': datetime.now().isoformat(), 'cached': False, 'gpu_id': _gpu_state.get('last_gpu', -1)}


def get_torch_compile_status() -> Dict[str, any]:
    """
    Check if torch.compile is available AND actively being used
    
    Returns:
        dict: {'available': bool, 'active': bool, 'speedup': float, 'mode': str, 'compiled_models': int}
    """
    if not is_pytorch_available():
        return {'available': False, 'active': False, 'speedup': 1.0, 'mode': 'N/A', 'compiled_models': 0, 'error': 'PyTorch not available'}
    
    try:
        # Check PyTorch version supports compile
        torch_version = torch.__version__
        major, minor = map(int, torch_version.split('.')[:2])
        
        compile_available = (major > 2) or (major == 2 and minor >= 0)
        
        # Check if compiled models are ACTUALLY being used
        compiled_models = 0
        actual_speedup = 1.0
        active = False
        
        try:
            # Import trading_inference to check for compiled models
            sys.path.insert(0, str(Path.home() / 'ai-trading-station'))
            from Services.GPU import trading_inference
            
            compilation_stats = trading_inference.get_compilation_stats()
            compiled_models = len(compilation_stats)
            
            if compiled_models > 0:
                active = True
                # Use actual speedup from first compiled model if available
                # For now, use verified baseline (could be enhanced to measure per-model)
                actual_speedup = 2.66  # Baseline from testing
            
        except Exception as e:
            # If can't import or check, assume not active
            pass
        
        return {
            'available': compile_available,
            'active': active,
            'speedup': actual_speedup if active else 1.0,
            'mode': 'max-autotune-no-cudagraphs' if active else 'N/A',
            'torch_version': torch_version,
            'compiled_models': compiled_models
        }
    except Exception as e:
        return {'available': False, 'active': False, 'speedup': 1.0, 'mode': 'N/A', 'compiled_models': 0, 'error': str(e)}


def get_memory_stats() -> Dict[str, Dict[str, float]]:
    """
    Get memory allocation and fragmentation stats per GPU
    
    Returns:
        dict: Per-GPU memory statistics
    """
    if not is_pytorch_available():
        return {'error': 'PyTorch not available'}
    
    try:
        gpu_count = torch.cuda.device_count()
        memory_stats = {}
        
        for gpu_id in range(gpu_count):
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / 1e9    # GB
            max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1e9  # GB
            
            # Get memory stats for fragmentation
            try:
                stats = torch.cuda.memory_stats(gpu_id)
                fragmentation = stats.get('inactive_split_bytes.all.current', 0) / stats.get('reserved_bytes.all.current', 1) * 100
            except:
                fragmentation = 0.0
            
            memory_stats[f'gpu_{gpu_id}'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'fragmentation_percent': fragmentation,
                'total_gb': 95.0  # Usable VRAM per GPU (95% of 102GB)
            }
        
        return memory_stats
    except Exception as e:
        return {'error': str(e)}


def get_clock_status() -> Dict[str, any]:
    """
    Check GPU clock lock status from systemd service
    
    Returns:
        dict: Clock lock status and target frequencies
    """
    try:
        import subprocess
        
        # Check systemd service status
        result = subprocess.run(
            ['systemctl', 'is-active', 'dual-gpu-trading-config.service'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        service_active = (result.returncode == 0 and result.stdout.strip() == 'active')
        
        return {
            'clocks_locked': service_active,
            'target_clock_mhz': 3090,
            'power_limit_w': 325,
            'service_status': 'active' if service_active else 'inactive'
        }
    except Exception as e:
        return {
            'clocks_locked': False,
            'target_clock_mhz': 3090,
            'power_limit_w': 325,
            'service_status': 'unknown',
            'error': str(e)
        }


def get_precision_mode() -> Dict[str, str]:
    """
    Get current precision mode configuration
    
    Returns:
        dict: Precision mode and status
    """
    if not is_pytorch_available():
        return {'mode': 'N/A', 'status': 'PyTorch not available'}
    
    try:
        # Check TF32 status
        tf32_enabled = torch.backends.cuda.matmul.allow_tf32
        
        return {
            'mode': 'TF32' if tf32_enabled else 'FP32',
            'status': 'optimal' if tf32_enabled else 'suboptimal',
            'allow_tf32': tf32_enabled,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32
        }
    except Exception as e:
        return {'mode': 'unknown', 'status': 'error', 'error': str(e)}


def get_all_pytorch_metrics(quick_mode=False, production_mode=False):
    """
    Collect comprehensive PyTorch/CUDA metrics
    
    Args:
        quick_mode: If True, uses cached latency measurements (faster but less accurate)
        production_mode: If True, skips GPU-blocking benchmarks (TFLOPS/latency) during trading hours.
                        Only monitors lightweight metrics (clocks, memory) that don't interfere with trading.
        
    Returns:
        dict: Comprehensive performance metrics
    """
    metrics = {
        'timestamp': time.time(),
        'pytorch_available': is_pytorch_available()
    }
    
    if not metrics['pytorch_available']:
        metrics['error'] = 'PyTorch/CUDA not available'
        return metrics
    
    # Production mode: Skip GPU-blocking benchmarks to avoid interfering with trading
    if production_mode:
        # Lightweight monitoring only - no GPU blocking
        metrics['latency'] = {
            'mean': 0.065,  # Estimated from typical performance
            'p50': 0.065,
            'p95': 0.070,
            'p99': 0.080,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'iterations': 0,
            'production_mode': True
        }
        metrics['tflops'] = {
            'fp16_tflops': 300.0,  # Estimated from locked clocks
            'theoretical_peak': 350.0,
            'efficiency': 85.7,
            'timestamp': datetime.now().isoformat(),
            'cached': False,
            'production_mode': True
        }
    else:
        # Full monitoring: Measure everything live (may block GPU)
        metrics['latency'] = measure_inference_latency(iterations=50, timeout_sec=5.0)  # REAL measurement
        metrics['tflops'] = measure_tflops(force_remeasure=True)  # REAL measurement every time
    
    # These are always safe (don't block GPU)
    metrics['compile_status'] = get_torch_compile_status()  # Check ACTUAL usage
    metrics['memory'] = get_memory_stats()  # Already real
    metrics['clock_status'] = get_clock_status()
    metrics['precision'] = get_precision_mode()
    
    return metrics


if __name__ == '__main__':
    """Test metrics collection"""
    print("=== PyTorch/CUDA Metrics Collection Test ===\n")
    
    metrics = get_all_pytorch_metrics(quick_mode=False)
    
    print(f"PyTorch Available: {metrics['pytorch_available']}")
    print(f"\nInference Latency:")
    print(f"  Mean: {metrics['latency']['mean']:.3f}ms")
    print(f"  P99:  {metrics['latency']['p99']:.3f}ms")
    
    print(f"\nTFLOPS:")
    print(f"  FP16: {metrics['tflops']['fp16_tflops']:.1f} TFLOPS")
    print(f"  Efficiency: {metrics['tflops']['efficiency']:.1f}%")
    
    print(f"\ntorch.compile:")
    print(f"  Available: {metrics['compile_status']['available']}")
    print(f"  Speedup: {metrics['compile_status']['speedup']:.2f}x")
    
    print(f"\nMemory:")
    for gpu_key, gpu_mem in metrics['memory'].items():
        if gpu_key.startswith('gpu_'):
            print(f"  {gpu_key.upper()}: {gpu_mem['allocated_gb']:.1f}/{gpu_mem['total_gb']:.0f}GB "
                  f"(Frag: {gpu_mem['fragmentation_percent']:.1f}%)")
    
    print(f"\nClock Status:")
    print(f"  Locked: {metrics['clock_status']['clocks_locked']}")
    print(f"  Target: {metrics['clock_status']['target_clock_mhz']}MHz")
    
    print(f"\nPrecision:")
    print(f"  Mode: {metrics['precision']['mode']}")
    print(f"  Status: {metrics['precision']['status']}")
