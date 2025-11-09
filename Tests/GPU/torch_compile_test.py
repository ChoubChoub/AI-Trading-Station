#!/usr/bin/env python3
"""
torch.compile Performance Test
Tests the REAL 2x optimization for trading inference

Expected Results:
- Eager mode: ~0.5ms baseline
- Compiled (inductor): ~0.25ms (2x speedup target)
- Compiled (cudagraphs): ~0.20ms (2.5x speedup if applicable)

Hardware: 2x RTX PRO 6000 Blackwell (SM_12.0)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

import torch
import torch.nn as nn
import numpy as np

# Auto-configure GPU (uses __init__.py)
sys.path.insert(0, str(Path.home() / 'ai-trading-station'))
from Services.GPU import get_gpu_config

# Suppress compile warnings
warnings.filterwarnings('ignore', category=UserWarning)

class TradingModel(nn.Module):
    """
    Realistic trading model architecture
    Similar to actual inference workload
    """
    def __init__(self, input_size=100, hidden_size=512, output_size=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.network(x)


def measure_inference_time(
    model: nn.Module,
    input_data: torch.Tensor,
    num_iterations: int = 1000,
    warmup_iterations: int = 100,
    device: str = 'cuda:0'
) -> Tuple[float, float, List[float]]:
    """
    Measure inference latency with high precision
    
    Returns:
        (mean_ms, std_ms, all_times_ms)
    """
    model.eval()
    times = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_data)
    torch.cuda.synchronize()
    
    # Measurement
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = model(input_data)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    times_array = np.array(times)
    return float(times_array.mean()), float(times_array.std()), times


def test_compile_mode(
    model: nn.Module,
    input_data: torch.Tensor,
    mode: str,
    backend: str = 'inductor',
    num_iterations: int = 1000
) -> Dict:
    """
    Test a specific torch.compile configuration
    
    Args:
        model: Model to compile
        input_data: Test input
        mode: Compile mode ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs')
        backend: Compiler backend ('inductor', 'cudagraphs', 'aot_eager')
        num_iterations: Number of test iterations
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing: backend={backend}, mode={mode}")
    print(f"{'='*60}")
    
    # Clone model for clean compilation
    test_model = TradingModel(
        input_size=input_data.shape[1],
        hidden_size=512,
        output_size=10
    ).to(input_data.device)
    test_model.load_state_dict(model.state_dict())
    
    try:
        # Compile model
        print(f"Compiling model... (this takes 10-30 seconds)")
        compile_start = time.time()
        
        compiled_model = torch.compile(
            test_model,
            backend=backend,
            mode=mode,
            fullgraph=True
        )
        
        # First inference triggers compilation
        with torch.no_grad():
            _ = compiled_model(input_data)
        torch.cuda.synchronize()
        
        compile_time = time.time() - compile_start
        print(f"Compilation completed in {compile_time:.2f} seconds")
        
        # Measure performance
        print(f"Measuring inference latency ({num_iterations} iterations)...")
        mean_ms, std_ms, all_times = measure_inference_time(
            compiled_model,
            input_data,
            num_iterations=num_iterations,
            warmup_iterations=50  # Fewer warmup needed after compile
        )
        
        results = {
            'backend': backend,
            'mode': mode,
            'compile_time_sec': compile_time,
            'mean_latency_ms': mean_ms,
            'std_latency_ms': std_ms,
            'min_latency_ms': float(np.min(all_times)),
            'max_latency_ms': float(np.max(all_times)),
            'p50_latency_ms': float(np.percentile(all_times, 50)),
            'p95_latency_ms': float(np.percentile(all_times, 95)),
            'p99_latency_ms': float(np.percentile(all_times, 99)),
            'coefficient_variation': std_ms / mean_ms,
            'success': True,
            'error': None
        }
        
        print(f"✅ Mean latency: {mean_ms:.4f}ms ± {std_ms:.4f}ms")
        print(f"   P50: {results['p50_latency_ms']:.4f}ms")
        print(f"   P95: {results['p95_latency_ms']:.4f}ms")
        print(f"   P99: {results['p99_latency_ms']:.4f}ms")
        
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        results = {
            'backend': backend,
            'mode': mode,
            'success': False,
            'error': str(e)
        }
    
    return results


def main():
    """Run comprehensive torch.compile performance tests"""
    print("="*60)
    print("torch.compile Performance Test")
    print("="*60)
    
    # Get GPU configuration
    config = get_gpu_config()
    print(f"\nGPU Configuration:")
    print(f"  GPUs: {config['gpu_count']}")
    print(f"  Memory: {config['memory_allocated_gb']} GB")
    print(f"  TF32: {config['enable_tf32']}")
    print(f"  Deterministic: {config['deterministic']}")
    
    # Create model and test data
    device = 'cuda:0'
    batch_size = 1  # Trading inference is typically batch=1
    input_size = 100
    
    print(f"\nModel Architecture:")
    print(f"  Input: {input_size}")
    print(f"  Hidden: 512 x 3 layers")
    print(f"  Output: 10")
    print(f"  Batch size: {batch_size}")
    
    model = TradingModel(input_size=input_size).to(device)
    input_data = torch.randn(batch_size, input_size, device=device)
    
    # Test 1: Eager mode baseline
    print("\n" + "="*60)
    print("BASELINE: Eager Mode (no compilation)")
    print("="*60)
    
    mean_ms, std_ms, all_times = measure_inference_time(
        model, input_data, num_iterations=1000, warmup_iterations=100
    )
    
    baseline_results = {
        'mode': 'eager',
        'backend': 'none',
        'mean_latency_ms': mean_ms,
        'std_latency_ms': std_ms,
        'min_latency_ms': float(np.min(all_times)),
        'max_latency_ms': float(np.max(all_times)),
        'p50_latency_ms': float(np.percentile(all_times, 50)),
        'p95_latency_ms': float(np.percentile(all_times, 95)),
        'p99_latency_ms': float(np.percentile(all_times, 99)),
        'coefficient_variation': std_ms / mean_ms,
        'compile_time_sec': 0.0
    }
    
    print(f"✅ Baseline: {mean_ms:.4f}ms ± {std_ms:.4f}ms")
    print(f"   P50: {baseline_results['p50_latency_ms']:.4f}ms")
    print(f"   P95: {baseline_results['p95_latency_ms']:.4f}ms")
    print(f"   P99: {baseline_results['p99_latency_ms']:.4f}ms")
    
    # Test 2-5: Different torch.compile configurations
    compile_configs = [
        ('inductor', 'default'),
        ('inductor', 'reduce-overhead'),
        ('inductor', 'max-autotune'),
        ('inductor', 'max-autotune-no-cudagraphs'),
    ]
    
    compile_results = []
    for backend, mode in compile_configs:
        result = test_compile_mode(
            model, input_data, mode=mode, backend=backend, num_iterations=1000
        )
        if result['success']:
            compile_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\nBaseline (Eager): {baseline_results['mean_latency_ms']:.4f}ms")
    
    if compile_results:
        print("\nCompiled Results:")
        best_result = None
        best_speedup = 0
        
        for result in compile_results:
            speedup = baseline_results['mean_latency_ms'] / result['mean_latency_ms']
            print(f"  {result['backend']:12s} {result['mode']:30s}: "
                  f"{result['mean_latency_ms']:7.4f}ms "
                  f"({speedup:.2f}x speedup)")
            
            if speedup > best_speedup:
                best_speedup = speedup
                best_result = result
        
        if best_result:
            print(f"\n🏆 Best Configuration:")
            print(f"   Backend: {best_result['backend']}")
            print(f"   Mode: {best_result['mode']}")
            print(f"   Latency: {best_result['mean_latency_ms']:.4f}ms")
            print(f"   Speedup: {best_speedup:.2f}x")
            print(f"   Compile Time: {best_result['compile_time_sec']:.2f}s")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path.home() / f'gpu_optimization_logs/torch_compile_test_{timestamp}.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': timestamp,
        'hardware': {
            'gpu_count': config['gpu_count'],
            'memory_gb': config['memory_allocated_gb'],
            'compute_capability': config['compute_capability'],
            'interconnect': config['interconnect']
        },
        'configuration': {
            'tf32_enabled': config['enable_tf32'],
            'deterministic': config['deterministic'],
            'pytorch_version': torch.__version__
        },
        'baseline': baseline_results,
        'compiled': compile_results,
        'best_speedup': best_speedup if compile_results else 0
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
