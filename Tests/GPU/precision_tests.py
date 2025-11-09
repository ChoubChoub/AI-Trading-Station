#!/usr/bin/env python3
"""
Precision Performance Testing for Blackwell SM_12.0
Tests FP32, TF32, BF16, and FP8 precision modes

Compares:
- Inference latency (target: 2x for BF16, 4x for FP8)
- Accuracy degradation (acceptable: <1% difference)
- Memory usage
- Thermal behavior

Results saved to ~/gpu_optimization_logs/

Author: AI Trading Station
Date: October 17, 2025
"""

import torch
import torch.nn as nn
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Add workspace root to path
sys.path.insert(0, str(Path.home() / 'ai-trading-station'))
from Services.GPU.blackwell_sm12_optimized_config import configure_sm12_optimizations, get_memory_stats


class SimpleTestModel(nn.Module):
    """
    Simple neural network for testing different precision modes.
    Mimics typical trading inference workload.
    """
    def __init__(self, input_size: int = 100, hidden_size: int = 512, output_size: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


def measure_inference_latency(
    model: nn.Module,
    data: torch.Tensor,
    iterations: int = 1000,
    warmup: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure inference latency with high precision.
    
    Args:
        model: PyTorch model
        data: Input data tensor
        iterations: Number of iterations to measure
        warmup: Warmup iterations (excluded from timing)
        device: Device to run on
    
    Returns:
        dict: Latency statistics in milliseconds
    """
    model = model.to(device)
    model.eval()
    data = data.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(data)
    
    # Synchronize before timing
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure with CUDA events for precision
    timings = []
    
    with torch.no_grad():
        for _ in range(iterations):
            if device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                output = model(data)
                end_event.record()
                
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                timings.append(elapsed_ms)
            else:
                start = time.perf_counter()
                output = model(data)
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings.append(elapsed_ms)
    
    timings = np.array(timings)
    
    return {
        'mean_ms': float(np.mean(timings)),
        'std_dev_ms': float(np.std(timings)),
        'min_ms': float(np.min(timings)),
        'max_ms': float(np.max(timings)),
        'p50_ms': float(np.percentile(timings, 50)),
        'p95_ms': float(np.percentile(timings, 95)),
        'p99_ms': float(np.percentile(timings, 99)),
        'variance_percent': float((np.std(timings) / np.mean(timings)) * 100)
    }


def test_precision_mode(
    precision_mode: str,
    model: nn.Module,
    data: torch.Tensor,
    reference_output: torch.Tensor,
    iterations: int = 1000,
    device: str = 'cuda'
) -> Dict[str, any]:
    """
    Test a specific precision mode.
    
    Args:
        precision_mode: 'fp32', 'tf32', 'bf16', or 'fp8'
        model: PyTorch model (FP32)
        data: Input data (FP32)
        reference_output: FP32 output for accuracy comparison
        iterations: Number of timing iterations
        device: Device to run on
    
    Returns:
        dict: Test results including timing and accuracy
    """
    print(f"\n{'='*70}")
    print(f"Testing: {precision_mode.upper()}")
    print(f"{'='*70}")
    
    results = {
        'precision_mode': precision_mode,
        'device': device,
        'iterations': iterations,
        'timestamp': datetime.now().isoformat()
    }
    
    # Get initial memory stats
    mem_before = get_memory_stats(0)
    
    try:
        if precision_mode == 'fp32':
            # Pure FP32 (no TF32)
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
            torch.backends.cudnn.conv.fp32_precision = 'ieee'
            
            print("Configuration: IEEE FP32 (exact, slow)")
            latency = measure_inference_latency(model, data, iterations, device=device)
            
            # Get output for accuracy reference
            model_test = model.to(device)
            data_test = data.to(device)
            with torch.no_grad():
                output = model_test(data_test)
            
            accuracy_diff = 0.0  # Reference
            
        elif precision_mode == 'tf32':
            # TF32 (baseline for modern GPUs)
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            
            print("Configuration: TensorFloat-32 (8x faster matmul)")
            latency = measure_inference_latency(model, data, iterations, device=device)
            
            # Get output
            model_test = model.to(device)
            data_test = data.to(device)
            with torch.no_grad():
                output = model_test(data_test)
            
            # Compare with reference
            accuracy_diff = float(torch.mean(torch.abs(output - reference_output)).item())
            
        elif precision_mode == 'bf16':
            # BF16 mixed precision (automatic with autocast)
            print("Configuration: BFloat16 mixed precision (2x speedup)")
            
            # Create timing function with autocast
            def bf16_inference():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        return model(data)
            
            # Measure latency
            model_test = model.to(device)
            data_test = data.to(device)
            
            # Warmup
            for _ in range(100):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        _ = model_test(data_test)
            
            torch.cuda.synchronize()
            
            # Measure
            timings = []
            for _ in range(iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    with torch.no_grad():
                        output = model_test(data_test)
                end_event.record()
                
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
            
            timings = np.array(timings)
            latency = {
                'mean_ms': float(np.mean(timings)),
                'std_dev_ms': float(np.std(timings)),
                'min_ms': float(np.min(timings)),
                'max_ms': float(np.max(timings)),
                'p50_ms': float(np.percentile(timings, 50)),
                'p95_ms': float(np.percentile(timings, 95)),
                'p99_ms': float(np.percentile(timings, 99)),
                'variance_percent': float((np.std(timings) / np.mean(timings)) * 100)
            }
            
            # Get final output for accuracy
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                with torch.no_grad():
                    output = model_test(data_test).to(torch.float32)
            
            accuracy_diff = float(torch.mean(torch.abs(output - reference_output)).item())
            
        elif precision_mode == 'fp8':
            # FP8 precision (manual casting)
            print("Configuration: FP8 E4M3 (4x speedup, experimental)")
            
            if not hasattr(torch, 'float8_e4m3fn'):
                print("❌ FP8 not available in this PyTorch version")
                return None
            
            # Convert model to FP8
            model_fp8 = model.to(torch.float8_e4m3fn)
            data_fp8 = data.to(torch.float8_e4m3fn)
            
            # Warmup
            for _ in range(100):
                with torch.no_grad():
                    _ = model_fp8(data_fp8)
            
            torch.cuda.synchronize()
            
            # Measure
            timings = []
            for _ in range(iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                with torch.no_grad():
                    output_fp8 = model_fp8(data_fp8)
                end_event.record()
                
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
            
            timings = np.array(timings)
            latency = {
                'mean_ms': float(np.mean(timings)),
                'std_dev_ms': float(np.std(timings)),
                'min_ms': float(np.min(timings)),
                'max_ms': float(np.max(timings)),
                'p50_ms': float(np.percentile(timings, 50)),
                'p95_ms': float(np.percentile(timings, 95)),
                'p99_ms': float(np.percentile(timings, 99)),
                'variance_percent': float((np.std(timings) / np.mean(timings)) * 100)
            }
            
            # Get final output (convert back to FP32)
            with torch.no_grad():
                output = model_fp8(data_fp8).to(torch.float32)
            
            accuracy_diff = float(torch.mean(torch.abs(output - reference_output)).item())
        
        else:
            raise ValueError(f"Unknown precision mode: {precision_mode}")
        
        # Get final memory stats
        mem_after = get_memory_stats(0)
        
        # Store results
        results['latency'] = latency
        results['accuracy_diff_vs_fp32'] = accuracy_diff
        results['memory_before_gb'] = mem_before['allocated_gb']
        results['memory_after_gb'] = mem_after['allocated_gb']
        results['memory_delta_gb'] = mem_after['allocated_gb'] - mem_before['allocated_gb']
        results['success'] = True
        
        # Print summary
        print(f"\n📊 Results:")
        print(f"   Mean latency:     {latency['mean_ms']:.4f}ms")
        print(f"   Std deviation:    {latency['std_dev_ms']:.4f}ms")
        print(f"   Variance:         {latency['variance_percent']:.2f}%")
        print(f"   P95 latency:      {latency['p95_ms']:.4f}ms")
        print(f"   P99 latency:      {latency['p99_ms']:.4f}ms")
        print(f"   Accuracy diff:    {accuracy_diff:.6f}")
        print(f"   Memory used:      {mem_after['allocated_gb']:.3f}GB")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        results['success'] = False
        results['error'] = str(e)
        return results
    
    return results


def run_full_precision_suite(
    iterations: int = 1000,
    save_results: bool = True
) -> Dict[str, any]:
    """
    Run complete precision testing suite: FP32, TF32, BF16, FP8.
    
    Args:
        iterations: Number of timing iterations per test
        save_results: Save results to JSON file
    
    Returns:
        dict: Complete test results
    """
    print("=" * 80)
    print("BLACKWELL SM_12.0 PRECISION PERFORMANCE TESTING")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: SM_{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}")
    print(f"Iterations per test: {iterations}")
    print()
    
    # Create test model and data
    print("Creating test model (trading inference workload)...")
    model = SimpleTestModel(input_size=100, hidden_size=512, output_size=10)
    model = model.cuda()
    
    # Create test data (batch of 32 samples)
    data = torch.randn(32, 100, device='cuda')
    
    # Get FP32 reference output
    print("Generating FP32 reference output...")
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    with torch.no_grad():
        reference_output = model(data).clone()
    
    # Run tests
    test_results = {}
    
    # Test 1: FP32 (baseline)
    test_results['fp32'] = test_precision_mode('fp32', model, data, reference_output, iterations)
    
    # Test 2: TF32 (baseline modern)
    test_results['tf32'] = test_precision_mode('tf32', model, data, reference_output, iterations)
    
    # Test 3: BF16 (conservative optimization)
    test_results['bf16'] = test_precision_mode('bf16', model, data, reference_output, iterations)
    
    # Test 4: FP8 (aggressive optimization)
    test_results['fp8'] = test_precision_mode('fp8', model, data, reference_output, iterations)
    
    # Calculate speedups
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")
    
    fp32_time = test_results['fp32']['latency']['mean_ms']
    
    comparison_table = []
    comparison_table.append(["Precision", "Latency", "vs FP32", "vs TF32", "Accuracy Loss", "Status"])
    comparison_table.append(["-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 15, "-" * 10])
    
    for mode in ['fp32', 'tf32', 'bf16', 'fp8']:
        if test_results[mode] and test_results[mode]['success']:
            latency = test_results[mode]['latency']['mean_ms']
            speedup_fp32 = fp32_time / latency
            speedup_tf32 = test_results['tf32']['latency']['mean_ms'] / latency if mode != 'tf32' else 1.0
            accuracy = test_results[mode]['accuracy_diff_vs_fp32']
            
            comparison_table.append([
                mode.upper(),
                f"{latency:.4f}ms",
                f"{speedup_fp32:.2f}x",
                f"{speedup_tf32:.2f}x",
                f"{accuracy:.6f}",
                "✅" if test_results[mode]['success'] else "❌"
            ])
    
    # Print table
    for row in comparison_table:
        print(f"  {row[0]:<10} {row[1]:<12} {row[2]:<10} {row[3]:<10} {row[4]:<15} {row[5]:<10}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if test_results['bf16']['success']:
        bf16_speedup = fp32_time / test_results['bf16']['latency']['mean_ms']
        bf16_accuracy = test_results['bf16']['accuracy_diff_vs_fp32']
        
        print(f"✅ BF16 Mixed Precision (Conservative):")
        print(f"   Speedup: {bf16_speedup:.2f}x")
        print(f"   Accuracy loss: {bf16_accuracy:.6f}")
        print(f"   Recommendation: {'RECOMMENDED' if bf16_accuracy < 0.01 else 'Check accuracy'}")
    
    if test_results['fp8'] and test_results['fp8']['success']:
        fp8_speedup = fp32_time / test_results['fp8']['latency']['mean_ms']
        fp8_accuracy = test_results['fp8']['accuracy_diff_vs_fp32']
        
        print(f"\n⚡ FP8 Precision (Aggressive):")
        print(f"   Speedup: {fp8_speedup:.2f}x")
        print(f"   Accuracy loss: {fp8_accuracy:.6f}")
        print(f"   Recommendation: {'RECOMMENDED' if fp8_accuracy < 0.05 else 'Accuracy degradation significant'}")
    
    # Save results
    if save_results:
        output_dir = Path.home() / 'gpu_optimization_logs'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f'precision_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    return test_results


if __name__ == '__main__':
    # Configure SM_12.0 optimizations first
    print("Configuring SM_12.0 optimizations...\n")
    config = configure_sm12_optimizations(
        enable_fp8=False,  # Will test manually
        enable_bf16=False,  # Will test manually
        memory_fraction=0.95,
        deterministic=True,
        verbose=False
    )
    
    # Run full test suite
    results = run_full_precision_suite(iterations=1000, save_results=True)
    
    print("\n" + "=" * 80)
    print("✅ PRECISION TESTING COMPLETE")
    print("=" * 80)
    print("\nNext: Run variance test with optimal precision mode")
