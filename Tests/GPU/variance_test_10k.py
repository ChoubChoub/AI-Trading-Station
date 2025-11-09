#!/usr/bin/env python3
"""
Comprehensive 10,000 Iteration Variance Test
Tests production stability with full optimization stack

Configuration Stack:
- Layer 1: Persistent environment variables (/etc/profile.d/)
- Layer 2: Auto-configured PyTorch (TF32, deterministic, 95% memory)
- Layer 3: torch.compile (2.66x speedup, max-autotune-no-cudagraphs)

Target Metrics:
- P50 latency: <0.10ms
- P95 latency: <0.15ms
- P99 latency: <0.20ms
- Max variance: <2ms (from sustained load)
- Coefficient of variation: <5%

Hardware: 2x RTX PRO 6000 Blackwell (SM_12.0)
"""
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

# Auto-configure GPU
sys.path.insert(0, str(Path.home() / 'ai-trading-station'))
from Services.GPU import get_gpu_config, trading_inference

print("="*60)
print("10,000 Iteration Variance Test")
print("="*60)

# Get configuration
config = get_gpu_config()
print(f"\nGPU Configuration:")
print(f"  GPUs: {config['gpu_count']}")
print(f"  TF32: {config['enable_tf32']}")
print(f"  Deterministic: {config['deterministic']}")
print(f"  Interconnect: {config['interconnect']}")

# Create test model (same as torch_compile_test)
class TradingModel(nn.Module):
    """Realistic trading model architecture"""
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

# Setup
device = 'cuda:0'
batch_size = 1
input_size = 100
num_iterations = 10000

print(f"\nTest Configuration:")
print(f"  Model: TradingModel (100 → 512 → 512 → 512 → 10)")
print(f"  Batch size: {batch_size}")
print(f"  Iterations: {num_iterations:,}")
print(f"  Device: {device}")

# Create and compile model
print(f"\nPreparing model...")
model = TradingModel(input_size=input_size).to(device).eval()
input_data = torch.randn(batch_size, input_size, device=device)

# Compile with optimal settings
compiled_model = trading_inference.compile_for_trading(
    model,
    backend='inductor',
    mode='max-autotune-no-cudagraphs',
    verbose=True
)

# First inference (triggers JIT compilation)
print(f"\nTriggering JIT compilation...")
with torch.no_grad():
    _ = compiled_model(input_data)
torch.cuda.synchronize()
print(f"✅ Compilation complete")

# Warmup
print(f"\nRunning warmup (100 iterations)...")
with torch.no_grad():
    for _ in range(100):
        _ = compiled_model(input_data)
torch.cuda.synchronize()
print(f"✅ Warmup complete")

# Main test
print(f"\nRunning {num_iterations:,} iteration variance test...")
print(f"This will take approximately {num_iterations * 0.062 / 1000:.1f} seconds...")

times = []
with torch.no_grad():
    for i in range(num_iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = compiled_model(input_data)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1:,}/{num_iterations:,} iterations...")

# Calculate statistics
times_array = np.array(times)
mean_ms = float(times_array.mean())
std_ms = float(times_array.std())
min_ms = float(times_array.min())
max_ms = float(times_array.max())
p50_ms = float(np.percentile(times_array, 50))
p95_ms = float(np.percentile(times_array, 95))
p99_ms = float(np.percentile(times_array, 99))
cv = std_ms / mean_ms

# Results
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"\n📊 Latency Statistics:")
print(f"   Mean:  {mean_ms:.4f}ms")
print(f"   Std:   {std_ms:.4f}ms")
print(f"   Min:   {min_ms:.4f}ms")
print(f"   Max:   {max_ms:.4f}ms")

print(f"\n📈 Percentiles:")
print(f"   P50:   {p50_ms:.4f}ms")
print(f"   P95:   {p95_ms:.4f}ms")
print(f"   P99:   {p99_ms:.4f}ms")

print(f"\n🎯 Variance Metrics:")
print(f"   Coefficient of Variation: {cv*100:.2f}%")
print(f"   Range (Max - Min): {max_ms - min_ms:.4f}ms")

# Check targets
print(f"\n✅ Target Validation:")
targets_met = []

if p50_ms < 0.10:
    print(f"   ✅ P50 < 0.10ms: {p50_ms:.4f}ms")
    targets_met.append(True)
else:
    print(f"   ⚠️  P50 < 0.10ms: {p50_ms:.4f}ms (target not met)")
    targets_met.append(False)

if p95_ms < 0.15:
    print(f"   ✅ P95 < 0.15ms: {p95_ms:.4f}ms")
    targets_met.append(True)
else:
    print(f"   ⚠️  P95 < 0.15ms: {p95_ms:.4f}ms (target not met)")
    targets_met.append(False)

if p99_ms < 0.20:
    print(f"   ✅ P99 < 0.20ms: {p99_ms:.4f}ms")
    targets_met.append(True)
else:
    print(f"   ⚠️  P99 < 0.20ms: {p99_ms:.4f}ms (target not met)")
    targets_met.append(False)

if (max_ms - min_ms) < 2.0:
    print(f"   ✅ Variance < 2ms: {max_ms - min_ms:.4f}ms")
    targets_met.append(True)
else:
    print(f"   ⚠️  Variance < 2ms: {max_ms - min_ms:.4f}ms (target not met)")
    targets_met.append(False)

if cv < 0.05:
    print(f"   ✅ CV < 5%: {cv*100:.2f}%")
    targets_met.append(True)
else:
    print(f"   ⚠️  CV < 5%: {cv*100:.2f}% (target not met)")
    targets_met.append(False)

# Overall status
targets_met_count = sum(targets_met)
total_targets = len(targets_met)

print(f"\n🏆 Overall Status: {targets_met_count}/{total_targets} targets met")

if all(targets_met):
    print(f"   ✅ EXCELLENT: All targets met - Production ready!")
elif targets_met_count >= 3:
    print(f"   ✅ GOOD: Most targets met - Production ready with monitoring")
else:
    print(f"   ⚠️  REVIEW: Some targets not met - Consider investigation")

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = Path.home() / f'gpu_optimization_logs/variance_test_10k_{timestamp}.json'
output_file.parent.mkdir(parents=True, exist_ok=True)

results = {
    'timestamp': timestamp,
    'configuration': {
        'gpu_count': config['gpu_count'],
        'tf32_enabled': config['enable_tf32'],
        'deterministic': config['deterministic'],
        'interconnect': config['interconnect'],
        'compile_backend': 'inductor',
        'compile_mode': 'max-autotune-no-cudagraphs'
    },
    'test_params': {
        'num_iterations': num_iterations,
        'batch_size': batch_size,
        'input_size': input_size,
        'device': device
    },
    'statistics': {
        'mean_ms': mean_ms,
        'std_ms': std_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'p50_ms': p50_ms,
        'p95_ms': p95_ms,
        'p99_ms': p99_ms,
        'coefficient_variation': cv,
        'range_ms': max_ms - min_ms
    },
    'targets': {
        'p50_target': 0.10,
        'p50_met': targets_met[0],
        'p95_target': 0.15,
        'p95_met': targets_met[1],
        'p99_target': 0.20,
        'p99_met': targets_met[2],
        'variance_target': 2.0,
        'variance_met': targets_met[3],
        'cv_target': 0.05,
        'cv_met': targets_met[4]
    },
    'status': 'EXCELLENT' if all(targets_met) else 'GOOD' if targets_met_count >= 3 else 'REVIEW',
    'production_ready': all(targets_met) or targets_met_count >= 3
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
print(f"\n{'='*60}")
