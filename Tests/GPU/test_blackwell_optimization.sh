#!/bin/bash

# Test script to verify Blackwell optimizations are working

echo "Testing Blackwell Max-Q Optimizations"
echo "======================================"
echo ""

# 1. Run the optimizer
echo "Step 1: Applying optimizations..."
sudo /home/youssefbahloul/ai-trading-station/Services/GPU/blackwell_maxq_optimizer.sh
echo ""

# 2. Test Python optimizations (using existing tested config)
echo "Step 2: Testing PyTorch optimizations..."
cd /home/youssefbahloul/ai-trading-station
python3 -c "
import sys
sys.path.append('Services/GPU')
from blackwell_sm12_optimized_config import configure_sm12_optimizations, get_optimal_precision_mode
print('Loading PyTorch with Blackwell optimizations...')
configure_sm12_optimizations()
precision = get_optimal_precision_mode()
print(f'Optimal precision mode: {precision}')
print('PyTorch configuration: SUCCESS')
"
echo ""

# 3. Run stress test
echo "Step 3: Running GPU stress test (30 seconds)..."
timeout 30 python3 -c "
import torch
import time

# Create large tensors
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device='cuda')
b = torch.randn(8192, 8192, dtype=torch.bfloat16, device='cuda')

print('Starting stress test...')
start = time.time()
iterations = 0

while time.time() - start < 30:
    c = torch.matmul(a, b)
    iterations += 1
    if iterations % 10 == 0:
        print(f'Iteration {iterations}, Elapsed: {time.time()-start:.1f}s')

torch.cuda.synchronize()
print(f'Completed {iterations} iterations in 30 seconds')
" &

STRESS_PID=$!

# Monitor while stress test runs
echo ""
echo "Monitoring GPU during stress test:"
for i in {1..6}; do
    sleep 5
    nvidia-smi --query-gpu=gpu_name,power.draw,clocks.sm,temperature.gpu --format=csv,noheader
done

wait $STRESS_PID

echo ""
echo "Test complete!"
echo ""

# 4. Show final state
echo "Final GPU State:"
nvidia-smi
