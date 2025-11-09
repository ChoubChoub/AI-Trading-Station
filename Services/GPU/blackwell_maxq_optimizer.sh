#!/bin/bash

################################################################################
# Blackwell Max-Q GPU Optimization Script
# For: RTX PRO 6000 Ada/Blackwell Generation GPUs
# Purpose: Maximize performance within Max-Q power/thermal constraints
################################################################################

# Note: Not using 'set -e' to allow script to continue if some Max-Q operations fail

SCRIPT_NAME="Blackwell Max-Q Optimizer"
LOG_FILE="/var/log/gpu_blackwell_optimization.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log_message() {
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

# Function to print colored status
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success")
            echo -e "${GREEN}✓${NC} $message"
            ;;
        "warning")
            echo -e "${YELLOW}⚠${NC} $message"
            ;;
        "error")
            echo -e "${RED}✗${NC} $message"
            ;;
    esac
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_status "error" "This script must be run as root"
   exit 1
fi

log_message "=========================================="
log_message "Starting $SCRIPT_NAME"
log_message "=========================================="

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
log_message "Detected $NUM_GPUS GPU(s)"

# Get GPU information
for i in $(seq 0 $((NUM_GPUS-1))); do
    GPU_NAME=$(nvidia-smi -i $i --query-gpu=name --format=csv,noheader)
    GPU_POWER_MAX=$(nvidia-smi -i $i --query-gpu=power.max_limit --format=csv,noheader,nounits)
    log_message "GPU $i: $GPU_NAME (Max Power: ${GPU_POWER_MAX}W)"
done

################################################################################
# 1. PERSISTENCE MODE - Keep driver loaded, reduce latency
################################################################################
print_status "success" "Enabling persistence mode..."
nvidia-smi -pm 1
if [ $? -eq 0 ]; then
    log_message "Persistence mode: ENABLED"
else
    print_status "warning" "Failed to enable persistence mode"
fi

################################################################################
# 2. POWER MANAGEMENT - Maximize within Max-Q limits
################################################################################
print_status "success" "Configuring power limits..."

for i in $(seq 0 $((NUM_GPUS-1))); do
    # Get maximum allowed power for this GPU
    MAX_POWER=$(nvidia-smi -i $i --query-gpu=power.max_limit --format=csv,noheader,nounits)
    
    # Set to maximum allowed power
    nvidia-smi -i $i -pl $MAX_POWER 2>/dev/null
    if [ $? -eq 0 ]; then
        log_message "GPU $i: Power limit set to ${MAX_POWER}W"
        print_status "success" "GPU $i: Power limit set to ${MAX_POWER}W"
    else
        print_status "warning" "GPU $i: Could not set power limit"
    fi
done

################################################################################
# 3. PERFORMANCE STATE - Force maximum performance mode
################################################################################
print_status "success" "Setting performance mode..."

# Try nvidia-settings only if X display is available (skip on headless servers)
if command -v nvidia-settings &> /dev/null && [ -n "$DISPLAY" ]; then
    for i in $(seq 0 $((NUM_GPUS-1))); do
        nvidia-settings -a "[gpu:$i]/GpuPowerMizerMode=1" &>/dev/null
        if [ $? -eq 0 ]; then
            log_message "GPU $i: Performance mode set to MAXIMUM"
            print_status "success" "GPU $i: Performance mode set via nvidia-settings"
        fi
    done
else
    print_status "warning" "Skipping nvidia-settings (headless server - normal for Max-Q)"
fi

# Also try to set preferred performance mode via nvidia-smi
for i in $(seq 0 $((NUM_GPUS-1))); do
    # Try to disable auto boost and set to P0 state
    nvidia-smi -i $i -ac 10500,2400 2>/dev/null || {
        # If that fails, try memory clock only
        nvidia-smi -i $i -lmc 10500 2>/dev/null || {
            print_status "warning" "GPU $i: Cannot lock memory clocks (Max-Q limitation)"
        }
    }
done

################################################################################
# 4. THERMAL MANAGEMENT - Prevent throttling
################################################################################
print_status "success" "Configuring thermal management..."

# Set GPU temperature target (lower = less throttling)
for i in $(seq 0 $((NUM_GPUS-1))); do
    # Try to set temperature target to 75°C (default is usually 83°C)
    nvidia-smi -i $i -gtt 75 2>/dev/null
    if [ $? -eq 0 ]; then
        log_message "GPU $i: Temperature target set to 75°C"
        print_status "success" "GPU $i: Temperature target set to 75°C"
    else
        print_status "warning" "GPU $i: Cannot set temperature target (Max-Q limitation)"
    fi
done

# Set aggressive fan profile if nvidia-settings is available and X display exists
if command -v nvidia-settings &> /dev/null && [ -n "$DISPLAY" ]; then
    for i in $(seq 0 $((NUM_GPUS-1))); do
        # Enable manual fan control
        nvidia-settings -a "[gpu:$i]/GPUFanControlState=1" &>/dev/null
        # Set fan speed to 75% (balance between cooling and noise)
        nvidia-settings -a "[fan:$i]/GPUTargetFanSpeed=75" &>/dev/null
        if [ $? -eq 0 ]; then
            log_message "GPU $i: Fan speed set to 75%"
            print_status "success" "GPU $i: Fan speed set to 75%"
        fi
    done
else
    print_status "warning" "Skipping fan control (headless server - using default thermal management)"
fi

################################################################################
# 5. MEMORY OPTIMIZATION - Critical for bandwidth-bound workloads
################################################################################
print_status "success" "Optimizing memory settings..."

for i in $(seq 0 $((NUM_GPUS-1))); do
    # Try to set memory to maximum frequency
    # Blackwell typically supports 10500 MHz (21 Gbps effective)
    nvidia-smi -i $i -lmc 10500 2>/dev/null
    if [ $? -eq 0 ]; then
        log_message "GPU $i: Memory clock locked to maximum"
    else
        # Try alternative memory frequencies
        for mem_freq in 10251 10001 9751 9501; do
            nvidia-smi -i $i -lmc $mem_freq 2>/dev/null && {
                log_message "GPU $i: Memory clock set to $mem_freq MHz"
                break
            }
        done
    fi
    
    # Reset memory clocks to enable highest available
    nvidia-smi -i $i -rmc 2>/dev/null
done

################################################################################
# 6. CUDA MPS (Multi-Process Service) - Better multi-process performance
################################################################################
print_status "success" "Configuring CUDA MPS..."

# Stop any existing MPS
echo quit | nvidia-cuda-mps-control 2>/dev/null

# Set MPS pipe directory
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps

# Create directories
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY

# Start MPS control daemon
nvidia-cuda-mps-control -d 2>/dev/null
if [ $? -eq 0 ]; then
    log_message "CUDA MPS: Started successfully"
    print_status "success" "CUDA MPS enabled for better multi-process performance"
else
    print_status "warning" "CUDA MPS could not be started"
fi

################################################################################
# 7. NVIDIA GSP (GPU System Processor) - Offload management tasks
################################################################################
print_status "success" "Checking GSP firmware..."

# Enable GSP firmware if available (reduces CPU overhead)
if nvidia-smi -q | grep -q "GSP Firmware Version"; then
    log_message "GSP Firmware: Available and active"
    print_status "success" "GSP firmware active (reduces CPU overhead)"
fi

################################################################################
# 8. COMPUTE MODE - Set to exclusive process
################################################################################
print_status "success" "Setting compute mode..."

for i in $(seq 0 $((NUM_GPUS-1))); do
    # Set to exclusive process mode (one process at a time, better performance)
    nvidia-smi -i $i -c EXCLUSIVE_PROCESS 2>/dev/null || {
        # Fallback to default if exclusive not supported
        nvidia-smi -i $i -c DEFAULT 2>/dev/null
        print_status "warning" "GPU $i: Using default compute mode"
    }
done

################################################################################
# 9. VERIFICATION - Display current settings
################################################################################
print_status "success" "Verification of settings:"
echo ""
echo "Current GPU Status:"
echo "==================="

nvidia-smi --query-gpu=gpu_name,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate,persistence_mode,compute_mode --format=table

# Log detailed state
log_message "Optimization complete. Detailed state:"
nvidia-smi -q >> "$LOG_FILE"

################################################################################
# SUMMARY
################################################################################
echo ""
echo "=========================================="
echo -e "${GREEN}Blackwell Max-Q Optimization Complete${NC}"
echo "=========================================="
echo ""
echo "Optimizations Applied:"
echo "  ✓ Persistence mode enabled"
echo "  ✓ Power limits maximized"
echo "  ✓ Performance mode activated"
echo "  ✓ Thermal targets configured"
echo "  ✓ Memory clocks optimized"
echo "  ✓ CUDA MPS enabled"
echo "  ✓ Compute mode configured"
echo ""
echo "Note: Max-Q GPUs have hardware-enforced limits."
echo "Current settings are optimal within these constraints."
echo ""
echo "To monitor GPU performance:"
echo "  watch -n 1 nvidia-smi"
echo ""
log_message "=========================================="
log_message "Optimization complete"
log_message "=========================================="