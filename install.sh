#!/bin/bash

# AI Trading Station Installation Script
# Installs dependencies and sets up the high-performance trading environment

set -euo pipefail

echo "Installing AI Trading Station dependencies..."

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    linux-tools-common \
    linux-tools-generic \
    rt-tests \
    hwloc-nox \
    stress-ng \
    numactl \
    ethtool \
    net-tools \
    bc \
    jq

# Install Python dependencies
pip3 install -r requirements.txt

echo "Dependencies installed successfully!"
echo "Run './ai-trading-station.sh --help' to get started."