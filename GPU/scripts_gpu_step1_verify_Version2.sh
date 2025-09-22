#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# OPERATION: Hardware Detection Verification
# WHY: Confirm RTX 6000 Pro Max Q cards are properly installed before any optimization
# EXPECTED OUTCOME: Both GPUs detected via lspci, basic system recognition
# FAILURE MODE: If GPUs not detected, check PCIe seating and power connections
# RTX 6000 PRO MAX Q LIMITATION: Professional cards require enterprise-grade power delivery (minimum 1200W PSU)
# -----------------------------------------------------------------------------

echo "=== STEP 1: VERIFYING GPU HARDWARE DETECTION ==="
echo "Checking if your RTX 6000 Pro Max Q cards are detected by the system..."
echo ""

# Ensure lspci is available
if ! command -v lspci >/dev/null 2>&1; then
  echo "❌ 'lspci' not found. Install pciutils first:"
  echo "   sudo apt update && sudo apt install -y pciutils"
  exit 1
fi

echo "🔍 Scanning PCIe bus for NVIDIA GPUs..."
# Match both 'VGA' and '3D controller' classes for NVIDIA devices
GPU_LINES="$(lspci | grep -i nvidia | grep -Ei 'vga|3d controller' || true)"
GPU_COUNT="$(printf "%s\n" "$GPU_LINES" | sed '/^\s*$/d' | wc -l | tr -d ' ')"

echo "Found ${GPU_COUNT} NVIDIA GPU(s) on PCIe bus:"
if [ -n "$GPU_LINES" ]; then
  echo "$GPU_LINES"
else
  echo "(none)"
fi

echo ""
echo "📋 Detailed GPU information:"
lspci -v | grep -A 20 -E "VGA.*NVIDIA|3D controller.*NVIDIA" | head -80 || true

echo ""
if [ "${GPU_COUNT}" -eq 2 ]; then
    echo "✅ SUCCESS: 2 NVIDIA GPUs detected (expected for dual RTX 6000 Pro Max Q)"
elif [ "${GPU_COUNT}" -eq 1 ]; then
    echo "⚠️  WARNING: Only 1 NVIDIA GPU detected (expected 2)"
    echo "   - Check second GPU seating and power connection"
    echo "   - Verify all PCIe power cables are connected"
    echo "   - Check BIOS: PCIe slots enabled, Above 4G Decoding enabled, Secure Boot off (for driver install later)"
elif [ "${GPU_COUNT}" -eq 0 ]; then
    echo "❌ ERROR: No NVIDIA GPUs detected"
    echo "   - Check GPU seating, power connections, and BIOS settings"
    echo "   - Ensure BIOS has PCIe slots enabled and Secure Boot disabled"
else
    echo "🤔 UNEXPECTED: ${GPU_COUNT} NVIDIA GPUs detected (expected 2)"
fi

echo ""
echo "⚡ Checking power and cooling requirements..."
echo "RTX 6000 Pro Max Q requirements:"
echo "  - Minimum 1200W PSU for dual card setup"
echo "  - Adequate PCIe power connectors (typically 2x 8-pin per card)"
echo "  - Enterprise cooling solution for sustained workloads"

echo ""
echo "➡️  Next: If 2 GPUs are detected, proceed to Step 2 (driver cleanup)."