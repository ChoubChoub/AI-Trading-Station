#!/usr/bin/env bash
set -euo pipefail

echo "=== STEP 3: PREPARING SYSTEM FOR GPU DRIVERS ==="
echo "Installing build tools and preparing your system for RTX 6000 Pro Max Q drivers..."
echo ""

# Ensure apt-get exists (Ubuntu/Debian)
if ! command -v apt-get >/dev/null 2>&1; then
  echo "❌ This script expects apt-get (Debian/Ubuntu). Aborting."
  exit 1
fi

# Resolve repo GPU folder (where this script lives) and log directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/step3_prepare_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Show Secure Boot status (informational)
if command -v mokutil >/dev/null 2>&1; then
  SB_STATE=$(mokutil --sb-state 2>/dev/null || true)
  echo "🔐 Secure Boot status: ${SB_STATE:-unknown}"
fi
echo ""

echo "📦 Updating package lists..."
sudo apt-get update -y

echo ""
echo "📦 Installing required packages:"
echo "  - build-essential, gcc, make, dkms, pkg-config"
echo "  - linux-headers-$(uname -r)"
echo "  - Vulkan libs: libvulkan1, mesa-vulkan-drivers, vulkan-tools"
sudo apt-get install -y \
  build-essential gcc make dkms pkg-config \
  "linux-headers-$(uname -r)" \
  libvulkan1 mesa-vulkan-drivers vulkan-tools

echo ""
echo "🧪 Verifying kernel headers for running kernel: $(uname -r)"
if dpkg -l "linux-headers-$(uname -r)" 2>/dev/null | awk '/^ii/{found=1} END{exit !found}'; then
  echo "✅ Headers installed for current kernel."
else
  echo "⚠️  Headers for $(uname -r) not found. Consider: sudo apt-get install linux-headers-generic, reboot to latest kernel, then re-run Step 3."
fi

echo ""
echo "🚫 Blacklisting nouveau to prevent conflicts..."
sudo bash -c "cat > /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF"

echo ""
echo "🔄 Updating initramfs (to apply nouveau blacklist)..."
sudo update-initramfs -u

echo ""
echo "📤 Attempting to unload nouveau (ignored if not loaded)..."
if lsmod | awk '{print $1}' | grep -qx 'nouveau'; then
  if sudo modprobe -r nouveau; then
    echo "✅ nouveau module unloaded."
  else
    echo "ℹ️  Could not unload nouveau now (likely in use). This is OK; the blacklist will take effect on next reboot."
  fi
else
  echo "✅ nouveau not loaded."
fi

# Make the log owned by the invoking user (handy when run with sudo)
INVOKER="${SUDO_USER:-$USER}"
if [ -n "${SUDO_USER:-}" ] && id "${INVOKER}" >/dev/null 2>&1; then
  chown "${INVOKER}:${INVOKER}" "${LOG_FILE}" 2>/dev/null || true
fi

echo ""
echo "✅ System preparation complete!"
echo "📋 Installed: build tools, kernel headers, Vulkan libs"
echo "🚫 Blacklisted: nouveau (effective next boot if it was loaded)"
echo "➡️  Next: STEP 4 (Install NVIDIA driver 535-open) — that step will reboot automatically."