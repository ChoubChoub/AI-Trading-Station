#!/usr/bin/env bash
set -euo pipefail

echo "=== STEP 4: INSTALLING NVIDIA DRIVERS (DETERMINISTIC) ==="
echo "Installing official NVIDIA drivers for your RTX 6000 Pro Max Q cards..."
echo ""
echo "Driver selection policy:"
echo "  - Prefer 535-open if available (as per Module_Version4)"
echo "  - Else fallback to the nearest available -open driver (550-open, then 570-open)"
echo "  - Enable nvidia-persistenced"
echo "  - Reboot after install"

# Ensure apt-get exists (Ubuntu/Debian)
if ! command -v apt-get >/dev/null 2>&1; then
  echo "❌ This script expects apt-get (Debian/Ubuntu). Aborting."
  exit 1
fi

# Resolve repo GPU folder (where this script lives) and log directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/step4_driver_install_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory and file exist and are writable by the invoking user
INVOKER="${SUDO_USER:-$USER}"
sudo mkdir -p "${LOG_DIR}"
sudo touch "${LOG_FILE}"
sudo chown "${INVOKER}:${INVOKER}" "${LOG_FILE}"

echo "📝 Logging to: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Show Secure Boot status (info)
if command -v mokutil >/dev/null 2>&1; then
  SB_STATE=$(mokutil --sb-state 2>/dev/null || true)
  echo "🔐 Secure Boot status: ${SB_STATE:-unknown}"
  echo "   Note: Secure Boot should be disabled or you must enroll MOK for the driver to load."
fi
echo ""

echo "📦 Updating package lists..."
sudo apt-get update -y

# Helper to check if a package has a candidate version
pkg_available() {
  local pkg="$1"
  local candidate
  candidate=$(apt-cache policy "$pkg" | awk '/Candidate:/ {print $2}')
  if [ -n "$candidate" ] && [ "$candidate" != "(none)" ]; then
    return 0
  else
    return 1
  fi
}

# Choose preferred driver package
PREFERRED_PKG=""
if pkg_available nvidia-driver-535-open; then
  PREFERRED_PKG="nvidia-driver-535-open"
elif pkg_available nvidia-driver-550-open; then
  PREFERRED_PKG="nvidia-driver-550-open"
elif pkg_available nvidia-driver-570-open; then
  PREFERRED_PKG="nvidia-driver-570-open"
else
  echo "⚠️  Could not find 535/550/570 open packages via apt. Attempting to use ubuntu-drivers to pick an open driver..."
  # Try to detect a recommended open driver slug and map it to a package
  if command -v ubuntu-drivers >/dev/null 2>&1; then
    SLUG=$(ubuntu-drivers devices 2>/dev/null | awk '/nvidia-driver-[0-9]+-open/ && /recommended/ {print $1; exit}')
    if [ -n "$SLUG" ]; then
      # SLUG format is like "nvidia:550-open" -> package "nvidia-driver-550-open"
      NUM=$(echo "$SLUG" | sed -n 's/^nvidia:\([0-9]\+\)-open/\1/p')
      if [ -n "$NUM" ]; then
        CANDIDATE="nvidia-driver-${NUM}-open"
        if pkg_available "$CANDIDATE"; then
          PREFERRED_PKG="$CANDIDATE"
        fi
      fi
    fi
  fi
fi

if [ -z "$PREFERRED_PKG" ]; then
  echo "❌ No suitable -open driver package found in your current apt sources."
  echo "   Options:"
  echo "   - Run: ubuntu-drivers devices (share output), or"
  echo "   - Add NVIDIA repo that provides 535-open for Ubuntu 24.04, or"
  echo "   - Allow installing the recommended non-open variant (not preferred)."
  exit 2
fi

echo ""
echo "🎯 Selected driver package: ${PREFERRED_PKG}"
echo "   If you prefer a different version, cancel now (Ctrl+C) and edit the script."
sleep 2

echo ""
echo "⬇️  Installing ${PREFERRED_PKG}..."
sudo apt-get install -y "${PREFERRED_PKG}"

echo ""
echo "⚙️  Enabling NVIDIA persistence daemon..."
sudo systemctl enable nvidia-persistenced || true
sudo systemctl start nvidia-persistenced || true
sudo systemctl status nvidia-persistenced --no-pager --lines=3 || true

echo ""
echo "🧪 Quick post-install check (may show limited info until after reboot):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not yet available in PATH (will be after reboot)."
fi

# Ensure the log remains owned by the invoking user
sudo chown "${INVOKER}:${INVOKER}" "${LOG_FILE}" 2>/dev/null || true

echo ""
echo "✅ Driver installation completed for ${PREFERRED_PKG}."
echo "🔁 Rebooting in 5 seconds to load the driver..."
sleep 5
sudo reboot