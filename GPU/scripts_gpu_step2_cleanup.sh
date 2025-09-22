#!/usr/bin/env bash
set -euo pipefail
echo "=== STEP 2: CLEANING UP OLD GPU DRIVERS ==="
echo "Removing any existing NVIDIA drivers and CUDA installations..."
echo "⚠️  Your display may flicker briefly during this process."
echo ""
if ! command -v apt-get >/dev/null 2>&1; then
  echo "❌ This script expects apt-get (Debian/Ubuntu). Aborting."; exit 1
fi
if command -v mokutil >/dev/null 2>&1; then
  SB_STATE=$(mokutil --sb-state 2>/dev/null || true)
  echo "🔐 Secure Boot status: ${SB_STATE:-unknown}"
  echo "   Note: For NVIDIA driver installation (Step 4), Secure Boot should be disabled or you'll need to enroll MOK."
fi
echo ""
read -rp "Type YES to proceed with cleanup (this will purge NVIDIA/CUDA packages): " CONFIRM
if [ "${CONFIRM:-}" != "YES" ]; then echo "Aborted by user."; exit 0; fi
echo ""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
LOG_DIR="${SCRIPT_DIR}/logs"; mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/step2_cleanup_$(date +%Y%m%d_%H%M%S).log"
echo "📝 Logging to: ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1
INVOKER="${SUDO_USER:-$USER}"
echo "📋 Gathering installed NVIDIA/CUDA packages..."
PKGS_LIST="$(dpkg -l | awk '/^ii/ {print $2}' | grep -E '^(cuda(|-.*)|nvidia(|-.*)|libcuda.*|libnvidia.*|libcublas.*|libcurand.*|libcufft.*|libcufile.*|libcusolver.*|libcusparse.*|libnpp.*|libnvjpeg.*|nsight.*)$' || true)"
if [ -n "${PKGS_LIST}" ]; then echo "Found packages to purge:"; echo "${PKGS_LIST}" | sed 's/^/  - /'; else echo "No NVIDIA/CUDA packages currently installed."; fi
readarray -t PKGS <<<"${PKGS_LIST}"
echo ""; echo "🧹 Purging packages (this may take a few minutes)..."
sudo apt-get update -y
if (( ${#PKGS[@]} > 0 )); then sudo apt-get purge -y "${PKGS[@]}"; else echo "(skip purge: none installed)"; fi
echo ""; echo "🧽 Autoremove and autoclean..."
sudo apt-get autoremove -y
sudo apt-get autoclean -y
echo ""; echo "🗑️  Removing leftover CUDA/NVIDIA directories and configs..."
CUDA_DIRS=(/usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*)
APT_LISTS=(/etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list)
MODPROBE_FILES=(/etc/modprobe.d/*nvidia* /etc/modprobe.d/*nouveau*)
shopt -s nullglob
for d in "${CUDA_DIRS[@]}"; do [ -e "$d" ] && echo "  rm -rf $d" && sudo rm -rf "$d"; done
for f in "${APT_LISTS[@]}"; do [ -e "$f" ] && echo "  rm -f $f" && sudo rm -f "$f"; done
for f in "${MODPROBE_FILES[@]}"; do [ -e "$f" ] && echo "  rm -f $f" && sudo rm -f "$f"; done
shopt -u nullglob
echo ""; echo "🔄 Updating linker cache and initramfs..."
sudo ldconfig; sudo update-initramfs -u
echo ""; echo "🔎 Post-clean verification:"
echo "Installed NVIDIA/CUDA packages after purge (should be none):"
dpkg -l | awk '/^ii/ {print $2" "$3}' | grep -E '^(cuda(|-.*)|nvidia(|-.*)|libcuda.*|libnvidia.*|libcublas.*|libcurand.*|libcufft.*|libcufile.*|libcusolver.*|libcusparse.*|libnpp.*|libnvjpeg.*|nsight.*) ' || true
if [ -n "${SUDO_USER:-}" ] && id "${INVOKER}" >/dev/null 2>&1; then chown "${INVOKER}:${INVOKER}" "${LOG_FILE}" 2>/dev/null || true; fi
echo ""; echo "✅ Cleanup complete! System ready for fresh driver installation."
echo "➡️  Next: STEP 3 (System preparation: build tools, headers, blacklist nouveau)."
