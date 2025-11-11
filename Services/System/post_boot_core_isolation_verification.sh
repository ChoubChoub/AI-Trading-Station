#!/bin/bash
# Run this script after reboot to verify core isolation is working

echo "=== POST-BOOT CORE ISOLATION VERIFICATION ==="
echo "Date: $(date)"
echo "Uptime: $(uptime -p)"
echo ""

# Create post-boot report file
POST_BOOT_REPORT="/var/log/post_boot_core_isolation_$(date +%Y%m%d_%H%M%S).txt"

# Function to log and display
log_and_display() {
    echo "$1"
    echo "$1" >> "$POST_BOOT_REPORT"
}

log_and_display "=== KERNEL COMMAND LINE VERIFICATION ==="
CMDLINE=$(cat /proc/cmdline)
log_and_display "Full kernel command line:"
log_and_display "$CMDLINE"
log_and_display ""

# Check each isolation parameter
check_kernel_param() {
    local param="$1"
    local description="$2"
    
    if echo "$CMDLINE" | grep -q "$param"; then
        PARAM_VALUE=$(echo "$CMDLINE" | grep -o "$param=[^[:space:]]*" | cut -d= -f2)
        log_and_display "✓ $description: $PARAM_VALUE"
        return 0
    else
        log_and_display "✗ $description: NOT FOUND"
        return 1
    fi
}

check_kernel_param "isolcpus" "CPU Isolation"
check_kernel_param "rcu_nocbs" "RCU No-Callbacks"
check_kernel_param "nohz_full" "No-Hz Full"
check_kernel_param "intel_pstate" "Intel P-State"
check_kernel_param "processor.max_cstate" "Max C-State"

log_and_display ""
log_and_display "=== CPU ISOLATION STATUS ==="

# Check isolated CPUs
if [ -f /sys/devices/system/cpu/isolated ]; then
    ISOLATED_CPUS=$(cat /sys/devices/system/cpu/isolated)
    if [ -n "$ISOLATED_CPUS" ]; then
        log_and_display "✓ Isolated CPUs: $ISOLATED_CPUS"
    else
        log_and_display "⚠ No CPUs currently isolated (may be normal - check cmdline)"
    fi
else
    log_and_display "⚠ Isolated CPU status not available"
fi

log_and_display ""
log_and_display "=== CPU GOVERNOR VERIFICATION ==="
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    GOVERNORS=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c)
    log_and_display "CPU Governor distribution:"
    echo "$GOVERNORS" | while read line; do
        log_and_display "  $line"
    done
else
    log_and_display "⚠ CPU frequency scaling not available"
fi

log_and_display ""
log_and_display "=== RUNNING IRQ AFFINITY CONFIGURATION ==="
if sudo /usr/local/bin/configure-nic-irq-affinity.sh >> "$POST_BOOT_REPORT" 2>&1; then
    log_and_display "✓ IRQ affinity configuration completed"
else
    log_and_display "⚠ IRQ affinity configuration had issues (check details above)"
fi

log_and_display ""
log_and_display "=== NETWORK OPTIMIZATION VERIFICATION ==="
OPTIMIZATIONS=(
    "net.core.rmem_max"
    "net.core.wmem_max"
    "net.core.busy_read"
    "net.core.busy_poll"
)

for opt in "${OPTIMIZATIONS[@]}"; do
    VALUE=$(sysctl -n "$opt" 2>/dev/null || echo "not available")
    log_and_display "  $opt: $VALUE"
done

log_and_display ""
log_and_display "=== SYSTEM PERFORMANCE BASELINE ==="

# Run performance validation
log_and_display "Running performance validation..."
if sudo /usr/local/bin/validate-isolation-performance.sh >> "$POST_BOOT_REPORT" 2>&1; then
    log_and_display "✓ Performance validation completed"
else
    log_and_display "⚠ Performance validation had issues"
fi

log_and_display ""
log_and_display "=== POST-BOOT VERIFICATION SUMMARY ==="

# Summary checks
SUMMARY_SCORE=0
SUMMARY_TOTAL=0

check_summary_item() {
    local description="$1"
    local condition="$2"
    
    SUMMARY_TOTAL=$((SUMMARY_TOTAL + 1))
    
    if eval "$condition" >/dev/null 2>&1; then
        log_and_display "✓ $description"
        SUMMARY_SCORE=$((SUMMARY_SCORE + 1))
    else
        log_and_display "✗ $description"
    fi
}

check_summary_item "Kernel isolation parameters present" "grep -q 'isolcpus=2,3' /proc/cmdline"
check_summary_item "Performance governor active" "grep -q 'performance' /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
check_summary_item "IRQ affinity configured" "test -f /var/log/nic-irq-affinity.log"
check_summary_item "System stable and responsive" "test $(cat /proc/loadavg | cut -d' ' -f1 | cut -d'.' -f1) -lt 5"
check_summary_item "Network interface operational" "ip link show | grep -q 'state UP'"

log_and_display ""
log_and_display "Overall Status: $SUMMARY_SCORE/$SUMMARY_TOTAL checks passed"

if [ $SUMMARY_SCORE -eq $SUMMARY_TOTAL ]; then
    log_and_display ""
    log_and_display "🎉 CORE ISOLATION IMPLEMENTATION SUCCESSFUL!"
    log_and_display ""
    log_and_display "Your system is now configured with:"
    log_and_display "  - Cores 2,3 isolated for NIC processing"
    log_and_display "  - Cores 0,1,4,5,6,7 available for AI trading applications"
    log_and_display "  - Performance optimizations active"
    log_and_display "  - Low-latency network configuration applied"
    log_and_display ""
    log_and_display "Next steps:"
    log_and_display "  1. Monitor system performance with: sudo /usr/local/bin/monitor-core-isolation.sh"
    log_and_display "  2. Begin AI trading station development on available cores"
    log_and_display "  3. Test under load and fine-tune as needed"
    log_and_display ""
    log_and_display "If issues arise, rollback with: sudo /usr/local/bin/rollback-core-isolation.sh"
    
    # Enable monitoring service
    if ! systemctl is-enabled nic-irq-affinity.service >/dev/null 2>&1; then
        sudo systemctl enable nic-irq-affinity.service
        log_and_display "✓ IRQ affinity service enabled for future boots"
    fi
    
else
    log_and_display ""
    log_and_display "⚠ SOME ISSUES DETECTED"
    log_and_display ""
    log_and_display "Please review the issues above. Common solutions:"
    log_and_display "  - If IRQ affinity failed: May be normal for some NICs"
    log_and_display "  - If CPU governors not set: Check cpupower service status"
    log_and_display "  - If isolation not showing: Verify GRUB update worked"
    log_and_display ""
    log_and_display "Full logs available in: $POST_BOOT_REPORT"
    log_and_display "For rollback: sudo /usr/local/bin/rollback-core-isolation.sh"
fi

echo ""
echo "Complete verification report saved to: $POST_BOOT_REPORT"
echo ""
echo "Useful commands for ongoing monitoring:"
echo "  sudo /usr/local/bin/monitor-core-isolation.sh    # System monitoring"
echo "  sudo /usr/local/bin/validate-isolation-performance.sh    # Performance testing"
echo "  sudo /usr/local/bin/rollback-core-isolation.sh   # Rollback if needed"
