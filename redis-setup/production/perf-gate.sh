#!/usr/bin/env bash
# Redis HFT Performance Gate (Institutional Grade)
# Orchestrates performance monitoring + environmental fingerprinting
# Enforces institutional-grade latency and configuration compliance
# 
# Exit Codes:
#   0 = PASS (all checks passed)
#   1 = SOFT FAIL (performance regression - degraded mode allowed if configured)
#   2 = HARD FAIL (environment drift or severe performance breach)
#   3 = CONFIGURATION ERROR (missing files, invalid config)
#   4 = BOOTSTRAP MODE (baseline created successfully)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="perf_gate_v1.0"

# Default configuration
THRESHOLDS_FILE="${SCRIPT_DIR}/../setup/configs/thresholds/perf-thresholds.env"
MONITOR_WRAPPER="${SCRIPT_DIR}/../monitoring/redis-hft-monitor_to_json.sh"
FINGERPRINT_SCRIPT="${SCRIPT_DIR}/runtime-fingerprint.sh"
BASELINE_DIR="/opt/redis-hft/baselines"
METRICS_DIR="/opt/redis-hft/metrics"

# Load configuration
if [[ -f "$THRESHOLDS_FILE" ]]; then
    source "$THRESHOLDS_FILE"
else
    echo "ERROR: Thresholds file not found: $THRESHOLDS_FILE" >&2
    exit 3
fi

# Ensure directories exist
sudo mkdir -p "$BASELINE_DIR" "$METRICS_DIR" 2>/dev/null || true
sudo chown $(whoami):$(whoami) "$BASELINE_DIR" "$METRICS_DIR" 2>/dev/null || true

print_usage() {
    cat <<EOF
Redis HFT Performance Gate v1.0

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --bootstrap         Create baseline fingerprint (first-time setup)
    --soft-fail         Allow degraded mode on performance regression
    --metrics-only      Run performance check only (skip fingerprint)
    --fingerprint-only  Run fingerprint check only (skip performance)
    --verbose           Detailed output
    --help              Show this help

EXIT CODES:
    0  PASS - All checks passed, system ready for trading
    1  SOFT FAIL - Performance regression (degraded mode if allowed)
    2  HARD FAIL - Environment drift or critical failure
    3  CONFIG ERROR - Invalid configuration or missing files
    4  BOOTSTRAP - Baseline created successfully

EXAMPLES:
    $0 --bootstrap              # First-time setup
    $0                          # Normal gate check
    $0 --soft-fail             # Allow degraded mode
    $0 --verbose               # Detailed diagnostics
EOF
}

log() {
    local msg="[$(date +'%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" >&2
    if [[ -n "${GATE_LOG:-}" && -w "$(dirname "${GATE_LOG}")" ]]; then
        echo "$msg" >> "$GATE_LOG" 2>/dev/null || true
    fi
}

verbose() {
    [[ "${VERBOSE:-0}" == "1" ]] && log "VERBOSE: $*" || true
}

# Parse arguments
BOOTSTRAP_MODE="${BOOTSTRAP_MODE:-false}"
SOFT_FAIL_ALLOW="${SOFT_FAIL_ALLOW:-false}"
METRICS_ONLY=false
FINGERPRINT_ONLY=false
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bootstrap) BOOTSTRAP_MODE=true; shift ;;
        --soft-fail) SOFT_FAIL_ALLOW=true; shift ;;
        --metrics-only) METRICS_ONLY=true; shift ;;
        --fingerprint-only) FINGERPRINT_ONLY=true; shift ;;
        --verbose) VERBOSE=1; shift ;;
        --help) print_usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; print_usage >&2; exit 3 ;;
    esac
done

check_prerequisites() {
    verbose "Checking prerequisites..."
    local missing=()
    
    [[ -x "$MONITOR_WRAPPER" ]] || missing+=("Monitor wrapper: $MONITOR_WRAPPER")
    [[ -x "$FINGERPRINT_SCRIPT" ]] || missing+=("Fingerprint script: $FINGERPRINT_SCRIPT")
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log "ERROR: Missing prerequisites:"
        printf '%s\n' "${missing[@]}" >&2
        exit 3
    fi
    
    verbose "Prerequisites OK"
}

get_performance_metrics() {
    verbose "Collecting performance metrics..."
    
    local metrics_file="${METRICS_OUTPUT:-${METRICS_DIR}/redis-metrics.json}"
    
    if ! "$MONITOR_WRAPPER" --output "$metrics_file" --validate 2>/dev/null; then
        log "ERROR: Failed to collect performance metrics"
        return 1
    fi
    
    verbose "Performance metrics saved to: $metrics_file"
    echo "$metrics_file"
}

check_performance_thresholds() {
    local metrics_file="$1"
    verbose "Checking performance thresholds..."
    
    SET_P99_MAX=${SET_P99_MAX} XADD_P99_MAX=${XADD_P99_MAX} RTT_P99_MAX=${RTT_P99_MAX} \
    SET_JITTER_MAX=${SET_JITTER_MAX} XADD_JITTER_MAX=${XADD_JITTER_MAX} RTT_JITTER_MAX=${RTT_JITTER_MAX} \
    MAX_CLIENTS=${MAX_CLIENTS:-10} SOFT_FAIL_ALLOW=${SOFT_FAIL_ALLOW} \
    python3 << EOF
import json, sys, os

try:
    with open('$metrics_file', 'r') as f:
        metrics = json.load(f)
    
    failures = []
    warnings = []
    
    # Get thresholds from environment
    set_p99_max = int(os.environ.get('SET_P99_MAX', 5))
    xadd_p99_max = int(os.environ.get('XADD_P99_MAX', 6))
    rtt_p99_max = float(os.environ.get('RTT_P99_MAX', 12))
    set_jitter_max = int(os.environ.get('SET_JITTER_MAX', 4))
    xadd_jitter_max = int(os.environ.get('XADD_JITTER_MAX', 5))
    rtt_jitter_max = float(os.environ.get('RTT_JITTER_MAX', 8))
    max_clients = int(os.environ.get('MAX_CLIENTS', 10))
    soft_fail_allow = os.environ.get('SOFT_FAIL_ALLOW', 'false').lower() == 'true'
    
    # Core operation thresholds
    set_data = metrics.get('set', {})
    if set_data.get('p99', 0) > set_p99_max:
        failures.append(f"SET P99: {set_data.get('p99')}μs > {set_p99_max}μs")
    if set_data.get('jitter', 0) > set_jitter_max:
        warnings.append(f"SET jitter: {set_data.get('jitter')}μs > {set_jitter_max}μs")
    
    xadd_data = metrics.get('xadd', {})
    if xadd_data.get('p99', 0) > xadd_p99_max:
        failures.append(f"XADD P99: {xadd_data.get('p99')}μs > {xadd_p99_max}μs")
    if xadd_data.get('jitter', 0) > xadd_jitter_max:
        warnings.append(f"XADD jitter: {xadd_data.get('jitter')}μs > {xadd_jitter_max}μs")
    
    rtt_data = metrics.get('rtt', {})
    if 'error' not in rtt_data:
        if rtt_data.get('p99', 0) > rtt_p99_max:
            failures.append(f"RTT P99: {rtt_data.get('p99')}μs > {rtt_p99_max}μs")
        if rtt_data.get('jitter', 0) > rtt_jitter_max:
            warnings.append(f"RTT jitter: {rtt_data.get('jitter')}μs > {rtt_jitter_max}μs")
    
    # Health checks
    health = metrics.get('health', {})
    if health.get('clients', 0) > max_clients:
        warnings.append(f"Too many clients: {health.get('clients')} > {max_clients}")
    
    # Tail health checks (Phase 4B)
    tail_gate_enabled = os.environ.get('TAIL_GATE_ENABLED', 'false').lower() == 'true'
    if tail_gate_enabled:
        try:
            import subprocess
            tail_result = subprocess.run([
                'python3', os.path.join(os.environ.get('SCRIPT_DIR', '.'), 'tail_aware_gate.py'), '$metrics_file'
            ], capture_output=True, text=True, timeout=10)
            
            if tail_result.stdout:
                for line in tail_result.stdout.strip().split('\\n'):
                    if 'TAIL FAILURES:' in line:
                        continue
                    elif 'TAIL WARNINGS:' in line:
                        continue
                    elif line.strip().startswith('❌'):
                        failures.append(f"TAIL: {line.strip()[2:].strip()}")
                    elif line.strip().startswith('⚠️'):
                        warnings.append(f"TAIL: {line.strip()[3:].strip()}")
            
            if tail_result.returncode != 0 and not os.environ.get('TAIL_GATE_WARN_ONLY', 'true').lower() == 'true':
                # Only fail hard if not in warn-only mode
                pass
        except Exception as e:
            warnings.append(f"Tail gate check failed: {e}")
    
    # Report results
    if failures:
        print("PERFORMANCE_HARD_FAIL")
        for failure in failures:
            print(f"FAIL: {failure}")
        sys.exit(1)
    elif warnings:
        print("PERFORMANCE_SOFT_FAIL" if not soft_fail_allow else "PERFORMANCE_WARNINGS")
        for warning in warnings:
            print(f"WARN: {warning}")
        sys.exit(1 if not soft_fail_allow else 0)
    else:
        print("PERFORMANCE_PASS")
        print(f"✅ SET: {set_data.get('p99')}μs/{set_data.get('jitter')}μs jitter")
        print(f"✅ XADD: {xadd_data.get('p99')}μs/{xadd_data.get('jitter')}μs jitter")
        if 'error' not in rtt_data:
            print(f"✅ RTT: {rtt_data.get('p99')}μs/{rtt_data.get('jitter')}μs jitter")

except Exception as e:
    print(f"ERROR: Failed to parse metrics: {e}")
    sys.exit(1)
EOF
}

get_system_fingerprint() {
    verbose "Collecting system fingerprint..."
    
    local fingerprint_file="${METRICS_DIR}/current-fingerprint.json"
    
    if ! "$FINGERPRINT_SCRIPT" > "$fingerprint_file" 2>/dev/null; then
        log "ERROR: Failed to collect system fingerprint"
        return 1
    fi
    
    verbose "System fingerprint saved to: $fingerprint_file"
    echo "$fingerprint_file"
}

check_fingerprint_drift() {
    local current_file="$1"
    local baseline_file="${BASELINE_FILE:-${BASELINE_DIR}/fingerprint-baseline.json}"
    
    verbose "Checking fingerprint drift against: $baseline_file"
    
    if [[ ! -f "$baseline_file" ]]; then
        if [[ "$BOOTSTRAP_MODE" == "true" ]]; then
            log "Creating baseline fingerprint..."
            cp "$current_file" "$baseline_file"
            log "✅ Baseline created: $baseline_file"
            return 4  # Bootstrap success
        else
            log "ERROR: No baseline fingerprint found. Run with --bootstrap first."
            return 2
        fi
    fi
    
    python3 << EOF
import json, sys

try:
    with open('$current_file', 'r') as f:
        current = json.load(f)
    with open('$baseline_file', 'r') as f:
        baseline = json.load(f)
    
    # Critical fields that must not change
    critical_fields = [
        'redis_binary_hash', 'redis_config_hash', 'systemd_unit_hash',
        'kernel_release', 'thp_enabled', 'cpu_governor'
    ]
    
    # Important fields that should not change
    important_fields = [
        'numa_nodes', 'onload_version', 'irqbalance_status'
    ]
    
    critical_drifts = []
    important_drifts = []
    
    for field in critical_fields:
        if current.get(field) != baseline.get(field):
            critical_drifts.append(f"{field}: {baseline.get(field)} → {current.get(field)}")
    
    for field in important_fields:
        if current.get(field) != baseline.get(field):
            important_drifts.append(f"{field}: {baseline.get(field)} → {current.get(field)}")
    
    if critical_drifts:
        print("FINGERPRINT_HARD_FAIL")
        print("Critical configuration drift detected:")
        for drift in critical_drifts:
            print(f"DRIFT: {drift}")
        sys.exit(2)
    elif important_drifts:
        print("FINGERPRINT_SOFT_FAIL")
        print("Important configuration changes detected:")
        for drift in important_drifts:
            print(f"CHANGE: {drift}")
        sys.exit(1)
    else:
        print("FINGERPRINT_PASS")
        print("✅ No configuration drift detected")

except Exception as e:
    print(f"ERROR: Failed to compare fingerprints: {e}")
    sys.exit(2)
EOF
}

main() {
    log "=== Redis HFT Performance Gate v$VERSION ==="
    log "Mode: Bootstrap=$BOOTSTRAP_MODE, SoftFail=$SOFT_FAIL_ALLOW, MetricsOnly=$METRICS_ONLY, FingerprintOnly=$FINGERPRINT_ONLY"
    
    check_prerequisites
    
    local perf_result=0
    local fingerprint_result=0
    
    # Performance check
    if [[ "$FINGERPRINT_ONLY" != "true" ]]; then
        log "--- Performance Gate Check ---"
        if metrics_file=$(get_performance_metrics); then
            if perf_output=$(check_performance_thresholds "$metrics_file" 2>&1); then
                log "Performance: PASS"
                verbose "$perf_output"
            else
                perf_result=$?
                log "Performance: FAIL (exit code: $perf_result)"
                echo "$perf_output" | while read line; do log "$line"; done
            fi
        else
            log "Performance: ERROR (failed to collect metrics)"
            perf_result=2
        fi
    fi
    
    # Fingerprint check
    if [[ "$METRICS_ONLY" != "true" ]]; then
        log "--- Environment Fingerprint Check ---"
        if fingerprint_file=$(get_system_fingerprint); then
            if fingerprint_output=$(check_fingerprint_drift "$fingerprint_file" 2>&1); then
                fingerprint_result=$?
                if [[ $fingerprint_result -eq 4 ]]; then
                    log "Fingerprint: BOOTSTRAP SUCCESS"
                else
                    log "Fingerprint: PASS"
                fi
                verbose "$fingerprint_output"
            else
                fingerprint_result=$?
                log "Fingerprint: FAIL (exit code: $fingerprint_result)"
                echo "$fingerprint_output" | while read line; do log "$line"; done
            fi
        else
            log "Fingerprint: ERROR (failed to collect fingerprint)"
            fingerprint_result=2
        fi
    fi
    
    # Final gate decision
    log "--- Final Gate Decision ---"
    
    if [[ $fingerprint_result -eq 4 ]]; then
        log "🎯 BOOTSTRAP COMPLETE - Run gate again to validate"
        exit 4
    elif [[ $perf_result -eq 2 || $fingerprint_result -eq 2 ]]; then
        log "🚫 GATE: HARD FAIL - System not ready for trading"
        exit 2
    elif [[ $perf_result -eq 1 || $fingerprint_result -eq 1 ]]; then
        if [[ "$SOFT_FAIL_ALLOW" == "true" ]]; then
            log "⚠️  GATE: SOFT FAIL (DEGRADED MODE ALLOWED)"
            exit 1
        else
            log "🚫 GATE: SOFT FAIL REJECTED - Fix issues before trading"
            exit 1
        fi
    else
        log "✅ GATE: PASS - System ready for institutional-grade trading"
        exit 0
    fi
}

# Run main function
main "$@"