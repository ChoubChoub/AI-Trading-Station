#!/usr/bin/env bash
# Redis HFT Monitor JSON Wrapper
# Non-intrusive adapter: runs monitor script and emits structured JSON
# Version: monitor_wrapper_v1

set -euo pipefail

VERSION="monitor_wrapper_v1"
MONITOR_SCRIPT="/opt/redis-hft/scripts/redis-hft-monitor.sh"

# Default options
MODE="invoke"
INPUT_FILE=""
OUTPUT_FILE=""
INCLUDE_RAW_HASH=0
VALIDATE=0
PRETTY=0

print_usage() {
  cat <<'EOF'
Redis HFT Monitor JSON Wrapper
Usage: redis-hft-monitor_to_json.sh [options]

Options:
  --invoke              (default) run the monitor script
  --file <path>         parse existing captured output
  --output <file>       write JSON atomically to file
  --include-raw-hash    include sha256 of raw input
  --validate            validate JSON with python3 if available
  --pretty              pretty-print JSON (multiline)
  --help                show this help

Exit codes:
  0 - Success
  1 - Missing required metrics (SET/XADD)
  2 - Input source failure
  3 - JSON validation failure
  4 - Internal parsing error
  5 - File I/O error
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --invoke) MODE="invoke"; shift ;;
    --file) INPUT_FILE="$2"; MODE="file"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --include-raw-hash) INCLUDE_RAW_HASH=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    --pretty) PRETTY=1; shift ;;
    --help) print_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; print_usage >&2; exit 4 ;;
  esac
done

# Get raw input
get_raw_input() {
  if [[ "$MODE" == "invoke" ]]; then
    if ! "$MONITOR_SCRIPT" 2>&1; then
      echo "ERROR: Monitor script failed with exit code $?" >&2
      exit 2
    fi
  elif [[ "$MODE" == "file" ]]; then
    if [[ ! -f "$INPUT_FILE" ]]; then
      echo "ERROR: Input file not found: $INPUT_FILE" >&2
      exit 5
    fi
    cat "$INPUT_FILE"
  fi
}

# Validate numeric value
is_numeric() {
  [[ $1 =~ ^[0-9]+(\.[0-9]+)?$ ]]
}

# Parse metrics from raw input
parse_metrics() {
  local raw_input="$1"
  local json_parts=()
  
  # Initialize variables
  local set_p50="" set_p95="" set_p99=""
  local xadd_p50="" xadd_p95="" xadd_p99=""
  local rtt_p50="" rtt_p95="" rtt_p99=""
  local mem_used="" clients="" ops_per_sec=""
  local stream_enabled=false
  local stream_p50="" stream_p95="" stream_p99="" stream_jitter="" stream_count=""
  local has_stream_error=false
  
  # Parse SET metrics
  if echo "$raw_input" | grep -E '^  SET μs: p50=([0-9]+) p95=([0-9]+) p99=([0-9]+)$' >/dev/null; then
    set_p50=$(echo "$raw_input" | grep -E '^  SET μs:' | sed -E 's/.*p50=([0-9]+).*/\1/')
    set_p95=$(echo "$raw_input" | grep -E '^  SET μs:' | sed -E 's/.*p95=([0-9]+).*/\1/')
    set_p99=$(echo "$raw_input" | grep -E '^  SET μs:' | sed -E 's/.*p99=([0-9]+).*/\1/')
  fi
  
  # Parse XADD metrics
  if echo "$raw_input" | grep -E '^  XADD μs: p50=([0-9]+) p95=([0-9]+) p99=([0-9]+)$' >/dev/null; then
    xadd_p50=$(echo "$raw_input" | grep -E '^  XADD μs:' | sed -E 's/.*p50=([0-9]+).*/\1/')
    xadd_p95=$(echo "$raw_input" | grep -E '^  XADD μs:' | sed -E 's/.*p95=([0-9]+).*/\1/')
    xadd_p99=$(echo "$raw_input" | grep -E '^  XADD μs:' | sed -E 's/.*p99=([0-9]+).*/\1/')
  fi
  
  # Parse RTT metrics
  if echo "$raw_input" | grep -E '^  Standard PING μs: p50=([0-9.]+) p95=([0-9.]+) p99=([0-9.]+)$' >/dev/null; then
    rtt_p50=$(echo "$raw_input" | grep -E '^  Standard PING μs:' | sed -E 's/.*p50=([0-9.]+).*/\1/')
    rtt_p95=$(echo "$raw_input" | grep -E '^  Standard PING μs:' | sed -E 's/.*p95=([0-9.]+).*/\1/')
    rtt_p99=$(echo "$raw_input" | grep -E '^  Standard PING μs:' | sed -E 's/.*p99=([0-9.]+).*/\1/')
  fi
  
  # Parse health metrics
  if echo "$raw_input" | grep -E '^  Memory Used: ' >/dev/null; then
    mem_used=$(echo "$raw_input" | grep -E '^  Memory Used:' | sed -E 's/^  Memory Used: (.+)$/\1/')
  fi
  
  if echo "$raw_input" | grep -E '^  Connected Clients: ([0-9]+)$' >/dev/null; then
    clients=$(echo "$raw_input" | grep -E '^  Connected Clients:' | sed -E 's/.*: ([0-9]+)$/\1/')
  fi
  
  if echo "$raw_input" | grep -E '^  Operations/sec: ([0-9]+)$' >/dev/null; then
    ops_per_sec=$(echo "$raw_input" | grep -E '^  Operations/sec:' | sed -E 's/.*: ([0-9]+)$/\1/')
  fi
  
  # Check for stream lag test
  if echo "$raw_input" | grep -E 'Measuring stream end-to-end consumer lag' >/dev/null; then
    stream_enabled=true
    if echo "$raw_input" | grep -E 'WARNING.*Consumer lag test produced no output' >/dev/null; then
      has_stream_error=true
    fi
    # Note: We don't have actual stream lag data in our samples, so we handle the error case
  fi
  
  # Validate required metrics
  if [[ -z "$set_p50" || -z "$set_p95" || -z "$set_p99" ]]; then
    echo "ERROR: Missing SET metrics" >&2
    exit 1
  fi
  
  if [[ -z "$xadd_p50" || -z "$xadd_p95" || -z "$xadd_p99" ]]; then
    echo "ERROR: Missing XADD metrics" >&2
    exit 1
  fi
  
  # Validate numeric values
  for val in "$set_p50" "$set_p95" "$set_p99" "$xadd_p50" "$xadd_p95" "$xadd_p99"; do
    if ! is_numeric "$val"; then
      echo "ERROR: Invalid numeric value: $val" >&2
      exit 4
    fi
  done
  
  # Calculate jitters
  local set_jitter=$((set_p99 - set_p50))
  local xadd_jitter=$((xadd_p99 - xadd_p50))
  
  # Build JSON
  local ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  
  # Start JSON object
  printf '{\n'
  printf '  "ts":"%s",\n' "$ts"
  printf '  "mode":"%s",\n' "$MODE"
  printf '  "source_version":"%s",\n' "$VERSION"
  
  # SET metrics
  printf '  "set":{"p50":%s,"p95":%s,"p99":%s,"jitter":%s},\n' "$set_p50" "$set_p95" "$set_p99" "$set_jitter"
  
  # XADD metrics
  printf '  "xadd":{"p50":%s,"p95":%s,"p99":%s,"jitter":%s},\n' "$xadd_p50" "$xadd_p95" "$xadd_p99" "$xadd_jitter"
  
  # RTT metrics
  if [[ -n "$rtt_p50" && -n "$rtt_p95" && -n "$rtt_p99" ]]; then
    printf '  "rtt":{"p50":%s,"p95":%s,"p99":%s},\n' "$rtt_p50" "$rtt_p95" "$rtt_p99"
  else
    printf '  "rtt":{"error":"unavailable"},\n'
  fi
  
  # Stream lag metrics
  if [[ "$stream_enabled" == true ]]; then
    if [[ "$has_stream_error" == true ]]; then
      printf '  "stream_lag":{"enabled":true,"error":"python_redis_missing"},\n'
    else
      # If we had actual stream data, it would go here
      printf '  "stream_lag":{"enabled":true,"error":"no_data"},\n'
    fi
  else
    printf '  "stream_lag":{"enabled":false},\n'
  fi
  
  # Health metrics
  printf '  "health":{"mem_used_human":"%s","clients":%s,"ops_per_sec":%s}' "$mem_used" "$clients" "$ops_per_sec"
  
  # Optional raw hash
  if [[ $INCLUDE_RAW_HASH -eq 1 ]]; then
    local raw_hash=$(echo "$raw_input" | sha256sum | cut -d' ' -f1)
    printf ',\n  "raw_sha256":"%s"' "$raw_hash"
  fi
  
  printf '\n}\n'
}

# Main execution
main() {
  local raw_input
  raw_input=$(get_raw_input)
  
  local json_output
  json_output=$(parse_metrics "$raw_input")
  
  # Validate JSON if requested
  if [[ $VALIDATE -eq 1 ]] && command -v python3 >/dev/null 2>&1; then
    if ! echo "$json_output" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
      echo "ERROR: JSON validation failed" >&2
      exit 3
    fi
  fi
  
  # Output JSON
  if [[ -n "$OUTPUT_FILE" ]]; then
    local temp_file="${OUTPUT_FILE}.tmp.$$"
    echo "$json_output" > "$temp_file" || { echo "ERROR: Failed to write temp file" >&2; exit 5; }
    mv "$temp_file" "$OUTPUT_FILE" || { echo "ERROR: Failed to move temp file" >&2; exit 5; }
  else
    echo "$json_output"
  fi
}

main "$@"