#!/usr/bin/env bash
# Redis HFT Monitor JSON Wrapper (Enhanced v2)
# Non-intrusive: parses existing monitor output or invokes it.
# Adds: pretty mode, op counts, stream lag parsing, RTT jitter.
# Version: monitor_wrapper_v2

set -euo pipefail
VERSION="monitor_wrapper_v2"
# Use workspace-relative path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="${SCRIPT_DIR}/../Scripts/Runtime/redis-hft-monitor.sh"

# Environment-provided operation counts (fallback defaults)
SET_COUNT=${COUNT_SET:-10000}
XADD_COUNT=${COUNT_XADD:-2000}
RTT_COUNT=${RTT_COUNT:-1000}

MODE="invoke"   # invoke | file
INPUT_FILE=""
OUTPUT_FILE=""
INCLUDE_RAW_HASH=0
VALIDATE=0
PRETTY=0
STRICT=0  # future use

print_usage() {
  cat <<'EOF'
Redis HFT Monitor JSON Wrapper (Enhanced v2)
Usage: redis-hft-monitor_to_json_v2.sh [options]
  --invoke               (default) run monitor script
  --file <path>          parse existing output file
  --output <file>        write JSON atomically to file
  --include-raw-hash     include sha256 of input and line count
  --validate             json.loads() validation if python3 available
  --pretty               pretty-print JSON (if python3 available)
  --help                 show this help

Exit Codes:
 0 success | 1 missing required metrics | 2 monitor failure | 3 json validation fail
 4 internal parse error | 5 file I/O error
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --invoke) MODE="invoke"; shift ;;
    --file) INPUT_FILE="$2"; MODE="file"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --include-raw-hash) INCLUDE_RAW_HASH=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    --pretty) PRETTY=1; shift ;;
    --help) print_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 4 ;;
  esac
done

get_raw() {
  if [[ $MODE == invoke ]]; then
    if ! "$MONITOR_SCRIPT" 2>&1; then
      echo "ERROR: Monitor script failed" >&2; exit 2
    fi
  else
    [[ -f $INPUT_FILE ]] || { echo "ERROR: File not found: $INPUT_FILE" >&2; exit 5; }
    cat "$INPUT_FILE"
  fi
}

is_num() { [[ $1 =~ ^[0-9]+(\.[0-9]+)?$ ]]; }

parse() {
  local raw="$1"
  local set_p50="" set_p95="" set_p99=""
  local xadd_p50="" xadd_p95="" xadd_p99=""
  local rtt_p50="" rtt_p95="" rtt_p99="" rtt_jitter=""
  local mem_used="" clients="" ops_per_sec=""
  local stream_enabled=false
  local stream_p50="" stream_p95="" stream_p99="" stream_jitter="" stream_count=""
  local stream_error=""

  # Patterns accept either μs or us
  local pat_time_unit='(μs|us)'

  # Extract SET
  local set_line
  set_line=$(echo "$raw" | grep -E "^  SET ${pat_time_unit}: p50=") || true
  if [[ -n $set_line ]]; then
    set_p50=$(sed -E 's/.*p50=([0-9]+).*/\1/' <<<"$set_line")
    set_p95=$(sed -E 's/.*p95=([0-9]+).*/\1/' <<<"$set_line")
    set_p99=$(sed -E 's/.*p99=([0-9]+).*/\1/' <<<"$set_line")
  fi

  # Extract XADD
  local xadd_line
  xadd_line=$(echo "$raw" | grep -E "^  XADD ${pat_time_unit}: p50=") || true
  if [[ -n $xadd_line ]]; then
    xadd_p50=$(sed -E 's/.*p50=([0-9]+).*/\1/' <<<"$xadd_line")
    xadd_p95=$(sed -E 's/.*p95=([0-9]+).*/\1/' <<<"$xadd_line")
    xadd_p99=$(sed -E 's/.*p99=([0-9]+).*/\1/' <<<"$xadd_line")
  fi

  # Extract RTT
  local rtt_line
  rtt_line=$(echo "$raw" | grep -E "^  Standard PING ${pat_time_unit}: p50=") || true
  if [[ -n $rtt_line ]]; then
    rtt_p50=$(sed -E 's/.*p50=([0-9.]+).*/\1/' <<<"$rtt_line")
    rtt_p95=$(sed -E 's/.*p95=([0-9.]+).*/\1/' <<<"$rtt_line")
    rtt_p99=$(sed -E 's/.*p99=([0-9.]+).*/\1/' <<<"$rtt_line")
    if is_num "$rtt_p50" && is_num "$rtt_p99"; then
      # RTT can be float; use awk for accurate subtraction
      rtt_jitter=$(awk -v a="$rtt_p99" -v b="$rtt_p50" 'BEGIN{printf "%.2f", (a-b)}')
    fi
  fi

  # Extract stream lag (full metrics line if present)
  local stream_line
  stream_line=$(echo "$raw" | grep -E "^  Stream end-to-end lag ${pat_time_unit}: p50=") || true
  if [[ -n $stream_line ]]; then
    stream_enabled=true
    if grep -q 'insufficient_samples' <<<"$stream_line"; then
      stream_error="insufficient_samples"
    else
      stream_p50=$(sed -E 's/.*p50=([0-9.]+).*/\1/' <<<"$stream_line")
      stream_p95=$(sed -E 's/.*p95=([0-9.]+).*/\1/' <<<"$stream_line")
      stream_p99=$(sed -E 's/.*p99=([0-9.]+).*/\1/' <<<"$stream_line")
      stream_jitter=$(sed -E 's/.*jitter=([0-9.]+).*/\1/' <<<"$stream_line")
      stream_count=$(sed -E 's/.*count=([0-9]+).*/\1/' <<<"$stream_line")
      [[ -z $stream_count ]] && stream_error="no_count"
    fi
  else
    # Check if stream lag was attempted but failed
    if grep -q 'Measuring stream end-to-end consumer lag' <<<"$raw"; then
      stream_enabled=true
      if grep -q 'WARNING.*Consumer lag test produced no output' <<<"$raw"; then
        stream_error="python_redis_missing"
      elif grep -q 'python_redis_missing' <<<"$raw"; then
        stream_error="python_redis_missing"
      elif grep -q 'no_data' <<<"$raw"; then
        stream_error="no_data"
      else
        stream_error="unknown_error"
      fi
    fi
  fi

  # Health
  mem_used=$(echo "$raw" | grep -E '^  Memory Used:' | head -n1 | sed -E 's/^  Memory Used: (.+)$/\1/' || true)
  clients=$(echo "$raw" | grep -E '^  Connected Clients:' | head -n1 | sed -E 's/.*: ([0-9]+)$/\1/' || true)
  ops_per_sec=$(echo "$raw" | grep -E '^  Operations/sec:' | head -n1 | sed -E 's/.*: ([0-9]+)$/\1/' || true)

  # Required metric check
  if [[ -z $set_p50 || -z $set_p95 || -z $set_p99 ]]; then echo "ERROR: Missing SET metrics" >&2; exit 1; fi
  if [[ -z $xadd_p50 || -z $xadd_p95 || -z $xadd_p99 ]]; then echo "ERROR: Missing XADD metrics" >&2; exit 1; fi

  # Validate numerics
  for v in $set_p50 $set_p95 $set_p99 $xadd_p50 $xadd_p95 $xadd_p99; do
    is_num "$v" || { echo "ERROR: Non-numeric metric: $v" >&2; exit 4; }
  done

  local set_jitter=$((set_p99-set_p50))
  local xadd_jitter=$((xadd_p99-xadd_p50))

  local ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  # JSON build (compact)
  {
    printf '{\n'
    printf '  "ts":"%s",\n' "$ts"
    printf '  "mode":"%s",\n' "$MODE"
    printf '  "source_version":"%s",\n' "$VERSION"
  printf '  "set":{"p50":%s,"p95":%s,"p99":%s,"jitter":%s,"count":%s},\n' "$set_p50" "$set_p95" "$set_p99" "$set_jitter" "$SET_COUNT"
  printf '  "xadd":{"p50":%s,"p95":%s,"p99":%s,"jitter":%s,"count":%s},\n' "$xadd_p50" "$xadd_p95" "$xadd_p99" "$xadd_jitter" "$XADD_COUNT"
    if [[ -n $rtt_p50 && -n $rtt_p95 && -n $rtt_p99 ]]; then
      # Guard against p99 < p50 (should not happen); if so zero jitter
      if is_num "$rtt_p50" && is_num "$rtt_p99"; then
        if awk -v a="$rtt_p99" -v b="$rtt_p50" 'BEGIN{exit (a>=b)}'; then
          rtt_jitter="0.00"
        fi
      fi
      printf '  "rtt":{"p50":%s,"p95":%s,"p99":%s,"jitter":%s,"count":%s},\n' "$rtt_p50" "$rtt_p95" "$rtt_p99" "${rtt_jitter:-0}" "$RTT_COUNT"
    else
      printf '  "rtt":{"error":"unavailable"},\n'
    fi
    if [[ $stream_enabled == true ]]; then
      if [[ -n $stream_error ]]; then
        printf '  "stream_lag":{"enabled":true,"error":"%s"},\n' "$stream_error"
      else
        printf '  "stream_lag":{"enabled":true,"p50":%s,"p95":%s,"p99":%s,"jitter":%s,"count":%s},\n' "$stream_p50" "$stream_p95" "$stream_p99" "$stream_jitter" "$stream_count"
      fi
    else
      printf '  "stream_lag":{"enabled":false},\n'
    fi
    if [[ -n $mem_used && -n $clients && -n $ops_per_sec ]]; then
      # Get Redis 8.2 specific metrics
      peak_time=$(redis-cli INFO memory 2>/dev/null | grep used_memory_peak_time | cut -d: -f2 | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'N/A')
      dynamic_hz=$(redis-cli CONFIG GET dynamic-hz 2>/dev/null | tail -1 || echo 'unknown')
      current_hz=$(redis-cli CONFIG GET hz 2>/dev/null | tail -1 || echo 'unknown')
      
      printf '  "health":{"mem_used_human":"%s","clients":%s,"ops_per_sec":%s,"redis_8_2":{"peak_memory_time":"%s","dynamic_hz":"%s","current_hz":"%s"}}' "$mem_used" "$clients" "$ops_per_sec" "$peak_time" "$dynamic_hz" "$current_hz"
    else
      printf '  "health":{"error":"missing"}'
    fi
    if [[ $INCLUDE_RAW_HASH -eq 1 ]]; then
      local raw_hash lines
      raw_hash=$(echo "$raw" | sha256sum | cut -d' ' -f1)
      lines=$(echo "$raw" | wc -l | tr -d ' ')
      printf ',\n  "raw_sha256":"%s",\n  "raw_lines":%s' "$raw_hash" "$lines"
    fi
    printf '\n}\n'
  }
}

main() {
  local raw
  raw=$(get_raw)
  local json
  json=$(parse "$raw")

  if [[ $VALIDATE -eq 1 && $PRETTY -eq 0 && $(command -v python3 || true) ]]; then
    echo "$json" | python3 -c 'import sys,json;json.load(sys.stdin)' >/dev/null || { echo "ERROR: JSON validation failed" >&2; exit 3; }
  fi
  # Pretty-print if requested
  if [[ $PRETTY -eq 1 && $(command -v python3 || true) ]]; then
    json=$(echo "$json" | python3 -c 'import sys,json;print(json.dumps(json.load(sys.stdin), indent=2, sort_keys=False))') || { echo "ERROR: JSON pretty failed" >&2; exit 3; }
  fi

  if [[ -n $OUTPUT_FILE ]]; then
    local tmp="${OUTPUT_FILE}.tmp.$$"
    echo "$json" > "$tmp" || { echo "ERROR: temp write failed" >&2; exit 5; }
    mv "$tmp" "$OUTPUT_FILE" || { echo "ERROR: atomic move failed" >&2; exit 5; }
  else
    echo "$json"
  fi
}

main "$@"
