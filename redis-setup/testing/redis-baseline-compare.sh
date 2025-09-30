#!/usr/bin/env bash
# Compare current Redis monitor run against a saved baseline thresholds.
# Usage:
#   1) Generate JSON-like simple key=val baseline:
#        bash redis-hft-monitor_v2.sh > latest.txt
#        grep -E "^(  SET|  XADD|  Stream end-to-end)" latest.txt | awk '{print $2,$3,$4,$5,$6}'
#   (Simpler approach implemented: run monitor and parse metrics directly.)
#   2) Define thresholds via env or a baseline file.
#
# Fast mode:
#   REDIS_HOST=127.0.0.1 REDIS_PORT=6379 ./redis-baseline-compare.sh
#   (Runs monitor script, extracts p50/p95/p99 for SET & XADD and compares.)
#
# Environment thresholds (defaults reasonable for current tuning):
#   BASE_SET_P99_MAX=7
#   BASE_XADD_P99_MAX=8
#   BASE_RTT_P99_MAX=20
#   BASE_LAG_P99_MAX=40 (only if CONSUMER_LAG_TEST=1)
#   DELTA_P99_WARN=2   (warn if p99 increases > this over last baseline stored)
#
# A last-run snapshot is stored in: redis-baseline-last.json (simple key=val lines)
#
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MONITOR="$SCRIPT_DIR/../monitoring/redis-hft-monitor_v2.sh"
[ -x "$MONITOR" ] || { echo "Monitor script not executable: $MONITOR"; exit 1; }

BASE_SET_P99_MAX=${BASE_SET_P99_MAX:-7}
BASE_XADD_P99_MAX=${BASE_XADD_P99_MAX:-8}
BASE_RTT_P99_MAX=${BASE_RTT_P99_MAX:-20}
BASE_LAG_P99_MAX=${BASE_LAG_P99_MAX:-40}
DELTA_P99_WARN=${DELTA_P99_WARN:-2}
LAST_FILE="$SCRIPT_DIR/redis-baseline-last.json"

TMP_OUT=$(mktemp)
# Run monitor (suppress color)
NO_COLOR=1 bash "$MONITOR" > "$TMP_OUT" 2>&1 || { echo "Monitor failed"; cat "$TMP_OUT"; rm -f "$TMP_OUT"; exit 1; }

parse_line() {
  # $1 pattern, $2 index (0-based) like p50=XXX
  grep -F "$1" "$TMP_OUT" | head -n1 | tr ' ' '\n' | grep p99= | sed 's/p99=//' || true
}

SET_P99=$(grep -F "SET μs:" "$TMP_OUT" | head -n1 | tr ' ' '\n' | grep p99= | sed 's/p99=//' || true)
XADD_P99=$(grep -F "XADD μs:" "$TMP_OUT" | head -n1 | tr ' ' '\n' | grep p99= | sed 's/p99=//' || true)
RTT_P99=$(grep -F "Standard PING μs:" "$TMP_OUT" | head -n1 | tr ' ' '\n' | grep p99= | sed 's/p99=//' || true)
LAG_P99=$(grep -F "Stream end-to-end lag μs:" "$TMP_OUT" | head -n1 | tr ' ' '\n' | grep p99= | sed 's/p99=//' || true)

fail=0
warn=0

check_metric() {
  local name=$1 val=$2 max=$3
  if [[ -z "$val" ]]; then
    echo "WARN: $name missing"; warn=1; return
  fi
  # Round numeric (strip decimals if integer expected)
  local ival=${val%%.*}
  if (( ival > max )); then
    echo "FAIL: $name p99=$val > threshold $max"
    fail=1
  else
    echo "PASS: $name p99=$val <= $max"
  fi
}

check_metric SET $SET_P99 $BASE_SET_P99_MAX
check_metric XADD $XADD_P99 $BASE_XADD_P99_MAX
check_metric RTT $RTT_P99 $BASE_RTT_P99_MAX
if [[ -n "$LAG_P99" ]]; then
  check_metric LAG $LAG_P99 $BASE_LAG_P99_MAX
fi

# Delta warnings vs last snapshot
if [[ -f "$LAST_FILE" ]]; then
  LAST_SET=$(grep '^SET_P99=' "$LAST_FILE" | cut -d= -f2 || true)
  LAST_XADD=$(grep '^XADD_P99=' "$LAST_FILE" | cut -d= -f2 || true)
  LAST_RTT=$(grep '^RTT_P99=' "$LAST_FILE" | cut -d= -f2 || true)
  LAST_LAG=$(grep '^LAG_P99=' "$LAST_FILE" | cut -d= -f2 || true)
  cmp_delta() {
    local name=$1 cur=$2 prev=$3
    if [[ -n "$cur" && -n "$prev" ]]; then
      local c=${cur%%.*}; local p=${prev%%.*}
      local diff=$(( c - p ))
      if (( diff > 0 && diff > DELTA_P99_WARN )); then
        echo "WARN: $name p99 regression: prev=$p now=$c (+$diff)"
        warn=1
      fi
    fi
  }
  cmp_delta SET $SET_P99 $LAST_SET
  cmp_delta XADD $XADD_P99 $LAST_XADD
  cmp_delta RTT $RTT_P99 $LAST_RTT
  if [[ -n "$LAG_P99" ]]; then cmp_delta LAG $LAG_P99 $LAST_LAG; fi
fi

# Write snapshot
{
  echo "SET_P99=$SET_P99"
  echo "XADD_P99=$XADD_P99"
  echo "RTT_P99=$RTT_P99"
  echo "LAG_P99=$LAG_P99"
  echo "TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$LAST_FILE"

rm -f "$TMP_OUT"

if (( fail == 1 )); then
  exit 2
fi
if (( warn == 1 )); then
  exit 1
fi
exit 0
