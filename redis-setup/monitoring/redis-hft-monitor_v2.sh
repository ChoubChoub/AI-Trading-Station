#!/usr/bin/env bash
# Redis HFT Performance Monitor (V2)
# Purpose: Reflect real Redis performance for HFT by reporting server-side command percentiles
# - No dependencies on Onload wrapper
# - No side effects (read-only except for temporary test keys/streams)
# - Uses Lua EVAL inside Redis to measure microsecond timings (P50/P95/P99)

set -euo pipefail

# Config
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-6379}
COUNT_SET=${COUNT_SET:-10000}     # ops for SET benchmark
COUNT_XADD=${COUNT_XADD:-2000}    # ops for XADD benchmark
STREAM_KEY=${STREAM_KEY:-test:stream}
TMP_PREFIX=${TMP_PREFIX:-test:latency}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $*"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $*"; }

REDIS_CLI=(redis-cli -h "$HOST" -p "$PORT" --raw)
command -v redis-cli >/dev/null 2>&1 || { error "redis-cli not found in PATH"; exit 1; }

echo -e "${BLUE}=== Redis HFT Performance Monitor (V2) ===${NC}"
echo "Date: $(date)"
echo "Target: $HOST:$PORT"
echo ""

# Quick availability check
if ! "${REDIS_CLI[@]}" PING >/dev/null 2>&1; then
  error "Cannot PING Redis at $HOST:$PORT"
  exit 1
fi
log "Redis ONLINE"

# Lua for server-side percentile measurement of SET
read -r -d '' LUA_SET <<'LUA' || true
local count = tonumber(ARGV[1])
local prefix = ARGV[2]
local times = {}
for i = 1, count do
  local t1 = redis.call('TIME')
  local start = tonumber(t1[1]) * 1000000 + tonumber(t1[2])
  redis.call('SET', prefix .. ':' .. i, 'v' .. i)
  local t2 = redis.call('TIME')
  local stop = tonumber(t2[1]) * 1000000 + tonumber(t2[2])
  times[#times+1] = stop - start
end
table.sort(times)
local len = #times
local p50 = times[math.floor(len * 0.50)] or 0
local p95 = times[math.floor(len * 0.95)] or 0
local p99 = times[math.floor(len * 0.99)] or 0
return {p50, p95, p99}
LUA

# Lua for server-side percentile measurement of XADD MAXLEN~
read -r -d '' LUA_XADD <<'LUA' || true
local count = tonumber(ARGV[1])
local stream = ARGV[2]
local times = {}
for i = 1, count do
  local t1 = redis.call('TIME')
  local start = tonumber(t1[1]) * 1000000 + tonumber(t1[2])
  redis.call('XADD', stream, 'MAXLEN', '~', '1000', '*', 'symbol', 'EURUSD', 'price', tostring(1.08 + i/100000), 'seq', tostring(i))
  local t2 = redis.call('TIME')
  local stop = tonumber(t2[1]) * 1000000 + tonumber(t2[2])
  times[#times+1] = stop - start
end
table.sort(times)
local len = #times
local p50 = times[math.floor(len * 0.50)] or 0
local p95 = times[math.floor(len * 0.95)] or 0
local p99 = times[math.floor(len * 0.99)] or 0
redis.call('DEL', stream)
return {p50, p95, p99}
LUA

# Run SET benchmark
log "Measuring server-side SET latency (P50/P95/P99 in μs) ..."
SET_OUT=$("${REDIS_CLI[@]}" EVAL "$LUA_SET" 0 "$COUNT_SET" "$TMP_PREFIX" || true)
if [[ -n "$SET_OUT" ]]; then
  # Expect 3 lines: p50, p95, p99
  mapfile -t SET_VALS <<<"$SET_OUT"
  echo "  SET μs: p50=${SET_VALS[0]} p95=${SET_VALS[1]} p99=${SET_VALS[2]}"
else
  warn "SET benchmark did not return values"
fi

# Run XADD benchmark
log "Measuring server-side XADD MAXLEN~ latency (P50/P95/P99 in μs) ..."
XADD_OUT=$("${REDIS_CLI[@]}" EVAL "$LUA_XADD" 0 "$COUNT_XADD" "$STREAM_KEY" || true)
if [[ -n "$XADD_OUT" ]]; then
  mapfile -t XADD_VALS <<<"$XADD_OUT"
  echo "  XADD μs: p50=${XADD_VALS[0]} p95=${XADD_VALS[1]} p99=${XADD_VALS[2]}"
else
  warn "XADD benchmark did not return values"
fi

  # Optional: Stream end-to-end consumer lag test (producer -> consumer group)
  # Enable with CONSUMER_LAG_TEST=1 (default off to keep baseline fast)
  # Env tunables:
  #   LAG_COUNT (default 1000) number of messages
  #   LAG_STREAM (default test:stream:lag)
  #   LAG_GROUP (default monitor:lag)
  if [[ "${CONSUMER_LAG_TEST:-0}" == "1" ]]; then
    if command -v python3 >/dev/null 2>&1; then
    log "Measuring stream end-to-end consumer lag (XADD -> XREADGROUP) ..."
    LAG_COUNT_VAL=${LAG_COUNT:-1000}
    LAG_STREAM_VAL=${LAG_STREAM:-test:stream:lag}
    LAG_GROUP_VAL=${LAG_GROUP:-monitor:lag}
    LAG_PY_OUT=$(HOST="$HOST" PORT="$PORT" LAG_COUNT="$LAG_COUNT_VAL" LAG_STREAM="$LAG_STREAM_VAL" LAG_GROUP="$LAG_GROUP_VAL" python3 - <<'PYCODE' 2>/dev/null || true)
import os, time, redis, statistics
  host=os.environ['HOST']; port=int(os.environ['PORT'])
  count=int(os.environ['LAG_COUNT']); stream=os.environ['LAG_STREAM']; group=os.environ['LAG_GROUP']
  r=redis.Redis(host=host, port=port, decode_responses=True)
  # Cleanup previous run if exists
  try:
    r.xgroup_destroy(stream, group)
  except Exception:
    pass
  try:
    r.delete(stream)
  except Exception:
    pass
  r.xgroup_create(stream, group, id='$', mkstream=True)
  lat=[]
  # Simple consumer loop in same process (captures server scheduling + delivery, minus inter-process wakeup variability)
  for i in range(count):
    t0=time.time_ns()  # producer timestamp (ns)
    r.xadd(stream, {'ts': str(t0), 'i': str(i)})
    # Read exactly one new message
    msgs=r.xreadgroup(group, 'c1', {stream: '>'}, count=1, block=1000)
    if not msgs:
      continue
    _, entries = msgs[0]
    if not entries:
      continue
    _id, fields = entries[0]
    try:
      ts=int(fields.get('ts','0'))
    except Exception:
      continue
    t1=time.time_ns()
    lat.append((t1 - ts)/1000.0)  # to microseconds
  if len(lat) >= 10:
    lat.sort()
    def pct(p):
      idx=min(int(len(lat)*p), len(lat)-1)
      return lat[idx]
    p50=pct(0.50); p95=pct(0.95); p99=pct(0.99)
    jitter=p99-p50
    print(f"Stream end-to-end lag μs: p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} jitter={jitter:.2f} count={len(lat)}")
  else:
    print("Stream end-to-end lag μs: insufficient_samples")
  # Cleanup stream to avoid buildup
  try:
    r.delete(stream)
  except Exception:
    pass
PYCODE
    if [[ -n "$LAG_PY_OUT" ]]; then
      echo "  $LAG_PY_OUT"
    else
      warn "Consumer lag test produced no output (python redis lib missing?)"
    fi
    else
    warn "Python not available; skipping consumer lag test"
    fi
  fi

# NOTE: OnLoad client path validation removed intentionally.
# This monitor is kept pure (loopback + server-side timing) because Redis
# is used strictly for in-host messaging in this deployment. If external
# NIC path latency comparison is ever needed, reintroduce a guarded block
# controlled by an explicit ENABLE_ONLOAD_CHECK=1 flag.

# Standard client RTT baseline (for comparison) 
if command -v python3 >/dev/null 2>&1; then
  log "Standard client RTT baseline (no OnLoad):"
  PY_RTT=$(REDIS_HOST="$HOST" REDIS_PORT="$PORT" python3 -c "
import os, time, redis
host = os.environ.get('REDIS_HOST','127.0.0.1')
port = int(os.environ.get('REDIS_PORT','6379'))
r = redis.Redis(host=host, port=port, decode_responses=True)
for _ in range(100): r.ping()
N, ts, pc = 1000, [], time.perf_counter
for _ in range(N):
    t0 = pc(); r.ping(); t1 = pc()
    ts.append((t1 - t0) * 1_000_000)
ts.sort()
pct = lambda p: ts[max(0, min(int(len(ts) * p), len(ts)-1))]
print(f'Standard PING μs: p50={pct(0.50):.2f} p95={pct(0.95):.2f} p99={pct(0.99):.2f}')
" 2>/dev/null || echo "N/A")
  
  if [[ -n "$PY_RTT" && "$PY_RTT" != "N/A" ]]; then
    echo "  $PY_RTT"
  else
    warn "Python redis client not available for baseline"
  fi
fi

# Health snapshot
MEM_USED=$("${REDIS_CLI[@]}" INFO memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
CLIENTS=$("${REDIS_CLI[@]}" INFO clients | grep connected_clients | cut -d: -f2 | tr -d '\r')
OPS=$("${REDIS_CLI[@]}" INFO stats | grep instantaneous_ops_per_sec | cut -d: -f2 | tr -d '\r')
echo ""
log "Health Snapshot:"
echo "  Memory Used: $MEM_USED"
echo "  Connected Clients: $CLIENTS"
echo "  Operations/sec: $OPS"

echo ""
echo -e "${BLUE}Done.${NC}"
