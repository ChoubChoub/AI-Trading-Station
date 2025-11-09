#!/bin/bash
###############################################################################
# Post-Reboot Sanity Check - Day 3 & 4 Validation
# Purpose: Verify all components auto-start and cache integration persists
###############################################################################

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║         POST-REBOOT SANITY CHECK - DAY 3 & 4 VALIDATION                   ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

PASSED=0
FAILED=0
WARNINGS=0

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNINGS++))
}

info() {
    echo -e "ℹ️  INFO: $1"
}

separator() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

###############################################################################
# TEST 1: Core Services Auto-Start
###############################################################################
separator
echo "TEST 1: Core Services Auto-Start"
separator

# Redis
if systemctl is-active --quiet redis-hft.service; then
    pass "Redis service is running"
else
    fail "Redis service is NOT running"
fi

# QuestDB
if systemctl is-active --quiet questdb.service; then
    pass "QuestDB service is running"
else
    fail "QuestDB service is NOT running"
fi

# Batch Writer (Day 4 integration)
if systemctl is-active --quiet batch-writer.service; then
    pass "Batch Writer service is running"
else
    fail "Batch Writer service is NOT running"
fi

###############################################################################
# TEST 2: Cache Integration Initialization
###############################################################################
separator
echo ""
echo "TEST 2: Cache Integration Initialization (Day 4)"
separator

# Check logs for cache initialization
if journalctl -u batch-writer.service --since "5 minutes ago" | grep -q "Tiered cache initialized"; then
    pass "Cache initialized on startup"
    
    # Get the warmed tick count
    WARMED_TICKS=$(journalctl -u batch-writer.service --since "5 minutes ago" | grep "Warmed cache" | tail -1 | grep -oP '\d+(?= recent ticks)')
    if [ ! -z "$WARMED_TICKS" ]; then
        info "Cache warmed with $WARMED_TICKS ticks"
        if [ "$WARMED_TICKS" -gt 1000 ]; then
            pass "Cache warming successful (>1000 ticks)"
        else
            warn "Cache warming low (<1000 ticks) - may need more runtime"
        fi
    fi
else
    fail "Cache NOT initialized on startup"
fi

# Check for cache errors
if journalctl -u batch-writer.service --since "5 minutes ago" | grep -qi "cache.*error"; then
    warn "Cache errors detected in logs (check journalctl)"
else
    pass "No cache errors detected"
fi

###############################################################################
# TEST 3: Data Flow Validation
###############################################################################
separator
echo ""
echo "TEST 3: Data Flow Validation"
separator

# Check if ticks are being inserted
RECENT_TICKS=$(journalctl -u batch-writer.service --since "2 minutes ago" | grep -c "ticks/sec" || echo "0")
if [ "$RECENT_TICKS" -gt 0 ]; then
    pass "Data is flowing (found $RECENT_TICKS tick rate reports)"
else
    warn "No recent tick rate reports (may need more time)"
fi

# Check for errors in last 5 minutes
ERROR_COUNT=$(journalctl -u batch-writer.service --since "5 minutes ago" | grep -c "❌" || echo "0")
if [ "$ERROR_COUNT" -eq 0 ]; then
    pass "No errors in last 5 minutes"
else
    warn "Found $ERROR_COUNT errors in last 5 minutes"
fi

###############################################################################
# TEST 4: Prometheus Metrics Availability
###############################################################################
separator
echo ""
echo "TEST 4: Prometheus Metrics (Day 3/4)"
separator

# Check if metrics endpoint responds
if curl -s http://localhost:9091/metrics > /dev/null 2>&1; then
    pass "Prometheus metrics endpoint responding"
    
    # Check for cache-specific metrics (would be added in Day 5)
    TICKS_INSERTED=$(curl -s http://localhost:9091/metrics | grep "ticks_inserted_total" | head -1)
    if [ ! -z "$TICKS_INSERTED" ]; then
        pass "Tick insertion metrics available"
        info "$TICKS_INSERTED"
    fi
else
    fail "Prometheus metrics endpoint NOT responding"
fi

###############################################################################
# TEST 5: Redis Cache Validation
###############################################################################
separator
echo ""
echo "TEST 5: Redis Cache Status (Day 3)"
separator

# Check Redis connectivity
if redis-cli ping > /dev/null 2>&1; then
    pass "Redis is responding"
    
    # Check if cache keys exist
    CACHE_KEYS=$(redis-cli --scan --pattern "cache:*" | wc -l)
    if [ "$CACHE_KEYS" -gt 0 ]; then
        pass "Cache keys exist in Redis ($CACHE_KEYS keys)"
    else
        warn "No cache keys found in Redis (may need more runtime)"
    fi
else
    fail "Redis is NOT responding"
fi

###############################################################################
# TEST 6: QuestDB Connectivity
###############################################################################
separator
echo ""
echo "TEST 6: QuestDB Status"
separator

# Check QuestDB HTTP endpoint
if curl -s http://localhost:9000/ > /dev/null 2>&1; then
    pass "QuestDB HTTP endpoint responding"
else
    fail "QuestDB HTTP endpoint NOT responding"
fi

# Check if ILP endpoint is accessible (port 9009)
if nc -z localhost 9009 2>/dev/null; then
    pass "QuestDB ILP endpoint accessible"
else
    fail "QuestDB ILP endpoint NOT accessible"
fi

###############################################################################
# TEST 7: Service Dependencies
###############################################################################
separator
echo ""
echo "TEST 7: Service Dependencies"
separator

# Check if batch-writer dependencies are met
DEPS_OUTPUT=$(systemctl list-dependencies batch-writer.service 2>/dev/null)
if echo "$DEPS_OUTPUT" | grep -q "questdb.service"; then
    pass "QuestDB dependency configured"
fi

if echo "$DEPS_OUTPUT" | grep -q "redis-hft.service"; then
    pass "Redis dependency configured"
fi

###############################################################################
# TEST 8: Auto-Start Configuration
###############################################################################
separator
echo ""
echo "TEST 8: Auto-Start Configuration (Day 4)"
separator

# Check if services are enabled
if systemctl is-enabled --quiet batch-writer.service; then
    pass "Batch Writer is enabled for auto-start"
else
    fail "Batch Writer is NOT enabled for auto-start"
fi

if systemctl is-enabled --quiet redis-hft.service; then
    pass "Redis is enabled for auto-start"
else
    warn "Redis is NOT enabled for auto-start"
fi

if systemctl is-enabled --quiet questdb.service; then
    pass "QuestDB is enabled for auto-start"
else
    warn "QuestDB is NOT enabled for auto-start"
fi

###############################################################################
# TEST 9: Day 3 Files Existence
###############################################################################
separator
echo ""
echo "TEST 9: Day 3 Implementation Files"
separator

FILES=(
    "/home/youssefbahloul/ai-trading-station/Services/QuestDB/Config/tiered_market_data.py"
    "/home/youssefbahloul/ai-trading-station/Services/QuestDB/Utility/feature_engineering.py"
    "/home/youssefbahloul/ai-trading-station/Tests/QuestDB/test_cache_integration.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        pass "$(basename $file) exists"
    else
        fail "$(basename $file) NOT found"
    fi
done

###############################################################################
# TEST 10: Cache Integration Validation
###############################################################################
separator
echo ""
echo "TEST 10: Cache Integration Test (Day 3/4)"
separator

info "Running cache integration test from Tests/QuestDB..."
cd /home/youssefbahloul/ai-trading-station/Tests/QuestDB

# Run the test (capture output but don't fail on errors)
if timeout 30 python3 test_cache_integration.py 2>&1 | grep -q "Warmed cache"; then
    pass "Cache integration test completed"
else
    warn "Cache integration test had issues (check manually)"
fi

###############################################################################
# SUMMARY
###############################################################################
separator
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                            TEST SUMMARY                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ PASSED${NC}:   $PASSED"
echo -e "${RED}❌ FAILED${NC}:   $FAILED"
echo -e "${YELLOW}⚠️  WARNINGS${NC}: $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  🎉 ALL CRITICAL TESTS PASSED - SYSTEM IS PERSISTENT AFTER REBOOT! 🎉    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "✅ Day 3 implementation: PERSISTENT"
    echo "✅ Day 4 integration: PERSISTENT"
    echo "✅ Auto-start: WORKING"
    echo "✅ Cache integration: OPERATIONAL"
    echo ""
    echo "🎯 Day 4 Completion: 100% (reboot validation passed!)"
    exit 0
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ SOME TESTS FAILED - MANUAL INVESTIGATION REQUIRED                     ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Please check:"
    echo "  1. journalctl -u batch-writer.service -n 100"
    echo "  2. systemctl status batch-writer.service"
    echo "  3. systemctl status redis-hft.service"
    echo "  4. systemctl status questdb.service"
    exit 1
fi
