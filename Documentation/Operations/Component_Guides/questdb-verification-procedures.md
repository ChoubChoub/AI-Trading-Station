# QuestDB Legacy Scripts Verification Report
**Date:** October 30, 2025  
**Purpose:** Verify accuracy of legacy classifications before migration  
**Status:** ✅ VERIFIED with corrections

---

## Executive Summary

**Original Classification:** 7 files marked as legacy  
**After Verification:** 5 files are truly legacy, 2 classifications need correction

**Critical Finding:** `QuestDB/scripts/redis_health_monitor.py` is NOT a simple duplicate - it's an **older version** that differs significantly from the production version.

---

## Detailed Verification Results

### ✅ VERIFIED AS LEGACY (5 files)

#### 1. `scripts/redis_to_questdb.py`
- **Classification:** ✅ CORRECT - Legacy
- **Rationale:** Superseded by v2
- **Evidence:**
  ```bash
  # Production service uses v2:
  ExecStart=/usr/bin/python3 /home/youssefbahloul/ai-trading-station/QuestDB/scripts/redis_to_questdb_v2.py
  ```
- **Service:** batch-writer.service (active, running since 20:23:02)
- **Action:** ✅ Safe to archive

#### 2. `scripts/redis_to_questdb_v2_day1_backup.py`
- **Classification:** ✅ CORRECT - Legacy
- **Rationale:** Backup copy from Day 1
- **Evidence:** Filename indicates backup, service uses `redis_to_questdb_v2.py`
- **Additional backups found:**
  - `redis_to_questdb_v2.py.backup_20251026_192449`
  - `redis_to_questdb_v2.py.backup_timezone_fix_20251026_195415`
- **Action:** ✅ Safe to archive (along with other .backup files)

#### 3. ⚠️ **`scripts/redis_health_monitor.py` - CLASSIFICATION NEEDS CORRECTION**
- **Original Classification:** "Moved to Monitoring/"
- **❌ INCORRECT RATIONALE** - This is NOT a simple move/copy
- **✅ CORRECT CLASSIFICATION:** Legacy - Older version superseded by enhanced version
- **Evidence:**
  ```
  QuestDB version:    113 lines, 4.2KB, modified Oct 27 11:40
  Monitoring version: 157 lines, 6.5KB, modified Oct 28 01:00 (NEWER)
  
  QuestDB header:   "Redis Connection Health Monitor"
  Monitoring header: "Redis Connection Health Monitor with Prometheus Metrics"
                    "Exports metrics to Prometheus for Grafana visualization"
  ```
- **Production Use:**
  ```bash
  Service: redis-health-monitor.service (PID 2451, active)
  WorkingDirectory: /home/youssefbahloul/ai-trading-station/Monitoring/Scripts/Runtime
  ExecStart: /usr/bin/python3 redis_health_monitor.py
  ```
- **Key Differences:**
  - Monitoring version has **Prometheus metrics integration** (new feature)
  - Monitoring version is **44 lines longer** (38% more code)
  - Files are **different**, not copies (diff shows differences)
  - Monitoring version is **newer** (Oct 28 vs Oct 27)
- **Corrected Rationale:** "Older version without Prometheus metrics - superseded by enhanced version in Monitoring/ with metrics export capability"
- **Action:** ✅ Safe to archive (it's the old version)

#### 4. `scripts/redis_health_monitor_with_alerts.py`
- **Classification:** ✅ CORRECT - Legacy
- **Rationale:** Old version superseded by newer monitoring
- **Evidence:**
  - 11KB file (larger than both other versions)
  - Modified Oct 27 11:40 (same date as basic version)
  - Not used by any service
  - Different approach (alerts instead of Prometheus metrics)
- **Action:** ✅ Safe to archive

#### 5. `scripts/create_table_pg.py`
- **Classification:** ✅ CORRECT - Legacy
- **Rationale:** PostgreSQL legacy - Old PostgreSQL table creation
- **Evidence:**
  ```python
  # Uses PostgreSQL wire protocol (port 8812)
  conn = await asyncpg.connect(host='localhost', port=8812, ...)
  # Creates table via SQL: "DROP TABLE IF EXISTS market_trades;"
  ```
- **Current Approach:** ILP (InfluxDB Line Protocol) via HTTP, not PostgreSQL wire protocol
- **References:** Only found in cleanup script: `phase2_archive_cleanup.sh`
- **Action:** ✅ Safe to archive

---

### ✅ VERIFIED AS LEGACY (Backups/Archives - 2 items)

#### 6. `backups/schema_metadata_20251020_203855.json`
- **Classification:** ✅ CORRECT - Legacy
- **Rationale:** Old schema backup from October 20
- **Evidence:**
  ```bash
  Directory contents:
  - backup_existing_data.sh (utility script)
  - backup_summary_20251020_203855.txt
  - crypto_ticks_backup_20251020_203855.csv
  - perf_test_backup_20251020_203855.csv
  - schema_metadata_20251020_203855.json (466 bytes)
  ```
- **Age:** 10 days old
- **Action:** ✅ Safe to archive

#### 7. `data.corrupted_20251024/`
- **Classification:** ✅ CORRECT - Legacy
- **Rationale:** Corrupted data archive from October 24
- **Evidence:**
  ```bash
  Location: /home/youssefbahloul/ai-trading-station/QuestDB/data.corrupted_20251024
  ```
- **Purpose:** Forensics/recovery reference
- **Age:** 6 days old
- **Action:** ✅ Safe to archive (or delete if recovery no longer needed)

---

## Additional Legacy Items Found (Not in Original List)

### Backup Files in scripts/ Directory
- `exchange_to_redis.py.backup_20251019_191454`
- `tiered_market_data.py.backup_20251026_080748`
- `tiered_market_data.py.backup_orjson_20251027_084009`
- `redis_to_questdb_v2.py.backup_20251026_192449`
- `redis_to_questdb_v2.py.backup_timezone_fix_20251026_195415`

**Action:** Should also be archived

---

## Summary Statistics

| Status | Count | Files |
|--------|-------|-------|
| ✅ Verified Legacy | 5 | redis_to_questdb.py, redis_to_questdb_v2_day1_backup.py, redis_health_monitor.py (older), redis_health_monitor_with_alerts.py, create_table_pg.py |
| ✅ Verified Legacy (Archives) | 2 | schema_metadata backup, data.corrupted_20251024/ |
| ⚠️ Classification Correction Needed | 1 | redis_health_monitor.py rationale |
| 📦 Additional Backups Found | 5 | Various .backup files |
| **TOTAL** | **13** | All safe to archive |

---

## Corrected Classification Table

| # | Script | Original Rationale | ✅ Corrected Rationale | Status |
|---|--------|-------------------|----------------------|--------|
| 1 | `redis_to_questdb.py` | Superseded by v2 | ✅ CORRECT | Verified |
| 2 | `redis_to_questdb_v2_day1_backup.py` | Backup copy | ✅ CORRECT | Verified |
| 3 | `redis_health_monitor.py` | ❌ "Moved to Monitoring/" | ✅ **"Older version (113 lines) without Prometheus metrics - superseded by enhanced version in Monitoring/ (157 lines) with metrics export"** | **Corrected** |
| 4 | `redis_health_monitor_with_alerts.py` | Old version | ✅ CORRECT (alerts approach deprecated) | Verified |
| 5 | `create_table_pg.py` | PostgreSQL legacy | ✅ CORRECT (now using ILP instead) | Verified |
| 6 | `schema_metadata_20251020_203855.json` | Old schema backup | ✅ CORRECT | Verified |
| 7 | `data.corrupted_20251024/` | Corrupted data archive | ✅ CORRECT | Verified |

---

## Key Insight: redis_health_monitor.py

**Why the correction matters:**

**Original rationale** ("Moved to Monitoring/") implies:
- Files are identical copies
- One location superseded another
- Simple relocation

**Actual situation:**
- Files are **different versions** (113 vs 157 lines)
- Monitoring version has **new features** (Prometheus metrics)
- QuestDB version is **older** (Oct 27) vs Monitoring (Oct 28)
- Production uses the **newer enhanced version**

**Impact:**
- Original rationale suggests both files have same functionality
- Corrected rationale clarifies it's an **older version** that was **enhanced**
- Important for understanding system evolution
- Prevents confusion during migration ("why are there two different files?")

---

## Production Services Verification

### Services Using QuestDB Scripts

| Service | Status | Script Used | Location |
|---------|--------|-------------|----------|
| **batch-writer.service** | ✅ Running | `redis_to_questdb_v2.py` | QuestDB/scripts/ |
| **redis-health-monitor.service** | ✅ Running (PID 2451) | `redis_health_monitor.py` | Monitoring/Scripts/Runtime/ |

**Conclusion:** Both production services use the **correct non-legacy versions**. All 7 legacy files are safe to archive.

---

## Archive Plan

### Phase 1: Create Archive Structure
```bash
mkdir -p QuestDB/Archive/Legacy/{scripts,backups,data}
```

### Phase 2: Move Legacy Scripts (5 files)
```bash
mv QuestDB/scripts/redis_to_questdb.py QuestDB/Archive/Legacy/scripts/
mv QuestDB/scripts/redis_to_questdb_v2_day1_backup.py QuestDB/Archive/Legacy/scripts/
mv QuestDB/scripts/redis_health_monitor.py QuestDB/Archive/Legacy/scripts/
mv QuestDB/scripts/redis_health_monitor_with_alerts.py QuestDB/Archive/Legacy/scripts/
mv QuestDB/scripts/create_table_pg.py QuestDB/Archive/Legacy/scripts/
```

### Phase 3: Move Backup Files (7 files)
```bash
mv QuestDB/scripts/*.backup* QuestDB/Archive/Legacy/scripts/
mv QuestDB/backups/schema_metadata_20251020_203855.json QuestDB/Archive/Legacy/backups/
```

### Phase 4: Move Corrupted Data (1 directory)
```bash
mv QuestDB/data.corrupted_20251024 QuestDB/Archive/Legacy/data/
```

### Phase 5: Create README
```bash
cat > QuestDB/Archive/Legacy/README.md <<'EOF'
# QuestDB Legacy Files Archive
Archived: October 30, 2025

## Scripts (5 files)
- redis_to_questdb.py - Superseded by v2
- redis_to_questdb_v2_day1_backup.py - Day 1 backup
- redis_health_monitor.py - Old version (113 lines), superseded by Monitoring/ version with Prometheus (157 lines)
- redis_health_monitor_with_alerts.py - Alerts approach deprecated
- create_table_pg.py - PostgreSQL wire protocol approach, now using ILP

## Backup Files (7 files)
- Various .backup files from script iterations

## Data (1 directory)
- data.corrupted_20251024/ - Corrupted data from October 24

All files verified safe to archive - no production dependencies.
EOF
```

---

## Validation Checklist

**Before archiving:**
- [x] Verify batch-writer.service uses `redis_to_questdb_v2.py`
- [x] Verify redis-health-monitor.service uses Monitoring version
- [x] Confirm no scripts reference legacy files
- [x] Check no cron jobs use legacy scripts
- [x] Verify systemd services don't reference legacy paths

**After archiving:**
- [ ] Test batch-writer.service starts successfully
- [ ] Test redis-health-monitor.service starts successfully  
- [ ] Run datafeed test to ensure no broken imports
- [ ] Check Grafana metrics still populate

---

## Conclusion

**Overall Assessment:** ✅ All 7 files correctly classified as legacy with one rationale correction

**Key Finding:** `redis_health_monitor.py` classification was correct (legacy), but rationale was misleading. It's not a simple relocation - it's an older version that was enhanced with Prometheus metrics.

**Safe to Proceed:** All files verified safe to archive. No production dependencies found.

**Recommendation:** Update documentation with corrected rationale for `redis_health_monitor.py` before proceeding with archive operation.

---

**Verification Completed:** October 30, 2025  
**Verified By:** System audit using service inspection, file comparison, and production validation  
**Approval:** ✅ Ready for archive migration
