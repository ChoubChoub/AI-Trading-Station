# QuestDB Cold Storage Setup - Completed
**Date:** October 18, 2025  
**Status:** ✅ OPERATIONAL

## Storage Configuration

### Hot Storage (NVMe)
- **Device:** Samsung 990 Pro 4TB (`/dev/nvme0n1p2`)
- **Mount:** `/` (root filesystem)
- **Data Path:** `/home/youssefbahloul/ai-trading-station/QuestDB/data/hot`
- **Capacity:** 3.6TB total, 3.4TB available
- **Usage:** Active trading data (30 days tick, 7 days orderbook)
- **Performance:** 7,350MB/s read/write

### Cold Storage (HDD)
- **Device:** Seagate ST8000DM004 8TB (`/dev/sda1`)
- **Serial:** ZR1687Y9
- **Mount:** `/mnt/hdd`
- **Data Path:** `/mnt/hdd/questdb/cold`
- **Capacity:** 7.3TB usable (8TB raw)
- **Usage:** Historical data (365+ days)
- **Filesystem:** ext4 with `largefile4` optimization
- **Mount Options:** `noatime,nodiratime` (performance optimized)
- **UUID:** 3c63310b-4f9c-4d6a-a44a-97f0e9cb86dd

## Setup Steps Completed

✅ 1. Wiped Windows remnants from HDD
✅ 2. Created GPT partition table
✅ 3. Created single partition spanning full disk
✅ 4. Formatted with ext4 (1% reserved, largefile4 optimized)
✅ 5. Created mount point at `/mnt/hdd`
✅ 6. Added persistent mount to `/etc/fstab`
✅ 7. Set ownership to `youssefbahloul:youssefbahloul`
✅ 8. Created QuestDB directory structure
✅ 9. Updated `server.conf` with cold storage path
✅ 10. Restarted QuestDB service
✅ 11. Verified write permissions

## Configuration

### /etc/fstab Entry
```
UUID=3c63310b-4f9c-4d6a-a44a-97f0e9cb86dd /mnt/hdd ext4 defaults,noatime,nodiratime 0 2
```

### QuestDB server.conf
```properties
# Cold Storage: HDD for historical data (365+ days) - ACTIVE
# Mounted: /dev/sda1 (8TB Seagate ST8000DM004) at /mnt/hdd
cairo.cold.storage.root=/mnt/hdd/questdb/cold
```

## Hot/Cold Tiering Strategy

### Hot Storage Retention (NVMe)
- **Tick data:** 30 days
- **Orderbook snapshots:** 7 days
- **Regime states:** 30 days
- **Expected size:** 500GB - 1TB

### Cold Storage Retention (HDD)
- **Tick data:** 365+ days
- **Orderbook snapshots:** 90 days
- **Regime states:** 365+ days
- **Expected size:** 2-5TB over 1 year

### Data Movement
QuestDB automatically moves data from hot to cold storage based on partition age:
- Partitions older than retention period → moved to cold storage
- Query engine transparently accesses both hot and cold data
- No manual intervention required

## Verification Commands

### Check Mounted Filesystems
```bash
df -h | grep -E "(nvme|sda)"
```

### Verify Cold Storage Directory
```bash
ls -la /mnt/hdd/questdb/cold/
```

### Test Write Permissions
```bash
touch /mnt/hdd/questdb/cold/test && rm /mnt/hdd/questdb/cold/test
```

### Check QuestDB Status
```bash
systemctl status questdb.service
```

### Monitor Storage Usage
```bash
# Hot storage
du -sh /home/youssefbahloul/ai-trading-station/QuestDB/data/hot

# Cold storage
du -sh /mnt/hdd/questdb/cold
```

## Performance Expectations

### Hot Storage (NVMe)
- Write latency: <1ms
- Read latency: <1ms
- Throughput: >200k inserts/sec
- Query performance: <100ms for recent data

### Cold Storage (HDD)
- Write latency: 5-10ms (archival, not latency-sensitive)
- Read latency: 10-50ms (historical analysis)
- Throughput: Sequential reads ~200MB/s
- Query performance: 1-5 seconds for historical queries

## Maintenance

### Disk Space Monitoring
Set up alerts when:
- Hot storage > 80% full (2.9TB used)
- Cold storage > 90% full (6.6TB used)

### Cleanup Strategy
If cold storage fills up:
1. Archive oldest data to external backup
2. Delete data older than 2 years
3. Compress rarely-accessed partitions

## Troubleshooting

### HDD Not Mounted After Reboot
```bash
sudo mount -a
systemctl status questdb.service
```

### QuestDB Can't Write to Cold Storage
```bash
# Check ownership
ls -ld /mnt/hdd/questdb/cold

# Fix if needed
sudo chown -R youssefbahloul:youssefbahloul /mnt/hdd/questdb
```

### Verify Partition Table
```bash
sudo parted /dev/sda print
```

## Backup Recommendations

1. **Critical data:** Replicate to cloud storage (S3, Wasabi)
2. **Cold storage:** Schedule weekly snapshots to external drive
3. **Hot storage:** Included in system backups (NVMe snapshot)

## Next Steps

- [ ] Configure data retention policies in QuestDB
- [ ] Set up monitoring for disk space alerts
- [ ] Create historical data download script
- [ ] Configure Redis → QuestDB pipeline
- [ ] Test hot/cold tiering with production data

---

**Documentation Owner:** Youssef Bahloul  
**Last Updated:** October 18, 2025  
**Version:** 1.0
