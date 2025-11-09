# Disk Identification - AI Trading Station
**Created:** October 18, 2025  
**Purpose:** Safety reference to prevent accidental data loss

## ⚠️ CRITICAL - DO NOT MODIFY THESE DEVICES

### Ubuntu System Disk (NEVER TOUCH)
- **Device:** `/dev/nvme0n1`
- **Model:** Samsung SSD 990 PRO 4TB
- **Serial:** S7KGNU0XB06256L
- **Size:** 3.6TB
- **Contains:** Ubuntu OS, /boot/efi, root filesystem
- **Partitions:**
  - `/dev/nvme0n1p1` - 1G - vfat - /boot/efi
  - `/dev/nvme0n1p2` - 3.6T - ext4 - / (root)

## ✅ Safe to Modify - QuestDB Cold Storage

### HDD for Cold Storage (Safe to format)
- **Device:** `/dev/sda`
- **Model:** Seagate ST8000DM004-2U91
- **Serial:** ZR1687Y9
- **Size:** 8TB (7.3TB usable)
- **Current Status:** Empty (only Windows metadata remnant)
- **Purpose:** QuestDB cold storage (historical crypto data 365+ days)

## Device Verification Command
```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,SERIAL
```

## Emergency Recovery
If you accidentally format the wrong drive:
1. **STOP IMMEDIATELY** - Do not write any more data
2. Use `testdisk` or `photorec` for recovery
3. Contact data recovery professional if critical

## Last Verified
- Date: October 18, 2025
- Verified by: System analysis before HDD formatting
- Ubuntu system intact: ✅ YES
