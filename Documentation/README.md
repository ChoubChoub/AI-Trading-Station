# AI Trading Station Documentation

**Last Updated:** 2025-11-09  
**Total Documents:** 177

## 📁 Structure

```
Documentation/
├── Architecture/      (3)  - High-level system architecture
├── Implementation/   (25)  - Implementation plans, strategies & future work (inc. VPS Brain)
├── Operations/       (20)  - Performance tuning, troubleshooting, networking
├── Setup/            (17)  - Installation guides & configuration
├── Reports/          (41)  - Dated incidents, audits, performance reports
└── ARCHIVE/          (71)  - Historical daily summaries & completed phases
```

## 🎯 Quick Navigation

### 🏗️ Understanding the System
- **[Architecture/](Architecture/)** - Core system designs (GPU, Trading, Backtesting)
- **[Implementation/](Implementation/)** - How we implemented features (CPU affinity, GPU optimization, etc.)

### 🔧 Running & Maintaining
- **[Operations/](Operations/)** - Performance tuning, troubleshooting guides, network configuration
- **[Setup/](Setup/)** - Installation guides for Grafana, Redis, QuestDB, Exchange ingestion

### 📊 History & Analysis
- **[Reports/](Reports/)** - Incident reports, audits, performance analysis (date-organized)
- **[ARCHIVE/](ARCHIVE/)** - Historical daily summaries and deprecated documentation

## 🔍 Finding Documents

### By Topic
```bash
# Performance tuning
ls Operations/Performance-Tuning/

# Network configuration
ls Operations/Networking/

# Component setup
ls Setup/

# Recent incidents
ls Reports/Incidents/
```

### By Date
```bash
# November 2025 reports
find Reports -name "*2025-11*"

# October incidents
ls Reports/Incidents/*2025-10*
```

## 📈 Document Counts

| Folder | Files | Purpose |
|--------|-------|---------|
| Architecture | 3 | High-level system design |
| Implementation | 25 | Implemented & planned strategies (inc. VPS Brain) |
| Operations | 20 | Operational procedures |
| Setup | 17 | Installation & configuration |
| Reports | 41 | Time-bound analysis |
| ARCHIVE | 71 | Historical daily summaries |

## 🧭 Where to Start

**New to the system?**
1. Read `Architecture/` for system overview
2. Check `Setup/` for installation guides
3. Review `Operations/` for daily operations

**Troubleshooting?**
- Check `Operations/Troubleshooting/`
- Review recent incidents in `Reports/Incidents/`

**Implementing a feature?**
- Review similar work in `Implementation/`
- Check relevant architecture in `Architecture/`

---

*Structure simplified 2025-11-09: Consolidated from 13 fragmented folders to 6 clear categories*