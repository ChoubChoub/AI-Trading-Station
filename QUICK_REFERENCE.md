# Production Control System - Quick Reference Card

**Version**: 1.0.0  
**Last Updated**: 2025-11-11

## 🚀 Quick Start (5 Steps)

```bash
# 1. Run setup script
cd /home/youssefbahloul/ai-trading-station
./setup-control-panel.sh

# 2. Install dependencies
cd ControlPanel
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
nano .env  # Add your credentials

# 4. Create log directory
sudo mkdir -p /var/log/trading
sudo chown $USER:$USER /var/log/trading

# 5. Test locally
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

## 📚 Documentation Files

| File | Purpose | Size |
|------|---------|------|
| `IMPLEMENTATION_SUMMARY.md` | Quick start & overview | 14K words |
| `Production_Control_System_COMPLETE.md` | Code templates | 41K words |
| `Production_Control_System_Implementation.md` | Deployment guide | 27K words |
| `setup-control-panel.sh` | Setup automation | 260 lines |

**Read them in this order**: SUMMARY → COMPLETE → IMPLEMENTATION

## 🔒 Security Checklist

- [ ] Domain purchased and DNS configured
- [ ] 2FA enabled (Authelia)
- [ ] SSL certificates obtained (Let's Encrypt)
- [ ] Environment variables configured (.env)
- [ ] Sudo permissions configured
- [ ] Firewall rules applied
- [ ] Recovery codes generated and stored offline
- [ ] Audit logging tested
- [ ] Emergency procedures documented
- [ ] Backup script configured

## 🎛️ API Endpoints

```bash
# Health check (no auth)
GET /health

# Emergency kill switch
POST /api/kill-switch
Body: {"confirmation": "CONFIRM_EMERGENCY_STOP"}

# Data feed control
POST /api/datafeed/{start|stop|restart|status}

# Service management
POST /api/service/{service}/{start|stop|restart|status}
Services: redis, questdb, trades, orderbook, writer, grafana, prometheus

# Power control (primary admin only)
POST /api/power/{on|off|status}
Body (for off): {"confirmation": "CONFIRM_POWER_OFF"}

# System status
GET /api/status/overview

# Audit logs
GET /api/audit/logs?limit=100
```

## 🌐 Access URLs (After Deployment)

- **Authelia Portal**: https://auth.aistation.trading
- **Control Panel**: https://ops.aistation.trading
- **Grafana**: https://grafana.aistation.trading
- **API Endpoint**: https://api.aistation.trading

## 🔧 Common Commands

### Service Management
```bash
# Check status
sudo systemctl status authelia
sudo systemctl status trading-control
sudo systemctl status nginx

# View logs
sudo journalctl -u authelia -f
sudo journalctl -u trading-control -f
sudo tail -f /var/log/nginx/error.log

# Restart services
sudo systemctl restart authelia
sudo systemctl restart trading-control
sudo systemctl restart nginx
```

### Configuration
```bash
# Edit Authelia config
sudo nano /etc/authelia/configuration.yml

# Edit Nginx config
sudo nano /etc/nginx/sites-available/aistation.trading

# Test Nginx config
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### SSL Certificates
```bash
# Check certificates
sudo certbot certificates

# Test renewal
sudo certbot renew --dry-run

# Force renewal
sudo certbot renew --force-renewal
```

### Audit Logs
```bash
# View recent audit logs
tail -f /var/log/trading/audit.jsonl

# Pretty print
tail -n 20 /var/log/trading/audit.jsonl | jq

# Verify chain integrity
python3 -c "from api.audit import AuditLogger; print(AuditLogger().verify_chain_integrity())"
```

## 🚨 Emergency Procedures

### Kill Switch Not Responding
```bash
# Manual service stop
sudo systemctl stop binance-trades
sudo systemctl stop binance-bookticker
sudo systemctl stop batch-writer

# Check logs
sudo journalctl -u trading-control -n 50
```

### Lost 2FA Access
```bash
# Use recovery codes (stored offline)
# Or reset via CLI:
sudo authelia users totp delete youssef
sudo authelia users totp generate youssef

# Reset password
sudo authelia users reset-password youssef
```

### Control Panel Down
```bash
# Check service
sudo systemctl status trading-control

# Check logs
sudo journalctl -u trading-control -n 50

# Restart service
sudo systemctl restart trading-control

# Check if port is listening
sudo ss -tulpn | grep :8080
```

### SSL Certificate Issues
```bash
# Check expiry
sudo certbot certificates

# Renew if needed
sudo certbot renew

# Check Nginx config
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

## 📊 System Architecture

```
Internet
    ↓
aistation.trading (Domain)
    ↓
Nginx (Port 443 - SSL/TLS)
    ↓
Authelia (Port 9091 - 2FA)
    ├─→ Grafana (Port 3000) - Monitoring
    ├─→ Control Panel (Port 8080) - Operations
    └─→ API (Port 8080) - Programmatic Access
```

## 🔐 User Roles

| Role | Access | Can Control |
|------|--------|-------------|
| **Admin** | Full | Kill switch, services, data feed, status |
| **Operator** | Read-only | Grafana monitoring only |
| **Primary Admin** | Full + Power | All above + power control |

## 📝 File Locations

```bash
# Control Panel
/home/youssefbahloul/ai-trading-station/ControlPanel/

# Logs
/var/log/trading/audit.jsonl
/var/log/nginx/access.log
/var/log/nginx/error.log

# Configurations
/etc/authelia/configuration.yml
/var/lib/authelia/users.yml
/etc/nginx/sites-available/aistation.trading
/etc/systemd/system/trading-control.service

# SSL Certificates
/etc/letsencrypt/live/aistation.trading/
```

## 🧪 Testing Checklist

Before production:
- [ ] Health endpoint responds
- [ ] Authelia login works
- [ ] 2FA setup successful
- [ ] Control panel UI loads
- [ ] System status displays correctly
- [ ] Service management works
- [ ] Kill switch tested (staging only!)
- [ ] Audit logging verified
- [ ] SSL certificates valid
- [ ] Backup restoration tested

## 📞 Support

### Troubleshooting Steps
1. Check service status
2. Review logs (journalctl)
3. Verify configurations
4. Test components individually
5. Check network connectivity
6. Review audit trail

### Documentation
- Full guides in `Documentation/Operations/`
- Code templates in `Production_Control_System_COMPLETE.md`
- Deployment guide in `Production_Control_System_Implementation.md`

### Emergency Access
- SSH to server
- Direct NanoKVM access (https://210.6.8.5:40443)
- Telegram bot (if configured)

## ⚠️ Important Notes

**NEVER**:
- Commit .env file to git
- Disable 2FA in production
- Skip confirmation for destructive actions
- Share recovery codes insecurely
- Modify audit logs

**ALWAYS**:
- Use strong passwords (16+ chars)
- Enable 2FA for all users
- Test in staging first
- Monitor audit logs
- Keep backups current
- Document changes

## 📦 Dependencies

```bash
# Python packages (see requirements.txt)
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiohttp==3.9.1
pydantic==2.5.0
redis==5.0.1
psutil==5.9.6
python-telegram-bot==20.7  # Optional

# System packages
nginx
certbot
python3-certbot-nginx
postgresql
redis-server
authelia
```

## 🎯 Success Metrics

- ✅ All services running
- ✅ SSL certificates valid
- ✅ 2FA working
- ✅ Audit logs intact
- ✅ Zero unauthorized access
- ✅ Regular backups running

## 📅 Maintenance Schedule

**Daily**:
- Check service health
- Review audit logs

**Weekly**:
- Review access logs
- Check disk space
- Verify backups

**Monthly**:
- Update system packages
- Review user access
- Test disaster recovery
- Review security alerts

---

**For detailed information**, see the full documentation files in `Documentation/Operations/`.

**Emergency contact**: Youssef Bahloul (Primary Admin)

**Last reviewed**: 2025-11-11
