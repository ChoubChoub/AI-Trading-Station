# Production Access & Control System - Implementation Guide

**Last Updated**: 2025-11-11  
**Status**: Implementation Ready  
**Version**: 1.0.0

## Overview

This guide provides step-by-step instructions for implementing the Production Access & Control System for the AI Trading Station. The system provides secure remote access, operational controls, and power management through a unified three-tier architecture.

## Architecture Summary

```
Internet → aistation.trading (Domain)
    ↓
Nginx Reverse Proxy (Port 443, SSL/TLS)
    ↓
Authelia 2FA Gateway (Port 9091)
    ├── grafana.aistation.trading    → Grafana (Port 3000 - Monitoring)
    ├── ops.aistation.trading        → Control Dashboard (Port 8080 - Operations)
    └── api.aistation.trading        → API Endpoint (Port 8080 - Programmatic Access)
```

## Implementation Phases

### Phase 0: Prerequisites (15 minutes)

#### Domain Requirements
- Purchase domain: `aistation.trading` (or your preferred domain)
- Recommended registrars:
  - Cloudflare Registrar: ~$38/year
  - Namecheap: ~$40/year
  - Google Domains: ~$40/year

#### DNS Configuration
Add the following DNS records:
```
Type    Name      Value              TTL
A       @         YOUR_SERVER_IP     Auto
A       auth      YOUR_SERVER_IP     Auto
A       grafana   YOUR_SERVER_IP     Auto
A       ops       YOUR_SERVER_IP     Auto
A       api       YOUR_SERVER_IP     Auto
```

#### Server Requirements
- **OS**: Ubuntu 22.04 LTS or newer
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 20GB available
- **Network**: Static IP or dynamic DNS
- **Access**: Root or sudo privileges

### Phase 1: Initial Setup (30 minutes)

#### 1.1 Run Setup Script

```bash
# Navigate to repository root
cd /home/youssefbahloul/ai-trading-station

# Make setup script executable
chmod +x setup-control-panel.sh

# Run setup script
./setup-control-panel.sh
```

This creates:
- `ControlPanel/` directory structure
- Configuration templates
- Environment variable examples
- Python package structure

#### 1.2 Configure Environment

```bash
cd ControlPanel

# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

Required environment variables:
```bash
# NanoKVM
NANOKVM_URL=https://210.6.8.5:40443
NANOKVM_USERNAME=admin
NANOKVM_PASSWORD=your_secure_password

# Optional: Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional: Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@aistation.trading
SMTP_PASSWORD=your_app_password

# Application
ENV=production
LOG_LEVEL=INFO
```

#### 1.3 Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### 1.4 Create Log Directory

```bash
# Create log directory
sudo mkdir -p /var/log/trading

# Set ownership
sudo chown $USER:$USER /var/log/trading

# Set permissions
chmod 755 /var/log/trading

# Create initial audit log
touch /var/log/trading/audit.jsonl
```

### Phase 2: Install Core Components (1-2 hours)

#### 2.1 Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    nginx \
    certbot \
    python3-certbot-nginx \
    postgresql \
    postgresql-contrib \
    redis-server \
    curl \
    wget \
    git \
    build-essential \
    python3-dev
```

#### 2.2 Install Authelia

```bash
# Download Authelia binary
cd /tmp
wget https://github.com/authelia/authelia/releases/download/v4.37.5/authelia-linux-amd64.tar.gz

# Extract and install
tar xzf authelia-linux-amd64.tar.gz
sudo mv authelia-linux-amd64 /usr/local/bin/authelia
sudo chmod +x /usr/local/bin/authelia

# Create directories
sudo mkdir -p /etc/authelia
sudo mkdir -p /var/lib/authelia

# Generate secrets
AUTHELIA_JWT_SECRET=$(openssl rand -hex 32)
AUTHELIA_SESSION_SECRET=$(openssl rand -hex 32)
AUTHELIA_STORAGE_KEY=$(openssl rand -hex 32)

# Save secrets to file for later use
cat > ~/authelia_secrets.txt << EOF
JWT_SECRET=$AUTHELIA_JWT_SECRET
SESSION_SECRET=$AUTHELIA_SESSION_SECRET
STORAGE_KEY=$AUTHELIA_STORAGE_KEY
EOF

echo "Secrets saved to ~/authelia_secrets.txt - Keep this file secure!"
```

#### 2.3 Configure Authelia

Create Authelia configuration:

```bash
sudo nano /etc/authelia/configuration.yml
```

Paste the following configuration (replace secrets with values from ~/authelia_secrets.txt):

```yaml
server:
  host: 0.0.0.0
  port: 9091

jwt_secret: YOUR_JWT_SECRET_HERE

default_redirection_url: https://ops.aistation.trading

totp:
  issuer: AI Trading Station
  period: 30
  skew: 1
  algorithm: SHA256
  digits: 6

# Optional: WebAuthn for hardware keys
webauthn:
  disable: false
  display_name: AI Trading Station
  attestation_conveyance_preference: indirect
  user_verification: preferred

authentication_backend:
  file:
    path: /var/lib/authelia/users.yml
    password:
      algorithm: argon2id
      iterations: 3
      memory: 65536
      parallelism: 4
      key_length: 32
      salt_length: 16

access_control:
  default_policy: deny
  rules:
    # Grafana - monitoring access
    - domain: grafana.aistation.trading
      policy: two_factor
      subject:
        - "group:operators"
        - "group:admins"
    
    # Control Panel - admin access only
    - domain: ops.aistation.trading
      policy: two_factor
      subject:
        - "group:admins"
    
    # API - admin access only
    - domain: api.aistation.trading
      policy: two_factor
      subject:
        - "group:admins"

session:
  name: authelia_session
  secret: YOUR_SESSION_SECRET_HERE
  expiration: 12h
  inactivity: 2h
  remember_me_duration: 7d
  domain: aistation.trading

regulation:
  max_retries: 3
  find_time: 2m
  ban_time: 10m

storage:
  encryption_key: YOUR_STORAGE_KEY_HERE
  local:
    path: /var/lib/authelia/db.sqlite3

notifier:
  filesystem:
    filename: /var/lib/authelia/notifications.txt
```

#### 2.4 Create Authelia Users

```bash
# Generate password hash for admin user
authelia crypto hash generate argon2id --password "YourSecurePassword"

# Create users file
sudo nano /var/lib/authelia/users.yml
```

Add users:

```yaml
users:
  youssef:
    displayname: "Youssef Bahloul"
    password: "$argon2id$v=19$m=65536,t=3,p=4$YOUR_HASH_HERE"
    email: your-email@example.com
    groups:
      - admins
      - operators
  
  # Add more users as needed
  # operator1:
  #   displayname: "Operator Name"
  #   password: "$argon2id$v=19$m=65536,t=3,p=4$HASH"
  #   email: operator@example.com
  #   groups:
  #     - operators
```

Set permissions:

```bash
sudo chown -R root:root /etc/authelia
sudo chown -R root:root /var/lib/authelia
sudo chmod 600 /var/lib/authelia/users.yml
```

#### 2.5 Create Authelia Systemd Service

```bash
sudo nano /etc/systemd/system/authelia.service
```

```ini
[Unit]
Description=Authelia authentication server
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/authelia --config /etc/authelia/configuration.yml
Restart=always
RestartSec=5
Environment=TZ=UTC
User=root
Group=root

[Install]
WantedBy=multi-user.target
```

Start Authelia:

```bash
sudo systemctl daemon-reload
sudo systemctl enable authelia
sudo systemctl start authelia
sudo systemctl status authelia
```

### Phase 3: Deploy Control Panel (1 hour)

#### 3.1 Create Control Panel Service

```bash
sudo nano /etc/systemd/system/trading-control.service
```

```ini
[Unit]
Description=AI Trading Station Control Panel
After=network.target authelia.service redis-server.service
Requires=authelia.service

[Service]
Type=simple
User=youssefbahloul
Group=youssefbahloul
WorkingDirectory=/home/youssefbahloul/ai-trading-station/ControlPanel
Environment="PATH=/home/youssefbahloul/ai-trading-station/ControlPanel/venv/bin"
EnvironmentFile=/home/youssefbahloul/ai-trading-station/ControlPanel/.env
ExecStart=/home/youssefbahloul/ai-trading-station/ControlPanel/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/var/log/trading

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-control
sudo systemctl start trading-control
sudo systemctl status trading-control
```

#### 3.2 Test Control Panel Locally

```bash
# Test API health
curl http://localhost:8080/health

# Expected response:
# {"status":"healthy","version":"1.0.0"}
```

### Phase 4: Configure Nginx Reverse Proxy (30 minutes)

#### 4.1 Create Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/aistation.trading
```

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name *.aistation.trading aistation.trading;
    return 301 https://$host$request_uri;
}

# Authelia Portal
server {
    listen 443 ssl http2;
    server_name auth.aistation.trading;
    
    ssl_certificate /etc/letsencrypt/live/aistation.trading/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aistation.trading/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    location / {
        proxy_pass http://localhost:9091;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Grafana Monitoring (existing service)
server {
    listen 443 ssl http2;
    server_name grafana.aistation.trading;
    
    ssl_certificate /etc/letsencrypt/live/aistation.trading/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aistation.trading/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header X-Content-Type-Options nosniff always;
    
    # Authelia forward auth
    location /authelia {
        internal;
        proxy_pass http://localhost:9091/api/verify;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URL $scheme://$http_host$request_uri;
    }
    
    location / {
        auth_request /authelia;
        error_page 401 =302 https://auth.aistation.trading/?rd=$scheme://$http_host$request_uri;
        error_page 403 = /error/403;
        
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Pass auth headers to Grafana
        auth_request_set $user $upstream_http_remote_user;
        auth_request_set $groups $upstream_http_remote_groups;
        proxy_set_header Remote-User $user;
        proxy_set_header Remote-Groups $groups;
    }
}

# Control Panel (Operations)
server {
    listen 443 ssl http2;
    server_name ops.aistation.trading;
    
    ssl_certificate /etc/letsencrypt/live/aistation.trading/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aistation.trading/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Authelia forward auth
    location /authelia {
        internal;
        proxy_pass http://localhost:9091/api/verify;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URL $scheme://$http_host$request_uri;
    }
    
    location / {
        auth_request /authelia;
        error_page 401 =302 https://auth.aistation.trading/?rd=$scheme://$http_host$request_uri;
        error_page 403 = /error/403;
        
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Pass auth headers to control panel
        auth_request_set $user $upstream_http_remote_user;
        auth_request_set $groups $upstream_http_remote_groups;
        proxy_set_header Remote-User $user;
        proxy_set_header Remote-Groups $groups;
        
        # WebSocket support for real-time updates
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# API Endpoint (for mobile/programmatic access)
server {
    listen 443 ssl http2;
    server_name api.aistation.trading;
    
    ssl_certificate /etc/letsencrypt/live/aistation.trading/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/aistation.trading/privkey.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff always;
    
    # CORS for mobile apps
    add_header Access-Control-Allow-Origin "https://ops.aistation.trading" always;
    add_header Access-Control-Allow-Credentials "true" always;
    add_header Access-Control-Allow-Methods "GET, POST, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Authorization, Content-Type, Remote-User, Remote-Groups" always;
    
    # Authelia forward auth
    location /authelia {
        internal;
        proxy_pass http://localhost:9091/api/verify;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URL $scheme://$http_host$request_uri;
    }
    
    location / {
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            return 204;
        }
        
        auth_request /authelia;
        error_page 401 =302 https://auth.aistation.trading/?rd=$scheme://$http_host$request_uri;
        error_page 403 = /error/403;
        
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Pass auth headers
        auth_request_set $user $upstream_http_remote_user;
        auth_request_set $groups $upstream_http_remote_groups;
        proxy_set_header Remote-User $user;
        proxy_set_header Remote-Groups $groups;
    }
}
```

#### 4.2 Enable Site

```bash
# Test Nginx configuration
sudo nginx -t

# Enable site
sudo ln -s /etc/nginx/sites-available/aistation.trading /etc/nginx/sites-enabled/

# Test again
sudo nginx -t
```

### Phase 5: SSL Certificates (15 minutes)

#### 5.1 Obtain Let's Encrypt Certificates

**IMPORTANT**: Ensure DNS records are propagated before running this step.

```bash
# Stop Nginx temporarily
sudo systemctl stop nginx

# Obtain wildcard certificate
sudo certbot certonly --standalone \
  -d aistation.trading \
  -d auth.aistation.trading \
  -d grafana.aistation.trading \
  -d ops.aistation.trading \
  -d api.aistation.trading \
  --email your-email@example.com \
  --agree-tos \
  --no-eff-email

# Start Nginx
sudo systemctl start nginx
```

#### 5.2 Configure Auto-Renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Certbot automatically sets up a systemd timer for renewal
sudo systemctl status certbot.timer
```

### Phase 6: Firewall Configuration (10 minutes)

```bash
# Allow SSH (if not already allowed)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw --force enable

# Check status
sudo ufw status verbose
```

### Phase 7: Testing & Validation (30 minutes)

#### 7.1 Test Authelia

```bash
# Check Authelia is running
curl -I http://localhost:9091

# Expected: HTTP 200 or redirect
```

#### 7.2 Test Control Panel Backend

```bash
# Test health endpoint
curl http://localhost:8080/health

# Expected: {"status":"healthy"}
```

#### 7.3 Test Full Authentication Flow

1. Open browser to `https://auth.aistation.trading`
2. Login with credentials (username: youssef, password: as configured)
3. Scan QR code with authenticator app (Google Authenticator, Authy, etc.)
4. Enter 6-digit code
5. Should see "Authentication successful" or redirect

#### 7.4 Test Control Panel Access

1. Navigate to `https://ops.aistation.trading`
2. Should redirect to Authelia for login
3. After successful authentication, should see Control Panel dashboard
4. Test system status refresh
5. Verify service list displays correctly

#### 7.5 Test Grafana Access

1. Navigate to `https://grafana.aistation.trading`
2. Should redirect to Authelia for login
3. After authentication, should access Grafana dashboards

### Phase 8: Post-Deployment (30 minutes)

#### 8.1 Configure Sudo Permissions for Service Control

Create sudoers file for service management:

```bash
sudo visudo -f /etc/sudoers.d/trading-control
```

Add:

```sudoers
# Allow trading-control service to manage specific services
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl start binance-trades
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop binance-trades
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart binance-trades
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl status binance-trades

youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl start binance-bookticker
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop binance-bookticker
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart binance-bookticker
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl status binance-bookticker

youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl start batch-writer
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop batch-writer
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart batch-writer
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl status batch-writer

youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl start redis-hft
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop redis-hft
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart redis-hft
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl status redis-hft

youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl start questdb
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop questdb
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart questdb
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl status questdb

youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl start grafana-server
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop grafana-server
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart grafana-server
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl status grafana-server

youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl start prometheus
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop prometheus
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart prometheus
youssefbahloul ALL=(ALL) NOPASSWD: /usr/bin/systemctl status prometheus
```

Test permissions:

```bash
# Should work without password prompt
sudo systemctl status binance-trades
```

#### 8.2 Set Up Monitoring Alerts

Edit Prometheus configuration to add alerts for control panel:

```bash
# Add to prometheus configuration
sudo nano /etc/prometheus/prometheus.yml
```

Add scrape config:

```yaml
scrape_configs:
  - job_name: 'trading-control'
    static_configs:
      - targets: ['localhost:9090']
```

#### 8.3 Document Emergency Procedures

Create emergency procedures document:

```bash
nano ~/emergency-procedures.md
```

Add critical information:
- Authelia recovery codes (store offline/encrypted)
- NanoKVM direct access credentials
- Telegram bot token (if configured)
- Emergency contact information
- Backup access methods

#### 8.4 Test Emergency Procedures

**Critical**: Test kill switch in non-production environment first!

1. Create test trading services (or use staging environment)
2. Test kill switch activation with confirmation
3. Verify all services stop correctly
4. Check audit log entries
5. Test service restart procedures
6. Document any issues

#### 8.5 Configure Backup & Recovery

```bash
# Create backup script
nano ~/backup-control-system.sh
```

```bash
#!/bin/bash
# Backup critical control system files

BACKUP_DIR="/home/youssefbahloul/backups/control-system-$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configurations
cp /etc/authelia/configuration.yml "$BACKUP_DIR/"
cp /var/lib/authelia/users.yml "$BACKUP_DIR/"
cp /etc/nginx/sites-available/aistation.trading "$BACKUP_DIR/"
cp -r /home/youssefbahloul/ai-trading-station/ControlPanel/config "$BACKUP_DIR/"
cp /home/youssefbahloul/ai-trading-station/ControlPanel/.env "$BACKUP_DIR/"

# Backup audit logs
cp /var/log/trading/audit.jsonl "$BACKUP_DIR/"

# Create archive
tar czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

Make executable and run:

```bash
chmod +x ~/backup-control-system.sh
./backup-control-system.sh
```

Add to crontab for daily backups:

```bash
crontab -e
```

Add line:

```
0 2 * * * /home/youssefbahloul/backup-control-system.sh
```

## Deployment Checklist

Use this checklist to verify deployment:

### Phase 0: Prerequisites
- [ ] Domain purchased and DNS configured
- [ ] Server meets minimum requirements
- [ ] Root/sudo access available
- [ ] Network connectivity verified

### Phase 1: Initial Setup
- [ ] Setup script executed successfully
- [ ] Environment variables configured
- [ ] Python virtual environment created
- [ ] Dependencies installed
- [ ] Log directory created with correct permissions

### Phase 2: Core Components
- [ ] System dependencies installed
- [ ] Authelia installed and configured
- [ ] Authelia users created
- [ ] Authelia service running
- [ ] Secrets stored securely

### Phase 3: Control Panel
- [ ] Control panel service created
- [ ] Service enabled and started
- [ ] Health endpoint responds correctly
- [ ] Logs show no errors

### Phase 4: Nginx Configuration
- [ ] Nginx configuration created
- [ ] Configuration syntax valid (`nginx -t`)
- [ ] Site enabled
- [ ] Nginx reloaded

### Phase 5: SSL Certificates
- [ ] Let's Encrypt certificates obtained
- [ ] All subdomains covered
- [ ] Auto-renewal configured
- [ ] Certificates valid

### Phase 6: Firewall
- [ ] Firewall rules configured
- [ ] SSH access maintained
- [ ] HTTP/HTTPS allowed
- [ ] Firewall enabled

### Phase 7: Testing
- [ ] Authelia accessible and working
- [ ] Control panel backend responds
- [ ] Full authentication flow works
- [ ] 2FA setup successful
- [ ] Control panel UI accessible
- [ ] Grafana access via Authelia works
- [ ] All services show correct status

### Phase 8: Post-Deployment
- [ ] Sudo permissions configured
- [ ] Service management tested
- [ ] Monitoring alerts configured
- [ ] Emergency procedures documented
- [ ] Emergency procedures tested
- [ ] Backup script created and tested
- [ ] Backup scheduled in crontab

### Security Verification
- [ ] All services use HTTPS
- [ ] 2FA required for all access
- [ ] Audit logging working
- [ ] Recovery codes generated and stored
- [ ] Sensitive files have correct permissions
- [ ] Environment variables not in version control
- [ ] Firewall blocking unauthorized access

## Troubleshooting

### Authelia Issues

**Problem**: Cannot access Authelia portal

```bash
# Check Authelia service
sudo systemctl status authelia

# Check logs
sudo journalctl -u authelia -n 50 --no-pager

# Test port
curl -I http://localhost:9091

# Restart service
sudo systemctl restart authelia
```

**Problem**: 2FA not working

```bash
# Regenerate TOTP secret
sudo authelia users totp generate youssef

# Reset password
sudo authelia users reset-password youssef
```

### Control Panel Issues

**Problem**: Control panel not responding

```bash
# Check service
sudo systemctl status trading-control

# Check logs
sudo journalctl -u trading-control -n 50 --no-pager

# Test directly
curl http://localhost:8080/health

# Restart service
sudo systemctl restart trading-control
```

**Problem**: Cannot stop/start services

```bash
# Check sudo permissions
sudo -l

# Test service control
sudo systemctl status binance-trades

# Fix permissions if needed
sudo visudo -f /etc/sudoers.d/trading-control
```

### Nginx Issues

**Problem**: 502 Bad Gateway

```bash
# Check Nginx error log
sudo tail -f /var/log/nginx/error.log

# Check upstream services
systemctl status authelia trading-control grafana-server

# Test backend directly
curl http://localhost:8080/health
curl http://localhost:9091
```

**Problem**: SSL certificate errors

```bash
# Check certificates
sudo certbot certificates

# Renew if needed
sudo certbot renew

# Check Nginx SSL config
sudo nginx -t
```

### General Debugging

```bash
# Check all services
systemctl status authelia trading-control nginx

# Check all ports
sudo ss -tulpn | grep -E ':(80|443|3000|8080|9091)'

# Check firewall
sudo ufw status verbose

# Check logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
sudo journalctl -u authelia -f
sudo journalctl -u trading-control -f
```

## Security Best Practices

1. **Never commit sensitive data**: Keep .env files out of version control
2. **Use strong passwords**: Minimum 16 characters, mixed case, numbers, symbols
3. **Enable 2FA for all users**: No exceptions for production access
4. **Regular backups**: Test backup restoration procedures
5. **Monitor audit logs**: Review regularly for suspicious activity
6. **Keep software updated**: Apply security patches promptly
7. **Limit access**: Use principle of least privilege
8. **Document procedures**: Keep emergency procedures up to date
9. **Test disaster recovery**: Regular drills for emergency scenarios
10. **Secure credentials**: Use password manager for sensitive values

## Support & Documentation

### Additional Resources
- Authelia Documentation: https://www.authelia.com/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Nginx Documentation: https://nginx.org/en/docs/
- Let's Encrypt: https://letsencrypt.org/docs/

### Internal Documentation
- [Architecture Overview](../Architecture/)
- [Component Guides](./Component_Guides/)
- [Runbooks](./Runbooks/)

---

**Implementation Status**: Ready for Deployment  
**Next Review**: After Phase 8 completion  
**Contact**: AI Trading Station Team
