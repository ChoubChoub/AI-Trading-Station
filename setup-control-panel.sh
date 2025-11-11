#!/bin/bash
#
# AI Trading Station - Production Access & Control System Setup Script
# 
# This script creates the directory structure and initial files for the 
# Production Access & Control System implementation.
#
# IMPORTANT: This script will NOT override any existing files.
#            All new files are created in a dedicated ControlPanel/ directory.
#
# Usage: ./setup-control-panel.sh
#
# Author: AI Trading Station Team
# Date: 2025-11-11
# Version: 1.0.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROL_PANEL_DIR="$BASE_DIR/ControlPanel"

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}AI Trading Station Control Panel Setup${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Check if ControlPanel directory already exists
if [ -d "$CONTROL_PANEL_DIR" ]; then
    echo -e "${YELLOW}WARNING: ControlPanel directory already exists!${NC}"
    echo -e "${YELLOW}This script will NOT override existing files.${NC}"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 1
    fi
fi

echo -e "${GREEN}Creating directory structure...${NC}"

# Create main directories
mkdir -p "$CONTROL_PANEL_DIR/api"
mkdir -p "$CONTROL_PANEL_DIR/frontend"
mkdir -p "$CONTROL_PANEL_DIR/tests"
mkdir -p "$CONTROL_PANEL_DIR/config"
mkdir -p "$CONTROL_PANEL_DIR/logs"

# Create API package
touch "$CONTROL_PANEL_DIR/api/__init__.py"

# Create test package  
touch "$CONTROL_PANEL_DIR/tests/__init__.py"

echo -e "${GREEN}✓ Directory structure created${NC}"

# Create requirements.txt
if [ ! -f "$CONTROL_PANEL_DIR/requirements.txt" ]; then
    cat > "$CONTROL_PANEL_DIR/requirements.txt" << 'EOF'
# AI Trading Station - Control Panel Dependencies
# Python 3.10+ required

# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# HTTP Client for NanoKVM integration
aiohttp==3.9.1
httpx==0.25.2

# Data Validation
pydantic==2.5.0
pydantic-settings==2.1.0

# Redis Client for metrics
redis==5.0.1
hiredis==2.2.3

# System Monitoring
psutil==5.9.6

# Configuration
pyyaml==6.0.1
python-dotenv==1.0.0

# Security & Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Logging & Monitoring
structlog==23.2.0
python-json-logger==2.0.7

# Optional: Telegram Bot (for emergency access)
python-telegram-bot==20.7

# Development Dependencies (optional)
# pytest==7.4.3
# pytest-asyncio==0.21.1
# pytest-cov==4.1.0
# black==23.12.0
# pylint==3.0.3
# mypy==1.7.1
EOF
    echo -e "${GREEN}✓ Created requirements.txt${NC}"
else
    echo -e "${YELLOW}⊙ requirements.txt already exists, skipping${NC}"
fi

# Create config.yaml
if [ ! -f "$CONTROL_PANEL_DIR/config/config.yaml" ]; then
    cat > "$CONTROL_PANEL_DIR/config/config.yaml" << 'EOF'
# AI Trading Station Control Panel Configuration
# 
# SECURITY NOTE: Store sensitive values in environment variables,
# not in this file. This file should be committed to version control
# with placeholder values only.

server:
  host: "0.0.0.0"
  port: 8080
  debug: false
  reload: false

security:
  # Authelia integration - user info comes from headers
  require_2fa: true
  admin_group: "admins"
  operators_group: "operators"
  
  # Power control restrictions
  power_control_user: "youssef"  # Only this user can control power
  power_control_networks:
    - "192.168.1.0/24"  # Only from local network
  
  # Session configuration
  session_timeout: 7200  # 2 hours
  inactivity_timeout: 900  # 15 minutes

services:
  # Allowed services for management
  allowed_services:
    redis: "redis-hft"
    questdb: "questdb"
    prometheus: "prometheus"
    grafana: "grafana-server"
    trades: "binance-trades"
    orderbook: "binance-bookticker"
    writer: "batch-writer"
  
  # Services to stop on kill switch
  trading_services:
    - "binance-trades"
    - "binance-bookticker"
    - "batch-writer"

nanokvm:
  # NanoKVM API configuration
  # Store credentials in environment variables:
  # NANOKVM_URL, NANOKVM_USERNAME, NANOKVM_PASSWORD
  base_url: "https://210.6.8.5:40443"
  verify_ssl: false  # Self-signed certificate
  timeout: 10
  
  # Graceful shutdown wait time (seconds)
  shutdown_wait: 30

redis:
  host: "localhost"
  port: 6379
  db: 0
  decode_responses: true
  
  # Streams to monitor
  streams:
    trades: "market:binance_spot:trades:BTCUSDT"
    orderbook: "market:binance_spot:orderbook:BTCUSDT"

audit:
  # Audit log configuration
  log_file: "/var/log/trading/audit.jsonl"
  log_level: "INFO"
  
  # Remote syslog (optional)
  remote_syslog:
    enabled: false
    host: ""
    port: 514
    protocol: "UDP"  # or TCP
  
  # Retention
  retention_days: 90

alerts:
  # Alert configuration
  telegram:
    enabled: false
    bot_token: ""  # Store in env: TELEGRAM_BOT_TOKEN
    chat_id: ""    # Store in env: TELEGRAM_CHAT_ID
  
  email:
    enabled: false
    smtp_host: ""
    smtp_port: 587
    smtp_user: ""  # Store in env: SMTP_USER
    smtp_password: ""  # Store in env: SMTP_PASSWORD
    from_address: "alerts@aistation.trading"
    to_addresses:
      - "admin@example.com"

monitoring:
  # Prometheus metrics
  metrics_enabled: true
  metrics_port: 9090
  
  # Health check
  health_check_interval: 30  # seconds
  
  # System status refresh
  status_refresh_interval: 5  # seconds
EOF
    echo -e "${GREEN}✓ Created config/config.yaml${NC}"
else
    echo -e "${YELLOW}⊙ config/config.yaml already exists, skipping${NC}"
fi

# Create .env.example
if [ ! -f "$CONTROL_PANEL_DIR/.env.example" ]; then
    cat > "$CONTROL_PANEL_DIR/.env.example" << 'EOF'
# AI Trading Station Control Panel - Environment Variables
# Copy this file to .env and fill in your actual values
# NEVER commit .env file to version control!

# NanoKVM Credentials
NANOKVM_URL=https://210.6.8.5:40443
NANOKVM_USERNAME=admin
NANOKVM_PASSWORD=your_nanokvm_password

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_AUTHORIZED_USERS=123456789,987654321

# SMTP Configuration (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@aistation.trading
SMTP_PASSWORD=your_smtp_password

# Application
ENV=production
LOG_LEVEL=INFO
DEBUG=false
EOF
    echo -e "${GREEN}✓ Created .env.example${NC}"
else
    echo -e "${YELLOW}⊙ .env.example already exists, skipping${NC}"
fi

# Create .gitignore for ControlPanel
if [ ! -f "$CONTROL_PANEL_DIR/.gitignore" ]; then
    cat > "$CONTROL_PANEL_DIR/.gitignore" << 'EOF'
# Environment variables
.env
*.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/*.log
*.log

# Test coverage
.coverage
htmlcov/
.pytest_cache/

# OS
.DS_Store
Thumbs.db
EOF
    echo -e "${GREEN}✓ Created .gitignore${NC}"
else
    echo -e "${YELLOW}⊙ .gitignore already exists, skipping${NC}"
fi

echo ""
echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Review the generated files in ControlPanel/"
echo "2. Copy .env.example to .env and fill in your credentials"
echo "3. Install Python dependencies:"
echo -e "   ${YELLOW}cd ControlPanel${NC}"
echo -e "   ${YELLOW}python3 -m venv venv${NC}"
echo -e "   ${YELLOW}source venv/bin/activate${NC}"
echo -e "   ${YELLOW}pip install -r requirements.txt${NC}"
echo ""
echo "4. Create log directory:"
echo -e "   ${YELLOW}sudo mkdir -p /var/log/trading${NC}"
echo -e "   ${YELLOW}sudo chown \$USER:\$USER /var/log/trading${NC}"
echo ""
echo "5. Review the implementation files:"
echo "   - api/main.py (main API implementation)"
echo "   - api/models.py (data models)"
echo "   - api/auth.py (authentication)"
echo "   - api/controls.py (control operations)"
echo "   - api/nanokvm.py (power management)"
echo "   - api/audit.py (audit logging)"
echo "   - frontend/index.html (web UI)"
echo ""
echo "6. Run the development server:"
echo -e "   ${YELLOW}uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload${NC}"
echo ""
echo -e "${BLUE}For production deployment:${NC}"
echo "See Documentation/Operations/Production_Control_System.md"
echo ""
echo -e "${YELLOW}IMPORTANT SECURITY NOTES:${NC}"
echo "- Never commit .env file to version control"
echo "- Store sensitive credentials in environment variables"
echo "- Enable 2FA authentication before production deployment"
echo "- Review and test all emergency procedures"
echo "- Set up audit log monitoring and alerts"
echo ""
