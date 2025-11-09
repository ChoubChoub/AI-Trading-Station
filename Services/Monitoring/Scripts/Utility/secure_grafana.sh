#!/bin/bash

# Grafana Security Hardening Script
# Purpose: Implement Opus security recommendations
# Date: October 22, 2025

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "  🔒 Grafana Security Hardening"
echo "==========================================${NC}"
echo ""

# Check if Grafana is installed
if ! systemctl is-active --quiet grafana-server; then
    echo -e "${RED}Error: Grafana is not running${NC}"
    echo "Please run ./phase3_setup.sh first"
    exit 1
fi

echo -e "${YELLOW}🔐 Step 1: Change Default Admin Password${NC}"
echo ""
echo "Current default credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
read -p "Enter new admin password: " -s NEW_PASSWORD
echo ""
read -p "Confirm new password: " -s NEW_PASSWORD_CONFIRM
echo ""

if [ "$NEW_PASSWORD" != "$NEW_PASSWORD_CONFIRM" ]; then
    echo -e "${RED}Error: Passwords do not match${NC}"
    exit 1
fi

if [ ${#NEW_PASSWORD} -lt 8 ]; then
    echo -e "${RED}Error: Password must be at least 8 characters${NC}"
    exit 1
fi

# Change password using Grafana CLI
echo -e "${YELLOW}Updating admin password...${NC}"
sudo grafana-cli admin reset-admin-password "$NEW_PASSWORD" >/dev/null 2>&1

echo -e "${GREEN}✓ Admin password changed successfully${NC}"
echo ""

# Step 2: Configure firewall (optional)
echo -e "${YELLOW}🔥 Step 2: Configure Firewall (Optional)${NC}"
echo ""
echo "Do you want to restrict Grafana access to localhost only?"
read -p "This prevents remote access. (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if UFW is available
    if command -v ufw &> /dev/null; then
        echo "Configuring UFW firewall..."
        
        # Allow from localhost
        sudo ufw allow from 127.0.0.1 to any port 3000 comment "Grafana localhost only"
        
        # Deny from everywhere else
        sudo ufw deny 3000 comment "Block remote Grafana access"
        
        echo -e "${GREEN}✓ Firewall rules configured${NC}"
        echo "  - Grafana accessible only from localhost"
        echo ""
    else
        echo -e "${YELLOW}⚠ UFW not found. Install with: sudo apt install ufw${NC}"
        echo ""
    fi
else
    echo "Skipping firewall configuration"
    echo ""
fi

# Step 3: Add basic auth to metrics endpoint (optional)
echo -e "${YELLOW}🔐 Step 3: Secure Metrics Endpoint (Optional)${NC}"
echo ""
echo "Do you want to add basic authentication to the metrics endpoint?"
echo "Note: This requires modifying redis_to_questdb_v2.py"
read -p "(y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}To add basic auth to metrics endpoint:${NC}"
    echo ""
    echo "1. Install python package:"
    echo "   pip3 install prometheus-flask-exporter"
    echo ""
    echo "2. Add to redis_to_questdb_v2.py:"
    cat << 'PYTHON_CODE'
   
   from flask import Flask, request, Response
   import functools
   
   def check_auth(username, password):
       return username == 'metrics' and password == 'YOUR_SECURE_PASSWORD'
   
   def authenticate():
       return Response('Authentication required', 401,
           {'WWW-Authenticate': 'Basic realm="Login Required"'})
   
   def requires_auth(f):
       @functools.wraps(f)
       def decorated(*args, **kwargs):
           auth = request.authorization
           if not auth or not check_auth(auth.username, auth.password):
               return authenticate()
           return f(*args, **kwargs)
       return decorated
   
   @app.route('/metrics')
   @requires_auth
   def metrics():
       # existing metrics code
PYTHON_CODE
    echo ""
    echo "3. Update Grafana datasource URL to:"
    echo "   http://metrics:YOUR_SECURE_PASSWORD@localhost:9091"
    echo ""
else
    echo "Skipping metrics endpoint authentication"
    echo ""
fi

# Step 4: Disable anonymous access
echo -e "${YELLOW}🚫 Step 4: Disable Anonymous Access${NC}"
echo ""

GRAFANA_INI="/etc/grafana/grafana.ini"

# Backup original config
sudo cp "$GRAFANA_INI" "$GRAFANA_INI.backup.$(date +%Y%m%d_%H%M%S)"

# Disable anonymous access
sudo sed -i 's/;enabled = false/enabled = false/' "$GRAFANA_INI"
sudo sed -i '/\[auth.anonymous\]/a enabled = false' "$GRAFANA_INI"

echo -e "${GREEN}✓ Anonymous access disabled${NC}"
echo ""

# Step 5: Enable HTTP strict transport security
echo -e "${YELLOW}🔒 Step 5: Enhanced Security Headers${NC}"
echo ""

# Add security headers
if ! grep -q "strict_transport_security" "$GRAFANA_INI"; then
    sudo bash -c "cat >> $GRAFANA_INI << 'INI_CONFIG'

# Security hardening
[security]
admin_password = 
secret_key = 
disable_gravatar = true
cookie_secure = false
cookie_samesite = lax
allow_embedding = false
strict_transport_security = true
x_content_type_options = true
x_xss_protection = true
INI_CONFIG
"
    echo -e "${GREEN}✓ Security headers configured${NC}"
else
    echo -e "${YELLOW}Security headers already configured${NC}"
fi
echo ""

# Step 6: Restart Grafana
echo -e "${YELLOW}🔄 Step 6: Restarting Grafana...${NC}"
sudo systemctl restart grafana-server
sleep 3

if systemctl is-active --quiet grafana-server; then
    echo -e "${GREEN}✓ Grafana restarted successfully${NC}"
else
    echo -e "${RED}Error: Grafana failed to restart${NC}"
    echo "Check logs: sudo journalctl -u grafana-server -n 50"
    exit 1
fi
echo ""

# Summary
echo -e "${GREEN}=========================================="
echo "  ✅ Security Hardening Complete!"
echo "==========================================${NC}"
echo ""
echo "🔒 Security Improvements Applied:"
echo "   ✓ Admin password changed (from default)"
echo "   ✓ Anonymous access disabled"
echo "   ✓ Security headers enabled"
echo "   ✓ Configuration backed up"
echo ""
echo "📝 Next Steps:"
echo "   1. Test login: http://localhost:3000"
echo "   2. Use new admin password"
echo "   3. Configure notification channels (email/Slack)"
echo "   4. Review security logs regularly"
echo ""
echo "🔧 Additional Recommendations:"
echo "   - Enable HTTPS with SSL certificates"
echo "   - Set up regular backup of Grafana database"
echo "   - Implement API key rotation policy"
echo "   - Enable audit logging"
echo ""
echo -e "${BLUE}Access Grafana: http://localhost:3000${NC}"
echo -e "${BLUE}Username: admin${NC}"
echo -e "${BLUE}Password: [your new password]${NC}"
echo ""
