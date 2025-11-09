#!/bin/bash
# Grafana Installation Script for Market Data Monitoring
# Installs Grafana OSS, configures data sources, and sets up dashboards

set -e  # Exit on error

echo "=========================================="
echo "  Grafana Installation for Market Data"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}❌ Please do not run as root. Run with sudo when needed.${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Step 1: Installing Grafana OSS...${NC}"
echo ""

# Add Grafana repository
if [ ! -f /etc/apt/sources.list.d/grafana.list ]; then
    echo "Adding Grafana repository..."
    sudo apt-get install -y apt-transport-https software-properties-common wget
    
    # Add GPG key
    sudo mkdir -p /etc/apt/keyrings/
    wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
    
    # Add repository
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
    
    echo -e "${GREEN}✓ Repository added${NC}"
else
    echo -e "${GREEN}✓ Grafana repository already configured${NC}"
fi

# Update and install
echo "Installing Grafana..."
sudo apt-get update
sudo apt-get install -y grafana

echo -e "${GREEN}✓ Grafana installed${NC}"
echo ""

echo -e "${YELLOW}📝 Step 2: Configuring Grafana...${NC}"
echo ""

# Configure Grafana
GRAFANA_INI="/etc/grafana/grafana.ini"

# Backup original config
if [ ! -f "${GRAFANA_INI}.backup" ]; then
    sudo cp "$GRAFANA_INI" "${GRAFANA_INI}.backup"
    echo -e "${GREEN}✓ Backed up original configuration${NC}"
fi

# Set up provisioning directories
sudo mkdir -p /etc/grafana/provisioning/datasources
sudo mkdir -p /etc/grafana/provisioning/dashboards
sudo mkdir -p /var/lib/grafana/dashboards

echo -e "${GREEN}✓ Provisioning directories created${NC}"
echo ""

echo -e "${YELLOW}🔗 Step 3: Configuring Data Sources...${NC}"
echo ""

# Create Prometheus data source
cat << 'EOF' | sudo tee /etc/grafana/provisioning/datasources/prometheus.yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9091
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "5s"
      httpMethod: GET
EOF

echo -e "${GREEN}✓ Prometheus data source configured (port 9091 - batch writer metrics)${NC}"

# Create QuestDB data source (PostgreSQL wire protocol)
cat << 'EOF' | sudo tee /etc/grafana/provisioning/datasources/questdb.yaml
apiVersion: 1

datasources:
  - name: QuestDB
    type: postgres
    access: proxy
    url: localhost:8812
    database: qdb
    user: admin
    secureJsonData:
      password: quest
    jsonData:
      sslmode: disable
      postgresVersion: 1400
      timescaledb: false
    editable: true
EOF

echo -e "${GREEN}✓ QuestDB data source configured (PostgreSQL wire port 8812)${NC}"
echo ""

echo -e "${YELLOW}📊 Step 4: Setting up Dashboard Provisioning...${NC}"
echo ""

# Create dashboard provisioning config
cat << 'EOF' | sudo tee /etc/grafana/provisioning/dashboards/market-data.yaml
apiVersion: 1

providers:
  - name: 'Market Data Dashboards'
    orgId: 1
    folder: 'Market Data'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

echo -e "${GREEN}✓ Dashboard provisioning configured${NC}"
echo ""

echo -e "${YELLOW}🔧 Step 5: Enabling and Starting Grafana...${NC}"
echo ""

# Enable and start Grafana
sudo systemctl daemon-reload
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# Wait for Grafana to start
echo "Waiting for Grafana to start..."
sleep 5

# Check if Grafana is running
if sudo systemctl is-active --quiet grafana-server; then
    echo -e "${GREEN}✓ Grafana is running${NC}"
else
    echo -e "${RED}❌ Grafana failed to start${NC}"
    sudo systemctl status grafana-server
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "  ✅ Grafana Installation Complete!"
echo "==========================================${NC}"
echo ""
echo "📍 Access Information:"
echo "   URL: http://localhost:3000"
echo "   Default Username: admin"
echo "   Default Password: admin"
echo "   (You'll be prompted to change password on first login)"
echo ""
echo "📊 Data Sources Configured:"
echo "   ✓ Prometheus (http://localhost:8000)"
echo "   ✓ QuestDB (PostgreSQL wire port 8812)"
echo ""
echo "📝 Next Steps:"
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Login with admin/admin"
echo "   3. Change your password"
echo "   4. Run: ./setup_dashboards.sh to import dashboards"
echo ""
echo "📁 Configuration Files:"
echo "   - Config: /etc/grafana/grafana.ini"
echo "   - Data Sources: /etc/grafana/provisioning/datasources/"
echo "   - Dashboards: /var/lib/grafana/dashboards/"
echo ""
echo "🔍 Useful Commands:"
echo "   sudo systemctl status grafana-server"
echo "   sudo systemctl restart grafana-server"
echo "   sudo journalctl -u grafana-server -f"
echo ""
