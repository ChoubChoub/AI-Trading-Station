# Operational Controls Implementation - Trading Station Command & Control

## Overview

Implement interactive controls for the trading system beyond passive monitoring:
- **Kill Switch**: Emergency stop all trading activity
- **Data Feed Control**: Start/stop market data collection
- **Service Management**: Start/stop individual services (Redis, QuestDB, collectors)
- **System Power**: Graceful shutdown/restart

---

## Architecture Options

### **Option 1: Grafana Plugins (Limited Controls)**

#### Available Approaches
1. **Button Panel Plugin** (3rd party)
2. **Webhook Alerts** (native, but triggered by thresholds, not buttons)
3. **Data Links** (opens URLs, can't POST commands)

#### Limitations
⚠️ **Grafana is designed for monitoring, not control**
- No native button/control panel support
- 3rd party plugins are limited and unmaintained
- Security concerns: Grafana users shouldn't have direct system control
- No audit trail for actions
- Cannot pass complex parameters
- Webhook alerts only trigger on metric thresholds (not manual button press)

#### Example: Button Panel Plugin
```json
// Limited functionality, unmaintained plugin
{
  "type": "cloudspout-button-panel",
  "requests": [
    {
      "method": "POST",
      "url": "http://localhost:8080/api/kill-switch",
      "payload": "{\"action\": \"stop_all\"}"
    }
  ]
}
```

**Verdict**: ❌ Not recommended for production trading controls

---

### **Option 2: Custom Control Dashboard (Web App) - RECOMMENDED**

Build a dedicated control interface separate from Grafana.

#### Architecture
```
┌─────────────────────────────────────────────────────┐
│                  User Interface                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Grafana (Monitoring)          Control Dashboard    │
│  Port 3000                     Port 8080            │
│  - View metrics                - Execute commands   │
│  - Alerts                      - Start/stop services│
│  - Read-only                   - Kill switch        │
│                                - Audit logs         │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│              Control API (FastAPI/Flask)            │
│  - Authentication (JWT tokens)                      │
│  - Authorization (role-based access)                │
│  - Rate limiting                                    │
│  - Audit logging                                    │
│  - Command validation                               │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│              System Services                        │
│  - Systemd services                                 │
│  - Trading processes                                │
│  - Data collectors                                  │
└─────────────────────────────────────────────────────┘
```

#### Tech Stack Options

**Option A: FastAPI + React (Modern, Production-Grade)**
```
Backend:  FastAPI (Python) - REST API
Frontend: React + Material-UI - Web interface
Auth:     JWT tokens + 2FA
Database: SQLite for audit logs
Deploy:   Systemd service on port 8080
```

**Option B: Streamlit (Rapid Development)**
```
Framework: Streamlit (Python) - All-in-one
Benefits:  Quick to build, Python-native
Drawbacks: Less customizable, not as secure
Use case:  Internal tool, trusted network
```

**Option C: Flask + Simple HTML/JS (Lightweight)**
```
Backend:  Flask (Python) - Minimal API
Frontend: Vanilla JS + Bootstrap
Benefits: Simple, low overhead
Use case: Quick implementation
```

---

### **Option 3: Telegram Bot (Mobile-First Control)**

Control system via Telegram messages from your phone.

#### Architecture
```
Telegram App (Your Phone)
    ↓ (Encrypted messages)
Telegram Bot API
    ↓
Python Bot (python-telegram-bot)
    ↓
System Commands (systemctl, kill switch)
```

#### Example Commands
```
/status - Show system status
/kill - Emergency stop all trading
/start_feed - Start data collection
/stop_feed - Stop data collection
/restart redis - Restart Redis
/shutdown - Graceful system shutdown
/logs - Get recent error logs
```

#### Benefits
✅ Access from anywhere (mobile-first)
✅ Encrypted by Telegram
✅ Push notifications for alerts
✅ Simple to implement (200 lines Python)
✅ No web hosting needed
✅ Rate limiting built-in

#### Drawbacks
⚠️ Depends on Telegram service
⚠️ Internet required (can't use if offline)
⚠️ Text-based interface (less visual)

---

## Detailed Implementation: Option 2 (Control Dashboard - RECOMMENDED)

### System Design

#### 1. Control API Backend (FastAPI)

**Features**:
- RESTful API for all control operations
- JWT token authentication
- Role-based access control (Admin, Operator, Viewer)
- Rate limiting (prevent accidental spam)
- Comprehensive audit logging
- Command validation and safety checks
- Dry-run mode for testing
- Rollback capabilities

**API Endpoints**:
```python
# Kill Switch
POST /api/kill-switch
  - Stop all trading immediately
  - Close all positions (if trading live)
  - Stop data collection
  - Log emergency stop

# Data Feed Control
POST /api/datafeed/start
POST /api/datafeed/stop
GET  /api/datafeed/status

# Service Management
POST /api/service/{name}/start    # redis, questdb, collectors, etc.
POST /api/service/{name}/stop
POST /api/service/{name}/restart
GET  /api/service/{name}/status
GET  /api/services/all

# System Control
POST /api/system/shutdown
POST /api/system/restart
GET  /api/system/health

# Audit & Logs
GET  /api/audit/logs
GET  /api/system/logs/{service}

# Configuration
GET  /api/config/symbols        # List monitored symbols
POST /api/config/symbols/add
POST /api/config/symbols/remove
```

#### 2. Frontend Dashboard (React/Streamlit)

**Dashboard Layout**:
```
┌────────────────────────────────────────────────────────┐
│  AI Trading Station - Control Panel                    │
│  User: admin@trading.com           Status: ✅ Online   │
├────────────────────────────────────────────────────────┤
│                                                         │
│  🚨 EMERGENCY CONTROLS                                 │
│  ┌─────────────────────────────────────────────┐      │
│  │  [🛑 KILL SWITCH]  Stop All Trading          │      │
│  │  Requires confirmation + password            │      │
│  └─────────────────────────────────────────────┘      │
│                                                         │
│  📊 DATA FEED CONTROL                                  │
│  ┌─────────────────────────────────────────────┐      │
│  │  Status: 🟢 Active (5/5 symbols)            │      │
│  │  [⏸️ Pause]  [▶️ Resume]  [🔄 Restart]        │      │
│  │  BTCUSDT ✅  ETHUSDT ✅  BNBUSDT ✅          │      │
│  │  SOLUSDT ✅  ADAUSDT ✅                      │      │
│  └─────────────────────────────────────────────┘      │
│                                                         │
│  ⚙️ SERVICE MANAGEMENT                                 │
│  ┌─────────────────────────────────────────────┐      │
│  │  Redis         🟢 Running  [Stop] [Restart] │      │
│  │  QuestDB       🟢 Running  [Stop] [Restart] │      │
│  │  Collectors    🟢 Running  [Stop] [Restart] │      │
│  │  Batch Writer  🟢 Running  [Stop] [Restart] │      │
│  │  Prometheus    🟢 Running  [Stop] [Restart] │      │
│  │  Grafana       🟢 Running  [Stop] [Restart] │      │
│  └─────────────────────────────────────────────┘      │
│                                                         │
│  💻 SYSTEM CONTROL                                     │
│  ┌─────────────────────────────────────────────┐      │
│  │  [📊 View Logs]  [🔄 Restart System]         │      │
│  │  [⏸️ Shutdown]  (Requires admin password)    │      │
│  └─────────────────────────────────────────────┘      │
│                                                         │
│  📜 AUDIT LOG (Last 10 actions)                        │
│  ┌─────────────────────────────────────────────┐      │
│  │  2025-10-22 21:15:23  admin   Started Redis │      │
│  │  2025-10-22 21:12:10  admin   Stopped feed  │      │
│  │  2025-10-22 20:58:45  admin   Kill switch   │      │
│  └─────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────┘
```

#### 3. Backend Implementation Example (FastAPI)

**File Structure**:
```
Trading/
├── ControlPanel/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── auth.py              # JWT authentication
│   │   ├── models.py            # Pydantic models
│   │   ├── services.py          # Service control logic
│   │   └── audit.py             # Audit logging
│   ├── frontend/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── styles.css
│   ├── config.yaml
│   └── requirements.txt
└── control-panel.service        # Systemd service
```

**Core Implementation (main.py)**:
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import subprocess
import logging
from datetime import datetime
import sqlite3

app = FastAPI(title="Trading Control Panel")
security = HTTPBearer()

# Audit logging
audit_db = sqlite3.connect('/var/log/trading/audit.db', check_same_thread=False)
audit_db.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user TEXT,
        action TEXT,
        service TEXT,
        result TEXT,
        ip_address TEXT
    )
''')

def log_audit(user: str, action: str, service: str, result: str, ip: str):
    """Log all control actions for compliance"""
    audit_db.execute(
        "INSERT INTO audit_log VALUES (NULL, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), user, action, service, result, ip)
    )
    audit_db.commit()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    # TODO: Implement proper JWT verification
    # For now, basic check
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return "admin"  # Return user from token

# Kill Switch - Most critical endpoint
@app.post("/api/kill-switch")
async def kill_switch(
    confirmation: str,
    user: str = Depends(verify_token)
):
    """Emergency stop all trading activity"""
    
    if confirmation != "CONFIRM_KILL_SWITCH":
        raise HTTPException(status_code=400, detail="Confirmation required")
    
    try:
        # Stop all trading services
        services = [
            'websocket-collectors',
            'batch-writer',
            'trading-engine'  # If you have a trading engine
        ]
        
        results = []
        for service in services:
            result = subprocess.run(
                ['sudo', 'systemctl', 'stop', service],
                capture_output=True,
                text=True,
                timeout=10
            )
            results.append(f"{service}: {'stopped' if result.returncode == 0 else 'failed'}")
        
        # Log to audit trail
        log_audit(user, "KILL_SWITCH", "all_trading", "SUCCESS", "127.0.0.1")
        
        # Send alert notification (email, Telegram, etc.)
        # send_alert(f"🚨 KILL SWITCH ACTIVATED by {user}")
        
        return {
            "status": "success",
            "message": "All trading services stopped",
            "details": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        log_audit(user, "KILL_SWITCH", "all_trading", f"FAILED: {str(e)}", "127.0.0.1")
        raise HTTPException(status_code=500, detail=str(e))

# Data Feed Control
@app.post("/api/datafeed/{action}")
async def control_datafeed(
    action: str,
    user: str = Depends(verify_token)
):
    """Start/Stop market data collection"""
    
    if action not in ['start', 'stop', 'restart']:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    try:
        result = subprocess.run(
            ['sudo', 'systemctl', action, 'websocket-collectors'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        log_audit(user, f"DATAFEED_{action.upper()}", "websocket-collectors", 
                 "SUCCESS" if result.returncode == 0 else "FAILED", "127.0.0.1")
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        
        return {
            "status": "success",
            "action": action,
            "service": "websocket-collectors",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Command timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Service Management
@app.post("/api/service/{service_name}/{action}")
async def control_service(
    service_name: str,
    action: str,
    user: str = Depends(verify_token)
):
    """Control individual system services"""
    
    # Whitelist of controllable services
    allowed_services = [
        'redis-server',
        'questdb',
        'websocket-collectors',
        'batch-writer',
        'prometheus',
        'grafana-server'
    ]
    
    if service_name not in allowed_services:
        raise HTTPException(status_code=400, detail="Service not allowed")
    
    if action not in ['start', 'stop', 'restart', 'status']:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    try:
        result = subprocess.run(
            ['sudo', 'systemctl', action, service_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        log_audit(user, f"{action.upper()}", service_name, 
                 "SUCCESS" if result.returncode == 0 else "FAILED", "127.0.0.1")
        
        return {
            "status": "success",
            "service": service_name,
            "action": action,
            "output": result.stdout,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get all service statuses
@app.get("/api/services/status")
async def get_all_services_status(user: str = Depends(verify_token)):
    """Get status of all monitored services"""
    
    services = [
        'redis-server',
        'questdb',
        'websocket-collectors',
        'batch-writer',
        'prometheus',
        'grafana-server'
    ]
    
    statuses = {}
    for service in services:
        result = subprocess.run(
            ['systemctl', 'is-active', service],
            capture_output=True,
            text=True
        )
        statuses[service] = result.stdout.strip()  # 'active', 'inactive', etc.
    
    return statuses

# Audit Log Retrieval
@app.get("/api/audit/logs")
async def get_audit_logs(
    limit: int = 100,
    user: str = Depends(verify_token)
):
    """Retrieve audit log entries"""
    
    cursor = audit_db.execute(
        "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    
    logs = []
    for row in cursor.fetchall():
        logs.append({
            "id": row[0],
            "timestamp": row[1],
            "user": row[2],
            "action": row[3],
            "service": row[4],
            "result": row[5],
            "ip_address": row[6]
        })
    
    return logs

# System shutdown (requires extra confirmation)
@app.post("/api/system/shutdown")
async def system_shutdown(
    confirmation: str,
    password: str,
    user: str = Depends(verify_token)
):
    """Graceful system shutdown"""
    
    if confirmation != "CONFIRM_SHUTDOWN" or password != "admin-password":
        raise HTTPException(status_code=403, detail="Invalid confirmation")
    
    log_audit(user, "SYSTEM_SHUTDOWN", "system", "INITIATED", "127.0.0.1")
    
    # Graceful shutdown sequence
    subprocess.run(['sudo', 'systemctl', 'stop', 'websocket-collectors'])
    subprocess.run(['sudo', 'systemctl', 'stop', 'batch-writer'])
    subprocess.run(['sudo', 'systemctl', 'stop', 'questdb'])
    subprocess.run(['sudo', 'systemctl', 'stop', 'redis-server'])
    
    # Schedule system shutdown in 1 minute (gives time to respond)
    subprocess.Popen(['sudo', 'shutdown', '-h', '+1'])
    
    return {"status": "shutdown_scheduled", "time": "1 minute"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

#### 4. Frontend Implementation (Simple HTML/JS)

**index.html**:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Control Panel</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .control-section {
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .emergency {
            background: #ff4444;
            border: 3px solid #cc0000;
        }
        button {
            background: #0f3460;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            font-size: 14px;
        }
        button:hover {
            background: #1a5490;
        }
        .kill-switch-btn {
            background: #cc0000;
            font-size: 18px;
            font-weight: bold;
            padding: 15px 30px;
        }
        .kill-switch-btn:hover {
            background: #ff0000;
        }
        .service-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        .status-active { color: #00ff00; }
        .status-inactive { color: #ff4444; }
        .audit-log {
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AI Trading Station - Control Panel</h1>
            <p>User: <span id="username">admin</span> | Status: <span id="system-status">🟢 Online</span></p>
        </div>

        <!-- Kill Switch -->
        <div class="control-section emergency">
            <h2>🚨 EMERGENCY CONTROLS</h2>
            <button class="kill-switch-btn" onclick="activateKillSwitch()">
                🛑 KILL SWITCH - STOP ALL TRADING
            </button>
            <p><small>Requires confirmation dialog</small></p>
        </div>

        <!-- Data Feed Control -->
        <div class="control-section">
            <h2>📊 DATA FEED CONTROL</h2>
            <p>Status: <span id="datafeed-status">🟢 Active (5/5 symbols)</span></p>
            <button onclick="controlDatafeed('stop')">⏸️ Pause Feed</button>
            <button onclick="controlDatafeed('start')">▶️ Resume Feed</button>
            <button onclick="controlDatafeed('restart')">🔄 Restart Feed</button>
        </div>

        <!-- Service Management -->
        <div class="control-section">
            <h2>⚙️ SERVICE MANAGEMENT</h2>
            <div id="services-list">
                <!-- Populated by JavaScript -->
            </div>
        </div>

        <!-- Audit Log -->
        <div class="control-section">
            <h2>📜 AUDIT LOG (Last 20 actions)</h2>
            <div id="audit-log" class="audit-log">
                <!-- Populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:8080/api';
        const TOKEN = 'your-secret-token';  // TODO: Implement proper JWT auth

        async function apiCall(endpoint, method = 'GET', body = null) {
            const options = {
                method,
                headers: {
                    'Authorization': `Bearer ${TOKEN}`,
                    'Content-Type': 'application/json'
                }
            };
            if (body) options.body = JSON.stringify(body);
            
            const response = await fetch(`${API_URL}${endpoint}`, options);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        }

        async function activateKillSwitch() {
            const confirmation = prompt('Type "CONFIRM_KILL_SWITCH" to activate emergency stop:');
            if (confirmation !== 'CONFIRM_KILL_SWITCH') {
                alert('Cancelled');
                return;
            }
            
            try {
                const result = await apiCall('/kill-switch', 'POST', { confirmation });
                alert('✅ Kill switch activated! All trading stopped.');
                loadAuditLog();
            } catch (error) {
                alert('❌ Error: ' + error.message);
            }
        }

        async function controlDatafeed(action) {
            try {
                const result = await apiCall(`/datafeed/${action}`, 'POST');
                alert(`✅ Datafeed ${action} successful`);
                loadServiceStatus();
                loadAuditLog();
            } catch (error) {
                alert('❌ Error: ' + error.message);
            }
        }

        async function controlService(serviceName, action) {
            try {
                const result = await apiCall(`/service/${serviceName}/${action}`, 'POST');
                alert(`✅ ${serviceName} ${action} successful`);
                loadServiceStatus();
                loadAuditLog();
            } catch (error) {
                alert('❌ Error: ' + error.message);
            }
        }

        async function loadServiceStatus() {
            try {
                const statuses = await apiCall('/services/status');
                const container = document.getElementById('services-list');
                container.innerHTML = '';
                
                for (const [service, status] of Object.entries(statuses)) {
                    const isActive = status === 'active';
                    const statusClass = isActive ? 'status-active' : 'status-inactive';
                    
                    container.innerHTML += `
                        <div class="service-row">
                            <span>${service} <span class="${statusClass}">${isActive ? '🟢 Running' : '🔴 Stopped'}</span></span>
                            <div>
                                ${isActive ? 
                                    `<button onclick="controlService('${service}', 'stop')">Stop</button>` :
                                    `<button onclick="controlService('${service}', 'start')">Start</button>`
                                }
                                <button onclick="controlService('${service}', 'restart')">Restart</button>
                            </div>
                        </div>
                    `;
                }
            } catch (error) {
                console.error('Failed to load service status:', error);
            }
        }

        async function loadAuditLog() {
            try {
                const logs = await apiCall('/audit/logs?limit=20');
                const container = document.getElementById('audit-log');
                container.innerHTML = logs.map(log => 
                    `${log.timestamp} | ${log.user.padEnd(10)} | ${log.action.padEnd(20)} | ${log.service.padEnd(20)} | ${log.result}`
                ).join('\n');
            } catch (error) {
                console.error('Failed to load audit log:', error);
            }
        }

        // Auto-refresh every 5 seconds
        setInterval(() => {
            loadServiceStatus();
            loadAuditLog();
        }, 5000);

        // Initial load
        loadServiceStatus();
        loadAuditLog();
    </script>
</body>
</html>
```

#### 5. Deployment as Systemd Service

**control-panel.service**:
```ini
[Unit]
Description=Trading Control Panel API
After=network.target

[Service]
Type=simple
User=trading
Group=trading
WorkingDirectory=/home/youssefbahloul/ai-trading-station/Trading/ControlPanel
ExecStart=/usr/bin/python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

# Security
PrivateTmp=yes
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/log/trading

# Environment
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

#### 6. Security Configuration

**Sudoers Configuration** (allow control panel to manage services):
```bash
# /etc/sudoers.d/trading-control
trading ALL=(ALL) NOPASSWD: /bin/systemctl start websocket-collectors
trading ALL=(ALL) NOPASSWD: /bin/systemctl stop websocket-collectors
trading ALL=(ALL) NOPASSWD: /bin/systemctl restart websocket-collectors
trading ALL=(ALL) NOPASSWD: /bin/systemctl start batch-writer
trading ALL=(ALL) NOPASSWD: /bin/systemctl stop batch-writer
trading ALL=(ALL) NOPASSWD: /bin/systemctl restart batch-writer
trading ALL=(ALL) NOPASSWD: /bin/systemctl start redis-server
trading ALL=(ALL) NOPASSWD: /bin/systemctl stop redis-server
trading ALL=(ALL) NOPASSWD: /bin/systemctl restart redis-server
trading ALL=(ALL) NOPASSWD: /bin/systemctl start questdb
trading ALL=(ALL) NOPASSWD: /bin/systemctl stop questdb
trading ALL=(ALL) NOPASSWD: /bin/systemctl restart questdb
trading ALL=(ALL) NOPASSWD: /bin/systemctl status *
trading ALL=(ALL) NOPASSWD: /bin/systemctl is-active *
```

---

## Implementation: Option 3 (Telegram Bot)

### Quick Implementation

**telegram_control_bot.py**:
```python
import os
import subprocess
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Bot token from @BotFather
BOT_TOKEN = "your-telegram-bot-token"
AUTHORIZED_USERS = [123456789]  # Your Telegram user ID

def is_authorized(update: Update) -> bool:
    return update.effective_user.id in AUTHORIZED_USERS

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show system status"""
    if not is_authorized(update):
        await update.message.reply_text("❌ Unauthorized")
        return
    
    services = ['redis-server', 'questdb', 'websocket-collectors', 'batch-writer']
    status_text = "📊 **System Status**\n\n"
    
    for service in services:
        result = subprocess.run(['systemctl', 'is-active', service], capture_output=True, text=True)
        status = result.stdout.strip()
        emoji = "🟢" if status == "active" else "🔴"
        status_text += f"{emoji} {service}: {status}\n"
    
    await update.message.reply_text(status_text)

async def kill_switch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Emergency stop all trading"""
    if not is_authorized(update):
        await update.message.reply_text("❌ Unauthorized")
        return
    
    await update.message.reply_text("🚨 Activating kill switch...")
    
    services = ['websocket-collectors', 'batch-writer']
    for service in services:
        subprocess.run(['sudo', 'systemctl', 'stop', service])
    
    await update.message.reply_text("✅ Kill switch activated. All trading stopped.")

async def start_feed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start data collection"""
    if not is_authorized(update):
        await update.message.reply_text("❌ Unauthorized")
        return
    
    subprocess.run(['sudo', 'systemctl', 'start', 'websocket-collectors'])
    await update.message.reply_text("✅ Data feed started")

async def stop_feed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop data collection"""
    if not is_authorized(update):
        await update.message.reply_text("❌ Unauthorized")
        return
    
    subprocess.run(['sudo', 'systemctl', 'stop', 'websocket-collectors'])
    await update.message.reply_text("✅ Data feed stopped")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("kill", kill_switch))
    app.add_handler(CommandHandler("start_feed", start_feed))
    app.add_handler(CommandHandler("stop_feed", stop_feed))
    
    print("🤖 Telegram bot started")
    app.run_polling()

if __name__ == '__main__':
    main()
```

**Setup**:
```bash
# Install dependencies
pip3 install python-telegram-bot

# Create bot via @BotFather on Telegram
# Get your user ID from @userinfobot

# Run bot
python3 telegram_control_bot.py

# Or as systemd service
sudo systemctl enable telegram-control-bot
sudo systemctl start telegram-control-bot
```

---

## Comparison Matrix

| Feature | Grafana Plugins | Custom Dashboard | Telegram Bot |
|---------|----------------|------------------|--------------|
| **Ease of Implementation** | ⭐⭐ Limited options | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Very Easy |
| **User Experience** | ⭐ Poor | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Security** | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ Robust | ⭐⭐⭐ Good |
| **Audit Trail** | ⭐ None | ⭐⭐⭐⭐⭐ Complete | ⭐⭐⭐ Basic |
| **Mobile Access** | ⭐⭐ Via browser | ⭐⭐⭐⭐ Responsive | ⭐⭐⭐⭐⭐ Native |
| **Production Ready** | ❌ No | ✅ Yes | ✅ Yes |
| **Customization** | ⭐ Limited | ⭐⭐⭐⭐⭐ Full control | ⭐⭐⭐ Moderate |
| **Development Time** | 2 hours | 1-2 days | 2-3 hours |
| **Maintenance** | ⭐⭐ Plugin updates | ⭐⭐⭐ Own code | ⭐⭐⭐⭐ Minimal |

---

## Recommendation

### **Primary: Custom Control Dashboard (Option 2)**
- Most professional and feature-rich
- Complete audit trail for compliance
- Best security model (JWT, RBAC)
- Can be integrated with Nginx reverse proxy (same domain)
- Suitable for team access

### **Secondary: Telegram Bot (Option 3)**
- Fastest to implement
- Great for personal use
- Mobile-first access
- Can run alongside custom dashboard as backup

### **Avoid: Grafana Plugins (Option 1)**
- Too limited for production controls
- No proper audit trail
- Security concerns

---

## Next Steps

**Decide on implementation approach**:

1. **Full Production** (Custom Dashboard):
   - Implement FastAPI backend
   - Build React/Streamlit frontend
   - Deploy on port 8080
   - Integrate with Nginx (same SSL/auth as Grafana)
   - Estimated time: 2 days

2. **Quick Solution** (Telegram Bot):
   - Implement Python bot (2-3 hours)
   - Test with authorized users
   - Deploy as systemd service
   - Can add custom dashboard later

3. **Hybrid** (Both):
   - Telegram bot for mobile/emergency access
   - Custom dashboard for detailed control
   - Best of both worlds

**Questions**:
1. Do you need team access or just personal control?
2. Is mobile access critical?
3. What's the timeline? (Quick bot vs. full dashboard)
4. Do you have regulatory/compliance requirements for audit logs?

---

**Document Created**: October 22, 2025  
**Purpose**: Implement operational controls beyond monitoring  
**Status**: Awaiting decision on implementation approach
