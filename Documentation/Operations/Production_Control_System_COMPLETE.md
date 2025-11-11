# Production Access & Control System - Complete Implementation Package

**Version**: 1.0.0  
**Date**: 2025-11-11  
**Status**: Ready for Deployment

## Overview

This document provides the complete implementation package for the AI Trading Station Production Access & Control System. It includes:

1. Setup script to create directory structure
2. Implementation templates for all components
3. Configuration files
4. Deployment procedures
5. Security guidelines

## Phase 1: Execute Setup Script

The setup script `setup-control-panel.sh` has been created in the repository root. Execute it to create the directory structure:

```bash
cd /home/youssefbahloul/ai-trading-station
chmod +x setup-control-panel.sh
./setup-control-panel.sh
```

This script creates:
- `ControlPanel/` directory with proper structure
- `ControlPanel/api/` for backend code
- `ControlPanel/frontend/` for web UI
- `ControlPanel/config/` for configuration files
- `ControlPanel/tests/` for unit tests
- Configuration templates (requirements.txt, config.yaml, .env.example)

## Phase 2: Implementation Templates

After running the setup script, you'll need to create the implementation files. Below are complete templates for each component.

### File Structure

```
ControlPanel/
├── api/
│   ├── __init__.py                    (created by setup)
│   ├── main.py                        (TO CREATE)
│   ├── models.py                      (TO CREATE)
│   ├── auth.py                        (TO CREATE)
│   ├── controls.py                    (TO CREATE)
│   ├── nanokvm.py                     (TO CREATE)
│   └── audit.py                       (TO CREATE)
├── frontend/
│   ├── index.html                     (TO CREATE)
│   ├── mobile.html                    (TO CREATE)
│   ├── dashboard.js                   (TO CREATE)
│   └── styles.css                     (TO CREATE)
├── tests/
│   ├── __init__.py                    (created by setup)
│   └── test_controls.py               (TO CREATE)
├── config/
│   └── config.yaml                    (created by setup)
├── .env.example                       (created by setup)
├── .gitignore                         (created by setup)
└── requirements.txt                   (created by setup)
```

## Template: api/main.py

**Purpose**: Main FastAPI application with all endpoints

**Implementation Approach**: This is the core application file that:
- Sets up FastAPI app with CORS middleware
- Imports and registers all control endpoints
- Provides health check
- Serves static files for frontend

**Key Features**:
- Health check endpoint (no auth required)
- All API endpoints require Authelia authentication
- Structured error handling
- Request/response logging

**Implementation Notes**:
```python
"""
AI Trading Station - Control Panel Main Application

This is the main FastAPI application for the control panel.
It integrates all control endpoints and serves the frontend UI.

Security: All endpoints (except /health) require Authelia 2FA authentication.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Import endpoint modules (to be created)
from . import auth, controls, audit

app = FastAPI(
    title="AI Trading Station Control Panel",
    description="Secure remote access and operational controls",
    version="1.0.0"
)

# CORS middleware for mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ops.aistation.trading",
        "https://api.aistation.trading"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Health check endpoint (no authentication required)
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "trading-control-panel"
    }

# Include routers from other modules
# app.include_router(controls.router, prefix="/api", tags=["controls"])
# app.include_router(audit.router, prefix="/api/audit", tags=["audit"])

# Serve static files (frontend)
# app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False  # Set to True for development
    )
```

**TODO**:
1. Implement full endpoint routing
2. Add error handlers
3. Add request logging
4. Integrate audit logging
5. Add rate limiting middleware

## Template: api/models.py

**Purpose**: Pydantic models for request/response validation

**Implementation Approach**:
- Define data models for all API endpoints
- Include validation rules
- Provide clear error messages

**Key Models Needed**:
- KillSwitchRequest (confirmation field)
- ServiceActionRequest (service name, action)
- PowerActionRequest (action, confirmation)
- SystemStatusResponse (services, metrics, power)
- AuditLogEntry (timestamp, user, action, etc.)

**Implementation Template**:
```python
"""
Pydantic Models for Control Panel API

All request/response models with validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
from datetime import datetime

class KillSwitchRequest(BaseModel):
    """Request model for emergency kill switch"""
    confirmation: str = Field(..., description="Must be 'CONFIRM_EMERGENCY_STOP'")
    
    @validator('confirmation')
    def validate_confirmation(cls, v):
        if v != "CONFIRM_EMERGENCY_STOP":
            raise ValueError("Invalid confirmation string")
        return v

class ServiceActionRequest(BaseModel):
    """Request model for service management"""
    service: str = Field(..., description="Service name")
    action: str = Field(..., description="Action: start, stop, restart, status")
    
    @validator('action')
    def validate_action(cls, v):
        allowed = ['start', 'stop', 'restart', 'status']
        if v not in allowed:
            raise ValueError(f"Action must be one of {allowed}")
        return v

class PowerActionRequest(BaseModel):
    """Request model for power control"""
    action: str = Field(..., description="Action: on, off, status")
    confirmation: Optional[str] = Field(None, description="Required for 'off' action")
    
    @validator('confirmation', always=True)
    def validate_confirmation(cls, v, values):
        if values.get('action') == 'off' and v != "CONFIRM_POWER_OFF":
            raise ValueError("Power off requires confirmation")
        return v

class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    timestamp: datetime
    services: Dict[str, bool]
    metrics: Dict[str, any]
    power: Dict[str, str]

class AuditLogEntry(BaseModel):
    """Audit log entry model"""
    timestamp: datetime
    user: str
    action: str
    target: str
    result: str
    details: Optional[Dict] = None
    previous_hash: str
    hash: str
```

## Template: api/auth.py

**Purpose**: Authelia authentication integration

**Implementation Approach**:
- Parse Authelia headers from requests
- Verify user authentication
- Check user groups for authorization
- Provide dependency injection for FastAPI

**Security Requirements**:
- All control endpoints must verify authentication
- Admin operations require 'admins' group
- Power control restricted to specific user

**Implementation Template**:
```python
"""
Authentication Integration with Authelia

Parses Authelia forward-auth headers and verifies user access.
"""

from fastapi import Request, HTTPException, Depends
from typing import Dict, List

class AuthInfo:
    """User authentication information from Authelia"""
    
    def __init__(self, username: str, groups: List[str]):
        self.username = username
        self.groups = groups
        self.is_admin = "admins" in groups
        self.is_operator = "operators" in groups

async def get_auth_info(request: Request) -> AuthInfo:
    """
    Extract authentication info from Authelia headers.
    
    Authelia sets these headers:
    - Remote-User: authenticated username
    - Remote-Groups: comma-separated list of groups
    
    Args:
        request: FastAPI request object
        
    Returns:
        AuthInfo object with user details
        
    Raises:
        HTTPException: If user not authenticated
    """
    username = request.headers.get("Remote-User")
    groups_header = request.headers.get("Remote-Groups", "")
    groups = [g.strip() for g in groups_header.split(",") if g.strip()]
    
    if not username:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please login via Authelia."
        )
    
    return AuthInfo(username=username, groups=groups)

async def require_admin(auth_info: AuthInfo = Depends(get_auth_info)) -> AuthInfo:
    """
    Dependency that requires admin role.
    
    Usage:
        @app.post("/api/admin-endpoint")
        async def admin_endpoint(auth: AuthInfo = Depends(require_admin)):
            # Only admins can access this
            pass
    """
    if not auth_info.is_admin:
        raise HTTPException(
            status_code=403,
            detail=f"Admin access required. User '{auth_info.username}' is not an admin."
        )
    return auth_info

async def require_primary_admin(
    auth_info: AuthInfo = Depends(require_admin)
) -> AuthInfo:
    """
    Dependency that requires primary admin (specific user).
    
    Used for power control operations.
    """
    if auth_info.username != "youssef":  # TODO: Make configurable
        raise HTTPException(
            status_code=403,
            detail=f"Power control restricted to primary admin only."
        )
    return auth_info
```

## Template: api/controls.py

**Purpose**: Control operations (kill switch, service management, etc.)

**Implementation Approach**:
- Implement all control endpoints
- Use subprocess for systemd control
- Integrate audit logging
- Handle errors gracefully

**Key Operations**:
1. Emergency kill switch
2. Data feed control (start/stop/restart)
3. Service management (individual services)
4. System status monitoring

**Implementation Template**:
```python
"""
Control Operations for Trading System

Implements all control endpoints: kill switch, service management, status.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
import subprocess
import asyncio
from datetime import datetime

from .auth import AuthInfo, require_admin
from .models import (
    KillSwitchRequest,
    ServiceActionRequest,
    SystemStatusResponse
)
from .audit import AuditLogger

router = APIRouter()
audit_logger = AuditLogger()

# Service name mappings
SERVICE_MAP = {
    "redis": "redis-hft",
    "questdb": "questdb",
    "prometheus": "prometheus",
    "grafana": "grafana-server",
    "trades": "binance-trades",
    "orderbook": "binance-bookticker",
    "writer": "batch-writer"
}

TRADING_SERVICES = ["binance-trades", "binance-bookticker", "batch-writer"]

@router.post("/kill-switch")
async def emergency_kill_switch(
    request: KillSwitchRequest,
    auth: AuthInfo = Depends(require_admin)
):
    """
    EMERGENCY: Stop all trading activity immediately.
    
    This endpoint stops all trading services and flushes Redis.
    Requires admin role and typed confirmation.
    
    Security: Logged to audit trail. Sends emergency notifications.
    """
    try:
        results = []
        
        # Stop all trading services
        for service in TRADING_SERVICES:
            try:
                result = subprocess.run(
                    ["sudo", "systemctl", "stop", service],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                results.append({
                    "service": service,
                    "stopped": result.returncode == 0,
                    "output": result.stdout if result.returncode == 0 else result.stderr
                })
            except subprocess.TimeoutExpired:
                results.append({
                    "service": service,
                    "stopped": False,
                    "error": "Timeout"
                })
            except Exception as e:
                results.append({
                    "service": service,
                    "stopped": False,
                    "error": str(e)
                })
        
        # Flush Redis (optional - be careful!)
        # subprocess.run(["redis-cli", "FLUSHALL"], timeout=5)
        
        # Log to audit trail
        audit_logger.log(
            user=auth.username,
            action="KILL_SWITCH_ACTIVATED",
            target="all_trading_services",
            result="SUCCESS",
            details={"services_stopped": results}
        )
        
        # TODO: Send emergency notifications (Telegram, email, SMS)
        
        return {
            "status": "emergency_stop_executed",
            "timestamp": datetime.utcnow().isoformat(),
            "services_stopped": len([r for r in results if r.get("stopped")]),
            "total_services": len(results),
            "results": results
        }
        
    except Exception as e:
        audit_logger.log(
            user=auth.username,
            action="KILL_SWITCH_FAILED",
            target="system",
            result="ERROR",
            details={"error": str(e)}
        )
        raise HTTPException(500, f"Kill switch failed: {str(e)}")

@router.post("/datafeed/{action}")
async def control_datafeed(
    action: str,
    auth: AuthInfo = Depends(require_admin)
):
    """
    Control market data collection.
    
    Actions:
    - start: Start all data feeds
    - stop: Stop all data feeds
    - restart: Restart all data feeds
    - status: Get feed status
    """
    if action not in ["start", "stop", "restart", "status"]:
        raise HTTPException(400, f"Invalid action: {action}")
    
    try:
        # Use the datafeed command if it exists
        result = subprocess.run(
            ["datafeed", action],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        success = result.returncode == 0
        
        audit_logger.log(
            user=auth.username,
            action=f"DATAFEED_{action.upper()}",
            target="market_data_collectors",
            result="SUCCESS" if success else "FAILED",
            details={
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
        )
        
        return {
            "action": action,
            "success": success,
            "output": result.stdout,
            "error": result.stderr if not success else None
        }
        
    except FileNotFoundError:
        raise HTTPException(404, "datafeed command not found")
    except subprocess.TimeoutExpired:
        raise HTTPException(408, "Command timed out")
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/service/{service}/{action}")
async def manage_service(
    service: str,
    action: str,
    auth: AuthInfo = Depends(require_admin)
):
    """
    Manage individual system services.
    
    Allowed services: redis, questdb, prometheus, grafana, trades, orderbook, writer
    Actions: start, stop, restart, status
    """
    if service not in SERVICE_MAP:
        raise HTTPException(400, f"Service '{service}' not allowed")
    
    if action not in ["start", "stop", "restart", "status"]:
        raise HTTPException(400, f"Invalid action: {action}")
    
    actual_service = SERVICE_MAP[service]
    
    try:
        result = subprocess.run(
            ["sudo", "systemctl", action, actual_service],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        success = result.returncode == 0
        
        audit_logger.log(
            user=auth.username,
            action=f"SERVICE_{action.upper()}",
            target=actual_service,
            result="SUCCESS" if success else "FAILED",
            details={
                "service": service,
                "output": result.stdout,
                "error": result.stderr if not success else None
            }
        )
        
        return {
            "service": service,
            "action": action,
            "success": success,
            "output": result.stdout,
            "error": result.stderr if not success else None
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(408, f"Service {action} timed out")
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/status/overview")
async def get_system_status(auth: AuthInfo = Depends(get_auth_info)):
    """
    Get comprehensive system status.
    
    Returns:
    - Service status (active/inactive)
    - Redis connectivity and metrics
    - Power status (if NanoKVM available)
    - Stream message counts
    """
    status = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
        "metrics": {},
        "power": {}
    }
    
    # Check all services
    for service_name, systemd_name in SERVICE_MAP.items():
        try:
            result = subprocess.run(
                ["systemctl", "is-active", systemd_name],
                capture_output=True,
                text=True,
                timeout=2
            )
            status["services"][service_name] = result.stdout.strip() == "active"
        except:
            status["services"][service_name] = False
    
    # Get Redis metrics
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        status["metrics"]["redis_connected"] = r.ping()
        
        # Get stream lengths
        try:
            status["metrics"]["trades_stream"] = r.xlen("market:binance_spot:trades:BTCUSDT")
            status["metrics"]["orderbook_stream"] = r.xlen("market:binance_spot:orderbook:BTCUSDT")
        except:
            pass
            
    except:
        status["metrics"]["redis_connected"] = False
    
    # Get power status (if NanoKVM available)
    # TODO: Integrate with NanoKVM API
    status["power"]["status"] = "unknown"
    
    return status
```

**TODO**:
1. Add timeout handling
2. Add retry logic for critical operations
3. Implement emergency notifications
4. Add Redis flush confirmation
5. Enhance error messages

## Template: api/nanokvm.py

**Purpose**: NanoKVM API integration for power management

**Implementation Approach**:
- HTTP client for NanoKVM REST API
- Authentication token management
- Power control operations (on/off/status)
- Graceful shutdown sequences

**Security Notes**:
- Only primary admin can control power
- Power off requires confirmation
- Self-signed certificate handling
- Network restrictions recommended

**Implementation Template**:
```python
"""
NanoKVM API Integration for Power Management

Provides remote power control via NanoKVM device.
"""

import aiohttp
import asyncio
from typing import Optional, Dict
from datetime import datetime, timedelta

class NanoKVMClient:
    """
    Client for NanoKVM REST API.
    
    Handles authentication, power control, and error handling.
    """
    
    def __init__(
        self,
        base_url: str = "https://210.6.8.5:40443",
        verify_ssl: bool = False
    ):
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.auth_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
    
    async def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate with NanoKVM API.
        
        Args:
            username: NanoKVM username
            password: NanoKVM password
            
        Returns:
            True if authentication successful
            
        Raises:
            aiohttp.ClientError: On connection errors
        """
        try:
            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{self.base_url}/api/auth/login",
                    json={"username": username, "password": password},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.auth_token = data.get("token")
                        # Assume token valid for 1 hour
                        self.token_expiry = datetime.utcnow() + timedelta(hours=1)
                        return True
                    else:
                        return False
        except Exception as e:
            print(f"NanoKVM authentication error: {e}")
            return False
    
    async def _ensure_authenticated(self):
        """Ensure we have a valid auth token."""
        if not self.auth_token or (
            self.token_expiry and datetime.utcnow() >= self.token_expiry
        ):
            # TODO: Re-authenticate with stored credentials
            raise Exception("NanoKVM not authenticated")
    
    async def get_power_status(self) -> Optional[Dict]:
        """
        Get current power status.
        
        Returns:
            Dictionary with power status information
            None if request fails
        """
        await self._ensure_authenticated()
        
        try:
            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                async with session.get(
                    f"{self.base_url}/api/power/status",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        return None
        except Exception as e:
            print(f"NanoKVM status error: {e}")
            return None
    
    async def power_on(self) -> bool:
        """
        Power on the system.
        
        Returns:
            True if successful
        """
        return await self._power_action("on")
    
    async def power_off(self) -> bool:
        """
        Power off the system.
        
        WARNING: This will shut down the system!
        Ensure all services are stopped first.
        
        Returns:
            True if successful
        """
        return await self._power_action("off")
    
    async def power_reset(self) -> bool:
        """
        Force reset the system.
        
        WARNING: This forces an immediate reset!
        Use only in emergencies.
        
        Returns:
            True if successful
        """
        return await self._power_action("reset")
    
    async def _power_action(self, action: str) -> bool:
        """Execute a power action."""
        await self._ensure_authenticated()
        
        try:
            connector = aiohttp.TCPConnector(ssl=self.verify_ssl)
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = {"Authorization": f"Bearer {self.auth_token}"}
                async with session.post(
                    f"{self.base_url}/api/power/{action}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            print(f"NanoKVM power {action} error: {e}")
            return False

# Global instance
nanokvm_client = NanoKVMClient()

async def graceful_shutdown_sequence():
    """
    Execute graceful shutdown before powering off.
    
    Steps:
    1. Stop all trading services
    2. Wait for buffers to flush
    3. Stop Redis persistence
    4. Power off via NanoKVM
    """
    # TODO: Implement graceful shutdown logic
    pass
```

**TODO**:
1. Implement token refresh logic
2. Add graceful shutdown sequence
3. Handle network timeouts
4. Add retry logic for critical operations
5. Store credentials securely

## Template: api/audit.py

**Purpose**: Immutable audit logging with blockchain-style hash chaining

**Implementation Approach**:
- Each log entry contains hash of previous entry
- SHA-256 hashing for tamper detection
- Persistent storage in JSON Lines format
- Optional remote syslog forwarding

**Security Requirements**:
- Logs must be tamper-evident
- Chain integrity verification
- Secure log storage with proper permissions

**Implementation Template**:
```python
"""
Immutable Audit Logging System

Implements blockchain-style audit logging with hash chaining
for tamper detection.
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

class AuditLogger:
    """
    Audit logger with blockchain-style immutability.
    
    Each log entry contains:
    - Timestamp
    - User
    - Action
    - Target
    - Result
    - Details (optional)
    - Hash of previous entry
    - Hash of this entry
    """
    
    def __init__(self, log_file: str = "/var/log/trading/audit.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.touch(exist_ok=True)
        
        # Load last entry to get chain head
        self.last_hash = self._get_last_hash()
    
    def _get_last_hash(self) -> str:
        """Get hash of last entry in log file."""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    return last_entry.get("hash", "0")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return "0"  # Genesis entry
    
    def log(
        self,
        user: str,
        action: str,
        target: str,
        result: str,
        details: Optional[Dict] = None
    ) -> Dict:
        """
        Create and append audit log entry.
        
        Args:
            user: Username performing action
            action: Action performed (e.g., "KILL_SWITCH_ACTIVATED")
            target: Target of action (e.g., "all_trading_services")
            result: Result of action (e.g., "SUCCESS", "FAILED")
            details: Optional additional details
            
        Returns:
            The created log entry
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": user,
            "action": action,
            "target": target,
            "result": result,
            "details": details or {},
            "previous_hash": self.last_hash
        }
        
        # Calculate hash of this entry
        entry_str = json.dumps(entry, sort_keys=True)
        entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
        
        # Update last hash for next entry
        self.last_hash = entry["hash"]
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        
        # TODO: Send to remote syslog if configured
        
        return entry
    
    def verify_chain_integrity(self, limit: Optional[int] = None) -> bool:
        """
        Verify integrity of audit log chain.
        
        Args:
            limit: Number of recent entries to verify (None = all)
            
        Returns:
            True if chain is valid, False if tampering detected
        """
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
                if limit:
                    lines = lines[-limit:]
                
                prev_hash = "0"
                
                for line in lines:
                    entry = json.loads(line)
                    
                    # Verify previous hash matches
                    if entry.get("previous_hash") != prev_hash:
                        return False
                    
                    # Verify entry hash
                    entry_copy = entry.copy()
                    stored_hash = entry_copy.pop("hash")
                    
                    calculated_hash = hashlib.sha256(
                        json.dumps(entry_copy, sort_keys=True).encode()
                    ).hexdigest()
                    
                    if calculated_hash != stored_hash:
                        return False
                    
                    prev_hash = stored_hash
                
                return True
                
        except (FileNotFoundError, json.JSONDecodeError):
            return False
    
    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """
        Get recent audit log entries.
        
        Args:
            limit: Number of recent entries to return
            
        Returns:
            List of log entries (newest first)
        """
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                
                # Get last N lines
                recent_lines = lines[-limit:] if len(lines) > limit else lines
                
                # Parse and reverse (newest first)
                entries = [json.loads(line) for line in recent_lines]
                entries.reverse()
                
                return entries
                
        except (FileNotFoundError, json.JSONDecodeError):
            return []

# Global audit logger instance
audit_logger = AuditLogger()
```

**TODO**:
1. Implement remote syslog forwarding
2. Add log rotation policy
3. Add encryption for sensitive details
4. Implement log archival
5. Add alerting on integrity failures

## Frontend Implementation

The frontend consists of four main files:

### 1. index.html - Main Control Panel UI

**Purpose**: Desktop-optimized control panel interface

**Key Features**:
- Emergency kill switch button (prominent)
- Power management card (NanoKVM integration)
- Data feed control card
- Service management with individual controls
- System metrics dashboard
- Real-time audit log viewer
- Confirmation modals for destructive actions

**Implementation Approach**:
- Pure HTML5/CSS3/JavaScript (no framework dependencies)
- Real-time updates via fetch API (5-second intervals)
- Confirmation dialogs for critical actions
- Responsive design

**TODO**: Create full HTML template (see existing design doc for complete code)

### 2. mobile.html - Progressive Web App

**Purpose**: Touch-optimized interface for mobile devices (especially iPhone)

**Key Features**:
- Touch-optimized buttons with haptic feedback
- iOS home screen support (add to home screen)
- Offline capability via service worker
- Dark theme optimized for OLED
- Safe area insets for notched displays

**TODO**: Create full mobile template

### 3. dashboard.js - Frontend Logic

**Purpose**: JavaScript logic for API interactions and UI updates

**Key Functions**:
- API client with authentication handling
- Real-time status polling
- Confirmation dialog management
- Error handling and user feedback
- WebSocket support (optional for real-time updates)

**TODO**: Implement full JavaScript client

### 4. styles.css - Styling

**Purpose**: Professional styling with responsive design

**Features**:
- Dark theme optimized for 24/7 monitoring
- Color-coded status indicators
- Responsive grid layout
- Accessibility considerations
- Print stylesheet (for documentation)

**TODO**: Create full stylesheet

## Configuration Files

### config.yaml Structure

The `config/config.yaml` file (created by setup script) contains all application configuration:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  debug: false

security:
  require_2fa: true
  admin_group: "admins"
  power_control_user: "youssef"

services:
  allowed_services:
    redis: "redis-hft"
    questdb: "questdb"
    # ... more services
  
  trading_services:
    - "binance-trades"
    - "binance-bookticker"
    - "batch-writer"

nanokvm:
  base_url: "https://210.6.8.5:40443"
  verify_ssl: false
  timeout: 10
  shutdown_wait: 30

audit:
  log_file: "/var/log/trading/audit.jsonl"
  retention_days: 90

# ... more configuration
```

### Environment Variables (.env)

Sensitive values are stored in environment variables (never committed):

```bash
# NanoKVM Credentials
NANOKVM_URL=https://210.6.8.5:40443
NANOKVM_USERNAME=admin
NANOKVM_PASSWORD=secure_password

# Telegram (optional)
TELEGRAM_BOT_TOKEN=bot_token
TELEGRAM_CHAT_ID=chat_id

# SMTP (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@aistation.trading
SMTP_PASSWORD=app_password

# Application
ENV=production
LOG_LEVEL=INFO
DEBUG=false
```

## Testing Strategy

### Unit Tests Structure

Create `tests/test_controls.py`:

```python
"""
Unit tests for control operations.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint (no auth required)."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_kill_switch_requires_auth():
    """Test kill switch requires authentication."""
    response = client.post("/api/kill-switch", json={
        "confirmation": "CONFIRM_EMERGENCY_STOP"
    })
    assert response.status_code == 401

def test_kill_switch_requires_confirmation():
    """Test kill switch requires correct confirmation."""
    headers = {
        "Remote-User": "youssef",
        "Remote-Groups": "admins"
    }
    response = client.post(
        "/api/kill-switch",
        json={"confirmation": "WRONG"},
        headers=headers
    )
    assert response.status_code == 400

# TODO: Add more tests
```

### Integration Testing

Test end-to-end flows:
1. Authentication via Authelia
2. Kill switch activation
3. Service management
4. Power control
5. Audit log integrity

### Security Testing

1. Test unauthorized access attempts
2. Verify 2FA requirement
3. Test CSRF protection
4. Verify audit logging of all actions
5. Test rate limiting

## Deployment Procedure

### Pre-Deployment Checklist

- [ ] Review all implementation files
- [ ] Update configuration for production
- [ ] Set environment variables
- [ ] Test in staging environment
- [ ] Verify backup procedures
- [ ] Document emergency procedures
- [ ] Train operators on usage

### Deployment Steps

1. **Run setup script**: `./setup-control-panel.sh`
2. **Implement files**: Copy templates and customize
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Configure services**: Follow implementation guide
5. **Test locally**: Verify all functionality
6. **Deploy to production**: Follow security procedures
7. **Verify deployment**: Run all tests
8. **Monitor**: Check logs and metrics

## Security Hardening

### File Permissions

```bash
# Secure configuration files
chmod 600 ControlPanel/.env
chmod 644 ControlPanel/config/config.yaml

# Secure log directory
sudo chown youssefbahloul:youssefbahloul /var/log/trading
chmod 755 /var/log/trading
chmod 644 /var/log/trading/audit.jsonl

# Secure systemd service file
sudo chmod 644 /etc/systemd/system/trading-control.service
```

### Network Security

```bash
# Firewall rules
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP (redirect to HTTPS)
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Optional: Restrict SSH to specific IPs
# sudo ufw delete allow 22/tcp
# sudo ufw allow from YOUR_IP to any port 22
```

### Application Security

1. **Enable rate limiting** in Nginx
2. **Configure fail2ban** for brute force protection
3. **Set up intrusion detection** (OSSEC, Wazuh)
4. **Regular security audits** of audit logs
5. **Monitor for anomalies** in Grafana

## Monitoring & Alerting

### Prometheus Metrics

Add custom metrics to `api/main.py`:

```python
from prometheus_client import Counter, Histogram

kill_switch_activations = Counter(
    'control_panel_kill_switch_total',
    'Total kill switch activations'
)

api_request_duration = Histogram(
    'control_panel_request_duration_seconds',
    'API request duration',
    ['endpoint', 'method']
)

# Use in endpoints:
@app.post("/api/kill-switch")
async def emergency_kill_switch(...):
    kill_switch_activations.inc()
    # ... rest of implementation
```

### Alert Rules

Configure Prometheus alert rules:

```yaml
groups:
  - name: trading_control_panel
    rules:
      - alert: KillSwitchActivated
        expr: increase(control_panel_kill_switch_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "Emergency kill switch activated"
          
      - alert: ControlPanelDown
        expr: up{job="trading-control"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Control panel is down"
```

## Backup & Recovery

### Backup Script

```bash
#!/bin/bash
# Daily backup of critical files

BACKUP_DIR="/home/youssefbahloul/backups/control-panel"
DATE=$(date +%Y%m%d)

mkdir -p "$BACKUP_DIR"

# Backup configurations
cp /etc/authelia/configuration.yml "$BACKUP_DIR/authelia-config-$DATE.yml"
cp /var/lib/authelia/users.yml "$BACKUP_DIR/authelia-users-$DATE.yml"
cp /etc/nginx/sites-available/aistation.trading "$BACKUP_DIR/nginx-config-$DATE"

# Backup audit logs
cp /var/log/trading/audit.jsonl "$BACKUP_DIR/audit-log-$DATE.jsonl"

# Create archive
tar czf "$BACKUP_DIR-$DATE.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Cleanup old backups (keep 30 days)
find /home/youssefbahloul/backups -name "control-panel-*.tar.gz" -mtime +30 -delete
```

### Recovery Procedure

1. **Stop services**:
   ```bash
   sudo systemctl stop trading-control authelia nginx
   ```

2. **Restore from backup**:
   ```bash
   tar xzf control-panel-YYYYMMDD.tar.gz
   sudo cp authelia-config-*.yml /etc/authelia/configuration.yml
   sudo cp authelia-users-*.yml /var/lib/authelia/users.yml
   # ... restore other files
   ```

3. **Verify configurations**:
   ```bash
   sudo nginx -t
   sudo authelia --config /etc/authelia/configuration.yml validate
   ```

4. **Restart services**:
   ```bash
   sudo systemctl start authelia nginx trading-control
   ```

5. **Verify functionality**:
   ```bash
   curl https://ops.aistation.trading/health
   ```

## Maintenance Procedures

### Weekly Maintenance

1. **Review audit logs**:
   ```bash
   tail -n 100 /var/log/trading/audit.jsonl | jq
   ```

2. **Check service health**:
   ```bash
   systemctl status authelia trading-control nginx
   ```

3. **Verify SSL certificates**:
   ```bash
   sudo certbot certificates
   ```

4. **Check disk space**:
   ```bash
   df -h
   du -sh /var/log/trading/
   ```

### Monthly Maintenance

1. **Review and update configurations**
2. **Update system packages**
3. **Review user access**
4. **Test backup restoration**
5. **Review security alerts**
6. **Update documentation**

## Support Resources

### Documentation References

- [Implementation Guide](./Production_Control_System_Implementation.md)
- [Authelia Documentation](https://www.authelia.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Nginx Documentation](https://nginx.org/en/docs/)

### Troubleshooting Steps

1. Check service status
2. Review logs
3. Verify configurations
4. Test components individually
5. Check network connectivity
6. Review audit trail

### Emergency Contacts

- **Primary Admin**: Youssef Bahloul
- **System Access**: SSH, Telegram Bot, Direct NanoKVM
- **Recovery Codes**: Stored in secure offline location

---

## Implementation Status

- [x] Setup script created
- [x] Documentation completed
- [x] Templates provided
- [ ] Implementation (to be done by operator)
- [ ] Testing (after implementation)
- [ ] Production deployment (after testing)

## Next Steps

1. **Execute setup script** to create directory structure
2. **Review templates** and understand implementation
3. **Customize templates** for your environment
4. **Implement files** following templates
5. **Test locally** before production deployment
6. **Follow deployment guide** for production setup
7. **Verify security** before going live
8. **Monitor** and maintain system

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-11-11  
**Status**: Ready for Implementation  
**Security Review**: Required before production deployment
