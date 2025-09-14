#!/usr/bin/env python3
"""
AI Trading Station - Monitoring Dashboard Complete
==================================================
PURPOSE: Real-time dashboard and UI for monitoring the AI Trading Station
ROLE: Provides comprehensive visualization of system performance, trading metrics,
      and onload-trading wrapper status with a user-friendly interface.
INTEGRATION: Works with monitor_trading_system_optimized.py for data collection
==================================================
"""

import json
import time
import sys
import os
import subprocess
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import signal
import argparse

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    load_average: List[float]
    network_interfaces: List[str]
    core_isolation_status: str
    onload_status: str
    trading_latency: Optional[float] = None

@dataclass
class TradingMetrics:
    """Trading system performance metrics"""
    timestamp: str
    orders_processed: int
    mean_latency_us: float
    p99_latency_us: float
    errors_count: int
    uptime_seconds: int

class MonitoringDashboard:
    """
    Complete monitoring dashboard for AI Trading Station
    
    Provides real-time visualization of:
    - System performance metrics
    - Trading latency and throughput
    - OnLoad wrapper status
    - Core isolation monitoring
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.running = False
        self.metrics_history: List[SystemMetrics] = []
        self.trading_history: List[TradingMetrics] = []
        self.max_history = self.config.get('max_history_records', 1000)
        
        # OnLoad wrapper paths
        self.onload_wrapper_path = self._get_onload_wrapper_path()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, 'monitoring_config.json')
    
    def _get_onload_wrapper_path(self) -> str:
        """Get absolute path to onload-trading wrapper"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        return os.path.join(repo_root, 'scripts', 'onload-trading')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "refresh_interval_seconds": 2,
            "dashboard_title": "AI Trading Station Monitor",
            "show_detailed_metrics": True,
            "alert_thresholds": {
                "cpu_usage_percent": 80,
                "memory_usage_percent": 85,
                "latency_threshold_us": 10.0
            },
            "max_history_records": 1000,
            "monitoring_cores": [2, 3],
            "display_width": 120
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration.")
        
        return default_config
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        timestamp = datetime.now().isoformat()
        
        # CPU usage
        try:
            cpu_result = subprocess.run(['top', '-bn1'], capture_output=True, text=True, timeout=5)
            cpu_line = [line for line in cpu_result.stdout.split('\n') if 'Cpu(s)' in line][0]
            cpu_usage = float(cpu_line.split()[1].replace('%us,', ''))
        except:
            cpu_usage = 0.0
        
        # Memory usage
        try:
            mem_result = subprocess.run(['free'], capture_output=True, text=True, timeout=5)
            mem_lines = mem_result.stdout.split('\n')
            mem_line = [line for line in mem_lines if 'Mem:' in line][0]
            mem_parts = mem_line.split()
            total_mem = int(mem_parts[1])
            used_mem = int(mem_parts[2])
            memory_usage = (used_mem / total_mem) * 100
        except:
            memory_usage = 0.0
        
        # Load average
        try:
            with open('/proc/loadavg', 'r') as f:
                load_avg = [float(x) for x in f.read().split()[:3]]
        except:
            load_avg = [0.0, 0.0, 0.0]
        
        # Network interfaces
        try:
            ip_result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True, timeout=5)
            interfaces = []
            for line in ip_result.stdout.split('\n'):
                if ': ' in line and 'lo' not in line and 'vir' not in line:
                    interface = line.split(':')[1].strip().split('@')[0]
                    interfaces.append(interface)
        except:
            interfaces = ['unknown']
        
        # Core isolation status
        core_isolation_status = self._check_core_isolation()
        
        # OnLoad status
        onload_status = self._check_onload_status()
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            load_average=load_avg,
            network_interfaces=interfaces,
            core_isolation_status=core_isolation_status,
            onload_status=onload_status
        )
    
    def _check_core_isolation(self) -> str:
        """Check CPU core isolation status"""
        try:
            if os.path.exists('/sys/devices/system/cpu/isolated'):
                with open('/sys/devices/system/cpu/isolated', 'r') as f:
                    isolated = f.read().strip()
                    if isolated:
                        return f"Active (cores: {isolated})"
                    else:
                        return "Not configured"
            else:
                return "Cannot determine"
        except:
            return "Error checking"
    
    def _check_onload_status(self) -> str:
        """Check OnLoad availability and wrapper status"""
        try:
            # Check if onload command is available
            subprocess.run(['which', 'onload'], capture_output=True, check=True, timeout=5)
            onload_available = True
        except:
            onload_available = False
        
        # Check if wrapper exists and is executable
        wrapper_exists = os.path.exists(self.onload_wrapper_path)
        wrapper_executable = wrapper_exists and os.access(self.onload_wrapper_path, os.X_OK)
        
        if onload_available and wrapper_executable:
            return "Ready (OnLoad + Wrapper)"
        elif onload_available:
            return "OnLoad available, wrapper missing"
        elif wrapper_executable:
            return "Wrapper available, OnLoad missing"
        else:
            return "Not available"
    
    def _display_banner(self):
        """Display dashboard banner"""
        title = self.config['dashboard_title']
        width = self.config['display_width']
        
        print("═" * width)
        print(f"║{title:^{width-2}}║")
        print(f"║{'Real-time Monitoring Dashboard':^{width-2}}║")
        print("═" * width)
    
    def _display_metrics(self, metrics: SystemMetrics):
        """Display current metrics"""
        width = self.config['display_width']
        
        print(f"║ Timestamp: {metrics.timestamp:<{width-15}}║")
        print("├" + "─" * (width-2) + "┤")
        
        # System Performance
        print(f"║ SYSTEM PERFORMANCE{' ' * (width-20)}║")
        print(f"║ CPU Usage:      {metrics.cpu_usage:6.1f}%{' ' * (width-26)}║")
        print(f"║ Memory Usage:   {metrics.memory_usage:6.1f}%{' ' * (width-26)}║")
        print(f"║ Load Average:   {metrics.load_average[0]:.2f}, {metrics.load_average[1]:.2f}, {metrics.load_average[2]:.2f}{' ' * (width-40)}║")
        print("├" + "─" * (width-2) + "┤")
        
        # Trading Performance Infrastructure
        print(f"║ TRADING INFRASTRUCTURE{' ' * (width-24)}║")
        print(f"║ Core Isolation: {metrics.core_isolation_status:<{width-19}}║")
        print(f"║ OnLoad Status:  {metrics.onload_status:<{width-19}}║")
        print(f"║ Network IFs:    {', '.join(metrics.network_interfaces[:3]):<{width-19}}║")
        print("├" + "─" * (width-2) + "┤")
        
        # OnLoad Wrapper Status
        wrapper_status = "Available" if os.path.exists(self.onload_wrapper_path) else "Missing"
        print(f"║ ONLOAD WRAPPER{' ' * (width-16)}║")
        print(f"║ Status:         {wrapper_status:<{width-19}}║")
        print(f"║ Path:           {self.onload_wrapper_path:<{width-19}}║")
        
        # Performance alerts
        alerts = []
        if metrics.cpu_usage > self.config['alert_thresholds']['cpu_usage_percent']:
            alerts.append(f"HIGH CPU: {metrics.cpu_usage:.1f}%")
        if metrics.memory_usage > self.config['alert_thresholds']['memory_usage_percent']:
            alerts.append(f"HIGH MEMORY: {metrics.memory_usage:.1f}%")
        
        if alerts:
            print("├" + "─" * (width-2) + "┤")
            print(f"║ ALERTS{' ' * (width-8)}║")
            for alert in alerts:
                print(f"║ ⚠ {alert:<{width-5}}║")
        
        print("═" * width)
    
    def _display_controls(self):
        """Display control information"""
        width = self.config['display_width']
        print(f"║ CONTROLS: Press Ctrl+C to exit, 'q' + Enter to quit{' ' * (width-52)}║")
        print("═" * width)
    
    def start_dashboard(self):
        """Start the monitoring dashboard"""
        self.running = True
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            self.running = False
            print("\nShutting down dashboard...")
        
        signal.signal(signal.SIGINT, signal_handler)
        
        print(f"Starting {self.config['dashboard_title']}...")
        print(f"Configuration loaded from: {self.config_path}")
        print(f"OnLoad wrapper path: {self.onload_wrapper_path}")
        print(f"Refresh interval: {self.config['refresh_interval_seconds']} seconds")
        print("\nPress Ctrl+C to exit\n")
        
        try:
            while self.running:
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Get current metrics
                metrics = self._get_system_metrics()
                
                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                # Display dashboard
                self._display_banner()
                self._display_metrics(metrics)
                self._display_controls()
                
                # Wait for next refresh
                time.sleep(self.config['refresh_interval_seconds'])
                
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            print("\nDashboard stopped.")
    
    def export_metrics(self, output_file: str):
        """Export collected metrics to JSON file"""
        data = {
            'config': self.config,
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'trading_history': [asdict(t) for t in self.trading_history],
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics exported to: {output_file}")

def main():
    """Main entry point for monitoring dashboard"""
    parser = argparse.ArgumentParser(description='AI Trading Station Monitoring Dashboard')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--export', help='Export metrics to JSON file and exit')
    parser.add_argument('--version', action='version', version='monitor_dashboard_complete.py v1.0.0')
    
    args = parser.parse_args()
    
    # Create dashboard instance
    dashboard = MonitoringDashboard(config_path=args.config)
    
    if args.export:
        # Export mode
        print("Collecting metrics for export...")
        metrics = dashboard._get_system_metrics()
        dashboard.metrics_history.append(metrics)
        dashboard.export_metrics(args.export)
    else:
        # Interactive dashboard mode
        dashboard.start_dashboard()

if __name__ == "__main__":
    main()