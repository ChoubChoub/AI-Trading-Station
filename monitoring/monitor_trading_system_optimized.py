#!/usr/bin/env python3
"""
AI Trading Station - Optimized Trading System Monitor
====================================================
PURPOSE: High-performance monitoring core for the AI Trading Station
ROLE: Provides optimized data collection with minimal system impact,
      designed to work alongside trading applications without affecting latency.
INTEGRATION: Feeds data to monitor_dashboard_complete.py and external monitoring systems
====================================================
"""

import json
import time
import os
import sys
import psutil
import subprocess
import threading
import queue
import signal
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import argparse
import logging
from pathlib import Path

@dataclass
class CoreMetrics:
    """Core system metrics optimized for low-impact collection"""
    timestamp: float
    cpu_cores_usage: List[float]  # Per-core CPU usage
    isolated_cores_status: Dict[int, float]  # Usage of isolated trading cores
    memory_available_gb: float
    network_rx_bytes: int
    network_tx_bytes: int
    irq_counts: Dict[str, int]  # IRQ counts per CPU
    context_switches: int
    interrupts: int

@dataclass
class TradingProcessMetrics:
    """Trading process specific metrics"""
    timestamp: float
    process_id: Optional[int]
    cpu_usage: float
    memory_rss_mb: float
    memory_vms_mb: float
    threads_count: int
    open_files: int
    network_connections: int
    is_running: bool

@dataclass
class LatencyMetrics:
    """Network and system latency metrics"""
    timestamp: float
    network_latency_us: Optional[float]
    system_call_latency_us: Optional[float]
    context_switch_latency_us: Optional[float]
    onload_status: str

class OptimizedMonitor:
    """
    High-performance monitoring system with minimal overhead
    
    Designed to:
    - Collect metrics with <1% CPU overhead
    - Monitor trading-specific KPIs
    - Detect performance anomalies
    - Integrate with OnLoad wrapper monitoring
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.running = False
        
        # Data collection
        self.metrics_queue = queue.Queue(maxsize=10000)
        self.current_metrics = None
        
        # OnLoad wrapper monitoring
        self.onload_wrapper_path = self._get_onload_wrapper_path()
        self.trading_process_pid = None
        
        # Performance tracking
        self.last_network_stats = None
        self.last_cpu_times = None
        self.baseline_metrics = None
        
        # Setup logging
        self._setup_logging()
        
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        script_dir = Path(__file__).parent
        return script_dir / 'monitoring_config.json'
    
    def _get_onload_wrapper_path(self) -> Path:
        """Get absolute path to onload-trading wrapper"""
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent
        return repo_root / 'scripts' / 'onload-trading'
    
    def _setup_logging(self):
        """Setup optimized logging"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('log_file', '/tmp/trading_monitor.log')),
                logging.StreamHandler(sys.stdout) if self.config.get('console_logging', False) else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger('TradingMonitor')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring_interval_seconds": 0.1,  # 100ms for high-frequency monitoring
            "metrics_retention_seconds": 3600,   # 1 hour of metrics
            "trading_cores": [2, 3],             # Cores to monitor for trading
            "performance_mode": "optimized",     # optimized, detailed, minimal
            "enable_latency_monitoring": True,
            "enable_irq_monitoring": True,
            "enable_process_monitoring": True,
            "alert_thresholds": {
                "cpu_usage_percent": 90,
                "memory_usage_percent": 95,
                "context_switches_per_sec": 10000,
                "interrupts_per_sec": 50000,
                "isolated_core_usage_percent": 5  # Very low threshold for trading cores
            },
            "output_format": "json",
            "export_interval_seconds": 60,
            "log_level": "INFO",
            "log_file": "/tmp/trading_monitor.log",
            "console_logging": False
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    self.logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration.")
        
        return default_config
    
    def _collect_core_metrics(self) -> CoreMetrics:
        """Collect core system metrics with minimal overhead"""
        current_time = time.time()
        
        # CPU per-core usage (optimized collection)
        try:
            cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
        except:
            cpu_cores = [0.0] * psutil.cpu_count()
        
        # Isolated cores monitoring
        isolated_cores_status = {}
        trading_cores = self.config.get('trading_cores', [2, 3])
        for core in trading_cores:
            if core < len(cpu_cores):
                isolated_cores_status[core] = cpu_cores[core]
        
        # Memory (available only, faster than detailed breakdown)
        try:
            memory_info = psutil.virtual_memory()
            memory_available_gb = memory_info.available / (1024**3)
        except:
            memory_available_gb = 0.0
        
        # Network stats (delta calculation)
        network_rx_bytes = 0
        network_tx_bytes = 0
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                network_rx_bytes = net_io.bytes_recv
                network_tx_bytes = net_io.bytes_sent
        except:
            pass
        
        # IRQ monitoring (if enabled)
        irq_counts = {}
        if self.config.get('enable_irq_monitoring', True):
            irq_counts = self._collect_irq_stats()
        
        # System-wide counters
        context_switches = 0
        interrupts = 0
        try:
            # Fast system stats collection
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('ctxt'):
                        context_switches = int(line.split()[1])
                    elif line.startswith('intr'):
                        interrupts = int(line.split()[1])
                    if context_switches and interrupts:
                        break
        except:
            pass
        
        return CoreMetrics(
            timestamp=current_time,
            cpu_cores_usage=cpu_cores,
            isolated_cores_status=isolated_cores_status,
            memory_available_gb=memory_available_gb,
            network_rx_bytes=network_rx_bytes,
            network_tx_bytes=network_tx_bytes,
            irq_counts=irq_counts,
            context_switches=context_switches,
            interrupts=interrupts
        )
    
    def _collect_irq_stats(self) -> Dict[str, int]:
        """Collect IRQ statistics for performance monitoring"""
        irq_counts = {}
        try:
            with open('/proc/interrupts', 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # Get CPU count from header
                    header = lines[0].strip().split()
                    cpu_count = len([h for h in header if h.startswith('CPU')])
                    
                    for line in lines[1:]:
                        parts = line.strip().split()
                        if len(parts) > cpu_count:
                            irq_name = parts[0].rstrip(':')
                            # Sum interrupts across all CPUs
                            total_irqs = sum(int(parts[i+1]) for i in range(cpu_count) if parts[i+1].isdigit())
                            irq_counts[irq_name] = total_irqs
        except:
            pass
        
        return irq_counts
    
    def _monitor_trading_process(self) -> Optional[TradingProcessMetrics]:
        """Monitor specific trading process if running"""
        if not self.config.get('enable_process_monitoring', True):
            return None
        
        # Try to find trading process
        trading_process = None
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    # Look for processes launched with onload-trading wrapper
                    if 'onload' in cmdline.lower() or 'trading' in cmdline.lower():
                        trading_process = proc
                        self.trading_process_pid = proc.info['pid']
                        break
                except:
                    continue
        except:
            pass
        
        if not trading_process:
            return TradingProcessMetrics(
                timestamp=time.time(),
                process_id=None,
                cpu_usage=0.0,
                memory_rss_mb=0.0,
                memory_vms_mb=0.0,
                threads_count=0,
                open_files=0,
                network_connections=0,
                is_running=False
            )
        
        # Collect process metrics
        try:
            cpu_usage = trading_process.cpu_percent()
            memory_info = trading_process.memory_info()
            threads_count = trading_process.num_threads()
            
            # Count open files and network connections
            try:
                open_files = len(trading_process.open_files())
            except:
                open_files = 0
            
            try:
                network_connections = len(trading_process.connections())
            except:
                network_connections = 0
            
            return TradingProcessMetrics(
                timestamp=time.time(),
                process_id=trading_process.pid,
                cpu_usage=cpu_usage,
                memory_rss_mb=memory_info.rss / (1024**2),
                memory_vms_mb=memory_info.vms / (1024**2),
                threads_count=threads_count,
                open_files=open_files,
                network_connections=network_connections,
                is_running=True
            )
        except:
            return None
    
    def _measure_latency_metrics(self) -> LatencyMetrics:
        """Measure system latency metrics"""
        current_time = time.time()
        
        # Network latency (simplified ping to localhost)
        network_latency_us = None
        if self.config.get('enable_latency_monitoring', True):
            try:
                start_time = time.perf_counter()
                result = subprocess.run(['ping', '-c', '1', '-W', '1', 'localhost'], 
                                      capture_output=True, timeout=2)
                if result.returncode == 0:
                    elapsed = (time.perf_counter() - start_time) * 1_000_000
                    network_latency_us = elapsed
            except:
                pass
        
        # System call latency (measure getpid syscall)
        system_call_latency_us = None
        try:
            start_time = time.perf_counter()
            for _ in range(100):  # Average over 100 calls
                os.getpid()
            elapsed = (time.perf_counter() - start_time) * 1_000_000 / 100
            system_call_latency_us = elapsed
        except:
            pass
        
        # Context switch latency estimation
        context_switch_latency_us = None
        # This is a simplified estimation - real measurement would require more complex setup
        
        # OnLoad status check
        onload_status = self._check_onload_status()
        
        return LatencyMetrics(
            timestamp=current_time,
            network_latency_us=network_latency_us,
            system_call_latency_us=system_call_latency_us,
            context_switch_latency_us=context_switch_latency_us,
            onload_status=onload_status
        )
    
    def _check_onload_status(self) -> str:
        """Check OnLoad wrapper and driver status"""
        try:
            # Check if onload is available
            result = subprocess.run(['which', 'onload'], capture_output=True, timeout=1)
            onload_available = result.returncode == 0
            
            # Check wrapper
            wrapper_available = self.onload_wrapper_path.exists() and os.access(self.onload_wrapper_path, os.X_OK)
            
            if onload_available and wrapper_available:
                return "ready"
            elif onload_available:
                return "onload_only"
            elif wrapper_available:
                return "wrapper_only"
            else:
                return "unavailable"
        except:
            return "error"
    
    def _detect_anomalies(self, metrics: CoreMetrics) -> List[str]:
        """Detect performance anomalies"""
        alerts = []
        thresholds = self.config['alert_thresholds']
        
        # Check isolated core usage
        for core, usage in metrics.isolated_cores_status.items():
            if usage > thresholds['isolated_core_usage_percent']:
                alerts.append(f"High usage on isolated core {core}: {usage:.1f}%")
        
        # Check overall CPU
        avg_cpu = sum(metrics.cpu_cores_usage) / len(metrics.cpu_cores_usage)
        if avg_cpu > thresholds['cpu_usage_percent']:
            alerts.append(f"High overall CPU usage: {avg_cpu:.1f}%")
        
        # Check memory
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        memory_usage_percent = (1 - metrics.memory_available_gb / total_memory_gb) * 100
        if memory_usage_percent > thresholds['memory_usage_percent']:
            alerts.append(f"High memory usage: {memory_usage_percent:.1f}%")
        
        return alerts
    
    def start_monitoring(self):
        """Start the optimized monitoring system"""
        self.running = True
        self.logger.info("Starting optimized trading system monitor")
        
        # Set up signal handler
        def signal_handler(signum, frame):
            self.running = False
            self.logger.info("Shutdown signal received")
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                start_time = time.perf_counter()
                
                # Collect metrics
                core_metrics = self._collect_core_metrics()
                trading_metrics = self._monitor_trading_process()
                latency_metrics = self._measure_latency_metrics()
                
                # Detect anomalies
                alerts = self._detect_anomalies(core_metrics)
                
                # Store current metrics
                self.current_metrics = {
                    'core': asdict(core_metrics),
                    'trading_process': asdict(trading_metrics) if trading_metrics else None,
                    'latency': asdict(latency_metrics),
                    'alerts': alerts,
                    'collection_time_us': (time.perf_counter() - start_time) * 1_000_000
                }
                
                # Add to queue (non-blocking)
                try:
                    self.metrics_queue.put_nowait(self.current_metrics)
                except queue.Full:
                    # Remove oldest metrics if queue is full
                    try:
                        self.metrics_queue.get_nowait()
                        self.metrics_queue.put_nowait(self.current_metrics)
                    except queue.Empty:
                        pass
                
                # Log alerts
                for alert in alerts:
                    self.logger.warning(f"ALERT: {alert}")
                
                # Sleep for remaining interval
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.config['monitoring_interval_seconds'] - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        finally:
            self.running = False
            self.logger.info("Monitoring stopped")
    
    def export_metrics(self, output_file: str, duration_seconds: int = 60):
        """Export collected metrics to file"""
        metrics_data = []
        start_time = time.time()
        
        self.logger.info(f"Exporting metrics for {duration_seconds} seconds to {output_file}")
        
        while time.time() - start_time < duration_seconds and self.running:
            try:
                metrics = self.metrics_queue.get(timeout=1)
                metrics_data.append(metrics)
            except queue.Empty:
                continue
        
        # Export data
        export_data = {
            'config': self.config,
            'metrics': metrics_data,
            'export_timestamp': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'total_records': len(metrics_data)
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(metrics_data)} metrics records to {output_file}")
        print(f"Metrics exported: {len(metrics_data)} records in {output_file}")

def main():
    """Main entry point for optimized trading monitor"""
    parser = argparse.ArgumentParser(description='AI Trading Station Optimized Monitor')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--export', help='Export metrics to file and exit')
    parser.add_argument('--duration', '-d', type=int, default=60, help='Export duration in seconds')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--version', action='version', version='monitor_trading_system_optimized.py v1.0.0')
    
    args = parser.parse_args()
    
    # Create monitor instance
    monitor = OptimizedMonitor(config_path=args.config)
    
    if args.export:
        # Export mode - collect metrics then export
        print(f"Starting metrics collection for {args.duration} seconds...")
        
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=monitor.start_monitoring)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        # Export metrics
        monitor.export_metrics(args.export, args.duration)
        monitor.running = False
    else:
        # Continuous monitoring mode
        if args.daemon:
            print("Starting monitoring daemon...")
        else:
            print("Starting interactive monitoring (Ctrl+C to stop)...")
        
        monitor.start_monitoring()

if __name__ == "__main__":
    main()