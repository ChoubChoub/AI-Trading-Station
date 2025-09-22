"""
AI Trading Station Monitoring System
Monitors core isolation, IRQ affinity, hardware health, and latency performance
"""
import json
import time
import psutil
import subprocess
import logging
import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
import os
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class TradingSystemMonitor:
    def __init__(self, config_file='config/monitor_config.json'):
        """Initialize monitor with configuration"""
        self.config = self.load_config(config_file)
        self.alerts = []
        self.metrics = {}
    def load_config(self, config_file):
        """Load monitoring configuration"""
        config_path = Path(__file__).parent.parent / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration based on discovery
            return {
                "cpu": {
                    "isolated_cores": [2, 3],
                    "expected_governor": "performance",
                    "temp_threshold": 75
                },
                "network": {
                    "trading_interfaces": ["enp130s0f0"]
                },
                "gpu": {
                    "temp_threshold": 80
                },
                "monitoring": {
                    "interval": 5,
                    "log_path": "/var/log/trading_monitor"
                }
            }
    def check_core_isolation(self):
        """Verify isolated cores are free from interrupts and unwanted processes"""
        isolated_cores = self.config['cpu']['isolated_cores']
        violations = []
        # Check IRQ affinity
        irq_violations = self._check_irq_affinity(isolated_cores)
        if irq_violations:
            violations.extend(irq_violations)
        # Check process affinity
        process_violations = self._check_process_affinity(isolated_cores)
        if process_violations:
            violations.extend(process_violations)
        # Check CPU usage on isolated cores
        cpu_violations = self._check_isolated_cpu_usage(isolated_cores)
        if cpu_violations:
            violations.extend(cpu_violations)
        return violations
    def _check_irq_affinity(self, isolated_cores):
        """Check if any IRQs are assigned to isolated cores"""
        violations = []
        try:
            with open('/proc/interrupts', 'r') as f:
                lines = f.readlines()
            # Parse CPU columns
            cpu_columns = lines[0].strip().split()
            for line in lines[1:]:
                parts = line.strip().split()
                if not parts or ':' not in parts[0]:
                    continue
                irq_num = parts[0].rstrip(':')
                # Check interrupt counts for isolated cores
                for core in isolated_cores:
                    if core < len(cpu_columns):
                        col_idx = core + 1  # +1 for IRQ number column
                        if col_idx < len(parts):
                            count = parts[col_idx]
                            if count.isdigit() and int(count) > 0:
                                # Check if count is increasing
                                irq_desc = ' '.join(parts[len(cpu_columns)+1:])
                                violations.append({
                                    'type': 'irq_on_isolated_core',
                                    'core': core,
                                    'irq': irq_num,
                                    'count': int(count),
                                    'description': irq_desc
                                })
        except Exception as e:
            logger.error(f"Error checking IRQ affinity: {e}")
        return violations
    def _check_process_affinity(self, isolated_cores):
        """Check if non-trading processes are running on isolated cores"""
        violations = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_affinity']):
                try:
                    if proc.info['cpu_affinity']:
                        # Check if process is pinned to isolated cores
                        affinity = set(proc.info['cpu_affinity'])
                        isolated_set = set(isolated_cores)
                        if affinity & isolated_set:
                            # Allow kernel threads and trading processes
                            if not (proc.info['name'].startswith('[') or 
                                   self._is_trading_process(proc.info['name'])):
                                violations.append({
                                    'type': 'process_on_isolated_core',
                                    'pid': proc.info['pid'],
                                    'name': proc.info['name'],
                                    'cores': list(affinity & isolated_set)
                                })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error checking process affinity: {e}")
        return violations
    def _check_isolated_cpu_usage(self, isolated_cores):
        """Check CPU usage on isolated cores"""
        violations = []
        try:
            # Get per-CPU stats
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            for core in isolated_cores:
                if core < len(cpu_percent):
                    usage = cpu_percent[core]
                    # If no trading process is running, usage should be near 0
                    if not self._is_trading_active() and usage > 5:
                        violations.append({
                            'type': 'high_cpu_on_isolated_core',
                            'core': core,
                            'usage': usage
                        })
        except Exception as e:
            logger.error(f"Error checking CPU usage: {e}")
        return violations
    def _is_trading_process(self, process_name):
        """Check if process is a known trading application"""
        trading_processes = self.config.get('processes', {}).get('trading', [])
        return any(tp in process_name.lower() for tp in trading_processes)
    def _is_trading_active(self):
        """Check if any trading process is currently active"""
        trading_processes = self.config.get('processes', {}).get('trading', [])
        for proc in psutil.process_iter(['name']):
            if any(tp in proc.info['name'].lower() for tp in trading_processes):
                return True
        return False
    def check_hardware_health(self):
        """Monitor CPU and GPU temperatures"""
        health_issues = []
        # CPU temperature
        cpu_issues = self._check_cpu_temperature()
        if cpu_issues:
            health_issues.extend(cpu_issues)
        # GPU temperature
        gpu_issues = self._check_gpu_temperature()
        if gpu_issues:
            health_issues.extend(gpu_issues)
        return health_issues
    def _check_cpu_temperature(self):
        """Check CPU temperature using sensors"""
        issues = []
        threshold = self.config['cpu'].get('temp_threshold', 75)
        try:
            result = subprocess.run(['sensors', '-j'], capture_output=True, text=True)
            if result.returncode == 0:
                sensor_data = json.loads(result.stdout)
                for chip, values in sensor_data.items():
                    if 'coretemp' in chip or 'k10temp' in chip:
                        for sensor, data in values.items():
                            if isinstance(data, dict) and any(key.endswith('_input') for key in data):
                                for key, value in data.items():
                                    if key.endswith('_input') and value > threshold:
                                        issues.append({
                                            'type': 'high_cpu_temp',
                                            'sensor': sensor,
                                            'temperature': value,
                                            'threshold': threshold
                                        })
        except Exception as e:
            logger.error(f"Error checking CPU temperature: {e}")
        return issues
    def _check_gpu_temperature(self):
        """Check GPU temperature using nvidia-smi"""
        issues = []
        threshold = self.config['gpu'].get('temp_threshold', 80)
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,temperature.gpu', '--format=csv,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_idx = parts[0]
                        gpu_name = parts[1]
                        temp = int(parts[2])
                        if temp > threshold:
                            issues.append({
                                'type': 'high_gpu_temp',
                                'gpu_index': gpu_idx,
                                'gpu_name': gpu_name,
                                'temperature': temp,
                                'threshold': threshold
                            })
        except Exception as e:
            logger.error(f"Error checking GPU temperature: {e}")
        return issues
    def check_network_latency(self):
        """Monitor network interface latency and performance"""
        latency_issues = []
        for interface in self.config['network']['trading_interfaces']:
            issues = self._check_interface_stats(interface)
            if issues:
                latency_issues.extend(issues)
        return latency_issues
    def _check_interface_stats(self, interface):
        """Check network interface statistics for anomalies"""
        issues = []
        try:
            stats = psutil.net_io_counters(pernic=True).get(interface)
            if stats:
                # Check for errors or drops
                if stats.errin > 0 or stats.errout > 0:
                    issues.append({
                        'type': 'network_errors',
                        'interface': interface,
                        'errors_in': stats.errin,
                        'errors_out': stats.errout
                    })
                if stats.dropin > 0 or stats.dropout > 0:
                    issues.append({
                        'type': 'network_drops',
                        'interface': interface,
                        'drops_in': stats.dropin,
                        'drops_out': stats.dropout
                    })
        except Exception as e:
            logger.error(f"Error checking network stats: {e}")
        return issues
    def collect_metrics(self):
        """Collect all monitoring metrics"""
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._collect_cpu_metrics(),
            'memory': self._collect_memory_metrics(),
            'network': self._collect_network_metrics(),
            'gpu': self._collect_gpu_metrics()
        }
        return self.metrics
    def _collect_cpu_metrics(self):
        """Collect CPU metrics"""
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        return {
            'usage_per_core': cpu_percent,
            'isolated_cores': self.config['cpu']['isolated_cores'],
            'isolated_usage': [cpu_percent[i] for i in self.config['cpu']['isolated_cores'] 
                             if i < len(cpu_percent)]
        }
    def _collect_memory_metrics(self):
        """Collect memory metrics"""
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'percent': mem.percent,
            'used': mem.used
        }
    def _collect_network_metrics(self):
        """Collect network metrics"""
        metrics = {}
        for interface in self.config['network']['trading_interfaces']:
            stats = psutil.net_io_counters(pernic=True).get(interface)
            if stats:
                metrics[interface] = {
                    'bytes_sent': stats.bytes_sent,
                    'bytes_recv': stats.bytes_recv,
                    'packets_sent': stats.packets_sent,
                    'packets_recv': stats.packets_recv,
                    'errors': stats.errin + stats.errout,
                    'drops': stats.dropin + stats.dropout
                }
        return metrics
    def _collect_gpu_metrics(self):
        """Collect GPU metrics"""
        metrics = []
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        metrics.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'temperature': int(parts[2]),
                            'utilization': int(parts[3].rstrip(' %')),
                            'memory_used': parts[4],
                            'memory_total': parts[5]
                        })
        except Exception as e:
            logger.error(f"Error collecting GPU metrics: {e}")
        return metrics
    def run_checks(self):
        """Run all monitoring checks"""
        all_issues = []
        # Core isolation checks
        isolation_issues = self.check_core_isolation()
        if isolation_issues:
            all_issues.extend(isolation_issues)
        # Hardware health checks
        health_issues = self.check_hardware_health()
        if health_issues:
            all_issues.extend(health_issues)
        # Network latency checks
        latency_issues = self.check_network_latency()
        if latency_issues:
            all_issues.extend(latency_issues)
        # Collect metrics
        self.collect_metrics()
        return all_issues
    def format_alert(self, issue):
        """Format an issue as an alert message"""
        if issue['type'] == 'irq_on_isolated_core':
            return f"ALERT: IRQ {issue['irq']} ({issue['description']}) on isolated core {issue['core']} with {issue['count']} interrupts"
        elif issue['type'] == 'process_on_isolated_core':
            return f"ALERT: Process {issue['name']} (PID: {issue['pid']}) running on isolated cores {issue['cores']}"
        elif issue['type'] == 'high_cpu_on_isolated_core':
            return f"ALERT: High CPU usage ({issue['usage']:.1f}%) on isolated core {issue['core']}"
        elif issue['type'] == 'high_cpu_temp':
            return f"ALERT: High CPU temperature {issue['sensor']}: {issue['temperature']}°C (threshold: {issue['threshold']}°C)"
        elif issue['type'] == 'high_gpu_temp':
            return f"ALERT: High GPU temperature on {issue['gpu_name']} (GPU {issue['gpu_index']}): {issue['temperature']}°C (threshold: {issue['threshold']}°C)"
        elif issue['type'] == 'network_errors':
            return f"ALERT: Network errors on {issue['interface']}: {issue['errors_in']} in, {issue['errors_out']} out"
        elif issue['type'] == 'network_drops':
            return f"ALERT: Network drops on {issue['interface']}: {issue['drops_in']} in, {issue['drops_out']} out"
        else:
            return f"ALERT: {issue}"
def main():
    parser = argparse.ArgumentParser(description='AI Trading Station Monitor')
    parser.add_argument('--config', default='config/monitor_config.json', 
                       help='Path to configuration file')
    parser.add_argument('--once', action='store_true', 
                       help='Run checks once and exit')
    parser.add_argument('--interval', type=int, default=5,
                       help='Monitoring interval in seconds')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    args = parser.parse_args()
    monitor = TradingSystemMonitor(args.config)
    if args.once:
        # Run once and exit
        issues = monitor.run_checks()
        if args.json:
            output = {
                'timestamp': datetime.now().isoformat(),
                'issues': issues,
                'metrics': monitor.metrics
            }
            print(json.dumps(output, indent=2))
        else:
            if issues:
                print("\n=== MONITORING ALERTS ===")
                for issue in issues:
                    print(monitor.format_alert(issue))
            else:
                print("✓  All systems operating normally")
            print("\n=== SYSTEM METRICS ===")
            print(json.dumps(monitor.metrics, indent=2))
    else:
        # Continuous monitoring
        print("Starting AI Trading Station Monitor...")
        print(f"Monitoring interval: {args.interval} seconds")
        print("Press Ctrl+C to stop\n")
        try:
            while True:
                issues = monitor.run_checks()
                if issues:
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ALERTS DETECTED:")
                    for issue in issues:
                        print(f"  - {monitor.format_alert(issue)}")
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓  All systems normal")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
if __name__ == '__main__':
    main()
