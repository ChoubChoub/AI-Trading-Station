#!/usr/bin/env python3
"""
AI Trading Station - System Monitor V3
Fixed server thread management for continuous latency measurements
"""

import json
import time
import socket
import threading
import statistics
import subprocess
import psutil
import logging
from pathlib import Path
from datetime import datetime
from collections import deque

class TradingSystemMonitor:
    def __init__(self, config_path="/home/youssefbahloul/ai-trading-station/monitoring/config/monitor_config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.alerts = deque(maxlen=100)
        self.metrics = {}
        self.latency_server = None
        self.server_port = 12345
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Start persistent latency server
        self.start_latency_server()
    
    def start_latency_server(self):
        """Start a persistent latency echo server"""
        def server_loop():
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            try:
                server.bind(('127.0.0.1', self.server_port))
                server.listen(5)
                server.settimeout(1)  # Allow periodic checks
                
                while True:
                    try:
                        conn, _ = server.accept()
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        
                        # Handle echo requests
                        while True:
                            try:
                                data = conn.recv(1024)
                                if not data:
                                    break
                                conn.send(data)
                            except:
                                break
                        conn.close()
                    except socket.timeout:
                        continue
                    except:
                        break
            except Exception as e:
                self.logger.error(f"Server error: {e}")
            finally:
                server.close()
        
        # Start server in background thread
        self.latency_server = threading.Thread(target=server_loop, daemon=True)
        self.latency_server.start()
        time.sleep(0.5)  # Let server start
    
    def measure_trading_latency(self):
        """Measure persistent connection latency"""
        try:
            # Connect to our persistent server
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client.settimeout(2)
            
            client.connect(('127.0.0.1', self.server_port))
            
            message = b'TRADING_LATENCY_TEST'
            warmup = self.config['system']['latency_warmup']
            samples = self.config['system']['latency_samples']
            
            # Warmup phase
            for _ in range(warmup):
                client.send(message)
                client.recv(1024)
            
            # Actual measurements
            latencies = []
            for _ in range(samples):
                start = time.perf_counter_ns()
                client.send(message)
                response = client.recv(1024)
                end = time.perf_counter_ns()
                
                latency_us = (end - start) / 1000
                latencies.append(latency_us)
            
            client.close()
            
            # Calculate stats
            latencies.sort()
            return {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': latencies[int(len(latencies) * 0.95)],
                'p99': latencies[int(len(latencies) * 0.99)],
                'min': min(latencies),
                'max': max(latencies)
            }
        except Exception as e:
            self.logger.error(f"Latency test failed: {e}")
            # Try to restart server if connection failed
            if "Connection refused" in str(e):
                self.logger.info("Restarting latency server...")
                self.start_latency_server()
            return None
    
    def check_cpu_isolation(self):
        """Check CPU isolation status"""
        isolated_cpus = self.config['system']['isolated_cpus']
        violations = []
        
        for cpu in isolated_cpus:
            usage = psutil.cpu_percent(interval=0.1, percpu=True)[cpu]
            if usage > self.config['thresholds']['cpu_isolated_usage_percent']:
                violations.append(f"CPU{cpu}: {usage:.1f}%")
        
        return {
            'isolated_cpus': isolated_cpus,
            'violations': violations,
            'status': 'OK' if not violations else 'VIOLATION'
        }
    
    def check_irq_affinity(self):
        """Check IRQ affinity for isolated CPUs"""
        isolated_cpus = set(self.config['system']['isolated_cpus'])
        violations = []
        
        try:
            with open('/proc/interrupts', 'r') as f:
                lines = f.readlines()
            
            for line in lines[1:]:
                parts = line.split()
                if len(parts) > 4 and ':' in parts[0]:
                    irq_num = parts[0].rstrip(':')
                    
                    for iface in self.config['system']['network_interfaces']:
                        if iface in line:
                            try:
                                with open(f'/proc/irq/{irq_num}/smp_affinity_list', 'r') as f:
                                    affinity = f.read().strip()
                                    assigned_cpus = set()
                                    for cpu_range in affinity.split(','):
                                        if '-' in cpu_range:
                                            start, end = map(int, cpu_range.split('-'))
                                            assigned_cpus.update(range(start, end + 1))
                                        else:
                                            assigned_cpus.add(int(cpu_range))
                                    
                                    if assigned_cpus & isolated_cpus:
                                        violations.append(f"IRQ {irq_num} ({iface})")
                            except:
                                pass
        except Exception as e:
            self.logger.error(f"IRQ check failed: {e}")
        
        return {
            'violations': violations,
            'status': 'OK' if not violations else 'VIOLATION'
        }
    
    def check_network_interfaces(self):
        """Check network interface statistics"""
        stats = {}
        
        for iface in self.config['system']['network_interfaces']:
            try:
                net_stats = psutil.net_io_counters(pernic=True).get(iface)
                if net_stats:
                    stats[iface] = {
                        'packets_sent': net_stats.packets_sent,
                        'packets_recv': net_stats.packets_recv,
                        'dropin': net_stats.dropin,
                        'dropout': net_stats.dropout,
                        'status': 'OK' if (net_stats.dropin + net_stats.dropout) < self.config['thresholds']['network_drops_max'] else 'WARNING'
                    }
            except:
                stats[iface] = {'status': 'ERROR'}
        
        return stats
    
    def check_gpu_status(self):
        """Check GPU status using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpus.append({
                        'name': parts[0],
                        'memory_used': float(parts[1]),
                        'memory_total': float(parts[2]),
                        'temperature': float(parts[3]),
                        'utilization': float(parts[4]),
                        'status': 'OK'
                    })
            return gpus
        except:
            return []
    
    def collect_all_metrics(self):
        """Collect all system metrics"""
        self.logger.info("Collecting system metrics...")
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'latency': self.measure_trading_latency(),
            'cpu_isolation': self.check_cpu_isolation(),
            'irq_affinity': self.check_irq_affinity(),
            'network': self.check_network_interfaces(),
            'gpu': self.check_gpu_status()
        }
        
        # Check for alerts
        if metrics['latency'] and metrics['latency']['mean'] > self.config['thresholds']['latency_mean_microseconds']:
            self.alerts.append(f"High latency: {metrics['latency']['mean']:.2f}μs")
        
        if metrics['cpu_isolation']['violations']:
            self.alerts.append(f"CPU violations: {metrics['cpu_isolation']['violations']}")
        
        if metrics['irq_affinity']['violations']:
            self.alerts.append(f"IRQ violations: {metrics['irq_affinity']['violations']}")
        
        self.metrics = metrics
        return metrics
    
    def run_once(self):
        """Run monitoring once and display results"""
        metrics = self.collect_all_metrics()
        
        print("\n" + "="*60)
        print(f"🚀 AI TRADING STATION - SYSTEM STATUS")
        print(f"   Time: {metrics['timestamp']}")
        print("="*60)
        
        # Latency
        if metrics['latency']:
            lat = metrics['latency']
            status = "✅" if lat['mean'] < self.config['thresholds']['latency_mean_microseconds'] else "⚠️"
            print(f"\n📊 Trading Latency (Persistent Connection):")
            print(f"   {status} Mean: {lat['mean']:.2f}μs")
            print(f"   P99: {lat['p99']:.2f}μs")
            print(f"   Range: {lat['min']:.2f}μs - {lat['max']:.2f}μs")
        
        # CPU, IRQ, Network, GPU (abbreviated for space)
        print(f"\n🔧 CPU Isolation: {metrics['cpu_isolation']['status']}")
        print(f"🔌 IRQ Affinity: {metrics['irq_affinity']['status']}")
        
        print("\n" + "="*60)
        print("✅ Monitoring complete")
    
    def run_continuous(self):
        """Run continuous monitoring"""
        interval = self.config['monitoring']['interval_seconds']
        self.logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        while True:
            try:
                self.collect_all_metrics()
                if self.metrics.get('latency'):
                    self.logger.info(f"Latency: {self.metrics['latency']['mean']:.2f}μs")
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI Trading Station Monitor V3')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()
    
    monitor = TradingSystemMonitor()
    
    if args.once:
        monitor.run_once()
    else:
        monitor.run_continuous()

if __name__ == '__main__':
    main()
