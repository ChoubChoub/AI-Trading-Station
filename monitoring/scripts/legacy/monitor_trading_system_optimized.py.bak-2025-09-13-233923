#!/usr/bin/env python3
"""
AI Trading Station - Optimized Monitor with 4.5μs Latency
Using exact same method as your baseline test
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
        self.server_thread = None
        self.server_running = False
        self.server_port = 12345
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def measure_trading_latency(self):
        """Measure latency EXACTLY like your baseline test that gets 4.5μs"""
        port = self.server_port
        
        # Start fresh server for each measurement (like your baseline)
        self.server_running = True
        server_thread = threading.Thread(
            target=self._latency_server, 
            args=(port,), 
            daemon=True
        )
        server_thread.start()
        time.sleep(0.5)  # Let server start
        
        try:
            # EXACT COPY of your working latency_test_client
            latencies = []
            
            # Single persistent connection - THIS IS KEY!
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Critical: TCP_NODELAY for low latency
            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_sock.connect(('127.0.0.1', port))
            
            message = b"TRADING_LATENCY_TEST"
            
            # Warmup - critical for consistent low latency
            for _ in range(100):
                client_sock.send(message)
                client_sock.recv(1024)
            
            # Actual measurements
            for i in range(1000):
                start = time.perf_counter_ns()
                client_sock.send(message)
                response = client_sock.recv(1024)
                end = time.perf_counter_ns()
                
                latency_us = (end - start) / 1000
                latencies.append(latency_us)
            
            client_sock.close()
            self.server_running = False
            
            # Calculate statistics
            latencies.sort()
            return {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'p95': latencies[int(0.95 * len(latencies))],
                'p99': latencies[int(0.99 * len(latencies))],
                'min': min(latencies),
                'max': max(latencies)
            }
        except Exception as e:
            self.logger.error(f"Latency test failed: {e}")
            self.server_running = False
            return None
    
    def _latency_server(self, port):
        """TCP echo server - EXACT COPY of your baseline"""
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Critical: TCP_NODELAY on server too
            server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            server_sock.bind(('127.0.0.1', port))
            server_sock.listen(1)
            
            while self.server_running:
                try:
                    conn, addr = server_sock.accept()
                    # TCP_NODELAY on accepted connection
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while self.server_running:
                        data = conn.recv(1024)
                        if not data:
                            break
                        conn.send(data)  # Echo back immediately
                    conn.close()
                except:
                    break
        except:
            pass
        finally:
            try:
                server_sock.close()
            except:
                pass
    
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
                capture_output=True, text=True, timeout=5
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
        
        # Latency (Most Important!)
        if metrics['latency']:
            lat = metrics['latency']
            status = "✅" if lat['mean'] < self.config['thresholds']['latency_mean_microseconds'] else "⚠️"
            print(f"\n📊 Trading Latency (Persistent Connection):")
            print(f"   {status} Mean: {lat['mean']:.2f}μs")
            print(f"   Median: {lat['median']:.2f}μs")
            print(f"   P99: {lat['p99']:.2f}μs")
            print(f"   Range: {lat['min']:.2f}μs - {lat['max']:.2f}μs")
            
            if lat['mean'] < 5:
                print(f"   🏆 WORLD-CLASS PERFORMANCE!")
        
        # CPU Isolation
        cpu = metrics['cpu_isolation']
        print(f"\n🔧 CPU Isolation (Cores {cpu['isolated_cpus']}):")
        print(f"   Status: {cpu['status']}")
        
        # IRQ Affinity
        irq = metrics['irq_affinity']
        print(f"\n🔌 IRQ Affinity:")
        print(f"   Status: {irq['status']}")
        
        # Network
        print(f"\n🌐 Network Interfaces:")
        for iface, stats in metrics['network'].items():
            print(f"   {iface}: {stats['status']}")
        
        # GPU
        if metrics['gpu']:
            print(f"\n🎮 GPU Status:")
            for i, gpu in enumerate(metrics['gpu']):
                print(f"   GPU{i}: {gpu['temperature']:.0f}°C, {gpu['utilization']:.0f}% load")
        
        print("\n" + "="*60)
        print("✅ Monitoring complete")
        print("="*60)
    
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
    parser = argparse.ArgumentParser(description='AI Trading Station Optimized Monitor')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()
    
    monitor = TradingSystemMonitor()
    
    if args.once:
        monitor.run_once()
    else:
        monitor.run_continuous()

if __name__ == '__main__':
    main()
