#!/usr/bin/env python3
"""Trading-focused latency monitoring with Onload - tests LOCAL performance"""

import json
import time
import socket
import subprocess
import statistics
import argparse
import logging
from datetime import datetime
import threading
import os

class TradingLatencyMonitor:
    def __init__(self, test_type='local', port=12345, samples=1000):
        self.test_type = test_type
        self.port = port
        self.samples = samples
        self.onload_available = self.check_onload()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
    def check_onload(self):
        """Check if Onload is available"""
        try:
            result = subprocess.run(['which', 'onload'], capture_output=True, text=True)
            if result.returncode == 0:
                # Check Onload version
                ver_result = subprocess.run(['onload', '--version'], capture_output=True, text=True)
                self.logger.info(f"✅ Onload kernel bypass available")
                return True
        except:
            pass
        return False
    
    def test_local_tcp_latency(self):
        """Test local TCP latency (like your ai-trading-station.sh test)"""
        self.logger.info("🔍 Testing Local TCP Latency (Trading Performance)")
        
        # Start server in background thread
        server_thread = threading.Thread(target=self._tcp_server)
        server_thread.daemon = True
        server_thread.start()
        time.sleep(0.5)  # Let server start
        
        latencies = []
        self.logger.info(f"🧪 Running {self.samples} latency measurements...")
        
        for i in range(self.samples):
            if i % 100 == 0:
                self.logger.info(f"  Progress: {i}/{self.samples}")
            
            try:
                start = time.perf_counter()
                
                # Connect and send/receive
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client.connect(('127.0.0.1', self.port))
                
                # Send timestamp
                client.send(b'PING')
                response = client.recv(4)
                
                end = time.perf_counter()
                client.close()
                
                # Calculate round-trip time in microseconds
                latency_us = (end - start) * 1_000_000
                latencies.append(latency_us)
                
            except Exception as e:
                self.logger.error(f"Error in measurement: {e}")
                continue
        
        if latencies:
            return self._calculate_stats(latencies)
        return None
    
    def _tcp_server(self):
        """Simple TCP echo server for latency testing"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.bind(('127.0.0.1', self.port))
        server.listen(1)
        
        while True:
            try:
                conn, addr = server.accept()
                data = conn.recv(4)
                conn.send(data)  # Echo back
                conn.close()
            except:
                break
    
    def _calculate_stats(self, latencies):
        """Calculate statistics in microseconds"""
        latencies.sort()
        
        stats = {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'p95': latencies[int(len(latencies) * 0.95)],
            'p99': latencies[int(len(latencies) * 0.99)],
            'stddev': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        return stats
    
    def test_network_latency(self, interface='enp130s0f0', target='127.0.0.1'):
        """Test network interface latency"""
        self.logger.info(f"📡 Testing Network Latency on {interface} to {target}")
        
        cmd = ['ping', '-c', '100', '-i', '0.01', '-q']
        if interface:
            cmd.extend(['-I', interface])
        cmd.append(target)
        
        # Wrap with Onload if available
        if self.onload_available:
            cmd = ['onload', '--profile=latency'] + cmd
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if 'min/avg/max' in result.stdout:
                stats_line = [l for l in result.stdout.split('\n') if 'min/avg/max' in l][0]
                stats = stats_line.split('=')[1].strip().split('/')
                
                # Convert ms to microseconds
                return {
                    'min': float(stats[0]) * 1000,
                    'mean': float(stats[1]) * 1000,
                    'max': float(stats[2]) * 1000,
                    'unit': 'microseconds'
                }
        except Exception as e:
            self.logger.error(f"Network test failed: {e}")
        
        return None
    
    def run_comprehensive_test(self):
        """Run all latency tests"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'onload_enabled': self.onload_available,
            'tests': {}
        }
        
        # Test 1: Local TCP (most relevant for trading)
        self.logger.info("=" * 50)
        tcp_stats = self.test_local_tcp_latency()
        if tcp_stats:
            results['tests']['local_tcp'] = tcp_stats
            self.logger.info(f"   Mean: {tcp_stats['mean']:.2f}μs")
            self.logger.info(f"   Median: {tcp_stats['median']:.2f}μs")
            self.logger.info(f"   P95: {tcp_stats['p95']:.2f}μs")
            self.logger.info(f"   P99: {tcp_stats['p99']:.2f}μs")
            self.logger.info(f"   Range: {tcp_stats['min']:.2f}μs - {tcp_stats['max']:.2f}μs")
        
        # Test 2: Network interface
        self.logger.info("=" * 50)
        for iface in ['enp130s0f0', 'enp130s0f1']:
            net_stats = self.test_network_latency(interface=iface)
            if net_stats:
                results['tests'][f'network_{iface}'] = net_stats
                self.logger.info(f"{iface}: {net_stats['mean']:.0f}μs avg")
        
        return results
    
    def export_for_grafana(self, results):
        """Export metrics in Prometheus format"""
        metrics = []
        
        # Local TCP metrics (most important for trading)
        if 'local_tcp' in results['tests']:
            tcp = results['tests']['local_tcp']
            metrics.append(f'trading_latency_tcp_mean_us {tcp["mean"]:.2f}')
            metrics.append(f'trading_latency_tcp_p99_us {tcp["p99"]:.2f}')
            metrics.append(f'trading_latency_tcp_min_us {tcp["min"]:.2f}')
            metrics.append(f'trading_latency_tcp_max_us {tcp["max"]:.2f}')
        
        return '\n'.join(metrics)

def main():
    parser = argparse.ArgumentParser(description='Trading-focused latency monitoring')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--port', type=int, default=12345, help='Port for TCP test')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--prometheus', action='store_true', help='Output for Prometheus')
    
    args = parser.parse_args()
    
    monitor = TradingLatencyMonitor(samples=args.samples, port=args.port)
    
    print("🚀 Trading Latency Monitor - Onload Performance Test")
    print("=" * 50)
    
    results = monitor.run_comprehensive_test()
    
    if args.json:
        print(json.dumps(results, indent=2))
    elif args.prometheus:
        print(monitor.export_for_grafana(results))
    else:
        print("\n📊 Summary:")
        if 'local_tcp' in results['tests']:
            tcp = results['tests']['local_tcp']
            print(f"✅ Local TCP Latency (Trading Performance):")
            print(f"   Mean: {tcp['mean']:.2f}μs")
            print(f"   P99: {tcp['p99']:.2f}μs")
            
            # Compare to your baseline
            baseline = 5.42  # Your target baseline
            if tcp['mean'] < baseline:
                print(f"   🎯 EXCELLENT: {baseline - tcp['mean']:.2f}μs better than baseline!")
            else:
                print(f"   ⚠️  {tcp['mean'] - baseline:.2f}μs slower than baseline")

if __name__ == '__main__':
    main()
