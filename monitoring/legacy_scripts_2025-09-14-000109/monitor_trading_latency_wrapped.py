#!/usr/bin/env python3
"""Trading latency monitor designed to work with onload-trading wrapper"""

import json
import time
import socket
import statistics
import argparse
import logging
from datetime import datetime
import threading
import os
import sys

class TradingLatencyMonitor:
    def __init__(self, test_type='local', port=12345, samples=1000):
        self.test_type = test_type
        self.port = port
        self.samples = samples
        
        # Check if we're running under Onload
        self.onload_active = 'LD_PRELOAD' in os.environ and 'libonload' in os.environ.get('LD_PRELOAD', '')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if self.onload_active:
            self.logger.info("✅ Running with Onload kernel bypass acceleration")
        else:
            self.logger.info("⚠️  Not running with Onload - use 'onload-trading' wrapper for accurate results")
    
    def test_local_tcp_latency(self):
        """Test local TCP latency - matches your ai-trading-station.sh test"""
        self.logger.info("🔍 Testing Local TCP Latency (Trading Performance)")
        
        # Start server in background thread
        server_thread = threading.Thread(target=self._tcp_server)
        server_thread.daemon = True
        server_thread.start()
        time.sleep(0.5)  # Let server start
        
        latencies = []
        self.logger.info(f"🧪 Running {self.samples} latency measurements...")
        
        for i in range(self.samples):
            if i % 100 == 0 and i > 0:
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
        
        self.logger.info(f"🔧 Server listening on port {self.port}")
        
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

def main():
    parser = argparse.ArgumentParser(description='Trading latency monitor for onload-trading wrapper')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--port', type=int, default=12345, help='Port for TCP test')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    monitor = TradingLatencyMonitor(samples=args.samples, port=args.port)
    
    # Check if running with onload-trading wrapper
    if not monitor.onload_active:
        print("⚠️  WARNING: Not running with onload-trading wrapper!")
        print("   For accurate trading latency, run:")
        print("   onload-trading python3 monitor_trading_latency_wrapped.py")
        print("")
    
    results = monitor.test_local_tcp_latency()
    
    if results:
        print("")
        print("📊 Results:")
        print(f"   Mean: {results['mean']:.2f}μs")
        print(f"   Median: {results['median']:.2f}μs")
        print(f"   P95: {results['p95']:.2f}μs")
        print(f"   P99: {results['p99']:.2f}μs")
        print(f"   Range: {results['min']:.2f}μs - {results['max']:.2f}μs")
        
        # Compare to baseline
        baseline = 5.42  # Your target
        if monitor.onload_active:
            if results['mean'] < baseline:
                print(f"   🎯 EXCELLENT: {baseline - results['mean']:.2f}μs better than baseline!")
            elif results['mean'] < 10:
                print(f"   ✅ GOOD: Within trading latency target (<10μs)")
            else:
                print(f"   ⚠️  Above 10μs target")
        
        if args.json:
            print("")
            print(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'onload_active': monitor.onload_active,
                'stats': results
            }, indent=2))

if __name__ == '__main__':
    main()
