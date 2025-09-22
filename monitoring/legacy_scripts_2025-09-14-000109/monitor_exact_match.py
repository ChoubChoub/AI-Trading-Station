#!/usr/bin/env python3
"""Exact match of your comprehensive_onload_test.py latency test"""

import socket
import time
import threading
import statistics
import os

class ExactLatencyTest:
    def __init__(self):
        self.server_running = True
        
    def latency_test_server(self, port):
        """Server exactly as in comprehensive_onload_test.py"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.bind(('127.0.0.1', port))
        server.listen(1)
        
        print(f"🔧 Server listening on port {port}")
        
        while self.server_running:
            try:
                conn, addr = server.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                data = conn.recv(16)
                conn.send(data)
                conn.close()
            except:
                break
        server.close()
    
    def latency_test_client(self, port, samples):
        """Client exactly as in comprehensive_onload_test.py"""
        latencies = []
        
        print(f"🧪 Running {samples} latency measurements...")
        
        for i in range(samples):
            if i % 100 == 0:
                print(f"  Progress: {i}/{samples}")
            
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            start = time.perf_counter_ns()
            client.connect(('127.0.0.1', port))
            client.send(b'PING')
            data = client.recv(16)
            end = time.perf_counter_ns()
            
            client.close()
            
            latency_us = (end - start) / 1000
            latencies.append(latency_us)
        
        # Calculate stats
        latencies.sort()
        return {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'p95': latencies[int(len(latencies) * 0.95)],
            'p99': latencies[int(len(latencies) * 0.99)]
        }
    
    def run_latency_test(self, with_onload=True):
        """Run complete latency test"""
        port = 12346 if with_onload else 12345
        
        # Start server in background
        server_thread = threading.Thread(
            target=self.latency_test_server, 
            args=(port,), 
            daemon=True
        )
        server_thread.start()
        time.sleep(1)  # Let server start
        
        # Run client test
        try:
            results = self.latency_test_client(port, 1000)
            self.server_running = False
            return results
        except Exception as e:
            print(f"❌ Latency test failed: {e}")
            return None

def main():
    # Check Onload status
    if 'LD_PRELOAD' in os.environ and 'libonload' in os.environ['LD_PRELOAD']:
        print("✅ Running with Onload kernel bypass")
    else:
        print("⚠️  Not running with Onload")
    
    print("🔍 Test 3: Standard TCP Latency")
    
    tester = ExactLatencyTest()
    results = tester.run_latency_test(with_onload=False)
    
    if results:
        print(f"   Mean: {results['mean']:.2f}μs")
        print(f"   Median: {results['median']:.2f}μs")
        print(f"   P95: {results['p95']:.2f}μs")
        print(f"   P99: {results['p99']:.2f}μs")
        print(f"   Range: {results['min']:.2f}μs - {results['max']:.2f}μs")

if __name__ == '__main__':
    main()
