#!/usr/bin/env python3
"""Monitoring with persistent connection - matches your baseline exactly"""

import socket
import time
import threading
import statistics
import os

class PersistentLatencyTest:
    def __init__(self):
        self.server_running = True
    
    def latency_test_server(self, port=12345):
        """TCP echo server for latency testing - EXACT MATCH"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('127.0.0.1', port))
        server_sock.listen(1)
        
        print(f"🔧 Server listening on port {port}")
        
        try:
            while self.server_running:
                conn, addr = server_sock.accept()
                while self.server_running:
                    data = conn.recv(1024)
                    if not data:
                        break
                    conn.send(data)  # Echo back immediately
                conn.close()
        except:
            pass
        finally:
            server_sock.close()
    
    def latency_test_client(self, port=12345, num_tests=1000):
        """Measure TCP round-trip latency - EXACT MATCH with persistent connection"""
        latencies = []
        
        # SINGLE CONNECTION - this is the key!
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(('127.0.0.1', port))
        
        message = b"TRADING_LATENCY_TEST"
        
        # Warmup - critical for consistent results
        print("🔥 Warming up connection...")
        for _ in range(100):
            client_sock.send(message)
            client_sock.recv(1024)
        
        print(f"🧪 Running {num_tests} latency measurements...")
        
        for i in range(num_tests):
            start = time.perf_counter_ns()
            client_sock.send(message)
            response = client_sock.recv(1024)
            end = time.perf_counter_ns()
            
            latency_us = (end - start) / 1000  # Convert to microseconds
            latencies.append(latency_us)
            
            if i % 100 == 0:
                print(f"  Progress: {i}/{num_tests}")
        
        client_sock.close()
        self.server_running = False
        
        # Calculate statistics
        mean = statistics.mean(latencies)
        median = statistics.median(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        return {
            'mean': mean,
            'median': median,
            'p95': p95,
            'p99': p99,
            'min': min_lat,
            'max': max_lat
        }
    
    def run_test(self):
        """Run the complete test"""
        # Start server
        server_thread = threading.Thread(
            target=self.latency_test_server,
            daemon=True
        )
        server_thread.start()
        time.sleep(0.5)  # Let server start
        
        # Run client test
        results = self.latency_test_client()
        
        return results

def main():
    # Check Onload
    if 'LD_PRELOAD' in os.environ and 'libonload' in os.environ['LD_PRELOAD']:
        print("✅ Running with Onload kernel bypass")
    else:
        print("⚠️ Not running with Onload - use onload-trading wrapper")
    
    print("🔍 Test 3: Standard TCP Latency (Persistent Connection)")
    
    tester = PersistentLatencyTest()
    results = tester.run_test()
    
    if results:
        print(f"   Mean: {results['mean']:.2f}μs")
        print(f"   Median: {results['median']:.2f}μs")
        print(f"   P95: {results['p95']:.2f}μs")
        print(f"   P99: {results['p99']:.2f}μs")
        print(f"   Range: {results['min']:.2f}μs - {results['max']:.2f}μs")
        
        # Compare to your baseline
        baseline = 4.56
        if results['mean'] < baseline * 1.5:
            print(f"\n🎯 EXCELLENT: Close to baseline ({results['mean']/baseline:.1f}x)")
        elif results['mean'] < 10:
            print(f"\n✅ GOOD: Within trading target (<10μs)")

if __name__ == '__main__':
    main()
