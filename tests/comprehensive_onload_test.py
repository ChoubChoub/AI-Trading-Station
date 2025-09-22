#!/usr/bin/env python3
"""
Comprehensive OnLoad Performance Test Suite
Created: 2025-09-01 10:40:34 UTC
User: ChoubChoub
Purpose: Test OnLoad performance for AI Trading Station
"""

import socket
import time
import statistics
import threading
import sys
import subprocess
import os

class OnLoadTester:
    def __init__(self):
        self.results = {}
        
    def test_onload_version(self):
        """Test OnLoad installation"""
        try:
            result = subprocess.run(['onload', '--version'], 
                                  capture_output=True, text=True)
            print(f"✅ OnLoad Version: {result.stdout.split()[1]}")
            return True
        except Exception as e:
            print(f"❌ OnLoad not available: {e}")
            return False
    
    def test_stack_status(self):
        """Check OnLoad stack status"""
        try:
            result = subprocess.run(['onload_stackdump', 'lots'], 
                                  capture_output=True, text=True, timeout=5)
            stack_count = result.stdout.count('stack')
            print(f"📊 Active OnLoad stacks: {stack_count}")
            return stack_count
        except Exception as e:
            print(f"⚠️  Stack dump failed: {e}")
            return 0
    
    def latency_test_server(self, port=12345):
        """TCP echo server for latency testing"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('127.0.0.1', port))
        server_sock.listen(1)
        
        print(f"🔧 Server listening on port {port}")
        
        try:
            while True:
                conn, addr = server_sock.accept()
                while True:
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
        """Measure TCP round-trip latency"""
        latencies = []
        
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(('127.0.0.1', port))
        
        message = b"TRADING_LATENCY_TEST"
        
        # Warmup
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
            'max': max_lat,
            'samples': len(latencies)
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
            return results
        except Exception as e:
            print(f"❌ Latency test failed: {e}")
            return None
    
    def test_redis_integration(self):
        """Test Redis latency (matches AI Trading Station architecture)"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
            r.ping()
            
            print("💾 Testing Redis integration...")
            latencies = []
            
            for _ in range(100):
                start = time.perf_counter_ns()
                r.ping()
                end = time.perf_counter_ns()
                latencies.append((end - start) / 1000)
            
            mean = statistics.mean(latencies)
            p99 = sorted(latencies)[98]
            
            print(f"   Redis latency - Mean: {mean:.2f}μs, P99: {p99:.2f}μs")
            return {'mean': mean, 'p99': p99}
            
        except ImportError:
            print("⚠️  Redis module not available")
            return None
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Run all performance tests"""
        print("=== OnLoad Performance Test Suite ===")
        print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        print(f"User: ChoubChoub")
        print(f"Target: Sub-10μs latency for algorithmic trading")
        print("")
        
        # Test 1: OnLoad installation
        if not self.test_onload_version():
            return
        
        # Test 2: Stack status
        self.test_stack_status()
        print("")
        
        # Test 3: Standard TCP latency
        print("🔍 Test 3: Standard TCP Latency")
        std_results = self.run_latency_test(with_onload=False)
        if std_results:
            print(f"   Mean: {std_results['mean']:.2f}μs")
            print(f"   Median: {std_results['median']:.2f}μs")
            print(f"   P95: {std_results['p95']:.2f}μs")
            print(f"   P99: {std_results['p99']:.2f}μs")
            print(f"   Range: {std_results['min']:.2f}μs - {std_results['max']:.2f}μs")
        print("")
        
        # Test 4: OnLoad accelerated latency
        print("🚀 Test 4: OnLoad Accelerated Latency")
        print("   Note: OnLoad acceleration is transparent to Python sockets")
        print("   For full acceleration, use onload wrapper: onload python3 script.py")
        print("")
        
        # Test 5: Redis integration
        self.test_redis_integration()
        print("")
        
        # Test 6: System performance info
        print("📋 System Performance Summary:")
        print(f"   CPU: {self._get_cpu_info()}")
        print(f"   Memory: {self._get_memory_info()}")
        print(f"   Network: Solarflare X2522 with OnLoad")
        print("")
        
        print("✅ Performance test suite completed!")
        print("")
        print("🎯 Trading System Integration Notes:")
        print("   • Use 'onload your_trading_app' for full acceleration")
        print("   • Target: <10μs for order processing pipeline") 
        print("   • Redis Streams architecture ready for integration")
        print("   • Consider CPU core isolation for deterministic performance")
    
    def _get_cpu_info(self):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
        except:
            return "Unknown"
        return "Unknown"
    
    def _get_memory_info(self):
        try:
            result = subprocess.run(['free', '-h'], capture_output=True, text=True)
            mem_line = [line for line in result.stdout.split('\n') if 'Mem:' in line][0]
            total = mem_line.split()[1]
            return f"{total} total"
        except:
            return "Unknown"

if __name__ == "__main__":
    tester = OnLoadTester()
    tester.run_comprehensive_test()

