#!/usr/bin/env python3
"""
Network Path Latency Harness - Phase 4A
Establishes reproducible external RTT + command latency baseline

Purpose: Measure real network path vs loopback illusion
Author: AI Trading Station Phase 4A  
Date: September 28, 2025
"""

import redis
import time
import json
import argparse
import statistics
import sys
import os
import socket
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NetworkTestConfig:
    """Configuration for network path testing"""
    host: str
    port: int
    cpu_affinity: Optional[int]
    test_duration: int
    sample_count: int
    pipeline_size: int
    qps_target: Optional[int]
    
@dataclass 
class LatencyMetrics:
    """Latency measurement results"""
    p50: float
    p95: float
    p99: float
    p99_9: float
    jitter: float
    tail_span: float  # p99.9 - p99
    stability_index: float  # (p99 - p95) / p99
    sample_count: int
    min_latency: float
    max_latency: float

class NetworkLatencyHarness:
    """Network path latency measurement harness"""
    
    def __init__(self, config: NetworkTestConfig):
        self.config = config
        self.redis_client = None
        self.results_dir = "network_tests"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def setup_cpu_affinity(self):
        """Set CPU affinity for this process if specified"""
        if self.config.cpu_affinity is not None:
            try:
                import psutil
                p = psutil.Process()
                p.cpu_affinity([self.config.cpu_affinity])
                print(f"✅ Set CPU affinity to core {self.config.cpu_affinity}")
            except ImportError:
                print("⚠️  psutil not available, using taskset fallback")
                # Use taskset as fallback
                pid = os.getpid()
                subprocess.run([
                    'taskset', '-cp', str(self.config.cpu_affinity), str(pid)
                ], capture_output=True)
    
    def connect_redis(self) -> bool:
        """Establish Redis connection with validation"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=False  # Raw bytes for timing accuracy
            )
            
            # Test connection
            self.redis_client.ping()
            
            # Get connection info
            info = self.redis_client.connection_pool.connection_kwargs
            print(f"✅ Connected to Redis at {self.config.host}:{self.config.port}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            return False
    
    def measure_single_command_latency(self, command: str = "PING") -> List[float]:
        """Measure latency of single Redis commands"""
        latencies = []
        
        print(f"📏 Measuring {self.config.sample_count} {command} commands...")
        
        for i in range(self.config.sample_count):
            start_time = time.perf_counter()
            
            if command == "PING":
                self.redis_client.ping()
            elif command == "GET":
                # Use a key that doesn't exist for consistent behavior
                self.redis_client.get("nonexistent:benchmark:key")
            elif command == "SET":
                self.redis_client.set("benchmark:temp", b"x" * 64)
            
            end_time = time.perf_counter()
            latency_us = (end_time - start_time) * 1_000_000
            latencies.append(latency_us)
            
            # Progress indicator
            if (i + 1) % (self.config.sample_count // 10) == 0:
                progress = (i + 1) / self.config.sample_count * 100
                print(f"  Progress: {progress:.0f}% ({i+1}/{self.config.sample_count})")
        
        return latencies
    
    def measure_pipeline_latency(self, pipeline_size: int) -> List[float]:
        """Measure latency with Redis pipeline"""
        latencies = []
        iterations = self.config.sample_count // pipeline_size
        
        print(f"📏 Measuring {iterations} pipelines of size {pipeline_size}...")
        
        for i in range(iterations):
            pipe = self.redis_client.pipeline()
            
            # Build pipeline
            for _ in range(pipeline_size):
                pipe.ping()
            
            # Execute and measure
            start_time = time.perf_counter()
            results = pipe.execute()
            end_time = time.perf_counter()
            
            # Calculate per-command latency
            total_latency_us = (end_time - start_time) * 1_000_000
            per_command_latency = total_latency_us / pipeline_size
            latencies.append(per_command_latency)
        
        return latencies
    
    def measure_sustained_qps_latency(self, target_qps: int, duration: int = 10) -> List[float]:
        """Measure latency under sustained QPS load"""
        latencies = []
        interval = 1.0 / target_qps
        
        print(f"📊 Measuring latency under {target_qps} QPS for {duration}s...")
        
        start_time = time.perf_counter()
        end_time = start_time + duration
        commands_sent = 0
        
        while time.perf_counter() < end_time:
            loop_start = time.perf_counter()
            
            # Send command and measure
            cmd_start = time.perf_counter()
            self.redis_client.ping()
            cmd_end = time.perf_counter()
            
            latency_us = (cmd_end - cmd_start) * 1_000_000
            latencies.append(latency_us)
            commands_sent += 1
            
            # Rate limiting
            elapsed = time.perf_counter() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        actual_duration = time.perf_counter() - start_time
        actual_qps = commands_sent / actual_duration
        
        print(f"  Sent {commands_sent} commands in {actual_duration:.2f}s")
        print(f"  Actual QPS: {actual_qps:.0f} (target: {target_qps})")
        
        return latencies
    
    def calculate_metrics(self, latencies: List[float]) -> LatencyMetrics:
        """Calculate comprehensive latency metrics"""
        if not latencies:
            raise ValueError("No latency measurements provided")
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        # Percentiles
        p50 = sorted_latencies[int(n * 0.50)]
        p95 = sorted_latencies[int(n * 0.95)]
        p99 = sorted_latencies[int(n * 0.99)]
        p99_9 = sorted_latencies[int(n * 0.999)] if n >= 1000 else sorted_latencies[-1]
        
        # Derived metrics
        jitter = p99 - p50
        tail_span = p99_9 - p99
        stability_index = (p99 - p95) / p99 if p99 > 0 else 0.0
        
        return LatencyMetrics(
            p50=p50,
            p95=p95,
            p99=p99,
            p99_9=p99_9,
            jitter=jitter,
            tail_span=tail_span,
            stability_index=stability_index,
            sample_count=n,
            min_latency=min(latencies),
            max_latency=max(latencies)
        )
    
    def get_network_info(self) -> Dict:
        """Gather network path information"""
        info = {
            'client_host': socket.gethostname(),
            'redis_host': self.config.host,
            'redis_port': self.config.port,
            'is_loopback': self.config.host in ['127.0.0.1', 'localhost'],
            'cpu_affinity': self.config.cpu_affinity
        }
        
        # Try to get network interface info
        try:
            if self.config.host not in ['127.0.0.1', 'localhost']:
                # External host - try to get route info
                result = subprocess.run([
                    'ip', 'route', 'get', self.config.host
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    info['route_info'] = result.stdout.strip()
        except Exception:
            pass
        
        return info
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive network path latency test"""
        print(f"🚀 Starting comprehensive network latency test")
        print(f"   Target: {self.config.host}:{self.config.port}")
        print(f"   CPU Affinity: {self.config.cpu_affinity}")
        print(f"   Samples: {self.config.sample_count}")
        
        # Setup
        self.setup_cpu_affinity()
        
        if not self.connect_redis():
            return {}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'host': self.config.host,
                'port': self.config.port,
                'cpu_affinity': self.config.cpu_affinity,
                'sample_count': self.config.sample_count,
                'pipeline_size': self.config.pipeline_size
            },
            'network_info': self.get_network_info(),
            'tests': {}
        }
        
        # Test 1: Single PING commands
        print("\n🔍 Test 1: Single PING latency")
        ping_latencies = self.measure_single_command_latency("PING")
        results['tests']['ping'] = {
            'metrics': self.calculate_metrics(ping_latencies).__dict__,
            'raw_latencies': ping_latencies[:100]  # Store first 100 for analysis
        }
        
        # Test 2: Single GET commands
        print("\n🔍 Test 2: Single GET latency")
        get_latencies = self.measure_single_command_latency("GET")
        results['tests']['get'] = {
            'metrics': self.calculate_metrics(get_latencies).__dict__,
            'raw_latencies': get_latencies[:100]
        }
        
        # Test 3: Pipeline test (if configured)
        if self.config.pipeline_size > 1:
            print(f"\n🔍 Test 3: Pipeline latency (size {self.config.pipeline_size})")
            pipeline_latencies = self.measure_pipeline_latency(self.config.pipeline_size)
            results['tests']['pipeline'] = {
                'metrics': self.calculate_metrics(pipeline_latencies).__dict__,
                'raw_latencies': pipeline_latencies[:100]
            }
        
        # Test 4: QPS load test (if configured)  
        if self.config.qps_target:
            print(f"\n🔍 Test 4: Sustained QPS load ({self.config.qps_target} QPS)")
            qps_latencies = self.measure_sustained_qps_latency(
                self.config.qps_target, 
                self.config.test_duration
            )
            results['tests']['qps_load'] = {
                'metrics': self.calculate_metrics(qps_latencies).__dict__,
                'raw_latencies': qps_latencies[:100]
            }
        
        # Summary
        print("\n📊 Test Summary:")
        for test_name, test_data in results['tests'].items():
            metrics = test_data['metrics']
            print(f"   {test_name.upper()}: P99={metrics['p99']:.2f}μs, "
                  f"P99.9={metrics['p99_9']:.2f}μs, "
                  f"Tail Span={metrics['tail_span']:.2f}μs")
        
        return results
    
    def save_results(self, results: Dict, test_name: str = None):
        """Save test results to JSON file"""
        if not results:
            return
        
        if test_name is None:
            test_name = f"network_test_{self.config.host}_{self.config.port}_{int(time.time())}"
        
        filename = f"{self.results_dir}/{test_name}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Network Path Latency Harness')
    parser.add_argument('--host', default='127.0.0.1', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--cpu', type=int, help='CPU core for affinity')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--duration', type=int, default=10, help='Test duration for QPS tests')
    parser.add_argument('--pipeline', type=int, default=1, help='Pipeline size')
    parser.add_argument('--qps', type=int, help='Target QPS for load test')
    parser.add_argument('--output', help='Output filename (without extension)')
    
    args = parser.parse_args()
    
    config = NetworkTestConfig(
        host=args.host,
        port=args.port,
        cpu_affinity=args.cpu,
        test_duration=args.duration,
        sample_count=args.samples,
        pipeline_size=args.pipeline,
        qps_target=args.qps
    )
    
    harness = NetworkLatencyHarness(config)
    results = harness.run_comprehensive_test()
    
    if results:
        harness.save_results(results, args.output)
        
        # Print RTT inflation analysis if not loopback
        if not results['network_info']['is_loopback']:
            print("\n🔍 RTT Inflation Analysis:")
            print("   Compare these results with loopback baseline")
            print("   Expected inflation: 1.3x-3x for network path")

if __name__ == '__main__':
    main()