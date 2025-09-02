#!/usr/bin/env python3

"""
Comprehensive OnLoad Latency Testing Suite
==========================================
Purpose: Validate OnLoad kernel bypass networking performance
Target: Sub-microsecond latency with 99.9% efficiency
Hardware: Solarflare X2522 10GbE with OnLoad acceleration
"""

import os
import sys
import time
import json
import socket
import struct
import threading
import statistics
import subprocess
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class LatencyResult:
    """Container for latency measurement results."""
    mean: float
    median: float
    p95: float
    p99: float
    p99_9: float
    min_val: float
    max_val: float
    std_dev: float
    sample_count: int
    timestamp: str

@dataclass
class TestConfiguration:
    """Test configuration parameters."""
    host: str = "127.0.0.1"
    port: int = 12345
    message_size: int = 64
    iterations: int = 10000
    warmup_iterations: int = 1000
    use_onload: bool = True
    cpu_affinity: Optional[List[int]] = None
    test_duration: int = 60

class OnLoadTester:
    """Comprehensive OnLoad performance testing framework."""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.results: Dict[str, LatencyResult] = {}
        self.test_start_time = datetime.now()
        
        # Performance targets (microseconds)
        self.targets = {
            'mean_latency': 4.37,
            'p95_latency': 4.53,
            'p99_latency': 4.89,
            'critical_threshold': 10.0
        }
        
    def setup_environment(self) -> bool:
        """Setup the test environment and validate OnLoad availability."""
        print("Setting up OnLoad test environment...")
        
        # Check OnLoad availability
        if not self._check_onload_available():
            print("WARNING: OnLoad not available, running in simulation mode")
            self.config.use_onload = False
        
        # Set CPU affinity if specified
        if self.config.cpu_affinity:
            self._set_cpu_affinity(self.config.cpu_affinity)
        
        # Configure socket options for low latency
        self._configure_system_for_low_latency()
        
        return True
    
    def _check_onload_available(self) -> bool:
        """Check if OnLoad is available and functional."""
        try:
            result = subprocess.run(['onload', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"OnLoad detected: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return False
    
    def _set_cpu_affinity(self, cpu_list: List[int]) -> None:
        """Set CPU affinity for the current process."""
        try:
            import psutil
            process = psutil.Process()
            process.cpu_affinity(cpu_list)
            print(f"Set CPU affinity to cores: {cpu_list}")
        except ImportError:
            print("psutil not available, skipping CPU affinity")
        except Exception as e:
            print(f"Failed to set CPU affinity: {e}")
    
    def _configure_system_for_low_latency(self) -> None:
        """Configure system parameters for optimal latency testing."""
        # Set process priority
        try:
            os.nice(-10)  # Higher priority
        except PermissionError:
            print("WARNING: Cannot set process priority (not running as root)")
        
        # Configure socket buffer sizes
        self.socket_options = [
            (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1),
            (socket.SOL_SOCKET, socket.SO_REUSEPORT, 1),
            (socket.SOL_TCP, socket.TCP_NODELAY, 1),
        ]
    
    @contextmanager
    def onload_context(self):
        """Context manager for OnLoad execution."""
        if self.config.use_onload:
            # In a real implementation, this would configure OnLoad environment
            os.environ['EF_POLL_USEC'] = '0'
            os.environ['EF_INT_DRIVEN'] = '0'
            os.environ['EF_RX_TIMESTAMPING'] = '1'
            os.environ['EF_TX_TIMESTAMPING'] = '1'
        
        try:
            yield
        finally:
            # Cleanup OnLoad environment
            if self.config.use_onload:
                for key in ['EF_POLL_USEC', 'EF_INT_DRIVEN', 'EF_RX_TIMESTAMPING', 'EF_TX_TIMESTAMPING']:
                    os.environ.pop(key, None)
    
    def create_test_server(self) -> socket.socket:
        """Create a test server socket with optimal settings."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Apply socket options for low latency
        for level, option, value in self.socket_options:
            try:
                server.setsockopt(level, option, value)
            except OSError as e:
                print(f"Warning: Could not set socket option {option}: {e}")
        
        server.bind((self.config.host, self.config.port))
        server.listen(1)
        
        return server
    
    def create_test_client(self) -> socket.socket:
        """Create a test client socket with optimal settings."""
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Apply socket options for low latency
        for level, option, value in self.socket_options:
            try:
                client.setsockopt(level, option, value)
            except OSError as e:
                print(f"Warning: Could not set socket option {option}: {e}")
        
        return client
    
    def measure_latency_ping_pong(self) -> List[float]:
        """Measure round-trip latency using ping-pong test."""
        latencies = []
        
        def server_worker(server_socket):
            """Server worker for ping-pong test."""
            conn, addr = server_socket.accept()
            try:
                while True:
                    data = conn.recv(self.config.message_size)
                    if not data:
                        break
                    conn.send(data)  # Echo back
            except Exception:
                pass
            finally:
                conn.close()
        
        with self.onload_context():
            # Create server
            server = self.create_test_server()
            server_thread = threading.Thread(target=server_worker, args=(server,))
            server_thread.daemon = True
            server_thread.start()
            
            time.sleep(0.1)  # Allow server to start
            
            # Create client and perform ping-pong test
            client = self.create_test_client()
            client.connect((self.config.host, self.config.port))
            
            message = b'X' * self.config.message_size
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                client.send(message)
                client.recv(self.config.message_size)
            
            # Actual test
            print(f"Running ping-pong test: {self.config.iterations} iterations...")
            
            for i in range(self.config.iterations):
                start_time = time.time_ns()
                client.send(message)
                response = client.recv(self.config.message_size)
                end_time = time.time_ns()
                
                if len(response) == self.config.message_size:
                    latency_us = (end_time - start_time) / 1000.0  # Convert to microseconds
                    latencies.append(latency_us)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Completed {i + 1}/{self.config.iterations} iterations")
            
            client.close()
            server.close()
        
        return latencies
    
    def measure_latency_udp(self) -> List[float]:
        """Measure UDP latency for comparison."""
        latencies = []
        
        def udp_server_worker():
            """UDP server worker."""
            server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            server.bind((self.config.host, self.config.port + 1))
            
            try:
                while True:
                    data, addr = server.recvfrom(self.config.message_size)
                    server.sendto(data, addr)  # Echo back
            except Exception:
                pass
            finally:
                server.close()
        
        with self.onload_context():
            # Start UDP server
            server_thread = threading.Thread(target=udp_server_worker)
            server_thread.daemon = True
            server_thread.start()
            
            time.sleep(0.1)  # Allow server to start
            
            # Create UDP client
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            message = b'X' * self.config.message_size
            
            # Warmup
            for _ in range(self.config.warmup_iterations):
                client.sendto(message, (self.config.host, self.config.port + 1))
                client.recvfrom(self.config.message_size)
            
            # Actual test
            print(f"Running UDP latency test: {self.config.iterations} iterations...")
            
            for i in range(self.config.iterations):
                start_time = time.time_ns()
                client.sendto(message, (self.config.host, self.config.port + 1))
                response, addr = client.recvfrom(self.config.message_size)
                end_time = time.time_ns()
                
                if len(response) == self.config.message_size:
                    latency_us = (end_time - start_time) / 1000.0
                    latencies.append(latency_us)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Completed {i + 1}/{self.config.iterations} iterations")
            
            client.close()
        
        return latencies
    
    def measure_throughput(self) -> Dict[str, float]:
        """Measure network throughput."""
        throughput_results = {}
        
        def throughput_server():
            """Throughput server worker."""
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for level, option, value in self.socket_options:
                try:
                    server.setsockopt(level, option, value)
                except OSError:
                    pass
            
            server.bind((self.config.host, self.config.port + 2))
            server.listen(1)
            
            conn, addr = server.accept()
            total_bytes = 0
            start_time = time.time()
            
            try:
                while time.time() - start_time < 10:  # 10 second test
                    data = conn.recv(65536)
                    if not data:
                        break
                    total_bytes += len(data)
            except Exception:
                pass
            finally:
                conn.close()
                server.close()
            
            duration = time.time() - start_time
            throughput_mbps = (total_bytes * 8) / (duration * 1000000)  # Mbps
            throughput_results['throughput_mbps'] = throughput_mbps
        
        with self.onload_context():
            # Start throughput server
            server_thread = threading.Thread(target=throughput_server)
            server_thread.daemon = True
            server_thread.start()
            
            time.sleep(0.1)
            
            # Create client and send data
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for level, option, value in self.socket_options:
                try:
                    client.setsockopt(level, option, value)
                except OSError:
                    pass
            
            client.connect((self.config.host, self.config.port + 2))
            
            data_chunk = b'X' * 65536
            start_time = time.time()
            bytes_sent = 0
            
            try:
                while time.time() - start_time < 10:
                    bytes_sent += client.send(data_chunk)
            except Exception:
                pass
            finally:
                client.close()
            
            server_thread.join(timeout=1)
        
        return throughput_results
    
    def calculate_statistics(self, latencies: List[float]) -> LatencyResult:
        """Calculate comprehensive latency statistics."""
        if not latencies:
            return LatencyResult(0, 0, 0, 0, 0, 0, 0, 0, 0, datetime.now().isoformat())
        
        latencies_array = np.array(latencies)
        
        return LatencyResult(
            mean=float(np.mean(latencies_array)),
            median=float(np.median(latencies_array)),
            p95=float(np.percentile(latencies_array, 95)),
            p99=float(np.percentile(latencies_array, 99)),
            p99_9=float(np.percentile(latencies_array, 99.9)),
            min_val=float(np.min(latencies_array)),
            max_val=float(np.max(latencies_array)),
            std_dev=float(np.std(latencies_array)),
            sample_count=len(latencies),
            timestamp=datetime.now().isoformat()
        )
    
    def evaluate_performance(self, result: LatencyResult) -> Dict[str, str]:
        """Evaluate performance against targets."""
        evaluation = {}
        
        evaluation['mean_target'] = 'PASS' if result.mean <= self.targets['mean_latency'] else 'FAIL'
        evaluation['p95_target'] = 'PASS' if result.p95 <= self.targets['p95_latency'] else 'FAIL'
        evaluation['p99_target'] = 'PASS' if result.p99 <= self.targets['p99_latency'] else 'FAIL'
        evaluation['critical_check'] = 'PASS' if result.max_val <= self.targets['critical_threshold'] else 'FAIL'
        
        evaluation['overall'] = 'PASS' if all(v == 'PASS' for v in evaluation.values()) else 'FAIL'
        
        return evaluation
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive OnLoad testing suite."""
        print("=" * 60)
        print("COMPREHENSIVE ONLOAD LATENCY TESTING SUITE")
        print("=" * 60)
        
        if not self.setup_environment():
            return {"error": "Failed to setup test environment"}
        
        results = {
            'test_metadata': {
                'timestamp': self.test_start_time.isoformat(),
                'hostname': os.uname().nodename,
                'onload_enabled': self.config.use_onload,
                'cpu_affinity': self.config.cpu_affinity,
                'test_configuration': {
                    'message_size': self.config.message_size,
                    'iterations': self.config.iterations,
                    'warmup_iterations': self.config.warmup_iterations
                }
            },
            'performance_targets': self.targets,
            'test_results': {}
        }
        
        try:
            # Test 1: TCP Ping-Pong Latency
            print("\n1. TCP Ping-Pong Latency Test")
            print("-" * 30)
            tcp_latencies = self.measure_latency_ping_pong()
            tcp_result = self.calculate_statistics(tcp_latencies)
            tcp_evaluation = self.evaluate_performance(tcp_result)
            
            results['test_results']['tcp_ping_pong'] = {
                'statistics': tcp_result.__dict__,
                'evaluation': tcp_evaluation
            }
            
            self._print_latency_results("TCP Ping-Pong", tcp_result, tcp_evaluation)
            
            # Test 2: UDP Latency  
            print("\n2. UDP Latency Test")
            print("-" * 20)
            udp_latencies = self.measure_latency_udp()
            udp_result = self.calculate_statistics(udp_latencies)
            udp_evaluation = self.evaluate_performance(udp_result)
            
            results['test_results']['udp_ping_pong'] = {
                'statistics': udp_result.__dict__,
                'evaluation': udp_evaluation
            }
            
            self._print_latency_results("UDP", udp_result, udp_evaluation)
            
            # Test 3: Throughput Test
            print("\n3. Throughput Test")
            print("-" * 15)
            throughput_data = self.measure_throughput()
            results['test_results']['throughput'] = throughput_data
            
            if throughput_data:
                print(f"Throughput: {throughput_data.get('throughput_mbps', 0):.2f} Mbps")
            
        except Exception as e:
            print(f"Test error: {e}")
            results['error'] = str(e)
        
        # Overall assessment
        results['overall_assessment'] = self._generate_overall_assessment(results)
        
        return results
    
    def _print_latency_results(self, test_name: str, result: LatencyResult, evaluation: Dict[str, str]):
        """Print formatted latency results."""
        print(f"{test_name} Results:")
        print(f"  Mean:     {result.mean:.3f}μs (target: ≤{self.targets['mean_latency']}μs) [{evaluation['mean_target']}]")
        print(f"  Median:   {result.median:.3f}μs")
        print(f"  P95:      {result.p95:.3f}μs (target: ≤{self.targets['p95_latency']}μs) [{evaluation['p95_target']}]")
        print(f"  P99:      {result.p99:.3f}μs (target: ≤{self.targets['p99_latency']}μs) [{evaluation['p99_target']}]")
        print(f"  P99.9:    {result.p99_9:.3f}μs")
        print(f"  Min/Max:  {result.min_val:.3f}μs / {result.max_val:.3f}μs")
        print(f"  Std Dev:  {result.std_dev:.3f}μs")
        print(f"  Samples:  {result.sample_count}")
        print(f"  Overall:  [{evaluation['overall']}]")
    
    def _generate_overall_assessment(self, results: Dict) -> Dict:
        """Generate overall performance assessment."""
        assessment = {
            'onload_status': 'enabled' if self.config.use_onload else 'disabled',
            'tests_passed': 0,
            'tests_failed': 0,
            'performance_grade': 'UNKNOWN',
            'recommendations': []
        }
        
        # Count pass/fail results
        for test_name, test_data in results.get('test_results', {}).items():
            if isinstance(test_data, dict) and 'evaluation' in test_data:
                evaluation = test_data['evaluation']
                if evaluation.get('overall') == 'PASS':
                    assessment['tests_passed'] += 1
                else:
                    assessment['tests_failed'] += 1
        
        # Generate performance grade
        total_tests = assessment['tests_passed'] + assessment['tests_failed']
        if total_tests > 0:
            pass_rate = assessment['tests_passed'] / total_tests
            if pass_rate >= 1.0:
                assessment['performance_grade'] = 'EXCELLENT'
            elif pass_rate >= 0.8:
                assessment['performance_grade'] = 'GOOD'
            elif pass_rate >= 0.6:
                assessment['performance_grade'] = 'FAIR'
            else:
                assessment['performance_grade'] = 'POOR'
        
        # Generate recommendations
        if not self.config.use_onload:
            assessment['recommendations'].append("Enable OnLoad for optimal performance")
        
        if assessment['tests_failed'] > 0:
            assessment['recommendations'].append("Review system configuration and hardware")
            assessment['recommendations'].append("Check CPU isolation and IRQ affinity settings")
        
        return assessment

def main():
    """Main test execution."""
    # Configure test parameters
    config = TestConfiguration(
        host="127.0.0.1",
        port=12345,
        message_size=64,
        iterations=10000,
        warmup_iterations=1000,
        use_onload=True,
        cpu_affinity=[2, 3]  # Trading cores
    )
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        if '--iterations' in sys.argv:
            idx = sys.argv.index('--iterations')
            if idx + 1 < len(sys.argv):
                config.iterations = int(sys.argv[idx + 1])
        
        if '--no-onload' in sys.argv:
            config.use_onload = False
        
        if '--help' in sys.argv:
            print("Comprehensive OnLoad Test Suite")
            print("Usage: python3 comprehensive_onload_test.py [options]")
            print("Options:")
            print("  --iterations N    Number of test iterations (default: 10000)")
            print("  --no-onload      Disable OnLoad for baseline testing")
            print("  --help           Show this help message")
            return
    
    # Run tests
    tester = OnLoadTester(config)
    results = tester.run_comprehensive_test()
    
    # Save results to file
    output_file = f"onload_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    assessment = results.get('overall_assessment', {})
    print(f"OnLoad Status: {assessment.get('onload_status', 'unknown').upper()}")
    print(f"Tests Passed: {assessment.get('tests_passed', 0)}")
    print(f"Tests Failed: {assessment.get('tests_failed', 0)}")
    print(f"Performance Grade: {assessment.get('performance_grade', 'UNKNOWN')}")
    
    if assessment.get('recommendations'):
        print("\nRecommendations:")
        for rec in assessment['recommendations']:
            print(f"  • {rec}")
    
    # Return appropriate exit code
    return 0 if assessment.get('performance_grade') in ['EXCELLENT', 'GOOD'] else 1

if __name__ == "__main__":
    exit(main())