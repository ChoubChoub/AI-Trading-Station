#!/usr/bin/env python3

"""
Simple Real-World Trading Performance Test
==========================================
Purpose: Lightweight performance validation for trading systems
Target: Quick verification of sub-10μs latency targets
Usage: Fast validation without complex multi-process simulation
"""

import os
import sys
import time
import json
import socket
import threading
import statistics
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

class SimpleLatencyTest:
    """Simple latency testing for trading applications."""
    
    def __init__(self, use_onload: bool = True):
        self.use_onload = use_onload
        self.test_duration = 30  # seconds
        self.message_size = 64
        self.target_latencies = {
            'mean': 4.37,
            'p95': 4.53,
            'p99': 4.89
        }
    
    def setup_environment(self):
        """Setup test environment."""
        if self.use_onload:
            os.environ.update({
                'EF_POLL_USEC': '0',
                'EF_INT_DRIVEN': '0',
                'EF_RX_TIMESTAMPING': '1',
                'EF_TX_TIMESTAMPING': '1'
            })
            print("OnLoad environment configured")
        else:
            print("Running baseline test without OnLoad")
    
    def run_simple_ping_test(self) -> List[float]:
        """Run simple ping-pong latency test."""
        latencies = []
        
        def echo_server(server_socket):
            """Simple echo server."""
            conn, addr = server_socket.accept()
            conn.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
            
            try:
                while True:
                    data = conn.recv(self.message_size)
                    if not data:
                        break
                    conn.send(data)
            except:
                pass
            finally:
                conn.close()
        
        # Create server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        server.bind(('127.0.0.1', 12355))
        server.listen(1)
        
        # Start server thread
        server_thread = threading.Thread(target=echo_server, args=(server,))
        server_thread.daemon = True
        server_thread.start()
        
        time.sleep(0.1)  # Allow server to start
        
        # Create client
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
        client.connect(('127.0.0.1', 12355))
        
        message = b'X' * self.message_size
        
        # Warmup
        for _ in range(100):
            client.send(message)
            client.recv(self.message_size)
        
        # Test
        iterations = 5000
        print(f"Running simple ping test: {iterations} iterations...")
        
        for i in range(iterations):
            start_time = time.time_ns()
            client.send(message)
            response = client.recv(self.message_size)
            end_time = time.time_ns()
            
            if len(response) == self.message_size:
                latency_us = (end_time - start_time) / 1000.0
                latencies.append(latency_us)
            
            if (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1}/{iterations}")
        
        client.close()
        server.close()
        
        return latencies
    
    def run_market_data_simulation(self) -> List[float]:
        """Simulate market data processing latency."""
        processing_latencies = []
        
        def market_data_processor():
            """Process simulated market data messages."""
            for i in range(1000):
                start_time = time.time_ns()
                
                # Simulate market data processing
                # Parse message (simulated)
                symbol = "AAPL"
                price = 150.0 + (i % 100) * 0.01
                quantity = 1000 + (i % 500)
                
                # Simple trading logic (simulated)
                if price > 150.5:
                    action = "SELL"
                elif price < 149.5:
                    action = "BUY"
                else:
                    action = "HOLD"
                
                # Risk check (simulated)
                risk_ok = quantity < 5000
                
                # Order generation (simulated)
                if action != "HOLD" and risk_ok:
                    order_id = f"ORDER_{i}"
                    order_side = action
                    order_price = price
                    order_qty = min(quantity, 1000)
                
                end_time = time.time_ns()
                processing_latency = (end_time - start_time) / 1000.0  # microseconds
                processing_latencies.append(processing_latency)
                
                # Small delay to simulate realistic data rate
                time.sleep(0.0001)  # 100μs between messages
        
        print("Running market data processing simulation...")
        market_data_processor()
        
        return processing_latencies
    
    def run_order_execution_simulation(self) -> List[float]:
        """Simulate order execution latency."""
        execution_latencies = []
        
        print("Running order execution simulation...")
        
        for i in range(500):
            start_time = time.time_ns()
            
            # Simulate order validation
            order_valid = True
            
            # Simulate risk checks
            risk_passed = True
            
            # Simulate market access
            if order_valid and risk_passed:
                # Simulate exchange communication latency
                time.sleep(0.000002)  # 2μs simulated exchange latency
            
            end_time = time.time_ns()
            execution_latency = (end_time - start_time) / 1000.0
            execution_latencies.append(execution_latency)
            
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/500 executions")
        
        return execution_latencies
    
    def calculate_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not latencies:
            return {}
        
        latencies_array = np.array(latencies)
        
        return {
            'mean': float(np.mean(latencies_array)),
            'median': float(np.median(latencies_array)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'std_dev': float(np.std(latencies_array)),
            'count': len(latencies)
        }
    
    def evaluate_performance(self, stats: Dict[str, float]) -> Dict[str, str]:
        """Evaluate performance against targets."""
        if not stats:
            return {}
        
        evaluation = {}
        evaluation['mean_target'] = 'PASS' if stats['mean'] <= self.target_latencies['mean'] else 'FAIL'
        evaluation['p95_target'] = 'PASS' if stats['p95'] <= self.target_latencies['p95'] else 'FAIL'
        evaluation['p99_target'] = 'PASS' if stats['p99'] <= self.target_latencies['p99'] else 'FAIL'
        evaluation['critical_check'] = 'PASS' if stats['max'] <= 50.0 else 'FAIL'  # 50μs critical threshold
        
        # Overall evaluation
        passes = sum(1 for v in evaluation.values() if v == 'PASS')
        evaluation['overall'] = 'PASS' if passes == len(evaluation) else 'FAIL'
        
        return evaluation
    
    def print_test_results(self, test_name: str, stats: Dict[str, float], evaluation: Dict[str, str]):
        """Print formatted test results."""
        print(f"\n{test_name} Results:")
        print(f"  Mean:     {stats.get('mean', 0):.3f}μs (target: ≤{self.target_latencies['mean']}μs) [{evaluation.get('mean_target', 'N/A')}]")
        print(f"  Median:   {stats.get('median', 0):.3f}μs")
        print(f"  P95:      {stats.get('p95', 0):.3f}μs (target: ≤{self.target_latencies['p95']}μs) [{evaluation.get('p95_target', 'N/A')}]")
        print(f"  P99:      {stats.get('p99', 0):.3f}μs (target: ≤{self.target_latencies['p99']}μs) [{evaluation.get('p99_target', 'N/A')}]")
        print(f"  Min/Max:  {stats.get('min', 0):.3f}μs / {stats.get('max', 0):.3f}μs")
        print(f"  Std Dev:  {stats.get('std_dev', 0):.3f}μs")
        print(f"  Samples:  {stats.get('count', 0)}")
        print(f"  Overall:  [{evaluation.get('overall', 'N/A')}]")
    
    def run_all_tests(self) -> Dict:
        """Run all simple performance tests."""
        print("=" * 50)
        print("SIMPLE REAL-WORLD TRADING PERFORMANCE TEST")
        print("=" * 50)
        
        self.setup_environment()
        
        results = {
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'onload_enabled': self.use_onload,
                'test_duration': self.test_duration,
                'message_size': self.message_size,
                'target_latencies': self.target_latencies
            },
            'test_results': {}
        }
        
        try:
            # Test 1: Simple ping-pong
            print("\n1. Network Ping-Pong Test")
            ping_latencies = self.run_simple_ping_test()
            ping_stats = self.calculate_statistics(ping_latencies)
            ping_evaluation = self.evaluate_performance(ping_stats)
            
            results['test_results']['ping_pong'] = {
                'statistics': ping_stats,
                'evaluation': ping_evaluation
            }
            
            self.print_test_results("Network Ping-Pong", ping_stats, ping_evaluation)
            
            # Test 2: Market data processing
            print("\n2. Market Data Processing Test")
            md_latencies = self.run_market_data_simulation()
            md_stats = self.calculate_statistics(md_latencies)
            md_evaluation = self.evaluate_performance(md_stats)
            
            results['test_results']['market_data_processing'] = {
                'statistics': md_stats,
                'evaluation': md_evaluation
            }
            
            self.print_test_results("Market Data Processing", md_stats, md_evaluation)
            
            # Test 3: Order execution
            print("\n3. Order Execution Test")
            exec_latencies = self.run_order_execution_simulation()
            exec_stats = self.calculate_statistics(exec_latencies)
            exec_evaluation = self.evaluate_performance(exec_stats)
            
            results['test_results']['order_execution'] = {
                'statistics': exec_stats,
                'evaluation': exec_evaluation
            }
            
            self.print_test_results("Order Execution", exec_stats, exec_evaluation)
            
        except Exception as e:
            print(f"Test error: {e}")
            results['error'] = str(e)
        
        # Overall assessment
        results['overall_assessment'] = self.generate_overall_assessment(results)
        
        return results
    
    def generate_overall_assessment(self, results: Dict) -> Dict:
        """Generate overall performance assessment."""
        assessment = {
            'onload_status': 'enabled' if self.use_onload else 'disabled',
            'tests_completed': 0,
            'tests_passed': 0,
            'performance_grade': 'UNKNOWN',
            'recommendations': []
        }
        
        # Count test results
        test_results = results.get('test_results', {})
        for test_name, test_data in test_results.items():
            assessment['tests_completed'] += 1
            evaluation = test_data.get('evaluation', {})
            if evaluation.get('overall') == 'PASS':
                assessment['tests_passed'] += 1
        
        # Calculate performance grade
        if assessment['tests_completed'] > 0:
            pass_rate = assessment['tests_passed'] / assessment['tests_completed']
            if pass_rate >= 1.0:
                assessment['performance_grade'] = 'EXCELLENT'
            elif pass_rate >= 0.67:
                assessment['performance_grade'] = 'GOOD'
            elif pass_rate >= 0.33:
                assessment['performance_grade'] = 'FAIR'
            else:
                assessment['performance_grade'] = 'POOR'
        
        # Generate recommendations
        if not self.use_onload:
            assessment['recommendations'].append("Consider enabling OnLoad for better performance")
        
        if assessment['tests_passed'] < assessment['tests_completed']:
            assessment['recommendations'].append("Review system configuration for latency optimization")
            assessment['recommendations'].append("Check CPU isolation and network settings")
        
        if assessment['performance_grade'] in ['EXCELLENT', 'GOOD']:
            assessment['recommendations'].append("System meets trading performance requirements")
        
        return assessment

def main():
    """Main execution function."""
    use_onload = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if '--no-onload' in sys.argv:
            use_onload = False
        if '--help' in sys.argv:
            print("Simple Real-World Trading Performance Test")
            print("Usage: python3 simple_real_world_test.py [--no-onload] [--help]")
            print("Options:")
            print("  --no-onload    Run without OnLoad acceleration")
            print("  --help         Show this help message")
            return 0
    
    # Run the test
    test = SimpleLatencyTest(use_onload=use_onload)
    results = test.run_all_tests()
    
    # Save results
    output_file = f"simple_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    assessment = results.get('overall_assessment', {})
    print(f"OnLoad Status: {assessment.get('onload_status', 'unknown').upper()}")
    print(f"Tests Completed: {assessment.get('tests_completed', 0)}")
    print(f"Tests Passed: {assessment.get('tests_passed', 0)}")
    print(f"Performance Grade: {assessment.get('performance_grade', 'UNKNOWN')}")
    
    recommendations = assessment.get('recommendations', [])
    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    # Detailed results summary
    test_results = results.get('test_results', {})
    if test_results:
        print(f"\nDetailed Results:")
        for test_name, test_data in test_results.items():
            stats = test_data.get('statistics', {})
            evaluation = test_data.get('evaluation', {})
            
            mean_lat = stats.get('mean', 0)
            overall_result = evaluation.get('overall', 'UNKNOWN')
            
            print(f"  {test_name.replace('_', ' ').title()}: {mean_lat:.3f}μs [{overall_result}]")
    
    # Return appropriate exit code
    performance_grade = assessment.get('performance_grade', 'UNKNOWN')
    return 0 if performance_grade in ['EXCELLENT', 'GOOD'] else 1

if __name__ == "__main__":
    exit(main())