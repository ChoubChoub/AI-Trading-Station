#!/usr/bin/env python3
"""Network latency monitoring with Onload kernel bypass support"""

import json
import time
import subprocess
import statistics
import argparse
import logging
from datetime import datetime
import os
import sys

class OnloadLatencyMonitor:
    def __init__(self, interface='enp130s0f0', target='8.8.8.8', threshold_ms=0.5):
        self.interface = interface
        self.target = target
        self.threshold_ms = threshold_ms
        
        # Setup logging FIRST
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Now check for Onload
        self.onload_available = self.check_onload()
        
    def check_onload(self):
        """Check if Onload is available and configured"""
        try:
            result = subprocess.run(
                ['which', 'onload'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                self.logger.info("✅ Onload kernel bypass detected")
                return True
        except:
            pass
        self.logger.warning("⚠️  Onload not found - results will not reflect kernel bypass performance")
        return False
    
    def run_ping_test(self, count=100, interval=0.01):
        """Run ping test with or without Onload"""
        latencies = []
        
        # Build command
        ping_cmd = [
            'ping',
            '-c', str(count),
            '-i', str(interval),
            '-q',
            self.target
        ]
        
        # Add interface binding if specified
        if self.interface:
            ping_cmd.extend(['-I', self.interface])
        
        # Wrap with Onload if available
        if self.onload_available:
            cmd = ['onload', '--profile=latency'] + ping_cmd
            self.logger.info(f"Running with Onload kernel bypass on {self.interface}")
        else:
            cmd = ping_cmd
            self.logger.warning(f"Running WITHOUT kernel bypass (results not representative of trading performance)")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=count * interval + 10
            )
            
            # Parse ping output
            if 'min/avg/max' in result.stdout:
                stats_line = [l for l in result.stdout.split('\n') if 'min/avg/max' in l][0]
                stats = stats_line.split('=')[1].strip().split('/')
                
                return {
                    'min': float(stats[0]),
                    'avg': float(stats[1]),
                    'max': float(stats[2]),
                    'kernel_bypass': self.onload_available
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error("Ping test timed out")
        except Exception as e:
            self.logger.error(f"Error running ping test: {e}")
        
        return None
    
    def run_comprehensive_test(self):
        """Run comprehensive latency test similar to your existing script"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'interface': self.interface,
            'target': self.target,
            'onload_enabled': self.onload_available,
            'tests': {}
        }
        
        # Test different packet sizes
        packet_sizes = [64, 256, 1024, 1400]
        
        for size in packet_sizes:
            self.logger.info(f"Testing packet size: {size} bytes")
            
            ping_cmd = [
                'ping',
                '-c', '100',
                '-i', '0.01',
                '-s', str(size),
                '-q',
                self.target
            ]
            
            if self.interface:
                ping_cmd.extend(['-I', self.interface])
            
            if self.onload_available:
                cmd = ['onload', '--profile=latency', '--force-profiles'] + ping_cmd
            else:
                cmd = ping_cmd
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                
                if 'min/avg/max' in result.stdout:
                    stats_line = [l for l in result.stdout.split('\n') if 'min/avg/max' in l][0]
                    stats = stats_line.split('=')[1].strip().split('/')
                    
                    results['tests'][f'packet_{size}'] = {
                        'min_ms': float(stats[0]),
                        'avg_ms': float(stats[1]),
                        'max_ms': float(stats[2])
                    }
                    
                    # Check against threshold
                    if float(stats[1]) > self.threshold_ms:
                        self.logger.warning(f"High latency detected: {stats[1]}ms (threshold: {self.threshold_ms}ms)")
                    
            except Exception as e:
                self.logger.error(f"Error testing packet size {size}: {e}")
        
        return results
    
    def export_prometheus_metrics(self, results):
        """Export metrics in Prometheus format"""
        metrics = []
        
        # Add metadata
        metrics.append(f'# HELP network_latency_ms Network latency in milliseconds')
        metrics.append(f'# TYPE network_latency_ms gauge')
        
        for test_name, values in results.get('tests', {}).items():
            for metric_type in ['min', 'avg', 'max']:
                value = values.get(f'{metric_type}_ms', 0)
                metrics.append(
                    f'network_latency_ms{{interface="{self.interface}",target="{self.target}",'
                    f'test="{test_name}",type="{metric_type}",kernel_bypass="{self.onload_available}"}} {value}'
                )
        
        return '\n'.join(metrics)

def main():
    parser = argparse.ArgumentParser(description='Network latency monitoring with Onload support')
    parser.add_argument('--interface', default='enp130s0f0', help='Network interface')
    parser.add_argument('--target', default='8.8.8.8', help='Target IP for latency test')
    parser.add_argument('--threshold', type=float, default=0.5, help='Latency threshold in ms')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test')
    parser.add_argument('--prometheus', action='store_true', help='Output in Prometheus format')
    
    args = parser.parse_args()
    
    monitor = OnloadLatencyMonitor(
        interface=args.interface,
        target=args.target,
        threshold_ms=args.threshold
    )
    
    if args.comprehensive:
        results = monitor.run_comprehensive_test()
        
        if args.prometheus:
            print(monitor.export_prometheus_metrics(results))
        else:
            print(json.dumps(results, indent=2))
    else:
        results = monitor.run_ping_test()
        if results:
            print(f"\nLatency Results ({'WITH' if results['kernel_bypass'] else 'WITHOUT'} kernel bypass):")
            print(f"  Min: {results['min']:.3f} ms")
            print(f"  Avg: {results['avg']:.3f} ms")
            print(f"  Max: {results['max']:.3f} ms")

if __name__ == '__main__':
    main()
