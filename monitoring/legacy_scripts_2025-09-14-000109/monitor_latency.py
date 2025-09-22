#!/usr/bin/env python3
"""
Network latency monitoring for AI Trading Station
Measures and tracks network latency for trading interfaces
"""
import time
import json
import subprocess
import statistics
from datetime import datetime
from collections import deque
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class LatencyMonitor:
    def __init__(self, interface, target_host="8.8.8.8", history_size=1000):
        self.interface = interface
        self.target_host = target_host
        self.history = deque(maxlen=history_size)
        self.stats = {}
    def measure_latency(self):
        """Measure network latency using ping"""
        try:
            cmd = [
                'ping', '-c', '1', '-W', '1', '-q',
                '-I', self.interface,
                self.target_host
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse ping output
                for line in result.stdout.split('\n'):
                    if 'min/avg/max' in line:
                        # Extract RTT
                        parts = line.split('=')[1].strip().split('/')
                        return float(parts[1])  # avg RTT
            return None
        except Exception as e:
            logger.error(f"Error measuring latency: {e}")
            return None
    def update_stats(self):
        """Calculate latency statistics"""
        if len(self.history) > 0:
            latencies = list(self.history)
            self.stats = {
                'min': min(latencies),
                'max': max(latencies),
                'avg': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'stddev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'p99': sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 0 else 0
            }
    def monitor(self, duration=60, interval=0.1):
        """Monitor latency for specified duration"""
        start_time = time.time()
        while time.time() - start_time < duration:
            latency = self.measure_latency()
            if latency is not None:
                self.history.append(latency)
                self.update_stats()
                # Alert on high latency
                if latency > 1.0:  # 1ms threshold
                    logger.warning(f"High latency detected: {latency:.3f}ms")
            time.sleep(interval)
        return self.stats
    def report(self):
        """Generate latency report"""
        if not self.stats:
            return "No latency data collected"
        report = f"""
=== Latency Report for {self.interface} ===
Target: {self.target_host}
Samples: {len(self.history)}
Statistics:
  Min:    {self.stats['min']:.3f} ms
  Max:    {self.stats['max']:.3f} ms
  Avg:    {self.stats['avg']:.3f} ms
  Median: {self.stats['median']:.3f} ms
  StdDev: {self.stats['stddev']:.3f} ms
  P99:    {self.stats['p99']:.3f} ms
"""
        return report
def main():
    parser = argparse.ArgumentParser(description='Network Latency Monitor')
    parser.add_argument('--interface', default='enp130s0f0', 
                       help='Network interface to monitor')
    parser.add_argument('--target', default='8.8.8.8',
                       help='Target host for latency measurement')
    parser.add_argument('--duration', type=int, default=60,
                       help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=float, default=0.1,
                       help='Measurement interval in seconds')
    args = parser.parse_args()
    monitor = LatencyMonitor(args.interface, args.target)
    print(f"Monitoring latency on {args.interface} to {args.target}")
    print(f"Duration: {args.duration}s, Interval: {args.interval}s")
    print("Press Ctrl+C to stop early\n")
    try:
        monitor.monitor(args.duration, args.interval)
        print(monitor.report())
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"latency_report_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'interface': args.interface,
                'target': args.target,
                'stats': monitor.stats,
                'history': list(monitor.history)
            }, f, indent=2)
        print(f"\nResults saved to: {filename}")
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        print(monitor.report())
if __name__ == '__main__':
    main()
