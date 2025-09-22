#!/usr/bin/env python3
"""
Prometheus Exporter for AI Trading Station
Exports metrics including persistent connection latency
"""

from prometheus_client import start_http_server, Gauge
import time
import json
from monitor_trading_system_v2 import TradingSystemMonitor

# Define Prometheus metrics
latency_mean = Gauge('trading_latency_mean_microseconds', 'Trading latency mean (persistent connection)')
latency_p99 = Gauge('trading_latency_p99_microseconds', 'Trading latency P99 (persistent connection)')
latency_min = Gauge('trading_latency_min_microseconds', 'Trading latency minimum')
latency_max = Gauge('trading_latency_max_microseconds', 'Trading latency maximum')

cpu_isolation_violations = Gauge('trading_cpu_isolation_violations', 'Number of CPU isolation violations')
irq_violations = Gauge('trading_irq_violations', 'Number of IRQ affinity violations')

network_drops = Gauge('trading_network_drops_total', 'Total network packet drops', ['interface'])
gpu_temperature = Gauge('trading_gpu_temperature_celsius', 'GPU temperature', ['gpu_id'])
gpu_utilization = Gauge('trading_gpu_utilization_percent', 'GPU utilization', ['gpu_id'])

def update_metrics(monitor):
    """Update Prometheus metrics from monitor"""
    metrics = monitor.collect_all_metrics()
    
    # Update latency metrics (most important!)
    if metrics['latency']:
        latency_mean.set(metrics['latency']['mean'])
        latency_p99.set(metrics['latency']['p99'])
        latency_min.set(metrics['latency']['min'])
        latency_max.set(metrics['latency']['max'])
    
    # CPU isolation
    cpu_isolation_violations.set(len(metrics['cpu_isolation']['violations']))
    
    # IRQ violations
    irq_violations.set(len(metrics['irq_affinity']['violations']))
    
    # Network drops
    for iface, stats in metrics['network'].items():
        if 'dropin' in stats:
            network_drops.labels(interface=iface).set(stats['dropin'] + stats['dropout'])
    
    # GPU metrics
    for i, gpu in enumerate(metrics['gpu']):
        gpu_temperature.labels(gpu_id=str(i)).set(gpu['temperature'])
        gpu_utilization.labels(gpu_id=str(i)).set(gpu['utilization'])

def main():
    # Start Prometheus HTTP server
    start_http_server(9090)
    print("🚀 Prometheus exporter started on port 9090")
    print("   Metrics endpoint: http://localhost:9090/metrics")
    
    # Initialize monitor
    monitor = TradingSystemMonitor()
    
    # Update metrics every 10 seconds
    while True:
        try:
            update_metrics(monitor)
            print(f"Updated metrics - Latency: {monitor.metrics.get('latency', {}).get('mean', 'N/A'):.2f}μs")
            time.sleep(10)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error updating metrics: {e}")
            time.sleep(10)

if __name__ == '__main__':
    main()
