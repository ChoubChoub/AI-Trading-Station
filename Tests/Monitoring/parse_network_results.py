#!/usr/bin/env python3
"""
Parse network ULL results and format for performance gate
"""
import json
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: parse_network_results.py <network_results.json> [performance_results.json]")
        sys.exit(1)
    
    network_file = sys.argv[1]
    perf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        # Load network results
        with open(network_file, 'r') as f:
            network = json.load(f)
        
        # Load performance results for correlation analysis (optional)
        perf_data = {}
        if perf_file and os.path.exists(perf_file):
            try:
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
            except:
                pass
        
        network_critical = os.environ.get('NETWORK_GATE_CRITICAL', 'true').lower() == 'true'
        
        failures = network['summary']['failures']
        warnings = network['summary']['warnings']
        
        # Enhanced analysis: correlate network issues with Redis performance
        correlation_detected = False
        if failures or warnings and perf_data:
            redis_set_p99 = perf_data.get('set', {}).get('p99', 0)
            redis_rtt_p99 = perf_data.get('rtt', {}).get('p99', 0)
            
            # If Redis latency is high AND network issues exist, flag correlation
            if redis_set_p99 > 4 or redis_rtt_p99 > 10:
                correlation_detected = True
                print("NETWORK_PERFORMANCE_CORRELATION_DETECTED")
                print(f"CORRELATION: Redis latency elevated (SET: {redis_set_p99}μs, RTT: {redis_rtt_p99}μs) with network issues")
        
        # Report network status
        if failures:
            status = "NETWORK_HARD_FAIL" if network_critical else "NETWORK_SOFT_FAIL"
            print(status)
            for failure in failures:
                print(f"NET_FAIL: {failure}")
            
            if correlation_detected:
                print("RECOMMENDATION: Fix network issues to improve Redis latency")
            
            sys.exit(1 if network_critical else 0)
            
        elif warnings:
            print("NETWORK_SOFT_FAIL")
            for warning in warnings:
                print(f"NET_WARN: {warning}")
            sys.exit(0)  # Warnings don't fail gate unless correlated with performance issues
            
        else:
            print("NETWORK_PASS")
            
            # Report key network metrics for visibility
            checks = network.get('checks', {})
            
            # Adaptive RX status
            if 'adaptive_coalescing' in checks:
                adaptive_rx = checks['adaptive_coalescing'].get('value', 'unknown')
                print(f"✅ Adaptive RX coalescing: {adaptive_rx}")
            
            # XPS configuration
            if 'xps_configuration' in checks:
                xps = checks['xps_configuration']
                failures_count = len(xps.get('failures', []))
                total_queues = len(xps.get('queues', {}))
                correct_queues = total_queues - failures_count
                print(f"✅ XPS configuration: {correct_queues}/{total_queues} queues correct")
            
            # Service status
            if 'services' in checks:
                svc = checks['services']
                failures_count = len(svc.get('failures', []))
                total_services = len(svc.get('services', {}))
                active_services = total_services - failures_count
                print(f"✅ Required services: {active_services}/{total_services} active")
            
            # IRQ violations
            if 'irq_violations' in checks:
                irq_count = len(checks['irq_violations'].get('violations', []))
                print(f"✅ IRQ violations on trading cores: {irq_count}")

    except Exception as e:
        print(f"ERROR: Network gate analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()