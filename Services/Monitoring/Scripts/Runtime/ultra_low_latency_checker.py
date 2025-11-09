#!/usr/bin/env python3
"""
Ultra-Low Latency Network Checker for Redis Performance Gate
Integrates network configuration validation with existing gate framework
"""

import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

class UltraLowLatencyChecker:
    """Network ultra-low latency configuration checker"""
    
    def __init__(self):
        self.interface = os.environ.get('ULL_TRADING_INTERFACE', 'enp130s0f0')
        self.load_thresholds()
    
    def load_thresholds(self):
        """Load ULL thresholds from environment"""
        self.thresholds = {
            'adaptive_rx_expected': os.environ.get('ULL_ADAPTIVE_RX_EXPECTED', 'off'),
            'xps_expected_mask': os.environ.get('ULL_XPS_EXPECTED_MASK', '0c'),
            'ring_rx_expected': int(os.environ.get('ULL_RING_RX_EXPECTED', 2048)),
            'ring_tx_expected': int(os.environ.get('ULL_RING_TX_EXPECTED', 2048)),
            'rps_expected': os.environ.get('ULL_RPS_EXPECTED', '00'),
            'cpu_usage_threshold': int(os.environ.get('ULL_CPU_USAGE_THRESHOLD', 10)),
            'required_services': os.environ.get('ULL_REQUIRED_SERVICES', '').split(),
        }
        
        self.criticality = {
            'adaptive_rx': os.environ.get('ULL_ADAPTIVE_RX_CRITICAL', 'true').lower() == 'true',
            'xps': os.environ.get('ULL_XPS_CRITICAL', 'true').lower() == 'true',
            'irq_violations': os.environ.get('ULL_IRQ_VIOLATIONS_CRITICAL', 'true').lower() == 'true',
            'service_status': os.environ.get('ULL_SERVICE_STATUS_CRITICAL', 'true').lower() == 'true',
            'ring_buffers': os.environ.get('ULL_RING_BUFFER_CRITICAL', 'false').lower() == 'true',
            'rps': os.environ.get('ULL_RPS_CRITICAL', 'false').lower() == 'true',
        }
    
    def run_command(self, cmd: List[str]) -> Tuple[bool, str]:
        """Run command and return (success, output)"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            return False, ""
    
    def check_service_status(self) -> Dict:
        """Check systemd service status"""
        results = {}
        failures = []
        warnings = []
        
        for service in self.thresholds['required_services']:
            success, output = self.run_command(['systemctl', 'is-active', service])
            is_active = success and output == 'active'
            
            results[service] = {
                'active': is_active,
                'status': output if success else 'unknown'
            }
            
            if not is_active:
                msg = f"Service {service} not active ({output})"
                if self.criticality['service_status']:
                    failures.append(msg)
                else:
                    warnings.append(msg)
        
        return {
            'services': results,
            'failures': failures,
            'warnings': warnings
        }
    
    def check_adaptive_coalescing(self) -> Dict:
        """Check adaptive RX coalescing configuration"""
        success, output = self.run_command(['ethtool', '-c', self.interface])
        
        if not success:
            return {
                'status': 'error',
                'value': 'unknown',
                'failures': [f"Failed to read coalescing config: ethtool error"],
                'warnings': []
            }
        
        # Parse adaptive RX setting
        for line in output.split('\n'):
            if 'Adaptive RX:' in line:
                current_value = line.split(':')[1].strip().split()[0]
                expected = self.thresholds['adaptive_rx_expected']
                
                if current_value == expected:
                    return {
                        'status': 'ok',
                        'value': current_value,
                        'failures': [],
                        'warnings': []
                    }
                else:
                    msg = f"Adaptive RX coalescing: {current_value} (expected: {expected})"
                    result = {
                        'status': 'fail',
                        'value': current_value,
                        'warnings': []
                    }
                    
                    if self.criticality['adaptive_rx']:
                        result['failures'] = [msg]
                    else:
                        result['failures'] = []
                        result['warnings'] = [msg]
                    
                    return result
        
        return {
            'status': 'error',
            'value': 'not_found',
            'failures': ['Adaptive RX setting not found in ethtool output'],
            'warnings': []
        }
    
    def check_xps_configuration(self) -> Dict:
        """Check XPS CPU steering configuration"""
        failures = []
        warnings = []
        queue_results = {}
        expected_mask = self.thresholds['xps_expected_mask']
        
        # Check all TX queues
        tx_queues = []
        try:
            queue_dirs = os.listdir(f'/sys/class/net/{self.interface}/queues')
            tx_queues = [q for q in queue_dirs if q.startswith('tx-')]
        except OSError:
            return {
                'status': 'error',
                'queues': {},
                'failures': [f"Cannot access queue directories for {self.interface}"],
                'warnings': []
            }
        
        all_correct = True
        for queue in sorted(tx_queues):
            xps_file = f'/sys/class/net/{self.interface}/queues/{queue}/xps_cpus'
            try:
                with open(xps_file, 'r') as f:
                    current_mask = f.read().strip()
                
                # Normalize masks (remove leading zeros for comparison)
                current_normalized = current_mask.lstrip('0') or '0'
                expected_normalized = expected_mask.lstrip('0') or '0'
                
                is_correct = current_normalized == expected_normalized
                queue_results[queue] = {
                    'current': current_mask,
                    'expected': expected_mask,
                    'correct': is_correct
                }
                
                if not is_correct:
                    all_correct = False
                    msg = f"XPS {queue}: {current_mask} (expected: {expected_mask})"
                    
                    if self.criticality['xps']:
                        failures.append(msg)
                    else:
                        warnings.append(msg)
                        
            except IOError:
                all_correct = False
                msg = f"Cannot read XPS config for {queue}"
                failures.append(msg)
        
        return {
            'status': 'ok' if all_correct else 'fail',
            'queues': queue_results,
            'failures': failures,
            'warnings': warnings
        }
    
    def check_ring_buffers(self) -> Dict:
        """Check ring buffer configuration"""
        success, output = self.run_command(['ethtool', '-g', self.interface])
        
        if not success:
            return {
                'status': 'error',
                'rx': 'unknown',
                'tx': 'unknown',
                'failures': ['Failed to read ring buffer config'],
                'warnings': []
            }
        
        # Parse ring buffer settings
        current_rx = None
        current_tx = None
        in_current_section = False
        
        for line in output.split('\n'):
            if 'Current hardware' in line:
                in_current_section = True
                continue
            
            if in_current_section and line.strip():
                if line.startswith('RX:'):
                    current_rx = int(line.split(':')[1].strip())
                elif line.startswith('TX:'):
                    current_tx = int(line.split(':')[1].strip())
                    break
        
        failures = []
        warnings = []
        
        expected_rx = self.thresholds['ring_rx_expected']
        expected_tx = self.thresholds['ring_tx_expected']
        
        status = 'ok'
        if current_rx != expected_rx:
            msg = f"RX ring buffer: {current_rx} (expected: {expected_rx})"
            status = 'fail'
            if self.criticality['ring_buffers']:
                failures.append(msg)
            else:
                warnings.append(msg)
        
        if current_tx != expected_tx:
            msg = f"TX ring buffer: {current_tx} (expected: {expected_tx})"
            status = 'fail'
            if self.criticality['ring_buffers']:
                failures.append(msg)
            else:
                warnings.append(msg)
        
        return {
            'status': status,
            'rx': current_rx,
            'tx': current_tx,
            'failures': failures,
            'warnings': warnings
        }
    
    def check_rps_configuration(self) -> Dict:
        """Check RPS configuration"""
        rps_file = f'/sys/class/net/{self.interface}/queues/rx-0/rps_cpus'
        
        try:
            with open(rps_file, 'r') as f:
                current_rps = f.read().strip()
                
            expected = self.thresholds['rps_expected']
            
            if current_rps == expected:
                return {
                    'status': 'ok',
                    'value': current_rps,
                    'failures': [],
                    'warnings': []
                }
            else:
                msg = f"RPS configuration: {current_rps} (expected: {expected})"
                result = {
                    'status': 'fail',
                    'value': current_rps,
                    'warnings': []
                }
                
                if self.criticality['rps']:
                    result['failures'] = [msg]
                else:
                    result['failures'] = []
                    result['warnings'] = [msg]
                
                return result
                
        except IOError:
            return {
                'status': 'error',
                'value': 'unknown',
                'failures': ['Cannot read RPS configuration'],
                'warnings': []
            }
    
    def check_irq_violations(self) -> Dict:
        """Check for IRQ violations on trading cores"""
        failures = []
        warnings = []
        violations = []
        
        try:
            with open('/proc/interrupts', 'r') as f:
                lines = f.readlines()
            
            trading_cores = [2, 3]  # Cores 2,3 are isolated for trading
            # Allow small historical counts (startup/configuration activity)
            # Only flag excessive interrupt activity that suggests ongoing violations
            violation_threshold = int(os.environ.get('ULL_IRQ_VIOLATION_THRESHOLD', '10'))
            
            for line in lines:
                if self.interface in line:
                    parts = line.split()
                    if len(parts) > max(trading_cores):
                        irq_num = parts[0].rstrip(':')
                        
                        for core in trading_cores:
                            count = int(parts[1 + core])  # +1 because IRQ number is first
                            if count > violation_threshold:
                                violations.append({
                                    'irq': irq_num,
                                    'core': core,
                                    'count': count
                                })
            
            if violations:
                msg = f"IRQ violations on trading cores: {len(violations)} violations"
                if self.criticality['irq_violations']:
                    failures.append(msg)
                else:
                    warnings.append(msg)
                
                return {
                    'status': 'fail',
                    'violations': violations,
                    'failures': failures,
                    'warnings': warnings
                }
            else:
                return {
                    'status': 'ok',
                    'violations': [],
                    'failures': [],
                    'warnings': []
                }
                
        except IOError:
            return {
                'status': 'error',
                'violations': [],
                'failures': ['Cannot read /proc/interrupts'],
                'warnings': []
            }
    
    def check_all(self) -> Dict:
        """Run all ultra-low latency checks"""
        results = {
            'services': self.check_service_status(),
            'adaptive_coalescing': self.check_adaptive_coalescing(),
            'xps_configuration': self.check_xps_configuration(),
            'ring_buffers': self.check_ring_buffers(),
            'rps_configuration': self.check_rps_configuration(),
            'irq_violations': self.check_irq_violations(),
        }
        
        # Aggregate failures and warnings
        all_failures = []
        all_warnings = []
        
        for check_name, check_result in results.items():
            all_failures.extend(check_result.get('failures', []))
            all_warnings.extend(check_result.get('warnings', []))
        
        # Determine overall status
        if all_failures:
            overall_status = 'ULTRA_LOW_LATENCY_HARD_FAIL'
        elif all_warnings:
            overall_status = 'ULTRA_LOW_LATENCY_SOFT_FAIL'
        else:
            overall_status = 'ULTRA_LOW_LATENCY_PASS'
        
        return {
            'overall_status': overall_status,
            'interface': self.interface,
            'checks': results,
            'summary': {
                'failures': all_failures,
                'warnings': all_warnings,
                'total_checks': len(results)
            }
        }

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Low Latency Network Checker')
    parser.add_argument('--output', help='Output JSON file')
    parser.add_argument('--interface', help='Network interface to check')
    
    args = parser.parse_args()
    
    if args.interface:
        os.environ['ULL_TRADING_INTERFACE'] = args.interface
    
    checker = UltraLowLatencyChecker()
    results = checker.check_all()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    if results['overall_status'] == 'ULTRA_LOW_LATENCY_HARD_FAIL':
        sys.exit(2)
    elif results['overall_status'] == 'ULTRA_LOW_LATENCY_SOFT_FAIL':
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()