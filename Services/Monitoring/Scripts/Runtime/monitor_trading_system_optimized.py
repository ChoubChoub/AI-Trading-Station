#!/usr/bin/env python3
"""
AI Trading Station - Optimized Monitor with 4.5μs Latency
Using exact same method as your baseline test
"""

import json
import os
import sys
import time
import socket
import threading
import statistics
import subprocess
import psutil
import logging
from pathlib import Path
from datetime import datetime
from collections import deque

# Add QuestDB scripts to path
sys.path.insert(0, '/home/youssefbahloul/ai-trading-station/Services/QuestDB/Config')
from monitor_questdb import QuestDBMonitor

class TradingSystemMonitor:
    def __init__(self, config_path="/home/youssefbahloul/ai-trading-station/Services/Monitoring/Config/monitor_config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.alerts = deque(maxlen=100)
        self.metrics = {}
        self.server_thread = None
        self.server_running = False
        self.server_port = 12345
        self._collection_lock = threading.Lock()  # Prevent concurrent collections
        
        # Test counters for monitoring activity
        self.latency_test_count = 0
        self.redis_check_count = 0
        
        # Drop rate tracking (for monitoring active drops vs historical)
        self.previous_drops = {}  # {interface: {'drops': count, 'timestamp': time}}
        self.drop_rates = {}      # {interface: drops_per_second}
        
        # Traffic rate tracking (packets/sec and bandwidth)
        self.previous_traffic = {}  # {interface: {'tx_packets': count, 'rx_packets': count, 'tx_bytes': bytes, 'rx_bytes': bytes, 'timestamp': time}}
        self.traffic_rates = {}     # {interface: {'tx_pps': int, 'rx_pps': int, 'mbps': float}}
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['monitoring']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def measure_trading_latency(self):
        """Measure latency EXACTLY like your baseline test that gets 4.5μs"""
        port = self.server_port
        
        # Start fresh server for each measurement (like your baseline)
        self.server_running = True
        server_thread = threading.Thread(
            target=self._latency_server, 
            args=(port,), 
            daemon=True
        )
        server_thread.start()
        time.sleep(0.5)  # Let server start
        
        try:
            # EXACT COPY of your working latency_test_client
            latencies = []
            
            # Single persistent connection - THIS IS KEY!
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Critical: TCP_NODELAY for low latency
            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_sock.connect(('127.0.0.1', port))
            
            message = b"TRADING_LATENCY_TEST"
            
            # Warmup - critical for consistent low latency
            for _ in range(100):
                client_sock.send(message)
                client_sock.recv(1024)
            
            # Actual measurements
            for i in range(1000):
                start = time.perf_counter_ns()
                client_sock.send(message)
                response = client_sock.recv(1024)
                end = time.perf_counter_ns()
                
                latency_us = (end - start) / 1000
                latencies.append(latency_us)
            
            client_sock.close()
            self.server_running = False
            
            # Increment test counter
            self.latency_test_count += 1
            
            # Calculate statistics
            latencies.sort()
            mean_val = statistics.mean(latencies)
            p99_val = latencies[int(0.99 * len(latencies))]
            p95_val = latencies[int(0.95 * len(latencies))]
            p50_val = statistics.median(latencies)
            
            # Calculate jitter (standard deviation for consistency measure)
            stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0
            
            return {
                'mean': mean_val,
                'median': p50_val,
                'p95': p95_val,
                'p99': p99_val,
                'min': min(latencies),
                'max': max(latencies),
                'jitter': stdev,  # Standard deviation as jitter measure
                'test_count': self.latency_test_count
            }
        except Exception as e:
            self.logger.error(f"Latency test failed: {e}")
            self.server_running = False
            return None
    
    def _latency_server(self, port):
        """TCP echo server - EXACT COPY of your baseline"""
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Critical: TCP_NODELAY on server too
            server_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            server_sock.bind(('127.0.0.1', port))
            server_sock.listen(1)
            
            while self.server_running:
                try:
                    conn, addr = server_sock.accept()
                    # TCP_NODELAY on accepted connection
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while self.server_running:
                        data = conn.recv(1024)
                        if not data:
                            break
                        conn.send(data)  # Echo back immediately
                    conn.close()
                except:
                    break
        except:
            pass
        finally:
            try:
                server_sock.close()
            except:
                pass
    
    def measure_network_rtt_lan(self):
        """Measure network RTT to LAN gateway (ping test)"""
        try:
            # Get default gateway IP
            result = subprocess.run(['ip', 'route'], capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                return {'avg_us': None, 'gateway_ip': None, 'status': 'ERROR', 'error': 'Cannot get gateway'}
            
            gateway_ip = None
            for line in result.stdout.split('\n'):
                if line.startswith('default via'):
                    gateway_ip = line.split()[2]
                    break
            
            if not gateway_ip:
                return {'avg_us': None, 'gateway_ip': None, 'status': 'ERROR', 'error': 'No default gateway'}
            
            # Ping gateway with 20 packets, 50ms interval for accuracy
            ping_result = subprocess.run(
                ['ping', '-c', '20', '-i', '0.05', gateway_ip],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if ping_result.returncode != 0:
                return {'avg_us': None, 'gateway_ip': gateway_ip, 'status': 'ERROR', 'error': 'Ping failed'}
            
            # Parse RTT: "rtt min/avg/max/mdev = 0.123/0.145/0.189/0.021 ms"
            for line in ping_result.stdout.split('\n'):
                if 'rtt min/avg/max/mdev' in line:
                    # Extract values: min/avg/max/mdev
                    rtt_values = line.split('=')[1].strip().split()[0]  # "0.123/0.145/0.189/0.021"
                    parts = rtt_values.split('/')
                    avg_ms = float(parts[1])
                    mdev_ms = float(parts[3])  # mdev = jitter
                    avg_us = avg_ms * 1000  # Convert to microseconds
                    jitter_us = mdev_ms * 1000
                    return {
                        'avg_us': round(avg_us, 2), 
                        'jitter_us': round(jitter_us, 2),
                        'gateway_ip': gateway_ip, 
                        'status': 'OK'
                    }
            
            return {'avg_us': None, 'jitter_us': None, 'gateway_ip': gateway_ip, 'status': 'ERROR', 'error': 'Cannot parse ping output'}
            
        except subprocess.TimeoutExpired:
            return {'avg_us': None, 'gateway_ip': None, 'status': 'ERROR', 'error': 'Ping timeout'}
        except Exception as e:
            return {'avg_us': None, 'gateway_ip': None, 'status': 'ERROR', 'error': str(e)}
    
    def measure_network_rtt_internet(self):
        """Measure network RTT to Internet (Google DNS 8.8.8.8)"""
        try:
            # Ping 8.8.8.8 with 10 packets, 100ms interval
            ping_result = subprocess.run(
                ['ping', '-c', '10', '-i', '0.1', '8.8.8.8'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if ping_result.returncode != 0:
                return {'avg_ms': None, 'status': 'ERROR', 'error': 'Ping failed'}
            
            # Parse RTT: "rtt min/avg/max/mdev = 10.234/12.456/15.789/1.234 ms"
            for line in ping_result.stdout.split('\n'):
                if 'rtt min/avg/max/mdev' in line:
                    # Extract values: min/avg/max/mdev
                    rtt_values = line.split('=')[1].strip().split()[0]  # "10.234/12.456/15.789/1.234"
                    parts = rtt_values.split('/')
                    avg_ms = float(parts[1])
                    mdev_ms = float(parts[3])  # mdev = jitter
                    return {
                        'avg_ms': round(avg_ms, 2), 
                        'jitter_ms': round(mdev_ms, 2),
                        'status': 'OK'
                    }
            
            return {'avg_ms': None, 'jitter_ms': None, 'status': 'ERROR', 'error': 'Cannot parse ping output'}
            
        except subprocess.TimeoutExpired:
            return {'avg_ms': None, 'status': 'ERROR', 'error': 'Ping timeout'}
        except Exception as e:
            return {'avg_ms': None, 'status': 'ERROR', 'error': str(e)}
    
    def calculate_performance_grade(self, onload_mean_us, onload_p99_us):
        """Calculate performance grade based on latency metrics"""
        try:
            # Use P99 as primary metric (worst-case performance matters in HFT)
            latency = onload_p99_us
            
            if latency is None:
                return {'grade': 'UNKNOWN', 'emoji': '❓', 'reason': 'No data'}
            
            # Grading thresholds based on HFT industry standards
            if latency < 5.0:
                return {'grade': 'WORLD-CLASS', 'emoji': '🏆', 'reason': f'P99 {latency}μs < 5μs'}
            elif latency < 10.0:
                return {'grade': 'EXCELLENT', 'emoji': '✅', 'reason': f'P99 {latency}μs < 10μs'}
            elif latency < 20.0:
                return {'grade': 'GOOD', 'emoji': '👍', 'reason': f'P99 {latency}μs < 20μs'}
            elif latency < 50.0:
                return {'grade': 'ACCEPTABLE', 'emoji': '⚠️', 'reason': f'P99 {latency}μs < 50μs'}
            else:
                return {'grade': 'NEEDS IMPROVEMENT', 'emoji': '❌', 'reason': f'P99 {latency}μs >= 50μs'}
                
        except Exception as e:
            return {'grade': 'ERROR', 'emoji': '❌', 'reason': str(e)}
    
    def check_cpu_isolation(self):
        """Check CPU isolation status"""
        isolated_cpus = self.config['system']['isolated_cpus']
        violations = []
        
        for cpu in isolated_cpus:
            usage = psutil.cpu_percent(interval=0.1, percpu=True)[cpu]
            if usage > self.config['thresholds']['cpu_isolated_usage_percent']:
                violations.append(f"CPU{cpu}: {usage:.1f}%")
        
        return {
            'isolated_cpus': isolated_cpus,
            'violations': violations,
            'status': 'OK' if not violations else 'VIOLATION'
        }
    
    def check_irq_affinity(self):
        """Check IRQ affinity for isolated CPUs"""
        isolated_cpus = set(self.config['system']['isolated_cpus'])
        violations = []
        
        try:
            with open('/proc/interrupts', 'r') as f:
                lines = f.readlines()
            
            for line in lines[1:]:
                parts = line.split()
                if len(parts) > 4 and ':' in parts[0]:
                    irq_num = parts[0].rstrip(':')
                    
                    for iface in self.config['system']['network_interfaces']:
                        if iface in line:
                            try:
                                with open(f'/proc/irq/{irq_num}/smp_affinity_list', 'r') as f:
                                    affinity = f.read().strip()
                                    assigned_cpus = set()
                                    for cpu_range in affinity.split(','):
                                        if '-' in cpu_range:
                                            start, end = map(int, cpu_range.split('-'))
                                            assigned_cpus.update(range(start, end + 1))
                                        else:
                                            assigned_cpus.add(int(cpu_range))
                                    
                                    if assigned_cpus & isolated_cpus:
                                        violations.append(f"IRQ {irq_num} ({iface})")
                            except:
                                pass
        except Exception as e:
            self.logger.error(f"IRQ check failed: {e}")
        
        return {
            'violations': violations,
            'status': 'OK' if not violations else 'VIOLATION'
        }
    
    def check_network_interfaces(self):
        """Check network interface statistics with drop rate, traffic rate, and bandwidth calculation"""
        stats = {}
        current_time = time.time()
        
        for iface in self.config['system']['network_interfaces']:
            try:
                net_stats = psutil.net_io_counters(pernic=True).get(iface)
                if net_stats:
                    total_drops = net_stats.dropin + net_stats.dropout
                    
                    # Calculate drop rate (drops per second)
                    drop_rate = 0.0
                    if iface in self.previous_drops:
                        time_delta = current_time - self.previous_drops[iface]['timestamp']
                        if time_delta > 0:
                            drop_delta = total_drops - self.previous_drops[iface]['drops']
                            drop_rate = drop_delta / time_delta
                            self.drop_rates[iface] = drop_rate
                    
                    # Store current values for next comparison
                    self.previous_drops[iface] = {
                        'drops': total_drops,
                        'timestamp': current_time
                    }
                    
                    # Calculate traffic rate (packets/sec) and bandwidth (Mbps)
                    tx_pps = 0
                    rx_pps = 0
                    mbps = 0.0
                    link_speed_gbps = 10  # Default, will try to read actual
                    utilization_pct = 0.0
                    
                    if iface in self.previous_traffic:
                        time_delta = current_time - self.previous_traffic[iface]['timestamp']
                        if time_delta > 0:
                            # Packets per second
                            tx_delta = net_stats.packets_sent - self.previous_traffic[iface]['tx_packets']
                            rx_delta = net_stats.packets_recv - self.previous_traffic[iface]['rx_packets']
                            tx_pps = int(tx_delta / time_delta)
                            rx_pps = int(rx_delta / time_delta)
                            
                            # Bandwidth in Mbps
                            tx_bytes_delta = net_stats.bytes_sent - self.previous_traffic[iface]['tx_bytes']
                            rx_bytes_delta = net_stats.bytes_recv - self.previous_traffic[iface]['rx_bytes']
                            total_bytes_delta = tx_bytes_delta + rx_bytes_delta
                            mbps = (total_bytes_delta * 8) / (time_delta * 1_000_000)
                            
                            # Get link speed and calculate utilization
                            try:
                                with open(f'/sys/class/net/{iface}/speed', 'r') as f:
                                    link_speed_mbps = int(f.read().strip())
                                    link_speed_gbps = link_speed_mbps / 1000
                                    if link_speed_mbps > 0:
                                        utilization_pct = (mbps / link_speed_mbps) * 100
                            except:
                                pass  # Use default if cannot read
                            
                            self.traffic_rates[iface] = {
                                'tx_pps': tx_pps,
                                'rx_pps': rx_pps,
                                'mbps': mbps,
                                'link_speed_gbps': link_speed_gbps,
                                'utilization_pct': utilization_pct
                            }
                    
                    # Store current values for next comparison
                    self.previous_traffic[iface] = {
                        'tx_packets': net_stats.packets_sent,
                        'rx_packets': net_stats.packets_recv,
                        'tx_bytes': net_stats.bytes_sent,
                        'rx_bytes': net_stats.bytes_recv,
                        'timestamp': current_time
                    }
                    
                    # HFT-grade tiered status based on DROP RATE
                    # OK: < 0.5/s (multicast filtering normal)
                    # WARNING: 0.5-1.0/s (elevated, monitor)
                    # CRITICAL: >= 10/s (immediate action required)
                    thresholds = self.config['thresholds']
                    if drop_rate >= thresholds['network_drop_rate_critical']:
                        status = 'CRITICAL'
                    elif drop_rate >= thresholds['network_drop_rate_warning']:
                        status = 'WARNING'
                    elif drop_rate >= thresholds['network_drop_rate_ok']:
                        status = 'INFO'
                    else:
                        status = 'OK'
                    
                    # Determine overall interface status
                    if status == 'OK' and utilization_pct < 80:
                        interface_status = 'OPTIMAL'
                    elif status in ['OK', 'INFO'] and utilization_pct < 90:
                        interface_status = 'GOOD'
                    elif status == 'WARNING' or utilization_pct >= 90:
                        interface_status = 'DEGRADED'
                    else:
                        interface_status = 'CRITICAL'
                    
                    stats[iface] = {
                        'packets_sent': net_stats.packets_sent,
                        'packets_recv': net_stats.packets_recv,
                        'tx_pps': tx_pps,
                        'rx_pps': rx_pps,
                        'mbps': round(mbps, 2),
                        'link_speed_gbps': link_speed_gbps,
                        'utilization_pct': round(utilization_pct, 2),
                        'dropin': net_stats.dropin,
                        'dropout': net_stats.dropout,
                        'total_drops': total_drops,
                        'drop_rate': round(drop_rate, 2),  # drops per second
                        'status': status,
                        'interface_status': interface_status
                    }
            except Exception as e:
                stats[iface] = {'status': 'ERROR', 'error': str(e)}
        
        return stats
    
    def get_ring_buffer_sizes(self, interface):
        """Get NIC ring buffer sizes using ethtool"""
        try:
            result = subprocess.run(['ethtool', '-g', interface], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode != 0:
                return {'status': 'ERROR', 'error': 'ethtool command failed'}
            
            # Parse output - look for "Current hardware settings:" section
            lines = result.stdout.split('\n')
            in_current_section = False
            rx_ring = None
            tx_ring = None
            
            for line in lines:
                if 'Current hardware settings:' in line:
                    in_current_section = True
                    continue
                
                if in_current_section:
                    if 'RX:' in line and 'Mini' not in line and 'Jumbo' not in line:
                        try:
                            rx_ring = int(line.split(':')[1].strip())
                        except:
                            pass
                    elif 'TX:' in line:
                        try:
                            tx_ring = int(line.split(':')[1].strip())
                        except:
                            pass
            
            if rx_ring is not None and tx_ring is not None:
                return {
                    'rx_ring': rx_ring,
                    'tx_ring': tx_ring,
                    'status': 'OK'
                }
            else:
                return {'status': 'ERROR', 'error': 'Could not parse ring buffer sizes'}
                
        except subprocess.TimeoutExpired:
            return {'status': 'ERROR', 'error': 'ethtool timeout'}
        except Exception as e:
            return {'status': 'ERROR', 'error': str(e)}
    
    def check_network_ultra_low_latency(self):
        """Check ultra-low latency network configuration"""
        ull_config = self.config.get('network', {}).get('ultra_low_latency', {})
        if not ull_config:
            return {'status': 'DISABLED', 'message': 'ULL monitoring not configured'}
        
        interface = ull_config.get('primary_interface', 'enp130s0f0')
        results = {
            'interface': interface,
            'checks': {},
            'overall_status': 'OK',
            'violations': [],
            'warnings': []
        }
        
        try:
            # Check 1: Adaptive RX Coalescing
            try:
                ethtool_result = subprocess.run(['ethtool', '-c', interface], 
                                              capture_output=True, text=True, timeout=5)
                if ethtool_result.returncode == 0:
                    adaptive_rx = None
                    for line in ethtool_result.stdout.split('\n'):
                        if 'Adaptive RX:' in line:
                            adaptive_rx = line.split(':')[1].strip().split()[0]
                            break
                    
                    expected = ull_config.get('adaptive_rx_expected', 'off')
                    results['checks']['adaptive_rx'] = {
                        'current': adaptive_rx,
                        'expected': expected,
                        'status': 'OK' if adaptive_rx == expected else 'FAIL'
                    }
                    
                    if adaptive_rx != expected:
                        results['violations'].append(f"Adaptive RX: {adaptive_rx} (expected: {expected})")
                        results['overall_status'] = 'CRITICAL'
                else:
                    results['checks']['adaptive_rx'] = {'status': 'ERROR', 'message': 'ethtool failed'}
            except Exception as e:
                results['checks']['adaptive_rx'] = {'status': 'ERROR', 'message': str(e)}
            
            # Check 2: XPS Configuration
            try:
                expected_mask = ull_config.get('xps_expected_mask', '0c')
                xps_queues = {}
                xps_violations = 0
                
                queue_dir = f'/sys/class/net/{interface}/queues'
                if os.path.exists(queue_dir):
                    tx_queues = [q for q in os.listdir(queue_dir) if q.startswith('tx-')]
                    
                    for queue in tx_queues:
                        xps_file = f'{queue_dir}/{queue}/xps_cpus'
                        try:
                            with open(xps_file, 'r') as f:
                                current_mask = f.read().strip()
                            
                            # Normalize masks for comparison
                            current_norm = current_mask.lstrip('0') or '0'
                            expected_norm = expected_mask.lstrip('0') or '0'
                            is_correct = current_norm == expected_norm
                            
                            xps_queues[queue] = {
                                'current': current_mask,
                                'expected': expected_mask,
                                'correct': is_correct
                            }
                            
                            if not is_correct:
                                xps_violations += 1
                        except IOError:
                            xps_queues[queue] = {'status': 'ERROR', 'message': 'Cannot read XPS config'}
                            xps_violations += 1
                
                results['checks']['xps_config'] = {
                    'queues': xps_queues,
                    'violations': xps_violations,
                    'status': 'OK' if xps_violations == 0 else 'FAIL'
                }
                
                if xps_violations > 0:
                    results['violations'].append(f"XPS violations: {xps_violations} queues")
                    results['overall_status'] = 'CRITICAL'
            except Exception as e:
                results['checks']['xps_config'] = {'status': 'ERROR', 'message': str(e)}
            
            # Check 3: Service Status
            required_services = ull_config.get('required_services', [])
            service_status = {}
            service_failures = 0
            
            for service in required_services:
                try:
                    svc_result = subprocess.run(['systemctl', 'is-active', service], 
                                              capture_output=True, text=True, timeout=5)
                    is_active = svc_result.returncode == 0 and svc_result.stdout.strip() == 'active'
                    service_status[service] = {
                        'active': is_active,
                        'status': svc_result.stdout.strip()
                    }
                    
                    if not is_active:
                        service_failures += 1
                        results['violations'].append(f"Service {service} not active")
                except Exception as e:
                    service_status[service] = {'status': 'ERROR', 'message': str(e)}
                    service_failures += 1
            
            results['checks']['services'] = {
                'services': service_status,
                'failures': service_failures,
                'status': 'OK' if service_failures == 0 else 'FAIL'
            }
            
            if service_failures > 0:
                results['overall_status'] = 'CRITICAL'
            
            # Check 4: IRQ Violations (trading cores should have no NIC IRQs)
            # NOTE: Ignore low counts (<= 5) which are legacy boot artifacts
            # Only alert on counts > 5 which indicate ongoing IRQ problems
            try:
                irq_violations = []
                with open('/proc/interrupts', 'r') as f:
                    lines = f.readlines()
                
                trading_cores = self.config.get('cpu', {}).get('isolated_cores', [2, 3])
                IRQ_THRESHOLD = 5  # Ignore legacy boot artifacts
                
                for line in lines:
                    if interface in line:
                        parts = line.split()
                        if len(parts) > max(trading_cores):
                            irq_num = parts[0].rstrip(':')
                            for core in trading_cores:
                                if len(parts) > core + 1:  # +1 because IRQ number is first
                                    count = int(parts[core + 1])
                                    if count > IRQ_THRESHOLD:  # Only alert on active problems
                                        irq_violations.append({
                                            'irq': irq_num,
                                            'core': core,
                                            'count': count
                                        })
                
                results['checks']['irq_violations'] = {
                    'violations': irq_violations,
                    'count': len(irq_violations),
                    'status': 'OK' if len(irq_violations) == 0 else 'FAIL'
                }
                
                if irq_violations:
                    results['violations'].append(f"IRQ violations on trading cores: {len(irq_violations)}")
                    results['overall_status'] = 'CRITICAL'
                    
            except Exception as e:
                results['checks']['irq_violations'] = {'status': 'ERROR', 'message': str(e)}
            
            # Check 4: Ring Buffer Sizes
            ring_buffers = self.get_ring_buffer_sizes(interface)
            results['checks']['ring_buffers'] = ring_buffers
        
        except Exception as e:
            results['overall_status'] = 'ERROR'
            results['message'] = f"ULL check failed: {str(e)}"
        
        return results
    
    def check_gpu_status(self):
        """Check GPU status using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 5:
                    gpus.append({
                        'name': parts[0],
                        'memory_used': float(parts[1]),
                        'memory_total': float(parts[2]),
                        'temperature': float(parts[3]),
                        'utilization': float(parts[4]),
                        'status': 'OK'
                    })
            return gpus
        except:
            return []
    
    def check_gpu_vram_detailed(self):
        """
        Enhanced GPU monitoring with comprehensive metrics
        Monitors: VRAM, clocks, power, temperature, throttling, clock lock status
        Integrates GPU optimization monitoring from Services/GPU
        """
        try:
            # Get GPU optimization config
            gpu_config = self.config.get('gpu_optimization', {})
            if not gpu_config.get('enabled', False):
                # Fall back to basic GPU monitoring
                return {'status': 'DISABLED', 'message': 'GPU optimization monitoring disabled'}
            
            # Query comprehensive GPU metrics in single nvidia-smi call
            # Include both current clocks AND application clocks to detect lock configuration
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,'
                 'temperature.gpu,utilization.gpu,power.draw,power.limit,'
                 'clocks.gr,clocks.mem,clocks.applications.gr,clocks.applications.mem,'
                 'clocks_event_reasons.active',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return {'status': 'ERROR', 'error': 'nvidia-smi command failed'}
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                    
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 13:  # Now expecting 13 fields (added 2 application clock fields)
                    try:
                        gpu_idx = int(parts[0])
                        gpu_data = {
                            'index': gpu_idx,
                            'name': parts[1],
                            'memory_used': float(parts[2]),  # MB
                            'memory_total': float(parts[3]),  # MB
                            'temperature': float(parts[4]),  # °C
                            'utilization': float(parts[5]),  # %
                            'power_draw': float(parts[6]),  # W
                            'power_limit': float(parts[7]),  # W
                            'clock_graphics': int(float(parts[8])),  # Current MHz
                            'clock_memory': int(float(parts[9])),  # Current MHz
                            'clock_graphics_app': int(float(parts[10])),  # Application lock MHz
                            'clock_memory_app': int(float(parts[11])),  # Application lock MHz
                            'throttle_reasons': parts[12],  # Hex string
                            'status': 'OK'
                        }
                        
                        # Determine if clocks are locked at target
                        target_graphics = gpu_config.get('target_graphics_clock_mhz', 3090)
                        target_memory = gpu_config.get('target_memory_clock_mhz', 14001)
                        idle_threshold = gpu_config.get('idle_clock_threshold_mhz', 300)
                        
                        # Check APPLICATION clocks to see if they're configured at target
                        app_clocks_at_target = (
                            gpu_data['clock_graphics_app'] == target_graphics and 
                            gpu_data['clock_memory_app'] == target_memory
                        )
                        
                        # Check if current clocks are at target (active) or idle
                        current_at_target = (
                            gpu_data['clock_graphics'] == target_graphics and 
                            gpu_data['clock_memory'] == target_memory
                        )
                        current_at_idle = gpu_data['clock_graphics'] < idle_threshold
                        
                        # Determine overall clock status
                        gpu_data['clocks_locked'] = app_clocks_at_target
                        gpu_data['clocks_idle'] = current_at_idle
                        
                        # Status logic:
                        # - LOCKED: App clocks set to target (current may vary due to P-states)
                        # - NOT_LOCKED: App clocks NOT set to target
                        # - IDLE: GPU in low-power idle state
                        # Note: When app clocks are locked, GPU may use intermediate P-states (e.g., 2302 MHz)
                        # for power efficiency. This is normal behavior, not drift.
                        if app_clocks_at_target:
                            if current_at_target:
                                gpu_data['clocks_status'] = 'LOCKED'  # Active at max frequency
                            elif current_at_idle:
                                gpu_data['clocks_status'] = 'IDLE'  # Deep idle state
                            else:
                                # Intermediate P-state (e.g., 2302 MHz) - normal power management
                                gpu_data['clocks_status'] = 'LOCKED'  # App clocks locked, P-states OK
                        else:
                            gpu_data['clocks_status'] = 'NOT_LOCKED'  # App clocks not configured!
                        
                        # Parse throttling status (0x0000000000000000 = no throttling)
                        gpu_data['throttling_active'] = (gpu_data['throttle_reasons'] != '0x0000000000000000')
                        
                        gpus.append(gpu_data)
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse GPU {gpu_idx} data: {e}")
                        continue
            
            if not gpus:
                return {'status': 'ERROR', 'error': 'No GPUs detected'}
            
            # Calculate aggregate statistics
            total_vram_used = sum([g['memory_used'] for g in gpus]) / 1024  # GB
            total_vram = sum([g['memory_total'] for g in gpus]) / 1024  # GB
            total_vram_percent = (total_vram_used / total_vram * 100) if total_vram > 0 else 0
            
            return {
                'gpus': gpus,
                'total_vram_used_gb': round(total_vram_used, 3),
                'total_vram_gb': round(total_vram, 1),
                'total_vram_percent': round(total_vram_percent, 1),
                'gpu_count': len(gpus),
                'status': 'OK'
            }
            
        except subprocess.TimeoutExpired:
            return {'status': 'ERROR', 'error': 'nvidia-smi timeout'}
        except Exception as e:
            self.logger.error(f"GPU detailed check failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def decode_throttle_reasons(self, throttle_hex_str):
        """
        Decode NVIDIA throttle reasons from hex bitmask to human-readable list
        
        Args:
            throttle_hex_str: Hex string from nvidia-smi (e.g., "0x0000000000000008")
        
        Returns:
            List of human-readable throttle reason strings
        
        NVIDIA Throttle Reason Bitmask:
        0x0001 = GPU idle
        0x0002 = Application clock setting limit
        0x0004 = SW power cap
        0x0008 = Hardware thermal slowdown (thermal temp limit)
        0x0010 = Power brake assertion
        0x0020 = Display clock setting limit / Board power connector limit
        0x0040 = Software thermal slowdown (thermal slowdown temp limit)
        0x0080 = Hardware thermal slowdown (HW slowdown active)
        0x0100 = Hardware power brake slowdown (HW power brake active)
        0x0200 = Display clock setting (reserved - not used)
        0x0400 = GPU voltage limit
        0x0800 = Sync boost limit
        """
        # Throttle reason definitions
        THROTTLE_REASONS = {
            0x0001: "GPU Idle",
            0x0002: "Application Clock Limit (clocks locked at target)",
            0x0004: "SW Power Cap (power limit reached)",
            0x0008: "Thermal Slowdown (temperature limit)",
            0x0010: "Power Brake Assertion",
            0x0020: "Board Power Connector Limit",
            0x0040: "SW Thermal Slowdown",
            0x0080: "HW Thermal Slowdown (critical temp)",
            0x0100: "HW Power Brake Slowdown",
            0x0200: "Current Limit",
            0x0400: "Voltage Limit",
            0x0800: "Sync Boost Limit"
        }
        
        reasons = []
        
        try:
            # Convert hex string to integer
            throttle_value = int(throttle_hex_str, 16)
            
            # Check each bit flag
            for flag, description in sorted(THROTTLE_REASONS.items()):
                if throttle_value & flag:
                    reasons.append(description)
            
            if not reasons:
                reasons.append("No throttling (0x0)")
        
        except (ValueError, TypeError) as e:
            reasons.append(f"Invalid throttle value: {throttle_hex_str}")
            self.logger.warning(f"Failed to decode throttle reasons: {e}")
        
        return reasons
    
    def get_throttle_guidance(self, throttle_value, is_critical):
        """
        Provide actionable guidance based on throttle reasons
        
        Args:
            throttle_value: Integer value of throttle bitmask
            is_critical: Boolean indicating if hardware protection engaged
        
        Returns:
            String with specific action guidance
        """
        if is_critical:
            # Critical hardware protection - immediate action required
            guidance_parts = []
            
            if throttle_value & 0x0080:  # HW Thermal Slowdown
                guidance_parts.append("STOP WORKLOAD - GPU >85°C, check cooling fans and thermal paste immediately")
            
            if throttle_value & 0x0100:  # HW Power Brake
                guidance_parts.append("CHECK PSU - Verify power supply delivering adequate power, check PCIe cables")
            
            return " | ".join(guidance_parts) if guidance_parts else "Hardware protection engaged - immediate attention required"
        
        else:
            # Warning - provide monitoring guidance
            guidance_parts = []
            
            if throttle_value & 0x0008:  # Thermal Slowdown
                guidance_parts.append("Monitor temps (target <83°C)")
            
            if throttle_value & 0x0004:  # SW Power Cap
                guidance_parts.append("Check power limit (expected: 325W)")
            
            if throttle_value & (0x0010 | 0x0020):  # Power brake or connector
                guidance_parts.append("Verify PSU and PCIe connections")
            
            if throttle_value & (0x0040 | 0x0200 | 0x0400):  # SW thermal, current, voltage
                guidance_parts.append("Review GPU configuration")
            
            return " | ".join(guidance_parts) if guidance_parts else "Monitor GPU status"
    
    def check_gpu_optimization_config(self):
        """
        Verify GPU optimization configuration is active
        Checks: persistence mode, power limits, clock lock status
        Returns: OPTIMAL, DEGRADED, or UNOPTIMIZED with specific violations
        """
        try:
            gpu_config = self.config.get('gpu_optimization', {})
            if not gpu_config.get('enabled', False):
                return {'status': 'DISABLED', 'message': 'GPU optimization monitoring disabled'}
            
            violations = []
            warnings = []
            
            # Check 1: Persistence mode
            persistence_mode = False
            if gpu_config.get('check_persistence_mode', True):
                try:
                    result = subprocess.run(['nvidia-smi', '-q'],
                                          capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        persistence_enabled = 'Persistence Mode' in result.stdout and 'Enabled' in result.stdout
                        persistence_mode = persistence_enabled
                        if not persistence_enabled:
                            violations.append("Persistence mode not enabled (run: nvidia-smi -pm 1)")
                except Exception as e:
                    warnings.append(f"Could not check persistence mode: {str(e)}")
            
            # Check 2: Power limits (should be at max 325W per GPU)
            gpu_data = self.check_gpu_vram_detailed()
            if gpu_data.get('status') == 'OK' and gpu_data.get('gpus'):
                expected_power = gpu_config.get('max_power_limit_w', 325)
                power_tolerance = gpu_config.get('power_limit_tolerance_w', 10)
                
                for gpu in gpu_data['gpus']:
                    actual_power = gpu.get('power_limit', 0)
                    if actual_power < (expected_power - power_tolerance):
                        violations.append(
                            f"GPU{gpu['index']} power limit {actual_power:.0f}W < expected {expected_power}W"
                        )
                
                # Check 3: Clock lock configuration
                # Check if application clocks are properly configured
                for gpu in gpu_data['gpus']:
                    clock_status = gpu.get('clocks_status', 'UNKNOWN')
                    if clock_status == 'NOT_LOCKED':
                        app_gr = gpu.get('clock_graphics_app', 0)
                        app_mem = gpu.get('clock_memory_app', 0)
                        violations.append(
                            f"GPU{gpu['index']} clocks NOT LOCKED: app clocks at {app_gr}/{app_mem} MHz (expected 3090/14001 MHz)"
                        )
                    # DRIFT is normal during GPU ramp (idle→active or active→idle)
                    # Only flag DRIFT if sustained (not transient ramp state)
                    # Note: DRIFT = app clocks locked correctly, but current clocks ramping
                    # This is expected behavior and not a violation
                
                # Check 4: Throttling detection
                # CRITICAL: Ignore spurious throttle events on idle GPUs
                # NVIDIA occasionally reports transient throttle bits (e.g., 0x4) when GPU idle
                # Only flag throttling when GPU is under load
                idle_threshold = gpu_config.get('idle_clock_threshold_mhz', 300)
                for gpu in gpu_data['gpus']:
                    is_throttling = gpu.get('throttling_active', False)
                    is_idle = gpu.get('clock_graphics', 9999) < idle_threshold
                    
                    # Only violation if throttling AND NOT idle (real throttling under load)
                    if is_throttling and not is_idle:
                        violations.append(
                            f"GPU{gpu['index']} throttling active under load (reason: {gpu.get('throttle_reasons', 'unknown')})"
                        )
            
            # Determine clock lock status (check if clocks are configured correctly)
            clocks_locked = False
            
            # Check if clocks are locked (application clocks set to target)
            if gpu_data.get('status') == 'OK' and gpu_data.get('gpus'):
                all_gpus_proper = True
                for gpu in gpu_data['gpus']:
                    clock_status = gpu.get('clocks_status', 'UNKNOWN')
                    # LOCKED, IDLE, or DRIFT are acceptable (only NOT_LOCKED is bad)
                    # DRIFT = GPU ramping between idle/active, but app clocks still locked
                    if clock_status == 'NOT_LOCKED':
                        all_gpus_proper = False
                        break
                clocks_locked = all_gpus_proper
            
            # Determine overall status
            if len(violations) == 0 and len(warnings) == 0:
                status = 'OPTIMAL'
            elif len(violations) == 0:
                status = 'GOOD'
            elif len(violations) <= 2:
                status = 'DEGRADED'
            else:
                status = 'UNOPTIMIZED'
            
            return {
                'status': status,
                'violations': violations,
                'warnings': warnings,
                'checks_passed': len(violations) == 0,
                'checks': {
                    'clocks_locked': clocks_locked,
                    'persistence_mode': persistence_mode
                }
            }
            
        except Exception as e:
            self.logger.error(f"GPU optimization config check failed: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def check_pytorch_performance(self):
        """
        Collect PyTorch/CUDA performance metrics for monitoring dashboard
        
        Metrics collected:
        - Inference latency (P50/P95/P99)
        - TFLOPS (FP16 actual performance)
        - torch.compile status and speedup
        - Memory allocation per GPU
        - Clock lock status
        - Precision mode (TF32)
        
        Returns:
            dict: PyTorch performance metrics for dashboard display
        """
        try:
            # Import pytorch_metrics module from Services/GPU
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path.home() / 'ai-trading-station'))
            
            from Services.GPU.pytorch_metrics import get_all_pytorch_metrics
            
            # Check if in production mode (trading hours) - read from config file
            production_mode = self.config.get('monitoring', {}).get('production_mode', False)
            
            # Collect metrics
            # - production_mode=False: Full monitoring (blocks GPU ~2s for benchmarks)
            # - production_mode=True: Lightweight monitoring (no GPU blocking, uses estimates)
            pytorch_metrics = get_all_pytorch_metrics(quick_mode=False, production_mode=production_mode)
            
            if not pytorch_metrics.get('pytorch_available'):
                return {
                    'status': 'UNAVAILABLE',
                    'error': 'PyTorch/CUDA not available'
                }
            
            # Format metrics for dashboard display
            formatted = {
                'status': 'OK',
                'inference': {
                    'mean_ms': pytorch_metrics['latency']['mean'],
                    'p50_ms': pytorch_metrics['latency']['p50'],
                    'p95_ms': pytorch_metrics['latency']['p95'],
                    'p99_ms': pytorch_metrics['latency']['p99'],
                    'target_p99_ms': 0.20,  # Target from variance_test_10k.py
                    'pass': pytorch_metrics['latency']['p99'] < 0.20,
                    'timestamp': pytorch_metrics['latency'].get('timestamp', 'N/A'),
                    'iterations': pytorch_metrics['latency'].get('iterations', 0)
                },
                'tflops': {
                    'fp16': pytorch_metrics['tflops']['fp16_tflops'],
                    'theoretical_peak': pytorch_metrics['tflops']['theoretical_peak'],
                    'efficiency': pytorch_metrics['tflops']['efficiency'],
                    'status': 'EXCELLENT' if pytorch_metrics['tflops']['efficiency'] > 80 else 'GOOD',
                    'cached': pytorch_metrics['tflops'].get('cached', False),
                    'timestamp': pytorch_metrics['tflops'].get('timestamp', 'N/A'),
                    'gpu_id': pytorch_metrics['tflops'].get('gpu_id', -1)
                },
                'compile': {
                    'available': pytorch_metrics['compile_status']['available'],
                    'active': pytorch_metrics['compile_status'].get('active', False),
                    'speedup': pytorch_metrics['compile_status']['speedup'],
                    'mode': pytorch_metrics['compile_status']['mode'],
                    'compiled_models': pytorch_metrics['compile_status'].get('compiled_models', 0),
                    'status': 'ACTIVE' if pytorch_metrics['compile_status'].get('active', False) else ('AVAILABLE' if pytorch_metrics['compile_status']['available'] else 'INACTIVE')
                },
                'memory': {},
                'hardware': {
                    'clocks_locked': pytorch_metrics['clock_status']['clocks_locked'],
                    'target_clock_mhz': pytorch_metrics['clock_status']['target_clock_mhz'],
                    'power_limit_w': pytorch_metrics['clock_status']['power_limit_w'],
                    'precision_mode': pytorch_metrics['precision']['mode'],
                    'precision_status': pytorch_metrics['precision']['status'],
                    'interconnect': 'PCIe'  # No NVLink
                }
            }
            
            # Format memory stats per GPU
            for gpu_key, gpu_mem in pytorch_metrics['memory'].items():
                if gpu_key.startswith('gpu_') and 'error' not in gpu_mem:
                    gpu_id = gpu_key.split('_')[1]
                    formatted['memory'][f'gpu_{gpu_id}'] = {
                        'allocated_gb': gpu_mem['allocated_gb'],
                        'reserved_gb': gpu_mem['reserved_gb'],
                        'total_gb': gpu_mem['total_gb'],
                        'fragmentation_percent': gpu_mem['fragmentation_percent'],
                        'utilization_percent': (gpu_mem['allocated_gb'] / gpu_mem['total_gb']) * 100
                    }
            
            return formatted
            
        except ImportError as e:
            self.logger.warning(f"Could not import pytorch_metrics: {e}")
            return {
                'status': 'UNAVAILABLE',
                'error': f'pytorch_metrics module not found: {e}'
            }
        except Exception as e:
            self.logger.error(f"PyTorch performance check failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def check_cpu_temperature(self):
        """Check CPU temperature using sensors"""
        try:
            result = subprocess.run(['sensors', '-A'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return {'available': False}
            
            temperatures = []
            for line in result.stdout.split('\n'):
                if 'Core' in line and '°C' in line:
                    # Extract temperature value
                    temp_str = line.split('+')[1].split('°C')[0]
                    try:
                        temp = float(temp_str)
                        temperatures.append(temp)
                    except (ValueError, IndexError):
                        continue
            
            if not temperatures:
                return {'available': False}
            
            max_temp = max(temperatures)
            avg_temp = sum(temperatures) / len(temperatures)
            
            # Determine status
            if max_temp > self.config['cpu']['temp_critical']:
                status = 'CRITICAL'
            elif max_temp > self.config['cpu']['temp_warning']:
                status = 'WARNING'
            else:
                status = 'OK'
            
            return {
                'available': True,
                'max_temp': max_temp,
                'avg_temp': avg_temp,
                'all_temps': temperatures,
                'status': status
            }
            
        except Exception as e:
            self.logger.error(f"Temperature check failed: {e}")
            return {'available': False}
    
    def check_system_resources(self):
        """Check comprehensive system resources: memory, hugepages, PCIe, power, thermal"""
        try:
            resources = {}
            
            # === CPU Metrics ===
            # Get CPU load (as percentage across all cores)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            resources['cpu_load'] = round(cpu_percent, 1)
            
            # Get CPU temperature from existing method
            temp_data = self.check_cpu_temperature()
            resources['cpu_temp_max'] = temp_data.get('max_temp', 0) if temp_data.get('available') else 0
            resources['cpu_temp_avg'] = temp_data.get('avg_temp', 0) if temp_data.get('available') else 0
            resources['cpu_temp_status'] = temp_data.get('status', 'UNKNOWN') if temp_data.get('available') else 'UNAVAILABLE'
            
            # Get isolated cores from config or detect
            isolated_cpus = self.config.get('cpu', {}).get('isolated_cpus', [2, 3])
            resources['isolated_cores'] = isolated_cpus
            
            # === Memory Metrics ===
            vm = psutil.virtual_memory()
            resources['memory_used_gb'] = round(vm.used / (1024**3), 1)
            resources['memory_total_gb'] = round(vm.total / (1024**3), 1)
            resources['memory_percent'] = round(vm.percent, 1)
            
            # === Huge Pages ===
            # Check /proc/meminfo for HugePages
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    hugepages_total = 0
                    hugepages_free = 0
                    for line in meminfo.split('\n'):
                        if 'HugePages_Total' in line:
                            hugepages_total = int(line.split()[1])
                        elif 'HugePages_Free' in line:
                            hugepages_free = int(line.split()[1])
                    
                    hugepages_allocated = hugepages_total - hugepages_free
                    resources['hugepages_total'] = hugepages_total
                    resources['hugepages_allocated'] = hugepages_allocated
                    resources['hugepages_free'] = hugepages_free
                    resources['hugepages_status'] = 'OK' if hugepages_allocated > 0 else 'NONE'
            except Exception as e:
                self.logger.warning(f"Failed to read hugepages: {e}")
                resources['hugepages_total'] = 0
                resources['hugepages_allocated'] = 0
                resources['hugepages_status'] = 'ERROR'
            
            # === PCIe Metrics ===
            # Auto-detect Solarflare NIC PCIe info
            try:
                # First, find Solarflare NIC bus address
                lspci_result = subprocess.run(
                    ['lspci'],
                    capture_output=True, text=True, timeout=5
                )
                
                solarflare_bus = None
                if lspci_result.returncode == 0:
                    for line in lspci_result.stdout.split('\n'):
                        if 'Solarflare' in line or 'SFC' in line:
                            # Extract bus address (e.g., "82:00.0")
                            solarflare_bus = line.split()[0]
                            self.logger.info(f"Detected Solarflare NIC at {solarflare_bus}")
                            break
                
                pcie_gen = 'Unknown'
                pcie_width = 'Unknown'
                pcie_latency_ns = 0
                
                if solarflare_bus:
                    # Try to get detailed PCIe info (first without sudo, then with -n for non-interactive)
                    result = None
                    
                    # Try without sudo first
                    try:
                        result = subprocess.run(
                            ['lspci', '-vvv', '-s', solarflare_bus],
                            capture_output=True, text=True, timeout=5
                        )
                    except Exception:
                        pass
                    
                    # If no LnkSta found, try with sudo -n (non-interactive, won't hang)
                    if not result or 'LnkSta:' not in result.stdout:
                        try:
                            result = subprocess.run(
                                ['sudo', '-n', 'lspci', '-vvv', '-s', solarflare_bus],
                                capture_output=True, text=True, timeout=2
                            )
                        except subprocess.TimeoutExpired:
                            self.logger.warning("PCIe detection: sudo timed out (password required)")
                            result = None
                        except Exception as e:
                            self.logger.warning(f"PCIe detection: sudo failed ({e})")
                            result = None
                    
                    if result and result.returncode == 0 and 'LnkSta:' in result.stdout:
                        for line in result.stdout.split('\n'):
                            if 'LnkSta:' in line:
                                # Extract "Speed 8GT/s, Width x4 (downgraded)"
                                if '16GT/s' in line or '16.0GT/s' in line:
                                    pcie_gen = 'Gen4'
                                elif '8GT/s' in line or '8.0GT/s' in line:
                                    pcie_gen = 'Gen3'
                                elif '5GT/s' in line or '5.0GT/s' in line:
                                    pcie_gen = 'Gen2'
                                elif '2.5GT/s' in line:
                                    pcie_gen = 'Gen1'
                                
                                if 'Width x16' in line:
                                    pcie_width = 'x16'
                                elif 'Width x8' in line:
                                    pcie_width = 'x8'
                                elif 'Width x4' in line:
                                    pcie_width = 'x4'
                                elif 'Width x2' in line:
                                    pcie_width = 'x2'
                                elif 'Width x1' in line:
                                    pcie_width = 'x1'
                            
                            elif 'LnkCap:' in line and 'L0s' in line:
                                # Extract latency from LnkCap (e.g., "Exit Latency L0s unlimited, L1 <64us")
                                try:
                                    if 'L1 <' in line:
                                        latency_str = line.split('L1 <')[1].split()[0]
                                        if 'us' in latency_str:
                                            # Convert microseconds to nanoseconds
                                            pcie_latency_ns = int(latency_str.replace('us', '')) * 1000
                                        elif 'ns' in latency_str:
                                            pcie_latency_ns = int(latency_str.replace('ns', ''))
                                except:
                                    pcie_latency_ns = 120  # Conservative default
                    else:
                        # Sudo failed or not available - use graceful fallback
                        self.logger.warning("PCIe detailed info unavailable - configure passwordless sudo for 'lspci'")
                        pcie_gen = 'N/A'
                        pcie_width = 'sudo'
                        pcie_latency_ns = 0
                
                resources['pcie_gen'] = pcie_gen
                resources['pcie_width'] = pcie_width
                resources['pcie_latency_ns'] = pcie_latency_ns if pcie_latency_ns > 0 else 120
                resources['pcie_bus'] = solarflare_bus if solarflare_bus else 'Not found'
                
                # Status grading: Consider Gen3 x4+ as OK (motherboard limitation acceptable)
                # Gen4 x8+ = OPTIMAL, Gen3 x4+ = OK, Gen2 or x1/x2 = DEGRADED
                if pcie_gen == 'Unknown' or pcie_gen == 'N/A':
                    resources['pcie_status'] = 'UNAVAILABLE'
                elif pcie_gen == 'Gen4' and pcie_width in ['x8', 'x16']:
                    resources['pcie_status'] = 'OPTIMAL'
                elif pcie_gen == 'Gen3' and pcie_width in ['x4', 'x8', 'x16']:
                    resources['pcie_status'] = 'OK'  # Motherboard limitation acceptable
                elif pcie_gen == 'Gen3' and pcie_width in ['x2', 'x1']:
                    resources['pcie_status'] = 'DEGRADED'  # Too narrow
                elif pcie_gen in ['Gen2', 'Gen1']:
                    resources['pcie_status'] = 'DEGRADED'  # Too slow
                else:
                    resources['pcie_status'] = 'UNKNOWN'
                
            except Exception as e:
                self.logger.warning(f"Failed to read PCIe info: {e}")
                resources['pcie_gen'] = 'Unknown'
                resources['pcie_width'] = 'Unknown'
                resources['pcie_latency_ns'] = 0
                resources['pcie_status'] = 'UNKNOWN'
            
            # === Power & Thermal ===
            # Read power consumption using Intel RAPL or hwmon
            try:
                power_w = 0
                
                # Method 1: Intel RAPL (Running Average Power Limit) - most accurate for CPU
                rapl_path = Path('/sys/class/powercap/intel-rapl/intel-rapl:0')
                if rapl_path.exists():
                    try:
                        energy_file = rapl_path / 'energy_uj'
                        # Read energy counter twice with 100ms interval
                        with open(energy_file, 'r') as f:
                            energy1 = int(f.read().strip())
                        
                        time.sleep(0.1)  # 100ms interval
                        
                        with open(energy_file, 'r') as f:
                            energy2 = int(f.read().strip())
                        
                        # Calculate power: (energy_diff in microjoules) / (time in seconds) / 1,000,000 = watts
                        energy_diff = energy2 - energy1
                        if energy_diff < 0:  # Counter wrapped around
                            # RAPL counter is 32-bit, max value around 2^32
                            energy_diff += (2**32)
                        
                        power_w = (energy_diff / 0.1) / 1000000  # Convert μJ to W
                        self.logger.debug(f"RAPL power: {power_w:.1f}W")
                    except Exception as e:
                        self.logger.warning(f"RAPL read failed: {e}")
                        power_w = 0
                
                # Method 2: Check /sys/class/hwmon for power sensors (backup)
                if power_w == 0:
                    hwmon_path = Path('/sys/class/hwmon')
                    if hwmon_path.exists():
                        for hwmon_dir in hwmon_path.iterdir():
                            if hwmon_dir.is_dir():
                                # Look for power input files
                                power_files = list(hwmon_dir.glob('power*_input'))
                                for pf in power_files:
                                    try:
                                        with open(pf, 'r') as f:
                                            # Power is in microwatts, convert to watts
                                            power_uw = int(f.read().strip())
                                            power_w += power_uw / 1000000
                                    except:
                                        continue
                
                resources['power_w'] = round(power_w, 1) if power_w > 0 else 0
                resources['power_status'] = 'OK' if power_w > 0 else 'UNAVAILABLE'
                
            except Exception as e:
                self.logger.warning(f"Failed to read power: {e}")
                resources['power_w'] = 0
                resources['power_status'] = 'UNAVAILABLE'
            
            # Thermal status (based on CPU temp)
            if resources['cpu_temp_status'] == 'OK':
                resources['thermal_status'] = 'Normal'
                resources['thermal_emoji'] = '✅'
            elif resources['cpu_temp_status'] == 'WARNING':
                resources['thermal_status'] = 'Elevated'
                resources['thermal_emoji'] = '⚠️'
            elif resources['cpu_temp_status'] == 'CRITICAL':
                resources['thermal_status'] = 'Critical'
                resources['thermal_emoji'] = '🚨'
            else:
                resources['thermal_status'] = 'Unknown'
                resources['thermal_emoji'] = '❓'
            
            return resources
            
        except Exception as e:
            self.logger.error(f"System resources check failed: {e}")
            return {
                'cpu_load': 0,
                'cpu_temp_max': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 0,
                'hugepages_allocated': 0,
                'pcie_gen': 'Unknown',
                'pcie_width': 'Unknown',
                'pcie_latency_ns': 0,
                'power_w': 0,
                'thermal_status': 'Unknown',
                'error': str(e)
            }
    
    def check_redis_hft(self):
        """Check Redis HFT performance metrics"""
        try:
            if not self.config.get('redis_hft', {}).get('enabled', False):
                return {'enabled': False}
            
            redis_config = self.config['redis_hft']
            monitor_script = redis_config['monitor_script']
            
            # Run Redis monitoring script
            result = subprocess.run(
                [monitor_script],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode != 0:
                self.logger.error(f"Redis monitor script failed: {result.stderr}")
                return {'enabled': True, 'error': 'Monitor script failed'}
            
            # Parse JSON output
            redis_data = json.loads(result.stdout)
            
            # Check performance gate status
            perf_gate_script = redis_config['perf_gate_script']
            gate_result = subprocess.run(
                ['sudo', '-n', perf_gate_script, '--metrics-only'],
                capture_output=True,
                timeout=15
            )
            perf_gate_status = 'PASS' if gate_result.returncode == 0 else 'FAIL'
            
            # DEBUG: Log the actual gate result
            self.logger.info(f"[DEBUG] Perf-gate return code: {gate_result.returncode}, status: {perf_gate_status}")
            
            # Evaluate thresholds
            thresholds = redis_config['thresholds']
            violations = []
            warnings = []
            
            rtt = redis_data.get('rtt', {})
            if 'error' not in rtt:
                if rtt.get('p99', 0) > thresholds['rtt_p99_max_us']:
                    violations.append(f"RTT P99: {rtt['p99']:.2f}μs > {thresholds['rtt_p99_max_us']}μs")
                if rtt.get('jitter', 0) > thresholds['rtt_jitter_max_us']:
                    warnings.append(f"RTT jitter: {rtt['jitter']:.2f}μs > {thresholds['rtt_jitter_max_us']}μs")
            
            health = redis_data.get('health', {})
            if health.get('ops_per_sec', 0) < thresholds['ops_per_sec_min']:
                warnings.append(f"Ops/sec: {health['ops_per_sec']} < {thresholds['ops_per_sec_min']}")
            
            # Increment test counter
            self.redis_check_count += 1
            
            # DEBUG: Log what we're returning
            result_dict = {
                'enabled': True,
                'performance_gate': perf_gate_status,
                'metrics': redis_data,
                'violations': violations,
                'warnings': warnings,
                'status': 'OK' if not violations else 'DEGRADED',
                'check_count': self.redis_check_count
            }
            self.logger.info(f"[DEBUG] Returning perf_gate={perf_gate_status}, violations={len(violations)}, status={result_dict['status']}")
            return result_dict
            
        except subprocess.TimeoutExpired:
            self.logger.error("Redis monitoring timeout")
            return {'enabled': True, 'error': 'Timeout'}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Redis metrics: {e}")
            return {'enabled': True, 'error': 'JSON parse error'}
        except Exception as e:
            self.logger.error(f"Redis HFT check failed: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def check_questdb(self):
        """Check QuestDB performance metrics"""
        try:
            # Initialize QuestDB monitor
            questdb_monitor = QuestDBMonitor()
            
            # Collect all QuestDB metrics
            metrics = questdb_monitor.collect_all_metrics()
            
            # Calculate status based on metrics
            status = 'OK'
            warnings = []
            errors = []
            
            # Check service status
            if metrics['service']['status'] != 'active':
                errors.append(f"Service not active: {metrics['service']['status']}")
                status = 'ERROR'
            
            # Check connectivity
            if metrics['connectivity']['status'] != 'ok':
                errors.append(f"Connectivity failed: {metrics['connectivity'].get('message', 'Unknown')}")
                status = 'ERROR'
            
            # Check process status
            proc = metrics['process']
            if proc.get('status') != 'running':
                errors.append("Process not running")
                status = 'ERROR'
            
            # Check memory usage (warn if >90% of allocated 96GB)
            if proc.get('memory_gb', 0) > 86:  # 90% of 96GB
                warnings.append(f"High memory usage: {proc['memory_gb']:.1f}GB / 96GB allocated")
                if status == 'OK':
                    status = 'WARNING'
            
            # Check hot storage (warn if >80% full)
            storage = metrics.get('storage', {})
            if 'hot' in storage and 'error' not in storage['hot']:
                hot = storage['hot']
                if hot.get('usage_percent', 0) > 80:
                    warnings.append(f"Hot storage high: {hot['usage_percent']:.1f}%")
                    if status == 'OK':
                        status = 'WARNING'
            
            # Check cold storage (warn if >90% full)
            if 'cold' in storage and 'error' not in storage['cold']:
                cold = storage['cold']
                if cold.get('usage_percent', 0) > 90:
                    warnings.append(f"Cold storage high: {cold['usage_percent']:.1f}%")
                    if status == 'OK':
                        status = 'WARNING'
            
            return {
                'enabled': True,
                'status': status,
                'metrics': metrics,
                'warnings': warnings,
                'errors': errors
            }
            
        except Exception as e:
            self.logger.error(f"QuestDB check failed: {e}")
            return {
                'enabled': True,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def generate_alerts(self):
        """Generate alerts based on current metrics and thresholds"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Clear old alerts from this cycle (keep historical)
        # We append new alerts, deque automatically manages maxlen=100
        
        # 1. PERFORMANCE BENCHMARKS ALERTS
        # Onload Stack latency
        if self.metrics.get('latency'):
            lat = self.metrics['latency']
            if lat.get('mean') and lat['mean'] > self.config['thresholds']['latency_mean_microseconds']:
                self.alerts.append(f"[{timestamp}] ⚠️ Onload latency HIGH: {lat['mean']:.2f}μs (threshold: {self.config['thresholds']['latency_mean_microseconds']}μs)")
            if lat.get('p99') and lat['p99'] > 20.0:
                self.alerts.append(f"[{timestamp}] ⚠️ Onload P99 latency HIGH: {lat['p99']:.2f}μs (threshold: 20μs)")
            if lat.get('jitter') and lat['jitter'] > 5.0:
                self.alerts.append(f"[{timestamp}] ⚠️ Onload jitter HIGH: {lat['jitter']:.2f}μs (threshold: 5μs)")
        
        # Redis HFT Performance
        if self.metrics.get('redis_hft'):
            redis = self.metrics['redis_hft']
            thresholds = self.config['redis_hft']['thresholds']
            
            # Specific RTT P99 alert with detailed context
            if redis.get('rtt_p99') and redis['rtt_p99'] > thresholds['rtt_p99_max_us']:
                self.alerts.append(f"[{timestamp}] 🔴 Redis RTT P99 EXCEEDED: {redis['rtt_p99']:.2f}μs > {thresholds['rtt_p99_max_us']}μs threshold (P50: {redis.get('rtt_p50', 'N/A'):.2f}μs)")
            if redis.get('rtt_jitter') and redis['rtt_jitter'] > thresholds['rtt_jitter_max_us']:
                self.alerts.append(f"[{timestamp}] ⚠️ Redis jitter HIGH: {redis['rtt_jitter']:.2f}μs > {thresholds['rtt_jitter_max_us']}μs (indicates tail latency variance)")
            if redis.get('ops_per_sec') and redis['ops_per_sec'] < thresholds['ops_per_sec_min']:
                self.alerts.append(f"[{timestamp}] ⚠️ Redis ops/sec LOW: {redis['ops_per_sec']:.0f} < {thresholds['ops_per_sec_min']} threshold")
            if redis.get('performance_gate') == 'FAIL':
                # Enhanced performance gate alert with reason
                failure_reasons = []
                if redis.get('violations'):
                    failure_reasons = redis.get('violations', [])
                reason_str = " | ".join(failure_reasons[:2]) if failure_reasons else "Check logs for details"
                self.alerts.append(f"[{timestamp}] 🚨 PERFORMANCE GATE FAILED - {reason_str}")
        
        # Network RTT (LAN)
        if self.metrics.get('network_rtt_lan'):
            lan = self.metrics['network_rtt_lan']
            if lan.get('avg_us') and lan['avg_us'] > 200.0:
                self.alerts.append(f"[{timestamp}] ⚠️ LAN RTT HIGH: {lan['avg_us']:.0f}μs (threshold: 200μs)")
            if lan.get('jitter_us') and lan['jitter_us'] > 100.0:
                self.alerts.append(f"[{timestamp}] ⚠️ LAN jitter HIGH: {lan['jitter_us']:.0f}μs (threshold: 100μs)")
        
        # 2. LATENCY CONSISTENCY ALERTS (Jitter grading)
        # Already covered above in individual jitter checks
        
        # 3. NETWORK PERFORMANCE & CONFIGURATION ALERTS
        if self.metrics.get('network'):
            for iface, stats in self.metrics['network'].items():
                # Drop rate alerts
                if stats.get('drop_status') in ['CRITICAL', 'WARNING']:
                    drop_rate = stats.get('drop_rate', 0)
                    self.alerts.append(f"[{timestamp}] ⚠️ {iface} packet drops: {drop_rate:.1f}/s ({stats['drop_status']})")
                
                # Traffic anomalies (if available)
                if stats.get('status') == 'CRITICAL':
                    self.alerts.append(f"[{timestamp}] 🚨 {iface} status CRITICAL!")
        
        # HFT NIC Configuration
        if self.metrics.get('network_ull'):
            ull = self.metrics['network_ull']
            checks = ull.get('checks', {})
            
            # Coalescing check
            if checks.get('adaptive_rx', {}).get('status') == 'FAIL':
                self.alerts.append(f"[{timestamp}] ⚠️ NIC coalescing NOT optimal - interrupt coalescing enabled")
            
            # Ring buffer check
            if checks.get('ring_buffers', {}).get('status') == 'FAIL':
                self.alerts.append(f"[{timestamp}] ⚠️ NIC ring buffers NOT optimal - not at target size")
            
            # IRQ isolation check
            if checks.get('irq_violations', {}).get('count', 0) > 0:
                violations_count = checks['irq_violations']['count']
                self.alerts.append(f"[{timestamp}] ⚠️ NIC IRQ isolation violated - {violations_count} IRQs not on isolated cores")
        
        # 4. SYSTEM RESOURCES ALERTS
        if self.metrics.get('system_resources'):
            sys_res = self.metrics['system_resources']
            
            # CPU temperature
            if sys_res.get('cpu_temp_max') and sys_res['cpu_temp_max'] > self.config['cpu']['temp_warning']:
                temp = sys_res['cpu_temp_max']
                threshold = self.config['cpu']['temp_critical'] if temp > self.config['cpu']['temp_critical'] else self.config['cpu']['temp_warning']
                emoji = '🚨' if temp > self.config['cpu']['temp_critical'] else '⚠️'
                self.alerts.append(f"[{timestamp}] {emoji} CPU temp HIGH: {temp:.0f}°C (threshold: {threshold}°C)")
            
            # Memory usage
            if sys_res.get('memory_percent') and sys_res['memory_percent'] > 90:
                self.alerts.append(f"[{timestamp}] ⚠️ Memory usage HIGH: {sys_res['memory_percent']:.1f}% (threshold: 90%)")
            
            # HugePages - alert if NOT CONFIGURED (check total, not in-use count)
            # Note: HugePages may show as "free" even when configured because Onload
            # allocates them on-demand during network I/O, then releases them.
            # We check if they're configured (total > 0), not if actively used.
            if sys_res.get('hugepages_total', 0) == 0:
                # Only add alert if not already present in recent alerts
                recent_hugepages_alerts = [a for a in list(self.alerts)[-10:] if 'HugePages NOT' in a]
                if len(recent_hugepages_alerts) == 0:
                    self.alerts.append(f"[{timestamp}] ⚠️ HugePages NOT configured - allocate with: sysctl -w vm.nr_hugepages=1024")
            
            # PCIe degradation
            if sys_res.get('pcie_status'):
                pcie = sys_res['pcie_status']
                if 'Gen2' in str(pcie) or 'x1' in str(pcie) or 'x2' in str(pcie):
                    self.alerts.append(f"[{timestamp}] ⚠️ PCIe degraded: {pcie} (expected: Gen3 x4 or better)")
            
            # Power anomalies (if available)
            if sys_res.get('power_w') and sys_res['power_w'] > 150:
                self.alerts.append(f"[{timestamp}] ⚠️ Power consumption HIGH: {sys_res['power_w']:.0f}W (threshold: 150W)")
        
        # CPU Isolation violations
        if self.metrics.get('cpu_isolation'):
            cpu_iso = self.metrics['cpu_isolation']
            for core in cpu_iso.get('isolated_cpus', []):
                usage = cpu_iso.get('isolated_cpu_usage', {}).get(core, 0)
                if usage > self.config['thresholds']['cpu_isolated_usage_percent']:
                    self.alerts.append(f"[{timestamp}] ⚠️ Isolated CPU{core} usage HIGH: {usage:.1f}% (threshold: {self.config['thresholds']['cpu_isolated_usage_percent']}%)")
        
        # 5. ACCELERATORS ALERTS (Basic GPU monitoring - for compatibility)
        if self.metrics.get('gpu'):
            for i, gpu in enumerate(self.metrics['gpu']):
                # GPU temperature
                if gpu.get('temperature') and gpu['temperature'] > self.config['gpu']['temp_threshold']:
                    self.alerts.append(f"[{timestamp}] ⚠️ GPU{i} temp HIGH: {gpu['temperature']:.0f}°C (threshold: {self.config['gpu']['temp_threshold']}°C)")
                
                # GPU utilization (for detecting anomalies)
                if gpu.get('utilization') and gpu['utilization'] > self.config['gpu']['util_threshold']:
                    self.alerts.append(f"[{timestamp}] ⚠️ GPU{i} utilization HIGH: {gpu['utilization']:.0f}% (threshold: {self.config['gpu']['util_threshold']}%)")
        
        # 6. GPU OPTIMIZATION ALERTS (Enhanced monitoring)
        if self.metrics.get('gpu_detailed'):
            gpu_data = self.metrics['gpu_detailed']
            gpu_config = self.config.get('gpu_optimization', {})
            
            if gpu_data.get('status') == 'OK' and gpu_data.get('gpus'):
                for gpu in gpu_data['gpus']:
                    gpu_idx = gpu['index']
                    
                    # Temperature alerts (more granular thresholds)
                    temp = gpu['temperature']
                    if temp > gpu_config.get('temp_critical', 83):
                        self.alerts.append(f"[{timestamp}] 🚨 GPU{gpu_idx} CRITICAL temp: {temp:.0f}°C (>{gpu_config['temp_critical']}°C)")
                    elif temp > gpu_config.get('temp_warning', 75):
                        self.alerts.append(f"[{timestamp}] ⚠️ GPU{gpu_idx} temp elevated: {temp:.0f}°C (>{gpu_config['temp_warning']}°C)")
                    
                    # Throttling alerts (filter out benign reasons)
                    if gpu.get('throttling_active', False):
                        throttle_hex = gpu.get('throttle_reasons', 'unknown')
                        
                        # Check if this is benign throttling (idle or clock limits)
                        try:
                            throttle_value = int(throttle_hex, 16)
                            # Benign throttle reasons (normal operation):
                            # 0x0001 = GPU Idle (expected when not in use)
                            # 0x0002 = Application Clock Limit (we locked clocks)
                            # 0x0004 = SW Power Cap (normal on Max-Q - power budget working correctly)
                            # 0x0800 = Sync Boost Limit (normal boost management)
                            BENIGN_FLAGS = 0x0001 | 0x0002 | 0x0004 | 0x0800
                            # Check if ONLY benign flags are set
                            is_benign_only = (throttle_value & ~BENIGN_FLAGS) == 0
                            
                            # Check for critical hardware throttling
                            # Critical flags: 0x0080 (HW thermal), 0x0100 (HW power brake)
                            CRITICAL_FLAGS = 0x0080 | 0x0100
                            is_critical = (throttle_value & CRITICAL_FLAGS) != 0
                        except (ValueError, TypeError):
                            is_benign_only = False
                            is_critical = False
                        
                        # Only alert if there are problematic throttle reasons
                        if not is_benign_only:
                            throttle_reasons = self.decode_throttle_reasons(throttle_hex)
                            reasons_text = "; ".join(throttle_reasons)
                            # Use 🚨 for critical hardware throttling, ⚠️ for warnings
                            alert_icon = "🚨" if is_critical else "⚠️"
                            severity = "CRITICAL" if is_critical else "WARNING"
                            # Get actionable guidance
                            guidance = self.get_throttle_guidance(throttle_value, is_critical)
                            # Construct complete alert with reasons and guidance
                            self.alerts.append(f"[{timestamp}] {alert_icon} GPU{gpu_idx} THROTTLING {severity} - {reasons_text} → ACTION: {guidance}")
                    
                    # Clock drift alerts (only during load)
                    if gpu.get('utilization', 0) > 10:  # Only check during active usage
                        if gpu.get('clocks_status') == 'DRIFT':
                            expected_gr = gpu_config.get('target_graphics_clock_mhz', 3090)
                            expected_mem = gpu_config.get('target_memory_clock_mhz', 14001)
                            actual_gr = gpu['clock_graphics']
                            actual_mem = gpu['clock_memory']
                            self.alerts.append(
                                f"[{timestamp}] ⚠️ GPU{gpu_idx} clock drift: {actual_gr}/{actual_mem} MHz (expected: {expected_gr}/{expected_mem} MHz)"
                            )
                    
                    # Power limit alerts
                    expected_power = gpu_config.get('max_power_limit_w', 325)
                    if gpu['power_limit'] < expected_power:
                        self.alerts.append(
                            f"[{timestamp}] ⚠️ GPU{gpu_idx} power limit LOW: {gpu['power_limit']:.0f}W (expected: {expected_power}W)"
                        )
                
                # VRAM usage alerts (aggregate across all GPUs)
                vram_percent = gpu_data.get('total_vram_percent', 0)
                vram_used_gb = gpu_data.get('total_vram_used_gb', 0)
                
                if vram_percent > gpu_config.get('vram_critical_percent', 95):
                    self.alerts.append(
                        f"[{timestamp}] 🚨 VRAM CRITICAL: {vram_percent:.1f}% ({vram_used_gb:.1f}GB used)"
                    )
                elif vram_percent > gpu_config.get('vram_warning_percent', 90):
                    self.alerts.append(
                        f"[{timestamp}] ⚠️ VRAM elevated: {vram_percent:.1f}% ({vram_used_gb:.1f}GB used)"
                    )
        
        # GPU configuration status alerts
        if self.metrics.get('gpu_optimization_config'):
            gpu_opt_config = self.metrics['gpu_optimization_config']
            if gpu_opt_config.get('status') == 'UNOPTIMIZED':
                violations = gpu_opt_config.get('violations', [])
                for violation in violations[:3]:  # Show top 3 violations
                    self.alerts.append(f"[{timestamp}] ⚠️ GPU config: {violation}")
        
        # PyTorch/CUDA Performance alerts (REAL MONITORING)
        if self.metrics.get('pytorch_performance'):
            pytorch = self.metrics['pytorch_performance']
            
            if pytorch.get('status') == 'OK':
                # Inference latency alerts
                inference = pytorch.get('inference', {})
                p99_ms = inference.get('p99_ms', 0)
                target_ms = inference.get('target_p99_ms', 0.20)
                
                if p99_ms > target_ms:
                    self.alerts.append(
                        f"[{timestamp}] ⚠️ INFERENCE LATENCY HIGH: P99={p99_ms:.3f}ms (target: <{target_ms}ms)"
                    )
                
                # TFLOPS performance alerts (>20% drop from baseline)
                tflops = pytorch.get('tflops', {})
                current_tflops = tflops.get('fp16', 0)
                baseline_tflops = 302.1  # Verified baseline from testing
                tflops_threshold = baseline_tflops * 0.80  # 20% drop threshold
                
                if current_tflops > 0 and current_tflops < tflops_threshold:
                    drop_percent = ((baseline_tflops - current_tflops) / baseline_tflops) * 100
                    self.alerts.append(
                        f"[{timestamp}] ⚠️ GPU PERFORMANCE DROP: {current_tflops:.1f} TFLOPS ({drop_percent:.0f}% below baseline {baseline_tflops:.1f})"
                    )
                
                # torch.compile status - only alert in production mode (when models should be running)
                # In monitoring mode, no models are running so torch.compile inactive is expected
                production_mode = self.config.get('monitoring', {}).get('production_mode', False)
                if production_mode:  # Only care about torch.compile when actually trading
                    compile_status = pytorch.get('compile', {})
                    available = compile_status.get('available', False)
                    active = compile_status.get('active', False)
                    
                    if available and not active:
                        self.alerts.append(
                            f"[{timestamp}] ⚠️ torch.compile NOT ACTIVE (models should be compiled during trading)"
                        )
                
                # Memory fragmentation alerts (already checked in GPU section, but add aggregate)
                memory = pytorch.get('memory', {})
                max_frag = 0
                for gpu_id, mem_stats in memory.items():
                    if isinstance(mem_stats, dict):
                        frag = mem_stats.get('fragmentation_percent', 0)
                        max_frag = max(max_frag, frag)
                
                if max_frag > 5.0:
                    self.alerts.append(
                        f"[{timestamp}] ⚠️ GPU MEMORY FRAGMENTATION HIGH: {max_frag:.1f}% (>5% threshold)"
                    )

    def collect_all_metrics(self):
        """Collect all system metrics - THREAD-SAFE with lock"""
        # Prevent concurrent collections (background worker + manual refresh)
        if not self._collection_lock.acquire(blocking=False):
            self.logger.warning("Metric collection already in progress, skipping concurrent request")
            return self.metrics
        
        try:
            # Collect latency metrics
            self.metrics['latency'] = self.measure_trading_latency()
            
            # Collect network RTT metrics (LAN and Internet)
            self.metrics['network_rtt_lan'] = self.measure_network_rtt_lan()
            self.metrics['network_rtt_internet'] = self.measure_network_rtt_internet()
            
            # Calculate performance grade based on latency
            latency_data = self.metrics['latency']
            if latency_data is not None:
                onload_mean = latency_data.get('mean')
                onload_p99 = latency_data.get('p99')
                self.metrics['performance_grade'] = self.calculate_performance_grade(onload_mean, onload_p99)
            else:
                self.metrics['performance_grade'] = {'grade': 'UNKNOWN', 'emoji': '❓', 'reason': 'No latency data'}
            
            # Collect system metrics
            self.metrics['cpu_isolation'] = self.check_cpu_isolation()
            self.metrics['irq_affinity'] = self.check_irq_affinity()
            self.metrics['cpu_temperature'] = self.check_cpu_temperature()
            self.metrics['system_resources'] = self.check_system_resources()
            
            # Collect network metrics
            self.metrics['network'] = self.check_network_interfaces()
            self.metrics['network_ull'] = self.check_network_ultra_low_latency()
            
            # Collect GPU metrics (basic + detailed)
            self.metrics['gpu'] = self.check_gpu_status()  # Basic (for compatibility)
            self.metrics['gpu_detailed'] = self.check_gpu_vram_detailed()  # Enhanced monitoring
            self.metrics['gpu_optimization_config'] = self.check_gpu_optimization_config()  # Config status
            self.metrics['pytorch_performance'] = self.check_pytorch_performance()  # PyTorch/CUDA metrics
            
            # Collect Redis metrics
            self.metrics['redis_hft'] = self.check_redis_hft()
            
            # Collect QuestDB metrics
            self.metrics['questdb'] = self.check_questdb()
            
            # Generate alerts based on collected metrics
            self.generate_alerts()
            
            self.metrics['collection_timestamp'] = time.time()
            return self.metrics
        finally:
            self._collection_lock.release()
    
    def run_once(self):
        """Run monitoring once and display results"""
        metrics = self.collect_all_metrics()
        
        print("\n" + "="*60)
        print(f"🚀 AI TRADING STATION - SYSTEM STATUS")
        print(f"   Time: {metrics['timestamp']}")
        print("="*60)
        
        # Onload Stack Benchmark (Localhost)
        if metrics['latency']:
            lat = metrics['latency']
            status = "✅" if lat['mean'] < self.config['thresholds']['latency_mean_microseconds'] else "⚠️"
            print(f"\n📊 Onload Stack Benchmark (Localhost Loopback):")
            print(f"   What it measures: Kernel bypass + memory operations (NOT physical network)")
            print(f"   {status} Mean: {lat['mean']:.2f}μs")
            print(f"   Median: {lat['median']:.2f}μs")
            print(f"   P99: {lat['p99']:.2f}μs")
            print(f"   Range: {lat['min']:.2f}μs - {lat['max']:.2f}μs")
            
            if lat['mean'] < 5:
                print(f"   🏆 WORLD-CLASS stack performance!")
        
        # CPU Isolation
        cpu = metrics['cpu_isolation']
        print(f"\n🔧 CPU Isolation (Cores {cpu['isolated_cpus']}):")
        print(f"   Status: {cpu['status']}")
        
        # IRQ Affinity
        irq = metrics['irq_affinity']
        print(f"\n🔌 IRQ Affinity:")
        print(f"   Status: {irq['status']}")
        
        # Network
        print(f"\n🌐 Network Interfaces:")
        for iface, stats in metrics['network'].items():
            print(f"   {iface}: {stats['status']}")
        
        # GPU
        if metrics['gpu']:
            print(f"\n🎮 GPU Status:")
            for i, gpu in enumerate(metrics['gpu']):
                print(f"   GPU{i}: {gpu['temperature']:.0f}°C, {gpu['utilization']:.0f}% load")
        
        print("\n" + "="*60)
        print("✅ Monitoring complete")
        print("="*60)
    
    def run_continuous(self):
        """Run continuous monitoring"""
        interval = self.config['monitoring']['interval_seconds']
        self.logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        while True:
            try:
                self.collect_all_metrics()
                if self.metrics.get('latency'):
                    self.logger.info(f"Latency: {self.metrics['latency']['mean']:.2f}μs")
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='AI Trading Station Optimized Monitor')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()
    
    monitor = TradingSystemMonitor()
    
    if args.once:
        monitor.run_once()
    else:
        monitor.run_continuous()

if __name__ == '__main__':
    main()


