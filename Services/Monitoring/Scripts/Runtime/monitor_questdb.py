#!/usr/bin/env python3
"""
QuestDB Monitoring Module
Collects performance metrics for QuestDB time-series database
Integrated with AI Trading Station monitoring system
"""

import os
import psutil
import subprocess
import json
from datetime import datetime
from pathlib import Path


class QuestDBMonitor:
    """Monitor QuestDB performance and storage metrics"""
    
    def __init__(self):
        self.host = "localhost"
        self.port = 9000
        self.hot_storage = "/home/youssefbahloul/ai-trading-station/Services/QuestDB/data/hot"
        self.cold_storage = "/mnt/hdd/questdb/cold"
        self.process_name = "java"  # QuestDB runs as Java process
        self.service_name = "questdb.service"
        
    def get_process_info(self):
        """Get QuestDB process information"""
        try:
            # Find QuestDB process by checking command line
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
                try:
                    cmdline = ' '.join(proc.info.get('cmdline', []))
                    if 'questdb' in cmdline.lower() and 'ServerMain' in cmdline:
                        return {
                            'pid': proc.info['pid'],
                            'memory_gb': proc.info['memory_info'].rss / (1024**3),
                            'cpu_percent': proc.info['cpu_percent'],
                            'status': 'running'
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {'status': 'not_found', 'memory_gb': 0, 'cpu_percent': 0}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'memory_gb': 0, 'cpu_percent': 0}
    
    def get_service_status(self):
        """Get systemd service status"""
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', self.service_name],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.stdout.strip()  # 'active', 'inactive', etc.
        except Exception:
            return 'unknown'
    
    def get_storage_metrics(self):
        """Get storage usage for hot and cold storage"""
        metrics = {}
        
        # Hot storage (NVMe)
        try:
            hot_path = Path(self.hot_storage)
            if hot_path.exists():
                stat = os.statvfs(hot_path)
                total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
                used_gb = ((stat.f_blocks - stat.f_bfree) * stat.f_frsize) / (1024**3)
                avail_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                
                metrics['hot'] = {
                    'total_gb': round(total_gb, 2),
                    'used_gb': round(used_gb, 2),
                    'available_gb': round(avail_gb, 2),
                    'usage_percent': round((used_gb / total_gb) * 100, 1),
                    'path': str(hot_path)
                }
        except Exception as e:
            metrics['hot'] = {'error': str(e)}
        
        # Cold storage (HDD)
        try:
            cold_path = Path(self.cold_storage)
            if cold_path.exists():
                stat = os.statvfs(cold_path)
                total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
                used_gb = ((stat.f_blocks - stat.f_bfree) * stat.f_frsize) / (1024**3)
                avail_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                
                metrics['cold'] = {
                    'total_gb': round(total_gb, 2),
                    'used_gb': round(used_gb, 2),
                    'available_gb': round(avail_gb, 2),
                    'usage_percent': round((used_gb / total_gb) * 100, 1),
                    'path': str(cold_path)
                }
        except Exception as e:
            metrics['cold'] = {'error': str(e)}
        
        return metrics
    
    def get_table_count(self):
        """Count tables by checking hot storage directory"""
        try:
            db_path = Path(self.hot_storage) / "db"
            if db_path.exists():
                # Count directories in db/ (each is a table)
                tables = [d for d in db_path.iterdir() if d.is_dir()]
                return len(tables)
            return 0
        except Exception:
            return 0
    
    def test_connectivity(self):
        """Test if QuestDB is responding to queries"""
        try:
            result = subprocess.run(
                ['curl', '-s', '--max-time', '3', '-G', 
                 f'http://{self.host}:{self.port}/exec',
                 '--data-urlencode', 'query=SELECT 1'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'dataset' in result.stdout:
                return {'status': 'ok', 'response_time_ms': '<1000'}
            else:
                return {'status': 'error', 'message': 'Invalid response'}
        except subprocess.TimeoutExpired:
            return {'status': 'timeout'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_nvme_io_stats(self):
        """Get NVMe I/O statistics"""
        try:
            result = subprocess.run(
                ['iostat', '-x', '-d', 'nvme0n1', '1', '2'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Parse last line with actual metrics
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if 'nvme0n1' in line:
                    parts = line.split()
                    if len(parts) >= 14:
                        return {
                            'read_mb_s': float(parts[5]) / 1024,  # Convert KB/s to MB/s
                            'write_mb_s': float(parts[6]) / 1024,
                            'utilization_percent': float(parts[-1])
                        }
            return {'error': 'Could not parse iostat output'}
        except Exception as e:
            return {'error': str(e)}
    
    def collect_all_metrics(self):
        """Collect all QuestDB metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'process': self.get_process_info(),
            'service': {
                'status': self.get_service_status()
            },
            'storage': self.get_storage_metrics(),
            'tables': {
                'count': self.get_table_count()
            },
            'connectivity': self.test_connectivity(),
            'nvme': self.get_nvme_io_stats(),
            'config': {
                'hot_storage': self.hot_storage,
                'cold_storage': self.cold_storage,
                'allocated_memory_gb': 96
            }
        }


def main():
    """Standalone test of QuestDB monitoring"""
    monitor = QuestDBMonitor()
    
    print("=" * 80)
    print("QuestDB Performance Monitor - Test Run")
    print("=" * 80)
    
    metrics = monitor.collect_all_metrics()
    print(json.dumps(metrics, indent=2))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    proc = metrics['process']
    print(f"Process Status: {proc.get('status', 'unknown')}")
    if proc.get('pid'):
        print(f"PID: {proc['pid']}")
        print(f"Memory: {proc['memory_gb']:.2f} GB")
        print(f"CPU: {proc['cpu_percent']:.1f}%")
    
    print(f"\nService Status: {metrics['service']['status']}")
    print(f"Connectivity: {metrics['connectivity']['status']}")
    print(f"Tables: {metrics['tables']['count']}")
    
    storage = metrics['storage']
    if 'hot' in storage and 'error' not in storage['hot']:
        hot = storage['hot']
        print(f"\nHot Storage (NVMe): {hot['used_gb']:.1f} GB / {hot['total_gb']:.1f} GB ({hot['usage_percent']:.1f}%)")
    
    if 'cold' in storage and 'error' not in storage['cold']:
        cold = storage['cold']
        print(f"Cold Storage (HDD): {cold['used_gb']:.1f} GB / {cold['total_gb']:.1f} GB ({cold['usage_percent']:.1f}%)")
    
    if 'error' not in metrics['nvme']:
        nvme = metrics['nvme']
        print(f"\nNVMe I/O: Read {nvme['read_mb_s']:.1f} MB/s, Write {nvme['write_mb_s']:.1f} MB/s, Util {nvme['utilization_percent']:.1f}%")


if __name__ == "__main__":
    main()
