#!/usr/bin/env python3
"""
Extended Tail Sampler - Phase 4B
Long-run tail metric collection with p99.9 precision and burst classification

Purpose: Production-ready tail governance for HFT Redis
Author: AI Trading Station Phase 4B
Date: September 28, 2025
"""

import redis
import time
import json
import os
import sys
import psutil
import signal
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics
import subprocess

@dataclass
class TailWindow:
    """Tail measurement window results"""
    window_start: str
    samples: int
    duration_seconds: float
    p50: float
    p95: float
    p99: float
    p99_9: float
    p99_9_confidence: str  # HIGH, MEDIUM, LOW based on sample count
    tail_span: float  # p99.9 - p99
    stability_index: float  # (p99 - p95) / p99
    burst_count: int
    burst_classification: str
    classification_confidence: str
    tail_baseline_ratio: float  # Current tail_span / rolling baseline
    min_latency: float
    max_latency: float
    raw_samples: List[float]  # Limited to first 100 for analysis

@dataclass
class SystemSnapshot:
    """System state snapshot for burst classification"""
    voluntary_ctxt_switches: int
    nonvoluntary_ctxt_switches: int
    minor_faults: int
    major_faults: int
    cpu_percent: float
    memory_percent: float
    timestamp: float

class ExtendedTailSampler:
    """Extended tail sampler for HFT Redis monitoring"""
    
    def __init__(self, 
                 host: str = '127.0.0.1',
                 port: int = 6379,
                 cpu_affinity: Optional[int] = 4,
                 window_size: int = 5000,
                 sampling_interval: float = 0.001):
        
        self.host = host
        self.port = port
        self.cpu_affinity = cpu_affinity
        self.window_size = window_size
        self.sampling_interval = sampling_interval
        
        # State management
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.state_dir = os.path.join(script_dir, "..", "..", "..", "redis", "State")
        self.tail_state_file = os.path.join(self.state_dir, "tail-run.json")
        self.redis_client = None
        self.running = True
        
        # Tail thresholds (from GPT recommendation)
        self.P99_9_MAX_RTT = float(os.getenv('P99_9_MAX_RTT', '20.0'))
        self.TAIL_SPAN_MAX_RTT = float(os.getenv('TAIL_SPAN_MAX_RTT', '8.0'))
        self.TAIL_BURST_LIMIT = int(os.getenv('TAIL_BURST_LIMIT', '6'))
        self.TAIL_BURST_DELTA_US = float(os.getenv('TAIL_BURST_DELTA_US', '6.0'))
        
        # Create state directory
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n📊 Received signal {signum}, shutting down tail sampler...")
        self.running = False
        
    def setup_cpu_affinity(self):
        """Set CPU affinity for consistent measurements"""
        if self.cpu_affinity is not None:
            try:
                p = psutil.Process()
                p.cpu_affinity([self.cpu_affinity])
                print(f"✅ Tail sampler pinned to CPU {self.cpu_affinity}")
            except Exception as e:
                print(f"⚠️  Failed to set CPU affinity: {e}")
    
    def connect_redis(self) -> bool:
        """Establish Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=False
            )
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            return False
    
    def get_system_snapshot(self) -> SystemSnapshot:
        """Capture system state for burst classification"""
        try:
            # Get Redis process stats if available
            redis_pid = None
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'redis-server' in proc.info.get('name', ''):
                    redis_pid = proc.info['pid']
                    break
            
            if redis_pid:
                redis_proc = psutil.Process(redis_pid)
                cpu_percent = redis_proc.cpu_percent()
                memory_percent = redis_proc.memory_percent()
                
                # Get context switches from /proc/pid/status
                with open(f'/proc/{redis_pid}/status', 'r') as f:
                    status = f.read()
                
                vol_switches = 0
                nonvol_switches = 0
                for line in status.split('\n'):
                    if 'voluntary_ctxt_switches' in line:
                        vol_switches = int(line.split()[1])
                    elif 'nonvoluntary_ctxt_switches' in line:
                        nonvol_switches = int(line.split()[1])
                
                # Get memory stats
                mem_info = redis_proc.memory_info()
                minor_faults = getattr(mem_info, 'minorfaults', 0)
                major_faults = getattr(mem_info, 'majorfaults', 0)
                
            else:
                # Fallback to system-wide stats
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                vol_switches = 0
                nonvol_switches = 0
                minor_faults = 0
                major_faults = 0
            
            return SystemSnapshot(
                voluntary_ctxt_switches=vol_switches,
                nonvoluntary_ctxt_switches=nonvol_switches,
                minor_faults=minor_faults,
                major_faults=major_faults,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                timestamp=time.time()
            )
            
        except Exception as e:
            print(f"⚠️  Failed to get system snapshot: {e}")
            return SystemSnapshot(0, 0, 0, 0, 0.0, 0.0, time.time())
    
    def classify_burst(self, 
                      latencies: List[float], 
                      burst_count: int,
                      system_before: SystemSnapshot,
                      system_after: SystemSnapshot) -> Tuple[str, str]:
        """Classify tail burst causes using heuristics"""
        
        if burst_count == 0:
            return "NONE", "HIGH"
        
        # Calculate deltas
        vol_delta = system_after.voluntary_ctxt_switches - system_before.voluntary_ctxt_switches
        nonvol_delta = system_after.nonvoluntary_ctxt_switches - system_before.nonvoluntary_ctxt_switches
        fault_delta = (system_after.minor_faults - system_before.minor_faults + 
                      system_after.major_faults - system_before.major_faults)
        
        # Classification heuristics
        confidence = "MEDIUM"
        
        # High involuntary context switches suggests scheduling pressure
        if nonvol_delta > 10:
            return "SCHED", confidence
        
        # Check for memory/allocation issues
        if fault_delta > 5 or system_after.memory_percent > 90:
            return "ALLOC", confidence
        
        # Check IRQ activity (simplified - could enhance with /proc/interrupts)
        cpu_spike = system_after.cpu_percent > 80
        if cpu_spike and burst_count > self.TAIL_BURST_LIMIT:
            return "IRQ", confidence
        
        # High voluntary context switches might indicate contention
        if vol_delta > 50:
            return "SCHED", "LOW"
        
        # Default to unknown for isolated bursts
        if burst_count <= 2:
            confidence = "LOW"
        
        return "UNKNOWN", confidence
    
    def calculate_p99_9_confidence(self, sample_count: int) -> str:
        """Calculate confidence level for P99.9 measurement based on sample count"""
        if sample_count >= 5000:
            return "HIGH"
        elif sample_count >= 3000:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_tail_baseline_ratio(self, current_tail_span: float) -> float:
        """Calculate current tail span ratio vs rolling baseline"""
        try:
            if os.path.exists(self.tail_state_file):
                with open(self.tail_state_file, 'r') as f:
                    data = json.load(f)
                    windows = data.get('windows', [])
                    
                    if len(windows) >= 12:
                        # Get last 12 windows for baseline
                        recent_spans = [w.get('tail_span', 0) for w in windows[-12:]]
                        baseline_median = statistics.median(recent_spans)
                        
                        if baseline_median > 0:
                            return current_tail_span / baseline_median
            
            # Default to 1.0 if no baseline available
            return 1.0
            
        except Exception:
            return 1.0
    
    def sample_tail_window(self) -> Optional[TailWindow]:
        """Sample a full tail measurement window"""
        print(f"📏 Starting tail window sampling ({self.window_size} samples)...")
        
        window_start = datetime.now().isoformat()
        start_time = time.perf_counter()
        
        # Get initial system state
        system_before = self.get_system_snapshot()
        
        # Collect latency samples
        latencies = []
        failed_samples = 0
        
        for i in range(self.window_size):
            try:
                sample_start = time.perf_counter()
                self.redis_client.ping()
                sample_end = time.perf_counter()
                
                latency_us = (sample_end - sample_start) * 1_000_000
                latencies.append(latency_us)
                
                # Rate limiting (if configured)
                if self.sampling_interval > 0:
                    elapsed = time.perf_counter() - sample_start
                    sleep_time = self.sampling_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Progress indicator
                if (i + 1) % (self.window_size // 10) == 0:
                    progress = (i + 1) / self.window_size * 100
                    print(f"  Progress: {progress:.0f}% ({i+1}/{self.window_size})")
                
                # Check for shutdown signal
                if not self.running:
                    print("🛑 Shutdown requested, stopping sample collection")
                    break
                    
            except Exception as e:
                failed_samples += 1
                if failed_samples > self.window_size * 0.01:  # >1% failure rate
                    print(f"❌ Too many failed samples ({failed_samples}), aborting window")
                    return None
        
        if len(latencies) < self.window_size * 0.95:  # <95% success rate
            print(f"⚠️  Insufficient samples collected ({len(latencies)}/{self.window_size})")
            return None
        
        # Get final system state
        system_after = self.get_system_snapshot()
        
        # Calculate metrics
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        # Percentiles
        p50 = sorted_latencies[int(n * 0.50)]
        p95 = sorted_latencies[int(n * 0.95)]
        p99 = sorted_latencies[int(n * 0.99)]
        p99_9 = sorted_latencies[int(n * 0.999)] if n >= 1000 else sorted_latencies[-1]
        
        # Derived metrics
        tail_span = p99_9 - p99
        stability_index = (p99 - p95) / p99 if p99 > 0 else 0.0
        
        # Burst analysis
        burst_threshold = p99 + self.TAIL_BURST_DELTA_US
        burst_count = sum(1 for lat in latencies if lat > burst_threshold)
        
        # Classify bursts
        burst_classification, classification_confidence = self.classify_burst(
            latencies, burst_count, system_before, system_after
        )
        
        # Calculate confidence and baseline metrics
        p99_9_confidence = self.calculate_p99_9_confidence(n)
        tail_baseline_ratio = self.get_tail_baseline_ratio(tail_span)
        
        # Create window result
        window = TailWindow(
            window_start=window_start,
            samples=n,
            duration_seconds=duration,
            p50=p50,
            p95=p95,
            p99=p99,
            p99_9=p99_9,
            p99_9_confidence=p99_9_confidence,
            tail_span=tail_span,
            stability_index=stability_index,
            burst_count=burst_count,
            burst_classification=burst_classification,
            classification_confidence=classification_confidence,
            tail_baseline_ratio=tail_baseline_ratio,
            min_latency=min(latencies),
            max_latency=max(latencies),
            raw_samples=latencies[:100]  # Store first 100 for analysis
        )
        
        print(f"✅ Window complete: P99={p99:.2f}μs, P99.9={p99_9:.2f}μs, "
              f"Tail Span={tail_span:.2f}μs, Bursts={burst_count}")
        
        return window
    
    def assess_tail_health(self, window: TailWindow) -> Dict[str, str]:
        """Assess tail health against thresholds"""
        issues = []
        
        if window.p99_9 > self.P99_9_MAX_RTT:
            issues.append(f"P99.9 HIGH: {window.p99_9:.2f}μs > {self.P99_9_MAX_RTT}μs")
        
        if window.tail_span > self.TAIL_SPAN_MAX_RTT:
            issues.append(f"TAIL_SPAN HIGH: {window.tail_span:.2f}μs > {self.TAIL_SPAN_MAX_RTT}μs")
        
        if window.burst_count > self.TAIL_BURST_LIMIT:
            issues.append(f"BURST COUNT HIGH: {window.burst_count} > {self.TAIL_BURST_LIMIT}")
        
        if window.burst_classification in ["SCHED", "ALLOC"] and window.confidence != "LOW":
            issues.append(f"CLASSIFIED ISSUE: {window.burst_classification} ({window.confidence})")
        
        if not issues:
            return {"status": "HEALTHY", "message": "All tail metrics within thresholds"}
        else:
            return {"status": "CONCERNING", "issues": issues}
    
    def save_tail_state(self, window: TailWindow):
        """Save tail window to state file with rotation policy"""
        try:
            # Load existing state
            tail_history = []
            baseline_stats = {}
            
            if os.path.exists(self.tail_state_file):
                with open(self.tail_state_file, 'r') as f:
                    data = json.load(f)
                    tail_history = data.get('windows', [])
                    baseline_stats = data.get('baseline_stats', {})
            
            # Add new window
            tail_history.append(asdict(window))
            
            # Implement rotation policy - keep only last N windows
            max_windows = int(os.getenv('TAIL_HISTORY_RETENTION', '48'))
            if len(tail_history) > max_windows:
                # Archive old windows before rotation (optional)
                archived_count = len(tail_history) - max_windows
                print(f"📁 Rotating {archived_count} old tail windows")
                
                # Keep only recent windows
                tail_history = tail_history[-max_windows:]
            
            # Update baseline statistics from recent windows
            if len(tail_history) >= 12:
                recent_spans = [w.get('tail_span', 0) for w in tail_history[-12:]]
                recent_p99s = [w.get('p99', 0) for w in tail_history[-12:]]
                
                baseline_stats = {
                    'tail_span_median': statistics.median(recent_spans),
                    'tail_span_p95': sorted(recent_spans)[int(len(recent_spans) * 0.95)],
                    'p99_median': statistics.median(recent_p99s),
                    'window_count': len(recent_spans),
                    'last_updated': datetime.now().isoformat()
                }
            
            # Save updated state
            state_data = {
                'last_updated': datetime.now().isoformat(),
                'thresholds': {
                    'p99_9_max_rtt': self.P99_9_MAX_RTT,
                    'tail_span_max_rtt': self.TAIL_SPAN_MAX_RTT,
                    'tail_burst_limit': self.TAIL_BURST_LIMIT,
                    'tail_burst_delta_us': self.TAIL_BURST_DELTA_US
                },
                'baseline_stats': baseline_stats,
                'retention_policy': {
                    'max_windows': max_windows,
                    'current_count': len(tail_history)
                },
                'windows': tail_history
            }
            
            # Atomic write
            temp_file = self.tail_state_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            os.rename(temp_file, self.tail_state_file)
            
            print(f"💾 Tail state saved to {self.tail_state_file}")
            
        except Exception as e:
            print(f"❌ Failed to save tail state: {e}")
    
    def run_continuous_sampling(self, window_interval: int = 300):
        """Run continuous tail sampling with specified interval"""
        print(f"🚀 Starting continuous tail sampling")
        print(f"   Window size: {self.window_size} samples")
        print(f"   Interval: {window_interval} seconds")
        print(f"   Thresholds: P99.9<{self.P99_9_MAX_RTT}μs, Tail Span<{self.TAIL_SPAN_MAX_RTT}μs")
        print(f"   Press Ctrl+C to stop gracefully")
        
        self.setup_cpu_affinity()
        
        if not self.connect_redis():
            return False
        
        window_count = 0
        
        while self.running:
            try:
                print(f"\n📊 Starting tail window #{window_count + 1}")
                
                # Sample tail window
                window = self.sample_tail_window()
                
                if window is None:
                    print("❌ Window sampling failed, skipping...")
                    time.sleep(10)  # Brief pause before retry
                    continue
                
                # Assess health
                health = self.assess_tail_health(window)
                
                # Report results
                print(f"📈 Window Results:")
                print(f"   P99: {window.p99:.2f}μs")
                print(f"   P99.9: {window.p99_9:.2f}μs") 
                print(f"   Tail Span: {window.tail_span:.2f}μs")
                print(f"   Bursts: {window.burst_count} ({window.burst_classification})")
                print(f"   Health: {health['status']}")
                
                if health['status'] == 'CONCERNING':
                    print(f"⚠️  Issues detected:")
                    for issue in health.get('issues', []):
                        print(f"     • {issue}")
                
                # Save state
                self.save_tail_state(window)
                
                window_count += 1
                
                # Wait for next window (if continuing)
                if self.running and window_interval > 0:
                    print(f"⏳ Waiting {window_interval}s for next window...")
                    for i in range(window_interval):
                        if not self.running:
                            break
                        time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n🛑 Keyboard interrupt received")
                break
            except Exception as e:
                print(f"❌ Unexpected error in sampling loop: {e}")
                time.sleep(10)
        
        print(f"📊 Tail sampling completed. {window_count} windows collected.")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extended Tail Sampler for HFT Redis')
    parser.add_argument('--host', default='127.0.0.1', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--cpu', type=int, default=4, help='CPU core for affinity')
    parser.add_argument('--window-size', type=int, default=5000, help='Samples per window')
    parser.add_argument('--interval', type=int, default=300, help='Seconds between windows (0=single window)')
    parser.add_argument('--sampling-rate', type=float, default=0.0, help='Delay between samples (0=max rate)')
    
    args = parser.parse_args()
    
    sampler = ExtendedTailSampler(
        host=args.host,
        port=args.port,
        cpu_affinity=args.cpu,
        window_size=args.window_size,
        sampling_interval=args.sampling_rate
    )
    
    if args.interval > 0:
        sampler.run_continuous_sampling(args.interval)
    else:
        # Single window mode
        sampler.setup_cpu_affinity()
        if sampler.connect_redis():
            window = sampler.sample_tail_window()
            if window:
                health = sampler.assess_tail_health(window)
                sampler.save_tail_state(window)
                print(f"\n📊 Single Window Results:")
                print(f"   Health: {health['status']}")
                if health['status'] == 'CONCERNING':
                    for issue in health.get('issues', []):
                        print(f"   Issue: {issue}")

if __name__ == '__main__':
    main()