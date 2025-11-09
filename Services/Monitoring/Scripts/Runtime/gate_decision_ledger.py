#!/usr/bin/env python3
"""
Gate Decision Ledger - Phase 4B Enhancement
Lightweight append-only audit trail for performance gate decisions

Purpose: Forensic trail for gate decisions with tail summary
Author: AI Trading Station Phase 4B
Date: September 28, 2025
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Optional

class GateDecisionLedger:
    """Lightweight append-only ledger for gate decisions"""
    
    def __init__(self, ledger_path: str = None):
        if ledger_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Updated to use centralized Monitoring/logs directory
            ledger_path = os.path.join(script_dir, "..", "..", "logs", "gate-decisions.log")
        self.ledger_path = ledger_path
        
        # Ensure log directory exists
        log_dir = os.path.dirname(self.ledger_path)
        os.makedirs(log_dir, exist_ok=True)
        
    def log_decision(self, 
                    gate_result: str,
                    performance_metrics: Dict,
                    tail_summary: Optional[Dict] = None,
                    fingerprint_hash: Optional[str] = None,
                    decision_reason: str = "",
                    failures: list = None,
                    warnings: list = None,
                    thresholds: Dict = None):
        """Log a gate decision with context"""
        
        timestamp = datetime.now().isoformat()
        
        # Extract key metrics for compact logging
        perf_summary = {}
        if 'set' in performance_metrics:
            perf_summary['set_p99'] = performance_metrics['set'].get('p99', 0)
            perf_summary['set_jitter'] = performance_metrics['set'].get('jitter', 0)
        if 'xadd' in performance_metrics:
            perf_summary['xadd_p99'] = performance_metrics['xadd'].get('p99', 0)
            perf_summary['xadd_jitter'] = performance_metrics['xadd'].get('jitter', 0)
        if 'rtt' in performance_metrics:
            perf_summary['rtt_p99'] = performance_metrics['rtt'].get('p99', 0)
            perf_summary['rtt_jitter'] = performance_metrics['rtt'].get('jitter', 0)
        if 'health' in performance_metrics:
            perf_summary['ops_per_sec'] = performance_metrics['health'].get('ops_per_sec', 0)
            perf_summary['clients'] = performance_metrics['health'].get('clients', 0)
        
        # Create decision record with enhanced failure tracking
        decision_record = {
            'timestamp': timestamp,
            'gate_result': gate_result,
            'performance_summary': perf_summary,
            'thresholds': thresholds or {},
            'failures': failures or [],
            'warnings': warnings or [],
            'tail_summary': tail_summary or {},
            'fingerprint_hash': fingerprint_hash,
            'decision_reason': decision_reason,
            'epoch': int(time.time())
        }
        
        # Append to ledger (atomic write)
        try:
            with open(self.ledger_path, 'a') as f:
                f.write(json.dumps(decision_record) + '\n')
        except Exception as e:
            # Silent fail - don't break gate operation
            pass
    
    def get_recent_decisions(self, count: int = 10) -> list:
        """Get recent gate decisions for analysis"""
        decisions = []
        
        try:
            if os.path.exists(self.ledger_path):
                with open(self.ledger_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines[-count:]:
                    try:
                        decision = json.loads(line.strip())
                        decisions.append(decision)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception:
            pass
            
        return decisions
    
    def rotate_ledger(self, max_entries: int = 1000):
        """Rotate ledger if it gets too large"""
        try:
            if not os.path.exists(self.ledger_path):
                return
                
            with open(self.ledger_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > max_entries:
                # Keep only recent entries
                recent_lines = lines[-max_entries:]
                
                # Write back
                with open(self.ledger_path, 'w') as f:
                    f.writelines(recent_lines)
                    
                print(f"📁 Rotated gate ledger: kept {len(recent_lines)} recent entries")
                
        except Exception:
            pass

# Standalone utility functions
def log_gate_decision(gate_result: str, 
                     performance_file: str,
                     tail_state_file: Optional[str] = None,
                     decision_reason: str = "",
                     gate_details_json: str = None):
    """Utility function to log gate decision from files"""
    
    ledger = GateDecisionLedger()
    
    # Load performance metrics
    performance_metrics = {}
    try:
        with open(performance_file, 'r') as f:
            performance_metrics = json.load(f)
    except Exception:
        pass
    
    # Parse gate details if provided (contains failures, warnings, thresholds)
    failures = []
    warnings = []
    thresholds = {}
    if gate_details_json:
        try:
            gate_details = json.loads(gate_details_json)
            failures = gate_details.get('failures', [])
            warnings = gate_details.get('warnings', [])
            thresholds = gate_details.get('thresholds', {})
        except Exception:
            pass
    
    # Load tail summary if available
    tail_summary = None
    if tail_state_file and os.path.exists(tail_state_file):
        try:
            with open(tail_state_file, 'r') as f:
                tail_data = json.load(f)
                windows = tail_data.get('windows', [])
                
                if windows:
                    latest_window = windows[-1]
                    tail_summary = {
                        'p99_9': latest_window.get('p99_9'),
                        'tail_span': latest_window.get('tail_span'),
                        'burst_count': latest_window.get('burst_count'),
                        'classification': latest_window.get('burst_classification'),
                        'confidence': latest_window.get('p99_9_confidence'),
                        'baseline_ratio': latest_window.get('tail_baseline_ratio', 1.0)
                    }
        except Exception:
            pass
    
    # Log the decision
    ledger.log_decision(
        gate_result=gate_result,
        performance_metrics=performance_metrics,
        tail_summary=tail_summary,
        decision_reason=decision_reason,
        failures=failures,
        warnings=warnings,
        thresholds=thresholds
    )

def main():
    """Command line interface for gate decision logging"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gate Decision Ledger')
    parser.add_argument('--result', choices=['PASS', 'SOFT_FAIL', 'HARD_FAIL'], 
                       help='Gate result')
    parser.add_argument('--metrics', help='Performance metrics JSON file')
    parser.add_argument('--tail-state', help='Tail state JSON file')
    parser.add_argument('--reason', default='', help='Decision reason')
    parser.add_argument('--gate-details', help='Gate details JSON string with failures/warnings/thresholds')
    parser.add_argument('--show-recent', type=int, help='Show recent N decisions')
    parser.add_argument('--rotate', action='store_true', help='Rotate ledger')
    
    args = parser.parse_args()
    
    ledger = GateDecisionLedger()
    
    if args.show_recent:
        decisions = ledger.get_recent_decisions(args.show_recent)
        for decision in decisions:
            print(f"{decision['timestamp']}: {decision['gate_result']} - {decision.get('decision_reason', '')}")
    
    elif args.rotate:
        ledger.rotate_ledger()
    
    else:
        log_gate_decision(args.result, args.metrics, args.tail_state, args.reason, args.gate_details)
        print(f"✅ Logged gate decision: {args.result}")

if __name__ == '__main__':
    main()