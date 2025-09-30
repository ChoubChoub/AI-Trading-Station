#!/usr/bin/env python3
"""
Tail-Aware Performance Gate Enhancement
Phase 4B - Integrates tail metrics into existing performance gate

Purpose: Add P99.9, tail span, and burst classification to gate checks
Author: AI Trading Station Phase 4B  
Date: September 28, 2025
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TailAssessment:
    """Tail health assessment result"""
    status: str  # HEALTHY, WARNING, CONCERNING
    issues: List[str]
    metrics: Dict[str, float]
    classification: str
    confidence: str

class TailAwareGate:
    """Tail-aware performance gate checker"""
    
    def __init__(self):
        # Load tail thresholds from environment
        self.p99_9_max_rtt = float(os.getenv('P99_9_MAX_RTT', '20.0'))
        self.tail_span_max_rtt = float(os.getenv('TAIL_SPAN_MAX_RTT', '8.0'))
        self.tail_burst_limit = int(os.getenv('TAIL_BURST_LIMIT', '3'))
        self.tail_burst_delta_us = float(os.getenv('TAIL_BURST_DELTA_US', '6.0'))
        
        # Gate behavior controls
        self.tail_gate_enabled = os.getenv('TAIL_GATE_ENABLED', 'false').lower() == 'true'
        self.tail_gate_warn_only = os.getenv('TAIL_GATE_WARN_ONLY', 'true').lower() == 'true'
        self.consecutive_threshold = int(os.getenv('TAIL_GATE_CONSECUTIVE_THRESHOLD', '2'))
        
        # State file for historical tracking
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.tail_state_file = os.path.join(script_dir, '..', 'state', 'tail-run.json')
    
    def load_tail_state(self) -> Optional[Dict]:
        """Load tail state history"""
        try:
            if os.path.exists(self.tail_state_file):
                with open(self.tail_state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load tail state: {e}", file=sys.stderr)
        return None
    
    def assess_current_metrics(self, metrics: Dict) -> TailAssessment:
        """Assess current performance metrics for tail health"""
        issues = []
        
        # Extract RTT metrics
        rtt_data = metrics.get('rtt', {})
        if 'error' in rtt_data:
            return TailAssessment(
                status="WARNING",
                issues=["RTT measurement failed"],
                metrics={},
                classification="MEASUREMENT_ERROR",
                confidence="HIGH"
            )
        
        # Basic metrics
        p99 = rtt_data.get('p99', 0)
        jitter = rtt_data.get('jitter', 0)
        
        # Calculate derived tail metrics (if available)
        # Note: Current monitor doesn't provide p99.9, so we estimate
        p99_9_estimate = p99 + (jitter * 1.5)  # Conservative estimate
        tail_span_estimate = jitter * 1.5
        
        tail_metrics = {
            'p99': p99,
            'p99_9_estimate': p99_9_estimate,
            'tail_span_estimate': tail_span_estimate,
            'jitter': jitter
        }
        
        # Assess against thresholds
        if p99_9_estimate > self.p99_9_max_rtt:
            issues.append(f"P99.9 estimate HIGH: {p99_9_estimate:.2f}μs > {self.p99_9_max_rtt}μs")
        
        if tail_span_estimate > self.tail_span_max_rtt:
            issues.append(f"Tail span estimate HIGH: {tail_span_estimate:.2f}μs > {self.tail_span_max_rtt}μs")
        
        # Classify based on jitter patterns
        classification = "UNKNOWN"
        confidence = "LOW"
        
        if jitter > 8:
            classification = "HIGH_JITTER"
            confidence = "MEDIUM"
        elif jitter > 4:
            classification = "MODERATE_JITTER"
            confidence = "MEDIUM"
        else:
            classification = "STABLE"
            confidence = "HIGH"
        
        # Determine overall status
        if not issues:
            status = "HEALTHY"
        elif len(issues) == 1 and "estimate" in issues[0]:
            status = "WARNING"
        else:
            status = "CONCERNING"
        
        return TailAssessment(
            status=status,
            issues=issues,
            metrics=tail_metrics,
            classification=classification,
            confidence=confidence
        )
    
    def assess_tail_history(self) -> Optional[TailAssessment]:
        """Assess tail health from historical data"""
        tail_state = self.load_tail_state()
        
        if not tail_state or 'windows' not in tail_state:
            return None
        
        windows = tail_state['windows']
        if not windows:
            return None
        
        # Get recent windows (last 3)
        recent_windows = windows[-3:]
        issues = []
        
        # Check for consecutive tail issues
        consecutive_span_breaches = 0
        consecutive_p99_9_breaches = 0
        
        for window in recent_windows:
            if window.get('tail_span', 0) > self.tail_span_max_rtt:
                consecutive_span_breaches += 1
            if window.get('p99_9', 0) > self.p99_9_max_rtt:
                consecutive_p99_9_breaches += 1
        
        # Assess patterns
        if consecutive_span_breaches >= self.consecutive_threshold:
            issues.append(f"Persistent tail span issues: {consecutive_span_breaches} consecutive breaches")
        
        if consecutive_p99_9_breaches >= self.consecutive_threshold:
            issues.append(f"Persistent P99.9 issues: {consecutive_p99_9_breaches} consecutive breaches")
        
        # Get latest window metrics
        latest_window = recent_windows[-1]
        tail_metrics = {
            'p99': latest_window.get('p99', 0),
            'p99_9': latest_window.get('p99_9', 0),
            'tail_span': latest_window.get('tail_span', 0),
            'burst_count': latest_window.get('burst_count', 0)
        }
        
        # Classification from historical data
        classification = latest_window.get('burst_classification', 'UNKNOWN')
        confidence = latest_window.get('confidence', 'LOW')
        
        # Determine status
        if not issues:
            status = "HEALTHY"
        elif consecutive_span_breaches >= self.consecutive_threshold:
            status = "CONCERNING"
        else:
            status = "WARNING"
        
        return TailAssessment(
            status=status,
            issues=issues,
            metrics=tail_metrics,
            classification=classification,
            confidence=confidence
        )
    
    def check_tail_health(self, metrics: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Main tail health check function
        Returns: (should_pass, failures, warnings)
        """
        failures = []
        warnings = []
        
        if not self.tail_gate_enabled:
            return True, failures, warnings
        
        # Assess current metrics
        current_assessment = self.assess_current_metrics(metrics)
        
        # Assess historical data if available
        historical_assessment = self.assess_tail_history()
        
        # Generate warnings/failures based on assessments
        if current_assessment.status == "CONCERNING":
            message = f"Current tail health: {current_assessment.status}"
            if self.tail_gate_warn_only:
                warnings.append(message)
                warnings.extend(current_assessment.issues)
            else:
                failures.append(message)
                failures.extend(current_assessment.issues)
        
        elif current_assessment.status == "WARNING":
            warnings.append(f"Current tail metrics elevated")
            warnings.extend(current_assessment.issues)
        
        # Historical assessment
        if historical_assessment and historical_assessment.status == "CONCERNING":
            message = f"Historical tail pattern: {historical_assessment.status}"
            if self.tail_gate_warn_only:
                warnings.append(message)
                warnings.extend(historical_assessment.issues)
            else:
                failures.append(message)
                failures.extend(historical_assessment.issues)
        
        # Log tail metrics for visibility
        if current_assessment.metrics:
            metrics_info = []
            for key, value in current_assessment.metrics.items():
                metrics_info.append(f"{key}={value:.2f}μs")
            warnings.append(f"Tail metrics: {', '.join(metrics_info)}")
        
        # Overall pass/fail decision
        should_pass = len(failures) == 0
        
        return should_pass, failures, warnings

def main():
    """Main function for standalone testing"""
    if len(sys.argv) != 2:
        print("Usage: python3 tail_aware_gate.py <metrics_file>")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        sys.exit(1)
    
    # Initialize tail gate
    gate = TailAwareGate()
    
    # Check tail health
    should_pass, failures, warnings = gate.check_tail_health(metrics)
    
    # Output results
    if failures:
        print("TAIL FAILURES:")
        for failure in failures:
            print(f"  ❌ {failure}")
    
    if warnings:
        print("TAIL WARNINGS:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    
    if should_pass and not warnings:
        print("✅ Tail health: PASS")
    
    # Exit code
    sys.exit(0 if should_pass else 1)

if __name__ == '__main__':
    main()