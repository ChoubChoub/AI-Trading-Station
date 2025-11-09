#!/usr/bin/env python3
# Redis Connection Health Monitor with Prometheus Metrics
# Monitors connection count, latency, and auto-restarts on critical issues
# Exports metrics to Prometheus for Grafana visualization

import asyncio
import subprocess
import time
import logging
from datetime import datetime
from prometheus_client import start_http_server, Gauge, Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus Metrics
redis_connections = Gauge('redis_connections_total', 'Total Redis connections (ESTABLISHED)')
redis_latency_ms = Gauge('redis_latency_milliseconds', 'Redis PING latency in milliseconds')
redis_health_status = Gauge('redis_health_status', 'Redis health status (0=healthy, 1=warning, 2=leak, 3=critical)')
redis_auto_restarts = Counter('redis_auto_restarts_total', 'Number of automatic batch-writer restarts')
redis_connection_leaks_detected = Counter('redis_connection_leaks_total', 'Number of connection leaks detected')

class RedisHealthMonitor:
    def __init__(self, metrics_port=9093):
        self.baseline_connections = 15   # Expected max (fixed from 25 - was double-counted)
        self.leak_threshold = 30         # Definite leak (fixed from 50)
        self.critical_threshold = 50     # Auto-restart threshold (fixed from 100)
        self.check_interval = 60        # Check every minute
        self.alert_cooldown = 300       # 5 min between alerts
        self.last_alert = 0
        self.metrics_port = metrics_port
        
    async def get_connection_count(self) -> int:
        """Get current Redis connection count (using Redis INFO for accuracy)"""
        # Use redis-cli INFO clients (same method as performance gate)
        result = subprocess.run(
            ["redis-cli", "INFO", "clients"],
            capture_output=True,
            text=True,
            timeout=1
        )
        
        for line in result.stdout.split('\n'):
            if line.startswith('connected_clients:'):
                return int(line.split(':')[1].strip())
        
        # Fallback to netstat if redis-cli fails
        # Count only client→server connections (where 6379 is remote/foreign port)
        result = subprocess.run(
            ["netstat", "-tn"], 
            capture_output=True, 
            text=True
        )
        
        established = [l for l in result.stdout.split('\n') 
                      if 'ESTABLISHED' in l and l.split()[4:5] and ':6379' in l.split()[4]]
        return len(established)
    
    async def check_redis_latency(self) -> float:
        """Test Redis response time"""
        start = time.perf_counter()
        result = subprocess.run(
            ["redis-cli", "PING"],
            capture_output=True,
            text=True,
            timeout=1
        )
        latency_ms = (time.perf_counter() - start) * 1000
        
        if "PONG" not in result.stdout:
            return -1  # Redis not responding
        return latency_ms
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        
        # Start Prometheus metrics server
        try:
            start_http_server(self.metrics_port)
            logger.info(f"� Prometheus metrics server started on port {self.metrics_port}")
        except OSError as e:
            logger.warning(f"Metrics server port {self.metrics_port} already in use, continuing without metrics: {e}")
        
        logger.info("�🚀 Redis Health Monitor started")
        logger.info(f"   Baseline: {self.baseline_connections} connections")
        logger.info(f"   Leak threshold: {self.leak_threshold} connections")
        logger.info(f"   Critical threshold: {self.critical_threshold} connections")
        
        consecutive_issues = 0
        
        while True:
            try:
                # Check connection count
                conn_count = await self.get_connection_count()
                
                # Check Redis latency
                latency = await self.check_redis_latency()
                
                # Update Prometheus metrics
                redis_connections.set(conn_count)
                if latency != -1:
                    redis_latency_ms.set(latency)
                
                # Determine health status
                health_status = 0  # healthy
                status = "✅ HEALTHY"
                if conn_count > self.baseline_connections:
                    status = "⚠️ WARNING"
                    health_status = 1
                if conn_count > self.leak_threshold:
                    status = "🚨 LEAK DETECTED"
                    health_status = 2
                if conn_count > self.critical_threshold:
                    health_status = 3
                
                redis_health_status.set(health_status)
                
                logger.info(f"{status} | Connections: {conn_count} | Latency: {latency:.1f}ms")
                
                # Handle issues
                if conn_count > self.critical_threshold:
                    logger.critical(f"CRITICAL: {conn_count} connections! Auto-restarting...")
                    redis_auto_restarts.inc()
                    subprocess.run(["sudo", "systemctl", "restart", "batch-writer.service"])
                    await asyncio.sleep(30)  # Wait for restart
                    
                elif conn_count > self.leak_threshold:
                    consecutive_issues += 1
                    
                    if consecutive_issues >= 3:  # 3 consecutive high readings
                        if time.time() - self.last_alert > self.alert_cooldown:
                            logger.warning(f"CONNECTION LEAK: {conn_count} connections for 3+ minutes")
                            redis_connection_leaks_detected.inc()
                            self.last_alert = time.time()
                else:
                    consecutive_issues = 0
                
                # Check latency issues
                if latency > 5.0 and latency != -1:
                    logger.warning(f"HIGH LATENCY: Redis responded in {latency:.1f}ms")
                elif latency == -1:
                    logger.error("Redis not responding to PING!")
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            await asyncio.sleep(self.check_interval)

async def main():
    monitor = RedisHealthMonitor()
    await monitor.monitor_loop()

if __name__ == "__main__":
    asyncio.run(main())
