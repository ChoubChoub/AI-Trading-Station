#!/usr/bin/env python3
"""
Binance Trades WebSocket Collector for QuestDB Ingestion Pipeline

Collects real-time trade data from Binance WebSocket API and writes to Redis streams
following the new architecture: market:binance_spot:trades:{symbol}

This collector complements binance_bookticker_collector.py to provide complete
market data (trades + orderbook) for the QuestDB ingestion pipeline.

Architecture:
- WebSocket: wss://stream.binance.com:9443 (combined streams)
- Redis: market:binance_spot:trades:BTCUSDT, etc.
- QuestDB: market_trades table (via redis_to_questdb_v2.py)

Features:
- Auto-reconnection with exponential backoff
- Health monitoring
- systemd service compatible
- Zero data loss design

Author: Claude (Opus)
Date: 2025-10-21
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import redis
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('binance_trades_collector')

# Redis Configuration (environment variable overrides)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_MAXLEN = int(os.getenv("REDIS_MAXLEN", "100000"))

# Binance WebSocket Configuration (environment variable overrides)
# Using combined streams for efficiency (single connection for all symbols)
SYMBOLS = os.getenv("TRADE_SYMBOLS", "BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,ADAUSDT").split(",")
STREAMS = "/".join([f"{symbol.lower()}@trade" for symbol in SYMBOLS])
BINANCE_WS_URL = f"wss://stream.binance.com:9443/stream?streams={STREAMS}"

# Connection parameters (environment variable overrides)
PING_INTERVAL = int(os.getenv("WS_PING_INTERVAL", "20"))  # Binance requires ping every 10 minutes
PING_TIMEOUT = int(os.getenv("WS_PING_TIMEOUT", "10"))
INITIAL_RETRY_DELAY = int(os.getenv("WS_RETRY_DELAY_INITIAL", "1"))  # seconds
MAX_RETRY_DELAY = int(os.getenv("WS_RETRY_DELAY_MAX", "30"))  # seconds

# Performance tracking
@dataclass
class CollectorMetrics:
    """Track collector performance metrics"""
    messages_received: int = 0
    messages_written: int = 0
    errors: int = 0
    reconnections: int = 0
    reconnection_reasons: dict = None
    write_latencies: list = None  # Last 100 write latencies
    start_time: float = 0.0
    last_message_time: float = 0.0
    
    def __post_init__(self):
        self.start_time = time.time()
        self.last_message_time = time.time()
        if self.reconnection_reasons is None:
            self.reconnection_reasons = {}
        if self.write_latencies is None:
            self.write_latencies = []


class BinanceTradesCollector:
    """
    Collects real-time trade data from Binance WebSocket API.
    
    Trade message format from Binance:
    {
        "stream": "btcusdt@trade",
        "data": {
            "e": "trade",          // Event type
            "E": 1638360219763,    // Event time (milliseconds)
            "s": "BTCUSDT",        // Symbol
            "t": 1270683622,       // Trade ID
            "p": "50452.43",       // Price
            "q": "0.00039000",     // Quantity
            "b": 8976191718,       // Buyer order ID
            "a": 8976191717,       // Seller order ID
            "T": 1638360219762,    // Trade time (milliseconds)
            "m": true,             // Is buyer the market maker?
            "M": true              // Ignore
        }
    }
    """
    
    def __init__(self):
        self.redis_client = redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        
        # Test Redis connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            sys.exit(1)
        
        self.running = True
        self.metrics = CollectorMetrics()
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    async def parse_and_write_trade(self, raw_message: str) -> bool:
        """
        Parse trade message and write to Redis stream
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            receive_time = time.time()
            msg = json.loads(raw_message)
            
            # Combined stream format has 'stream' and 'data' fields
            if 'stream' in msg and 'data' in msg:
                data = msg['data']
                
                # Verify this is a trade event
                if data.get('e') != 'trade':
                    return False
                
                # Extract trade data
                symbol = data['s']  # e.g., "BTCUSDT"
                
                # Convert to microseconds (matching orderbook collector)
                # Binance sends milliseconds, we store microseconds
                timestamp_ms = int(data['T'])  # Trade time (not event time)
                timestamp_us = timestamp_ms * 1000
                
                # Determine side based on 'm' flag
                # m = true means buyer is market maker (market sell order)
                # m = false means seller is market maker (market buy order)
                side = "sell" if data['m'] else "buy"
                
                # Prepare Redis stream data
                trade_data = {
                    "timestamp": str(timestamp_us),
                    "symbol": symbol,
                    "exchange": "binance_spot",
                    "price": data['p'],
                    "volume": data['q'],  # Binance uses 'q' for quantity
                    "side": side
                }
                
                # Write to Redis stream (using env var for maxlen)
                write_start = time.time()
                stream_key = f"market:binance_spot:trades:{symbol}"
                self.redis_client.xadd(
                    stream_key,
                    trade_data,
                    maxlen=REDIS_MAXLEN
                )
                write_latency = (time.time() - write_start) * 1000  # milliseconds
                
                # Track latency (keep last 100)
                self.metrics.write_latencies.append(write_latency)
                if len(self.metrics.write_latencies) > 100:
                    self.metrics.write_latencies.pop(0)
                
                self.metrics.messages_written += 1
                
                # Log sample trades periodically (every 1000 trades)
                if self.metrics.messages_written % 1000 == 0:
                    logger.info(
                        f"Sample trade: {symbol} {side} {data['q']} @ {data['p']} "
                        f"(Total: {self.metrics.messages_written:,} trades)"
                    )
                
                return True
                
            else:
                logger.warning(f"Unexpected message format: {raw_message[:100]}...")
                return False
                
        except Exception as e:
            logger.error(f"Error parsing/writing trade: {e}")
            logger.debug(f"Raw message: {raw_message[:200]}...")
            self.metrics.errors += 1
            return False
    
    async def connect_and_subscribe(self):
        """
        Connect to Binance WebSocket and process trade messages
        """
        retry_delay = INITIAL_RETRY_DELAY
        
        while self.running:
            try:
                logger.info(f"Connecting to Binance WebSocket...")
                logger.debug(f"URL: {BINANCE_WS_URL}")
                
                async with websockets.connect(
                    BINANCE_WS_URL,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PING_TIMEOUT,
                    close_timeout=10
                ) as websocket:
                    logger.info(f"Connected! Receiving trades for: {', '.join(SYMBOLS)}")
                    retry_delay = INITIAL_RETRY_DELAY  # Reset on successful connection
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        self.metrics.messages_received += 1
                        self.metrics.last_message_time = time.time()
                        
                        await self.parse_and_write_trade(message)
                        
                        # Print stats every 10 seconds
                        if self.metrics.messages_received % 2000 == 0:
                            self.print_stats()
                    
            except websockets.ConnectionClosed as e:
                reason = f"ConnectionClosed: {e.code if hasattr(e, 'code') else 'unknown'}"
                logger.warning(f"WebSocket connection closed: {e}")
                self.metrics.reconnections += 1
                self.metrics.reconnection_reasons[reason] = self.metrics.reconnection_reasons.get(reason, 0) + 1
                
            except Exception as e:
                reason = f"{type(e).__name__}: {str(e)[:50]}"
                logger.error(f"Connection error: {type(e).__name__}: {e}")
                self.metrics.errors += 1
                self.metrics.reconnections += 1
                self.metrics.reconnection_reasons[reason] = self.metrics.reconnection_reasons.get(reason, 0) + 1
                
            if self.running:
                logger.info(f"Reconnecting in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
    
    async def health_check_loop(self):
        """
        Monitor connection health and print periodic stats
        """
        while self.running:
            await asyncio.sleep(5)
            
            # Check if we're receiving messages
            time_since_last_msg = time.time() - self.metrics.last_message_time
            
            if time_since_last_msg > 30:
                logger.warning(
                    f"⚠️  No messages received for {time_since_last_msg:.1f}s - "
                    f"possible connection issue"
                )
            
            # Print detailed stats every minute
            if int(time.time()) % 60 < 5:
                self.print_detailed_stats()
    
    def print_stats(self):
        """Print basic statistics"""
        elapsed = time.time() - self.metrics.start_time
        rate = self.metrics.messages_written / elapsed if elapsed > 0 else 0
        
        # Calculate average latency
        avg_latency = (sum(self.metrics.write_latencies) / len(self.metrics.write_latencies) 
                      if self.metrics.write_latencies else 0)
        
        logger.info(
            f"Stats: {self.metrics.messages_written:,} trades written, "
            f"{rate:.1f} trades/sec, "
            f"avg latency: {avg_latency:.2f}ms, "
            f"{self.metrics.errors} errors, "
            f"{self.metrics.reconnections} reconnections"
        )
    
    def print_detailed_stats(self):
        """Print detailed statistics"""
        elapsed = time.time() - self.metrics.start_time
        if elapsed == 0:
            return
            
        # Calculate rates
        receive_rate = self.metrics.messages_received / elapsed
        write_rate = self.metrics.messages_written / elapsed
        error_rate = self.metrics.errors / elapsed * 60  # errors per minute
        
        # Calculate efficiency
        efficiency = (self.metrics.messages_written / self.metrics.messages_received * 100 
                     if self.metrics.messages_received > 0 else 0)
        
        # Calculate latency stats
        if self.metrics.write_latencies:
            sorted_latencies = sorted(self.metrics.write_latencies)
            avg_latency = sum(sorted_latencies) / len(sorted_latencies)
            p50_latency = sorted_latencies[len(sorted_latencies) // 2]
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 1 else sorted_latencies[0]
        else:
            avg_latency = p50_latency = p99_latency = 0
        
        # Format reconnection reasons
        reconnection_summary = ""
        if self.metrics.reconnection_reasons:
            reconnection_summary = "\\n  Reasons: " + ", ".join(
                [f"{reason}: {count}x" for reason, count in sorted(
                    self.metrics.reconnection_reasons.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]]  # Top 3 reasons
            )
        
        logger.info(
            f"\\n"
            f"=== DETAILED STATISTICS ===\\n"
            f"Runtime: {elapsed:.1f}s\\n"
            f"Messages received: {self.metrics.messages_received:,} ({receive_rate:.1f}/sec)\\n"
            f"Trades written: {self.metrics.messages_written:,} ({write_rate:.1f}/sec)\\n"
            f"Write efficiency: {efficiency:.1f}%\\n"
            f"Write latency: avg={avg_latency:.2f}ms, p50={p50_latency:.2f}ms, p99={p99_latency:.2f}ms\\n"
            f"Errors: {self.metrics.errors} ({error_rate:.1f}/min)\\n"
            f"Reconnections: {self.metrics.reconnections}{reconnection_summary}\\n"
            f"=========================="
        )
    
    async def health_endpoint_server(self):
        """Simple HTTP health endpoint for load balancer health checks"""
        try:
            from aiohttp import web
            
            async def health_handler(request):
                """Health check endpoint"""
                last_message_age = time.time() - self.metrics.last_message_time
                
                # Healthy if received message in last 60 seconds
                is_healthy = last_message_age < 60 and self.running
                
                health_data = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "last_message_age_seconds": round(last_message_age, 2),
                    "messages_written": self.metrics.messages_written,
                    "reconnections": self.metrics.reconnections,
                    "errors": self.metrics.errors,
                    "uptime_seconds": round(time.time() - self.metrics.start_time, 2)
                }
                
                status = 200 if is_healthy else 503
                return web.json_response(health_data, status=status)
            
            app = web.Application()
            app.router.add_get('/health', health_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            health_port = int(os.getenv("HEALTH_PORT", "8001"))
            site = web.TCPSite(runner, '0.0.0.0', health_port)
            await site.start()
            
            logger.info(f"Health endpoint available at http://0.0.0.0:{health_port}/health")
            
            # Keep server running
            while self.running:
                await asyncio.sleep(1)
                
            await runner.cleanup()
            
        except ImportError:
            logger.warning("aiohttp not installed, health endpoint disabled")
        except Exception as e:
            logger.error(f"Health endpoint error: {e}")
    
    async def run(self):
        """Main run loop"""
        logger.info("Starting Binance Trades Collector...")
        logger.info(f"Collecting trades for: {', '.join(SYMBOLS)}")
        logger.info(f"Writing to Redis streams: market:binance_spot:trades:{{symbol}}")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.connect_and_subscribe()),
            asyncio.create_task(self.health_check_loop()),
            asyncio.create_task(self.health_endpoint_server())
        ]
        
        try:
            # Run until shutdown
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled, shutting down...")
        finally:
            self.running = False
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Final stats
            self.print_detailed_stats()
            logger.info("Binance Trades Collector stopped.")


async def main():
    """Main entry point"""
    collector = BinanceTradesCollector()
    await collector.run()


if __name__ == "__main__":
    # Run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
