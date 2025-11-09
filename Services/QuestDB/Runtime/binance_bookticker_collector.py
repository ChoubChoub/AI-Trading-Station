#!/usr/bin/env python3
"""
Binance BookTicker WebSocket Collector

Collects real-time bookTicker (L1 orderbook) data from Binance Spot and writes
to Redis streams for consumption by QuestDB batch writer.

Features:
- Multi-symbol subscription (combined stream)
- Automatic reconnection with exponential backoff
- Health monitoring and metrics
- Compatible with redis_to_questdb_v2.py batch writer

Usage:
    python3 binance_bookticker_collector.py

Output Format (Redis):
    Stream: market:binance_spot:orderbook:{symbol}
    Fields: timestamp, symbol, exchange, bid_price, bid_qty, ask_price, ask_qty
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional
import redis.asyncio as redis
import websockets
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
# Use combined streams endpoint (no subscription needed)
SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "adausdt"]
STREAMS = [f"{s}@bookTicker" for s in SYMBOLS]
BINANCE_WS_URL = f"wss://stream.binance.com:9443/stream?streams={'/'.join(STREAMS)}"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_MAX_LEN = 100000  # Max entries per stream

# Reconnection settings
INITIAL_RETRY_DELAY = 1  # seconds
MAX_RETRY_DELAY = 30  # seconds
PING_INTERVAL = 20  # seconds
PONG_TIMEOUT = 10  # seconds

# Metrics
@dataclass
class CollectorMetrics:
    """Track collector performance metrics"""
    messages_received: int = 0
    messages_written: int = 0
    errors: int = 0
    reconnections: int = 0
    last_message_time: float = 0
    start_time: float = 0
    messages_by_symbol: Dict[str, int] = None  # Per-symbol message tracking
    
    def __post_init__(self):
        self.start_time = time.time()
        self.last_message_time = time.time()
        if self.messages_by_symbol is None:
            self.messages_by_symbol = {}
    
    def log_stats(self):
        """Log current statistics"""
        uptime = time.time() - self.start_time
        msg_rate = self.messages_received / uptime if uptime > 0 else 0
        
        # Base stats
        logger.info(
            f"Stats: {self.messages_received:,} msgs received | "
            f"{self.messages_written:,} written | "
            f"{msg_rate:.1f} msgs/sec | "
            f"{self.errors} errors | "
            f"{self.reconnections} reconnections"
        )
        
        # Per-symbol breakdown (every 5000 messages)
        if self.messages_received % 5000 == 0 and self.messages_by_symbol:
            symbol_stats = ", ".join([f"{s.upper()}: {count}" for s, count in sorted(self.messages_by_symbol.items())])
            logger.info(f"  Per-symbol: {symbol_stats}")


class BinanceBookTickerCollector:
    """Collects bookTicker data from Binance and writes to Redis"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = [s.lower() for s in symbols]
        self.redis_client: Optional[redis.Redis] = None
        self.websocket = None
        self.metrics = CollectorMetrics()
        self.running = False
        
    async def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = await redis.from_url(
                f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
                encoding="utf-8",
                decode_responses=False
            )
            await self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def disconnect_redis(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("✅ Disconnected from Redis")
    
    def build_subscribe_message(self) -> Optional[str]:
        """
        Build WebSocket subscription message.
        Note: Using combined stream endpoint, so no subscription needed.
        """
        return None  # Not needed for combined streams
    
    async def parse_and_write_message(self, raw_message: str) -> bool:
        """Parse bookTicker message and write to Redis"""
        try:
            msg = json.loads(raw_message)
            
            # Debug: Log first few messages
            if self.metrics.messages_received < 3:
                logger.info(f"📩 Received message: {json.dumps(msg, indent=2)[:200]}")
            
            # Handle bookTicker data from combined stream
            # Format: {"stream": "btcusdt@bookTicker", "data": {...}}
            if 'stream' in msg and 'data' in msg:
                data = msg['data']
                
                # Parse bookTicker data
                symbol = data['s']  # e.g., "BTCUSDT"
                bid_price = float(data['b'])
                bid_qty = float(data['B'])
                ask_price = float(data['a'])
                ask_qty = float(data['A'])
                
                # Generate timestamp (microseconds)
                timestamp_us = int(time.time() * 1_000_000)
                ws_receive_time = int(time.time() * 1_000_000)  # Capture receive time for latency analysis
                
                # Prepare Redis stream entry
                stream_key = f"market:binance_spot:orderbook:{symbol}"
                entry = {
                    "timestamp": timestamp_us,
                    "symbol": symbol,
                    "exchange": "binance_spot",
                    "bid_price": bid_price,
                    "bid_qty": bid_qty,
                    "ask_price": ask_price,
                    "ask_qty": ask_qty,
                    "ws_receive_time": ws_receive_time  # Track when we received from WebSocket
                }
                
                # Write to Redis stream
                await self.redis_client.xadd(stream_key, entry, maxlen=REDIS_MAX_LEN)
                
                self.metrics.messages_received += 1
                self.metrics.messages_written += 1
                self.metrics.last_message_time = time.time()
                
                # Track per-symbol metrics
                if symbol not in self.metrics.messages_by_symbol:
                    self.metrics.messages_by_symbol[symbol] = 0
                self.metrics.messages_by_symbol[symbol] += 1
                
                # Log stats every 1000 messages
                if self.metrics.messages_received % 1000 == 0:
                    self.metrics.log_stats()
                
                return True
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            self.metrics.errors += 1
            return False
        except Exception as e:
            logger.error(f"❌ Failed to process message: {e}")
            self.metrics.errors += 1
            return False
    
    async def health_check_loop(self):
        """Monitor connection health and restart if needed"""
        while self.running:
            await asyncio.sleep(5)
            
            # Check if we've received messages recently
            time_since_last_msg = time.time() - self.metrics.last_message_time
            
            if time_since_last_msg > 30:
                logger.warning(
                    f"⚠️  No messages received for {time_since_last_msg:.1f}s - "
                    "connection may be stale"
                )
    
    async def connect_and_subscribe(self):
        """Connect to Binance WebSocket and subscribe to streams"""
        retry_delay = INITIAL_RETRY_DELAY
        
        while self.running:
            try:
                logger.info(f"🔌 Connecting to Binance WebSocket...")
                
                async with websockets.connect(
                    BINANCE_WS_URL,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PONG_TIMEOUT,
                    close_timeout=10  # Cleaner shutdown handling
                ) as websocket:
                    self.websocket = websocket
                    logger.info(f"✅ Connected to Binance WebSocket")
                    logger.info(f"📡 Listening to {len(self.symbols)} bookTicker streams...")
                    
                    # Reset retry delay on successful connection
                    retry_delay = INITIAL_RETRY_DELAY
                    
                    # Listen for messages
                    async for message in websocket:
                        if not self.running:
                            break
                        
                        await self.parse_and_write_message(message)
                
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"⚠️  WebSocket connection closed: {e}")
                self.metrics.reconnections += 1
                
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                self.metrics.errors += 1
                self.metrics.reconnections += 1
            
            # Exponential backoff
            if self.running:
                logger.info(f"🔄 Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, MAX_RETRY_DELAY)
    
    async def run(self):
        """Main run loop"""
        self.running = True
        
        try:
            # Connect to Redis
            await self.connect_redis()
            
            logger.info("=" * 80)
            logger.info("🚀 Binance BookTicker Collector")
            logger.info("=" * 80)
            logger.info(f"Symbols: {', '.join([s.upper() for s in self.symbols])}")
            logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
            logger.info(f"Stream format: market:binance_spot:orderbook:{{symbol}}")
            logger.info("=" * 80)
            
            # Start health check loop
            health_check_task = asyncio.create_task(self.health_check_loop())
            
            # Start WebSocket connection loop
            await self.connect_and_subscribe()
            
            # Clean up health check
            health_check_task.cancel()
            try:
                await health_check_task
            except asyncio.CancelledError:
                pass
            
        except KeyboardInterrupt:
            logger.info("⚠️  Received interrupt signal")
        finally:
            self.running = False
            await self.disconnect_redis()
            
            # Final stats
            logger.info("=" * 80)
            logger.info("📊 Final Statistics")
            logger.info("=" * 80)
            self.metrics.log_stats()
            logger.info("=" * 80)
            logger.info("✅ Shutdown complete")


async def main():
    """Main entry point"""
    collector = BinanceBookTickerCollector(symbols=SYMBOLS)
    await collector.run()


if __name__ == "__main__":
    asyncio.run(main())
