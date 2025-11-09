#!/usr/bin/env python3
"""
Redis Streams → QuestDB High-Performance Batch Writer (v2 - ILP)
=================================================================
Day 1 Implementation: ILP Protocol for Maximum Throughput

Key Improvements over PostgreSQL wire protocol:
- QuestDB native ILP (InfluxDB Line Protocol) via HTTP
- 10-100x faster writes (10,000+ rows/sec)
- No prepared statement issues
- Production-proven pattern
- Batch size: 500 records (Opus recommendation)
- Target: 10,000+ inserts/sec, P99 latency <10ms

Architecture:
- Consume from Redis streams: market:binance_spot:trades:*
- Batch accumulation with 100ms timeout
- ILP line protocol via HTTP endpoint (port 9000)
- asyncpg pool kept for future query needs (Day 3)
- Prometheus metrics for monitoring

Reference: MARKET-DATA-ENHANCEMENT-IMPLEMENTATION-PLAN.md
Author: Claude (Day 1 Implementation - ILP optimization per Opus)
Date: 2025-10-21
"""

import asyncio
import json
import os
import signal
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import asyncpg
import aiohttp
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Day 4: Import tiered cache for cache population
try:
    # Add Config directory to path for tiered_market_data module
    sys.path.insert(0, '/home/youssefbahloul/ai-trading-station/Services/QuestDB/Config')
    from tiered_market_data import TieredMarketData, MarketTick
    TIERED_CACHE_AVAILABLE = True
except ImportError:
    print("⚠️  tiered_market_data not available - cache population disabled")
    TIERED_CACHE_AVAILABLE = False
    TieredMarketData = type(None)  # Use type(None) as placeholder for type hints
    MarketTick = type(None)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_UNIX_SOCKET = os.getenv("REDIS_UNIX_SOCKET", None)  # For ultra-low latency

# QuestDB configuration
QUESTDB_HOST = os.getenv("QUESTDB_HOST", "localhost")
QUESTDB_PG_PORT = int(os.getenv("QUESTDB_PG_PORT", 8812))  # PostgreSQL wire protocol (for queries)
QUESTDB_HTTP_PORT = int(os.getenv("QUESTDB_HTTP_PORT", 9000))  # HTTP REST for ILP writes
QUESTDB_ILP_PORT = int(os.getenv("QUESTDB_ILP_PORT", 9000))  # ILP over HTTP (same as HTTP_PORT)
QUESTDB_USER = os.getenv("QUESTDB_USER", "admin")
QUESTDB_PASSWORD = os.getenv("QUESTDB_PASSWORD", "quest")
QUESTDB_DATABASE = os.getenv("QUESTDB_DATABASE", "qdb")

# Performance tuning (Optimized for ILP protocol - Day 1 final)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2000))  # Increased from 500 for ILP throughput
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", 50))  # Reduced from 100ms for faster flushing
CONNECTION_POOL_MIN = int(os.getenv("CONNECTION_POOL_MIN", 5))  # Keep for query needs
CONNECTION_POOL_MAX = int(os.getenv("CONNECTION_POOL_MAX", 20))  # Keep for query needs
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))  # Increased from 6 to exceed 5k ticks/sec target

# Redis streams configuration (Day 2: trades + orderbook)
STREAM_PATTERNS = [
    "market:binance_spot:trades:*",      # Trade ticks
    "market:binance_spot:orderbook:*"    # OrderBook updates (bookTicker)
]
CONSUMER_GROUP = "questdb_writer_multistream_v2"
CONSUMER_PREFIX = "worker"
BLOCK_MS = 1000  # Block for 1 second waiting for messages
READ_COUNT = 500  # Read up to 500 messages at once

# Monitoring
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", 9091))  # Changed from 9090 to avoid conflicts
STATS_INTERVAL_SEC = 10

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Counters
metrics_messages_read = Counter(
    "redis_messages_read_total",
    "Total messages read from Redis streams",
    ["worker_id"]
)

metrics_ticks_inserted = Counter(
    "questdb_ticks_inserted_total",
    "Total ticks inserted into QuestDB",
    ["worker_id", "symbol"]
)

metrics_batches_flushed = Counter(
    "questdb_batches_flushed_total",
    "Total batches flushed to QuestDB",
    ["worker_id"]
)

metrics_errors = Counter(
    "pipeline_errors_total",
    "Total errors encountered",
    ["worker_id", "error_type"]
)

# Histograms
metrics_batch_size = Histogram(
    "questdb_batch_size",
    "Distribution of batch sizes",
    ["worker_id"],
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
)

metrics_insert_latency = Histogram(
    "questdb_insert_latency_seconds",
    "Insert latency in seconds",
    ["worker_id"],
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)

metrics_batch_age = Histogram(
    "questdb_batch_age_seconds",
    "Age of oldest message in batch when flushed",
    ["worker_id"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# Gauges
metrics_queue_depth = Gauge(
    "redis_queue_depth",
    "Current queue depth in Redis streams",
    ["stream"]
)

metrics_active_workers = Gauge(
    "pipeline_active_workers",
    "Number of active workers"
)

metrics_connection_pool_size = Gauge(
    "questdb_connection_pool_size",
    "Current connection pool size"
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradeMessage:
    """Parsed trade message from Redis stream."""
    timestamp: int  # Microseconds since epoch
    symbol: str
    exchange: str
    price: float
    volume: float
    side: bool  # True = buy, False = sell
    received_at: float = field(default_factory=time.time)  # For latency tracking


@dataclass
class OrderbookMessage:
    """Parsed orderbook message from Redis stream (bookTicker)."""
    timestamp: int  # Microseconds since epoch
    symbol: str
    exchange: str
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float
    received_at: float = field(default_factory=time.time)  # For latency tracking


@dataclass
class WorkerStats:
    """Statistics for a worker."""
    worker_id: int
    messages_read: int = 0
    ticks_inserted: int = 0
    batches_flushed: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    last_flush_time: float = field(default_factory=time.time)
    
    def get_rate(self) -> float:
        """Get inserts per second."""
        elapsed = time.time() - self.start_time
        return self.ticks_inserted / elapsed if elapsed > 0 else 0.0


# ============================================================================
# GLOBAL STATE
# ============================================================================

shutdown_event = asyncio.Event()
worker_stats: Dict[int, WorkerStats] = {}
db_pool: Optional[asyncpg.Pool] = None
http_session: Optional[aiohttp.ClientSession] = None  # Global session for ILP writes
tiered_data = None  # Type: Optional[TieredMarketData], set when initialized


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n⚠️  Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================

async def create_db_pool() -> asyncpg.Pool:
    """
    Create asyncpg connection pool to QuestDB.
    
    QuestDB supports PostgreSQL wire protocol on port 8812.
    Using QuestDB-optimized settings per Opus recommendations.
    """
    print(f"🔌 Connecting to QuestDB at {QUESTDB_HOST}:{QUESTDB_PG_PORT}...")
    
    pool = await asyncpg.create_pool(
        host=QUESTDB_HOST,
        port=QUESTDB_PG_PORT,
        user=QUESTDB_USER,
        password=QUESTDB_PASSWORD,
        database=QUESTDB_DATABASE,
        
        # QuestDB-optimized connection pool settings
        min_size=CONNECTION_POOL_MIN,
        max_size=CONNECTION_POOL_MAX,
        max_queries=50000,  # Reuse connections more
        max_inactive_connection_lifetime=300,
        
        # Timeouts (shorter for faster failure detection)
        timeout=10,  # Connection timeout
        command_timeout=5,  # Query timeout
        
        # Statement cache
        statement_cache_size=0,  # Disabled for QuestDB compatibility
    )
    
    # Test connection
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1, "Connection test failed"
    
    print(f"✅ Database pool created ({CONNECTION_POOL_MIN}-{CONNECTION_POOL_MAX} connections)")
    metrics_connection_pool_size.set(CONNECTION_POOL_MIN)
    
    return pool


# ============================================================================
# BATCH FLUSHING WITH ILP PROTOCOL (QuestDB Native High-Performance)
# ============================================================================

async def flush_batch_to_questdb(
    batch: List[TradeMessage],
    pool: asyncpg.Pool,  # Keep for future query needs (Day 3)
    worker_id: int
) -> bool:
    """
    Flush batch to QuestDB using ILP (InfluxDB Line Protocol) for maximum throughput.
    
    ILP is QuestDB's native high-performance ingestion protocol.
    Advantages over PostgreSQL wire protocol:
    - 10-100x faster for bulk writes (10,000+ rows/sec)
    - No prepared statement issues
    - Simpler implementation
    - Production-proven pattern
    
    Format: table_name,tag1=value1,tag2=value2 field1=value1,field2=value2 timestamp
    
    Args:
        batch: List of trade messages
        pool: asyncpg connection pool (kept for future query needs)
        worker_id: Worker ID for metrics
        
    Returns:
        True if successful, False otherwise
    """
    if not batch:
        return True
    
    start_time = time.time()
    batch_age = start_time - batch[0].received_at
    
    try:
        # Build ILP payload (newline-delimited)
        lines = []
        for msg in batch:
            # ILP format:
            # market_trades,symbol=BTCUSDT,exchange=binance_spot price=50000.0,volume=1.5,side=t 1634567890000000000
            # Tags: symbol, exchange (indexed automatically)
            # Fields: price, volume, side (values)
            # Timestamp: nanoseconds (msg.timestamp is microseconds)
            
            side_str = 't' if msg.side else 'f'  # Boolean as t/f
            line = (
                f"market_trades,"
                f"symbol={msg.symbol},"
                f"exchange={msg.exchange} "
                f"price={msg.price},"
                f"volume={msg.volume},"
                f"side={side_str} "
                f"{msg.timestamp * 1000}"  # Convert microseconds to nanoseconds
            )
            lines.append(line)
        
        payload = '\n'.join(lines)
        
        # Send to QuestDB ILP endpoint via HTTP (using persistent session)
        # Connection pooling eliminates TCP handshake overhead
        async with http_session.post(
            f'http://{QUESTDB_HOST}:{QUESTDB_HTTP_PORT}/write?precision=n',
            data=payload,
            headers={'Content-Type': 'text/plain'}
        ) as resp:
            # QuestDB returns 200 or 204 for successful ILP writes
            if resp.status not in (200, 204):
                error_text = await resp.text()
                raise Exception(f"ILP write failed (HTTP {resp.status}): {error_text}")
        
        # Record metrics
        elapsed = time.time() - start_time
        metrics_batch_size.labels(worker_id=worker_id).observe(len(batch))
        metrics_insert_latency.labels(worker_id=worker_id).observe(elapsed)
        metrics_batch_age.labels(worker_id=worker_id).observe(batch_age)
        metrics_batches_flushed.labels(worker_id=worker_id).inc()
        
        for msg in batch:
            metrics_ticks_inserted.labels(worker_id=worker_id, symbol=msg.symbol).inc()
        
        # Update worker stats
        if worker_id in worker_stats:
            stats = worker_stats[worker_id]
            stats.ticks_inserted += len(batch)
            stats.batches_flushed += 1
            stats.last_flush_time = time.time()
        
        # Populate tiered cache (Day 4 integration - Optimized per Copilot review)
        if TIERED_CACHE_AVAILABLE and tiered_data:
            try:
                # List comprehension for better performance (Copilot optimization)
                # CRITICAL: Use utcfromtimestamp() to match cache query's utcnow() timezone
                ticks = [
                    MarketTick(
                        timestamp=datetime.utcfromtimestamp(msg.timestamp / 1000000),
                        symbol=msg.symbol,
                        exchange=msg.exchange,
                        data_type='TRADE',
                        price=msg.price,
                        volume=msg.volume,
                        side='BUY' if msg.side else 'SELL'
                    )
                    for msg in batch
                ]
                await tiered_data.add_batch_to_cache(ticks)
                # Debug: Log cache population every 10 batches
                if stats.batches_flushed % 10 == 0:
                    print(f"🔧 Worker {worker_id} added {len(ticks)} ticks to cache (batch #{stats.batches_flushed})")
            except Exception as e:
                # Don't fail batch write if cache population fails
                print(f"⚠️  Worker {worker_id} cache population error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Worker {worker_id} batch flush error: {type(e).__name__}: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("".join(traceback.format_tb(e.__traceback__)[:3]))  # First 3 lines
        metrics_errors.labels(worker_id=worker_id, error_type="batch_flush").inc()
        
        if worker_id in worker_stats:
            worker_stats[worker_id].errors += 1
        
        return False


async def flush_orderbook_to_questdb(
    batch: List[OrderbookMessage],
    db_pool: asyncpg.Pool,
    worker_id: int
) -> bool:
    """
    Flush orderbook batch to QuestDB via ILP protocol over HTTP.
    
    ILP Format:
    market_orderbook,symbol=BTCUSDT,exchange=binance_spot bid_price=50000.0,bid_qty=1.5,ask_price=50001.0,ask_qty=2.0 1234567890000000
    """
    if not batch:
        return True
    
    start_time = time.time()
    batch_age = time.time() - batch[0].received_at if batch else 0
    
    try:
        # Build ILP protocol lines
        lines = []
        for msg in batch:
            # Table name with tags (indexed columns)
            line = f"market_orderbook,symbol={msg.symbol},exchange={msg.exchange}"
            
            # Field columns (float values)
            line += f" bid_price={msg.bid_price},bid_qty={msg.bid_qty}"
            line += f",ask_price={msg.ask_price},ask_qty={msg.ask_qty}"
            
            # Timestamp (nanoseconds)
            line += f" {msg.timestamp * 1000}"  # microseconds → nanoseconds
            
            lines.append(line)
        
        # Join with newlines
        ilp_payload = "\n".join(lines)
        
        # Send via HTTP POST using persistent session
        async with http_session.post(
            f"http://{QUESTDB_HOST}:{QUESTDB_HTTP_PORT}/write?precision=n",
            data=ilp_payload.encode('utf-8'),
            headers={'Content-Type': 'text/plain'}
        ) as resp:
            # QuestDB returns 200 or 204 for successful ILP writes
            if resp.status not in (200, 204):
                error_text = await resp.text()
                raise Exception(f"ILP orderbook write failed (HTTP {resp.status}): {error_text}")
        
        # Record metrics
        elapsed = time.time() - start_time
        metrics_batch_size.labels(worker_id=worker_id).observe(len(batch))
        metrics_insert_latency.labels(worker_id=worker_id).observe(elapsed)
        metrics_batch_age.labels(worker_id=worker_id).observe(batch_age)
        metrics_batches_flushed.labels(worker_id=worker_id).inc()
        
        for msg in batch:
            metrics_ticks_inserted.labels(worker_id=worker_id, symbol=msg.symbol).inc()
        
        # Update worker stats
        if worker_id in worker_stats:
            stats = worker_stats[worker_id]
            stats.ticks_inserted += len(batch)
            stats.batches_flushed += 1
            stats.last_flush_time = time.time()
        
        return True
        
    except Exception as e:
        print(f"❌ Worker {worker_id} orderbook batch flush error: {type(e).__name__}: {str(e)}")
        metrics_errors.labels(worker_id=worker_id, error_type="orderbook_batch_flush").inc()
        
        if worker_id in worker_stats:
            worker_stats[worker_id].errors += 1
        
        return False


# ============================================================================
# REDIS STREAM CONSUMER
# ============================================================================

async def parse_trade_message(data: Dict[bytes, bytes]) -> Optional[TradeMessage]:
    """
    Parse trade message from Redis stream.
    
    Expected format from exchange_to_redis.py:
    {
        b'timestamp': b'1697812345678900',  # microseconds
        b'symbol': b'BTCUSDT',
        b'exchange': b'BINANCE',
        b'price': b'42150.50',
        b'volume': b'0.125',
        b'side': b'buy'  # or 'sell'
    }
    """
    try:
        return TradeMessage(
            timestamp=int(data[b'timestamp']),
            symbol=data[b'symbol'].decode(),
            exchange=data[b'exchange'].decode(),
            price=float(data[b'price']),
            volume=float(data[b'volume']),
            side=(data[b'side'].decode().lower() == 'buy')
        )
    except Exception as e:
        print(f"⚠️  Failed to parse trade message: {e}")
        return None


async def parse_orderbook_message(data: Dict[bytes, bytes]) -> Optional[OrderbookMessage]:
    """
    Parse orderbook message from Redis stream (bookTicker format).
    
    Expected format from exchange_to_redis.py:
    {
        b'timestamp': b'1697812345678900',  # microseconds
        b'symbol': b'BTCUSDT',
        b'exchange': b'BINANCE',
        b'bid_price': b'42150.50',
        b'bid_qty': b'1.25',
        b'ask_price': b'42151.00',
        b'ask_qty': b'2.10'
    }
    """
    try:
        return OrderbookMessage(
            timestamp=int(data[b'timestamp']),
            symbol=data[b'symbol'].decode(),
            exchange=data[b'exchange'].decode(),
            bid_price=float(data[b'bid_price']),
            bid_qty=float(data[b'bid_qty']),
            ask_price=float(data[b'ask_price']),
            ask_qty=float(data[b'ask_qty'])
        )
    except Exception as e:
        print(f"⚠️  Failed to parse orderbook message: {e}")
        return None


async def ensure_consumer_group(redis_client: redis.Redis, stream: str):
    """Ensure consumer group exists for stream."""
    try:
        await redis_client.xgroup_create(
            stream,
            CONSUMER_GROUP,
            id="0",
            mkstream=True
        )
        print(f"✅ Created consumer group '{CONSUMER_GROUP}' for '{stream}'")
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            pass  # Already exists
        else:
            raise


async def discover_streams(redis_client: redis.Redis) -> List[str]:
    """Discover all streams matching patterns."""
    streams = []
    
    for pattern in STREAM_PATTERNS:
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                key_type = await redis_client.type(key)
                if key_type == b"stream":
                    if key_str not in streams:
                        streams.append(key_str)
            
            if cursor == 0:
                break
    
    return streams


# ============================================================================
# WORKER
# ============================================================================

async def worker(worker_id: int):
    """
    Worker coroutine that reads from Redis and writes to QuestDB.
    
    Args:
        worker_id: Unique worker identifier
    """
    global db_pool
    
    print(f"✅ Worker {worker_id} starting...")
    
    # Initialize stats
    stats = WorkerStats(worker_id=worker_id)
    worker_stats[worker_id] = stats
    metrics_active_workers.inc()
    
    # Connect to Redis
    if REDIS_UNIX_SOCKET:
        redis_client = redis.Redis(unix_socket_path=REDIS_UNIX_SOCKET, decode_responses=False)
    else:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    
    consumer_name = f"{CONSUMER_PREFIX}_{worker_id}"
    trade_batch: List[TradeMessage] = []
    orderbook_batch: List[OrderbookMessage] = []
    last_flush_time = time.time()
    
    try:
        # Discover streams - keep retrying until streams appear
        streams = []
        while not streams and not shutdown_event.is_set():
            streams = await discover_streams(redis_client)
            if not streams:
                if worker_id == 0:  # Only worker 0 prints to avoid spam
                    print(f"⏳ Waiting for streams matching patterns: {STREAM_PATTERNS}...")
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
        
        if not streams:
            return  # Shutdown event was set
        
        print(f"✅ Worker {worker_id}: Monitoring {len(streams)} streams")
        
        # Ensure consumer groups
        for stream in streams:
            await ensure_consumer_group(redis_client, stream)
        
        # Track last stream discovery time
        last_discovery_time = time.time()
        DISCOVERY_INTERVAL_SEC = 30  # Rediscover streams every 30 seconds
        
        # Main consumption loop
        while not shutdown_event.is_set():
            try:
                # Periodically rediscover streams (pick up new symbols)
                now = time.time()
                if now - last_discovery_time >= DISCOVERY_INTERVAL_SEC:
                    new_streams = await discover_streams(redis_client)
                    for stream in new_streams:
                        if stream not in streams:
                            await ensure_consumer_group(redis_client, stream)
                            streams.append(stream)
                            if worker_id == 0:
                                print(f"✅ Discovered new stream: {stream}")
                    last_discovery_time = now
                
                # Read from all streams
                stream_dict = {stream: ">" for stream in streams}
                
                messages = await redis_client.xreadgroup(
                    CONSUMER_GROUP,
                    consumer_name,
                    stream_dict,
                    count=READ_COUNT,
                    block=BLOCK_MS
                )
                
                if messages:
                    for stream, message_list in messages:
                        stream_name = stream.decode() if isinstance(stream, bytes) else stream
                        
                        # Determine stream type based on stream name
                        is_trade_stream = ":trades:" in stream_name
                        is_orderbook_stream = ":orderbook:" in stream_name
                        
                        for message_id, data in message_list:
                            stats.messages_read += 1
                            metrics_messages_read.labels(worker_id=worker_id).inc()
                            
                            # Parse message based on stream type
                            if is_trade_stream:
                                trade = await parse_trade_message(data)
                                if trade:
                                    trade_batch.append(trade)
                            elif is_orderbook_stream:
                                orderbook = await parse_orderbook_message(data)
                                if orderbook:
                                    orderbook_batch.append(orderbook)
                            
                            # Acknowledge message
                            await redis_client.xack(stream_name, CONSUMER_GROUP, message_id)
                
                # Check if we should flush (either batch)
                now = time.time()
                batch_timeout_exceeded = (now - last_flush_time) * 1000 >= BATCH_TIMEOUT_MS
                
                # Flush trades
                if len(trade_batch) >= BATCH_SIZE or (trade_batch and batch_timeout_exceeded):
                    await flush_batch_to_questdb(trade_batch, db_pool, worker_id)
                    trade_batch.clear()
                    last_flush_time = now
                
                # Flush orderbook
                if len(orderbook_batch) >= BATCH_SIZE or (orderbook_batch and batch_timeout_exceeded):
                    await flush_orderbook_to_questdb(orderbook_batch, db_pool, worker_id)
                    orderbook_batch.clear()
                    last_flush_time = now
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Worker {worker_id} error: {e}")
                stats.errors += 1
                metrics_errors.labels(worker_id=worker_id, error_type="worker_loop").inc()
                await asyncio.sleep(1)
        
        # Flush remaining batches on shutdown
        if trade_batch:
            await flush_batch_to_questdb(trade_batch, db_pool, worker_id)
        if orderbook_batch:
            await flush_orderbook_to_questdb(orderbook_batch, db_pool, worker_id)
            
    finally:
        await redis_client.close()
        metrics_active_workers.dec()
        print(f"✅ Worker {worker_id} shutdown complete")


# ============================================================================
# MONITORING
# ============================================================================

async def cache_query_generator():
    """
    Periodically query the cache to generate metrics for Grafana dashboard.
    Tests all three cache tiers (HOT, WARM, COLD) with appropriate time windows.
    
    Queries multiple symbols and data types every 5 seconds to populate:
    - cache_requests_total
    - cache_hits_total (by tier: HOT/WARM/COLD)
    - cache_latency_seconds
    - cache_size_entries
    
    CRITICAL FIX: Add periodic health check to prevent Redis connection issues
    """
    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
    DATA_TYPES = ['TRADE', 'ORDERBOOK']
    
    # Define time windows for each cache tier
    # HOT: <60s, WARM: 60-600s, COLD: >600s
    QUERY_WINDOWS = [
        ('HOT', 5),       # 5 seconds - should hit HOT cache (Opus recommendation)
        ('HOT', 30),      # 30 seconds - should hit HOT cache
        ('WARM', 60),     # 1 minute - should hit WARM cache (Opus recommendation)
        ('WARM', 180),    # 3 minutes - should hit WARM cache (Opus recommendation)
        ('COLD', 600),    # 10 minutes - should hit COLD cache (Opus recommendation)
        ('COLD', 1800),   # 30 minutes - should hit COLD cache
    ]
    
    print("🔍 Starting cache query generator for Grafana metrics...")
    print(f"   Testing {len(QUERY_WINDOWS)} time windows across HOT/WARM/COLD tiers")
    
    query_count = 0
    tier_hits = {'HOT': 0, 'WARM': 0, 'COLD': 0}
    last_health_check = time.time()  # ADDED: Track health check timing
    
    while not shutdown_event.is_set():
        try:
            # CRITICAL: Check Redis connection health every 60 seconds (Opus recommendation)
            if time.time() - last_health_check > 60:
                try:
                    await tiered_data.warm_cache.redis.ping()
                    last_health_check = time.time()
                    print("✅ Redis health check passed")
                except Exception as health_error:
                    print(f"⚠️  Redis health check failed: {health_error} - attempting reconnect")
                    try:
                        await tiered_data.warm_cache.connect()
                        print("✅ Redis reconnected successfully")
                        last_health_check = time.time()
                    except Exception as reconnect_error:
                        print(f"❌ Redis reconnect failed: {reconnect_error}")
            
            # Rotate through different time windows to hit all tiers
            for target_tier, seconds in QUERY_WINDOWS:
                for symbol in SYMBOLS[:3]:  # Use first 3 symbols for faster rotation
                    for data_type in DATA_TYPES:
                        if shutdown_event.is_set():
                            break
                        
                        try:
                            ticks, tier, latency_ms = await tiered_data.get_recent_ticks(
                                symbol=symbol,
                                exchange='binance_spot',
                                data_type=data_type,
                                seconds=seconds
                            )
                            
                            query_count += 1
                            tier_hits[tier] = tier_hits.get(tier, 0) + 1
                            
                            # Log progress every 50 queries with tier distribution
                            if query_count % 50 == 0:
                                print(f"🔍 Cache queries: {query_count} | Last: {symbol}:{data_type} ({seconds}s) | "
                                      f"Tier: {tier} | Latency: {latency_ms:.2f}ms | Ticks: {len(ticks)}")
                                print(f"   Distribution: HOT={tier_hits.get('HOT', 0)} "
                                      f"WARM={tier_hits.get('WARM', 0)} COLD={tier_hits.get('COLD', 0)}")
                        
                        except Exception as e:
                            print(f"⚠️  Cache query error for {symbol}:{data_type} ({seconds}s): {e}")
                
                # Small delay between time windows
                await asyncio.sleep(0.1)
            
            # Wait 5 seconds before next full round
            await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Cache query generator error: {e}")
            await asyncio.sleep(5)
    
    print(f"✅ Cache query generator stopped (total queries: {query_count})")
    print(f"   Final distribution: HOT={tier_hits.get('HOT', 0)} "
          f"WARM={tier_hits.get('WARM', 0)} COLD={tier_hits.get('COLD', 0)}")


async def stats_monitor():
    """Periodically print statistics."""
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(STATS_INTERVAL_SEC)
            
            print(f"\n{'='*80}")
            print(f"📊 Pipeline Statistics")
            print(f"{'='*80}")
            
            total_ticks = sum(s.ticks_inserted for s in worker_stats.values())
            total_errors = sum(s.errors for s in worker_stats.values())
            
            for worker_id in sorted(worker_stats.keys()):
                stats = worker_stats[worker_id]
                rate = stats.get_rate()
                print(f"Worker {worker_id}: {stats.ticks_inserted:,} ticks | "
                      f"{stats.batches_flushed:,} batches | "
                      f"{rate:,.0f} ticks/sec | "
                      f"{stats.errors} errors")
            
            print(f"\nTotal: {total_ticks:,} ticks | {total_errors} errors")
            print(f"{'='*80}\n")
            
        except asyncio.CancelledError:
            break


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point."""
    global db_pool, http_session, tiered_data
    
    print("=" * 80)
    print("🚀 Redis → QuestDB High-Performance Batch Writer (v2)")
    print("=" * 80)
    print(f"Workers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE} records")
    print(f"Batch timeout: {BATCH_TIMEOUT_MS}ms")
    print(f"Connection pool: {CONNECTION_POOL_MIN}-{CONNECTION_POOL_MAX}")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"QuestDB: {QUESTDB_HOST}:{QUESTDB_HTTP_PORT} (ILP via HTTP)")
    print(f"Stream patterns: {STREAM_PATTERNS}")
    print(f"Prometheus metrics: http://localhost:{PROMETHEUS_PORT}")
    print("=" * 80)
    print()
    
    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    print(f"✅ Prometheus metrics server started on port {PROMETHEUS_PORT}")
    
    # Create database connection pool (for future query needs)
    db_pool = await create_db_pool()
    
    # Create persistent HTTP session for ILP writes (connection pooling)
    timeout = aiohttp.ClientTimeout(total=5.0)
    connector = aiohttp.TCPConnector(
        limit=100,  # Max 100 concurrent connections
        limit_per_host=50,  # Max 50 per host
        ttl_dns_cache=300,  # DNS cache for 5 minutes
        keepalive_timeout=30  # Keep connections alive for 30 seconds
    )
    http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    print(f"✅ HTTP connection pool created (limit=100, keepalive=30s)")
    
    # Initialize tiered cache (Day 4 integration)
    if TIERED_CACHE_AVAILABLE:
        try:
            tiered_data = TieredMarketData()
            await tiered_data.start()
            print("✅ Tiered cache initialized and warmed")
        except Exception as e:
            print(f"⚠️  Tiered cache initialization failed: {e}")
            print("   Continuing without cache...")
    
    # Start workers
    tasks = []
    
    # Stats monitor
    tasks.append(asyncio.create_task(stats_monitor()))
    
    # Cache query generator (for Grafana metrics)
    if TIERED_CACHE_AVAILABLE and tiered_data:
        tasks.append(asyncio.create_task(cache_query_generator()))
        print("✅ Cache query generator started (for Grafana dashboard)")
    
    # Worker tasks
    for i in range(NUM_WORKERS):
        tasks.append(asyncio.create_task(worker(i)))
    
    try:
        # Wait for shutdown
        await shutdown_event.wait()
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for completion
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted, shutting down...")
    finally:
        # Close HTTP session
        if http_session:
            await http_session.close()
            print("✅ HTTP session closed")
        
        # Close database pool
        if db_pool:
            await db_pool.close()
            print("✅ Database pool closed")
    
    # Final stats
    print("\n📊 Final Statistics:")
    total_ticks = sum(s.ticks_inserted for s in worker_stats.values())
    total_batches = sum(s.batches_flushed for s in worker_stats.values())
    total_errors = sum(s.errors for s in worker_stats.values())
    
    print(f"Total ticks inserted: {total_ticks:,}")
    print(f"Total batches flushed: {total_batches:,}")
    print(f"Total errors: {total_errors}")
    
    if total_batches > 0:
        avg_batch_size = total_ticks / total_batches
        print(f"Average batch size: {avg_batch_size:.1f}")
    
    print("\n✅ Shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
