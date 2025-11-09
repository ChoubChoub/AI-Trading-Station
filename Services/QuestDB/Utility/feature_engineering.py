#!/usr/bin/env python3
"""
Feature Engineering Framework
Day 3 Implementation: Calculate alpha signals from raw market data

Features Implemented:
- Spread metrics (absolute, relative, z-score)
- Mid-price and fair value
- Orderbook imbalance
- VPIN (Volume-synchronized Probability of Informed Trading)
- Microstructure features

All calculations use raw data from tiered cache system.

Author: Claude Sonnet
Date: October 25, 2025
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import math
import time  # Added for feature caching

from tiered_market_data import TieredMarketData, MarketTick


@dataclass
class FeatureSet:
    """Container for calculated features"""
    timestamp: datetime
    symbol: str
    exchange: str
    
    # Spread features
    spread_abs: Optional[float] = None
    spread_bps: Optional[float] = None  # Basis points
    spread_zscore: Optional[float] = None
    
    # Price features
    mid_price: Optional[float] = None
    fair_value: Optional[float] = None  # Volume-weighted
    
    # Orderbook features
    book_imbalance: Optional[float] = None
    bid_ask_ratio: Optional[float] = None
    
    # Volume features
    volume_5s: Optional[float] = None
    volume_1m: Optional[float] = None
    volume_zscore: Optional[float] = None
    
    # Microstructure
    vpin: Optional[float] = None  # Volume-synchronized PIN
    trade_flow_imbalance: Optional[float] = None
    
    # Volatility
    volatility_1m: Optional[float] = None
    volatility_5m: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'spread_abs': self.spread_abs,
            'spread_bps': self.spread_bps,
            'spread_zscore': self.spread_zscore,
            'mid_price': self.mid_price,
            'fair_value': self.fair_value,
            'book_imbalance': self.book_imbalance,
            'bid_ask_ratio': self.bid_ask_ratio,
            'volume_5s': self.volume_5s,
            'volume_1m': self.volume_1m,
            'volume_zscore': self.volume_zscore,
            'vpin': self.vpin,
            'trade_flow_imbalance': self.trade_flow_imbalance,
            'volatility_1m': self.volatility_1m,
            'volatility_5m': self.volatility_5m
        }


class FeatureEngineer:
    """
    Feature engineering using tiered market data
    All features calculated from raw data, no pre-computation
    Includes feature caching (Opus optimization)
    """
    
    def __init__(self, tiered_data: TieredMarketData, cache_ttl: int = 1):
        self.tiered_data = tiered_data
        self._feature_cache = {}  # Simple TTL cache
        self._cache_ttl = cache_ttl  # 1 second TTL (Opus recommendation)
    
    def _get_cache_key(self, symbol: str, exchange: str) -> str:
        """Generate cache key for features"""
        return f"{symbol}:{exchange}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached features are still valid"""
        if cache_key not in self._feature_cache:
            return False
        
        cached = self._feature_cache[cache_key]
        age = time.time() - cached['timestamp']
        return age < self._cache_ttl
    
    async def calculate_spread_features(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, Optional[float]]:
        """
        Calculate spread-related features
        
        Returns:
            - spread_abs: Absolute spread (ask - bid)
            - spread_bps: Spread in basis points
            - spread_zscore: Z-score of spread (vs 1-hour history)
        """
        # Get latest orderbook tick
        ticks, tier, latency = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='ORDERBOOK',
            seconds=1
        )
        
        if not ticks:
            return {'spread_abs': None, 'spread_bps': None, 'spread_zscore': None}
        
        latest = ticks[0]
        
        # Calculate absolute spread
        if latest.ask_price and latest.bid_price:
            spread_abs = latest.ask_price - latest.bid_price
            mid_price = (latest.ask_price + latest.bid_price) / 2
            spread_bps = (spread_abs / mid_price) * 10000  # Basis points
        else:
            return {'spread_abs': None, 'spread_bps': None, 'spread_zscore': None}
        
        # Calculate z-score (need historical spreads)
        ticks_1h, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='ORDERBOOK',
            seconds=3600
        )
        
        if len(ticks_1h) > 10:
            spreads = [
                (t.ask_price - t.bid_price)
                for t in ticks_1h
                if t.ask_price and t.bid_price
            ]
            
            if len(spreads) > 1:
                mean_spread = statistics.mean(spreads)
                stdev_spread = statistics.stdev(spreads)
                
                if stdev_spread > 0:
                    spread_zscore = (spread_abs - mean_spread) / stdev_spread
                else:
                    spread_zscore = 0.0
            else:
                spread_zscore = None
        else:
            spread_zscore = None
        
        return {
            'spread_abs': spread_abs,
            'spread_bps': spread_bps,
            'spread_zscore': spread_zscore
        }
    
    async def calculate_price_features(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, Optional[float]]:
        """
        Calculate price-related features
        
        Returns:
            - mid_price: (bid + ask) / 2
            - fair_value: Volume-weighted price from recent trades
        """
        # Get latest orderbook
        book_ticks, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='ORDERBOOK',
            seconds=1
        )
        
        if not book_ticks:
            return {'mid_price': None, 'fair_value': None}
        
        latest_book = book_ticks[0]
        
        # Mid price
        if latest_book.ask_price and latest_book.bid_price:
            mid_price = (latest_book.ask_price + latest_book.bid_price) / 2
        else:
            mid_price = None
        
        # Fair value (VWAP from last 5 seconds)
        trade_ticks, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='TRADE',
            seconds=5
        )
        
        if trade_ticks:
            total_value = sum(t.price * t.volume for t in trade_ticks if t.price and t.volume)
            total_volume = sum(t.volume for t in trade_ticks if t.volume)
            
            if total_volume > 0:
                fair_value = total_value / total_volume
            else:
                fair_value = None
        else:
            fair_value = None
        
        return {
            'mid_price': mid_price,
            'fair_value': fair_value
        }
    
    async def calculate_orderbook_features(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, Optional[float]]:
        """
        Calculate orderbook imbalance features
        
        Returns:
            - book_imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
            - bid_ask_ratio: bid_qty / ask_qty
        """
        ticks, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='ORDERBOOK',
            seconds=1
        )
        
        if not ticks:
            return {'book_imbalance': None, 'bid_ask_ratio': None}
        
        latest = ticks[0]
        
        if latest.bid_qty and latest.ask_qty and latest.ask_qty > 0:
            book_imbalance = (latest.bid_qty - latest.ask_qty) / (latest.bid_qty + latest.ask_qty)
            bid_ask_ratio = latest.bid_qty / latest.ask_qty
        else:
            book_imbalance = None
            bid_ask_ratio = None
        
        return {
            'book_imbalance': book_imbalance,
            'bid_ask_ratio': bid_ask_ratio
        }
    
    async def calculate_volume_features(
        self,
        symbol: str,
        exchange: str
    ) -> Dict[str, Optional[float]]:
        """
        Calculate volume-related features
        
        Returns:
            - volume_5s: Total volume in last 5 seconds
            - volume_1m: Total volume in last 1 minute
            - volume_zscore: Z-score vs 1-hour average
        """
        # Get trades
        ticks_5s, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='TRADE',
            seconds=5
        )
        
        ticks_1m, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='TRADE',
            seconds=60
        )
        
        ticks_1h, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='TRADE',
            seconds=3600
        )
        
        # Calculate volumes
        volume_5s = sum(t.volume for t in ticks_5s if t.volume) if ticks_5s else None
        volume_1m = sum(t.volume for t in ticks_1m if t.volume) if ticks_1m else None
        
        # Calculate z-score (1-minute volumes over 1 hour)
        if len(ticks_1h) > 60:
            # Split into 1-minute buckets
            minute_volumes = []
            for i in range(0, len(ticks_1h), 60):
                bucket = ticks_1h[i:i+60]
                vol = sum(t.volume for t in bucket if t.volume)
                minute_volumes.append(vol)
            
            if len(minute_volumes) > 1 and volume_1m:
                mean_vol = statistics.mean(minute_volumes)
                stdev_vol = statistics.stdev(minute_volumes)
                
                if stdev_vol > 0:
                    volume_zscore = (volume_1m - mean_vol) / stdev_vol
                else:
                    volume_zscore = 0.0
            else:
                volume_zscore = None
        else:
            volume_zscore = None
        
        return {
            'volume_5s': volume_5s,
            'volume_1m': volume_1m,
            'volume_zscore': volume_zscore
        }
    
    async def calculate_vpin(
        self,
        symbol: str,
        exchange: str,
        lookback_seconds: int = 60,
        num_buckets: int = 50
    ) -> Optional[float]:
        """
        Calculate VPIN (Volume-synchronized Probability of Informed Trading)
        
        VPIN is a measure of order flow toxicity:
        - High VPIN = high probability of informed trading (toxic flow)
        - Used for liquidity risk assessment
        
        Reference: Easley, D., López de Prado, M. M., & O'Hara, M. (2012)
        """
        # Get trade ticks
        ticks, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='TRADE',
            seconds=lookback_seconds
        )
        
        if len(ticks) < num_buckets * 2:
            return None
        
        # Calculate total volume and volume per bucket
        total_volume = sum(t.volume for t in ticks if t.volume)
        if total_volume == 0:
            return None
        
        volume_per_bucket = total_volume / num_buckets
        
        # Create volume-synchronized buckets
        buckets = []
        current_bucket = {'buy_volume': 0.0, 'sell_volume': 0.0}
        bucket_volume = 0.0
        
        for tick in ticks:
            if not tick.volume or not tick.side is not None:
                continue
            
            # Add to current bucket
            if tick.side:  # Buy
                current_bucket['buy_volume'] += tick.volume
            else:  # Sell
                current_bucket['sell_volume'] += tick.volume
            
            bucket_volume += tick.volume
            
            # Check if bucket is full
            if bucket_volume >= volume_per_bucket:
                buckets.append(current_bucket)
                current_bucket = {'buy_volume': 0.0, 'sell_volume': 0.0}
                bucket_volume = 0.0
        
        if len(buckets) < 2:
            return None
        
        # Calculate VPIN
        order_imbalances = []
        for bucket in buckets:
            total = bucket['buy_volume'] + bucket['sell_volume']
            if total > 0:
                imbalance = abs(bucket['buy_volume'] - bucket['sell_volume']) / total
                order_imbalances.append(imbalance)
        
        if order_imbalances:
            vpin = statistics.mean(order_imbalances)
            return vpin
        else:
            return None
    
    async def calculate_trade_flow_imbalance(
        self,
        symbol: str,
        exchange: str,
        seconds: int = 60
    ) -> Optional[float]:
        """
        Calculate trade flow imbalance
        
        (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        Positive = net buying pressure
        Negative = net selling pressure
        """
        ticks, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='TRADE',
            seconds=seconds
        )
        
        if not ticks:
            return None
        
        buy_volume = sum(t.volume for t in ticks if t.side is True and t.volume)
        sell_volume = sum(t.volume for t in ticks if t.side is False and t.volume)
        
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            return (buy_volume - sell_volume) / total_volume
        else:
            return None
    
    async def calculate_volatility(
        self,
        symbol: str,
        exchange: str,
        seconds: int = 60
    ) -> Optional[float]:
        """
        Calculate realized volatility (standard deviation of returns)
        """
        ticks, _, _ = await self.tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange=exchange,
            data_type='TRADE',
            seconds=seconds
        )
        
        if len(ticks) < 2:
            return None
        
        # Calculate returns
        prices = [t.price for t in ticks if t.price]
        if len(prices) < 2:
            return None
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        if len(returns) > 1:
            volatility = statistics.stdev(returns)
            # Annualize (assuming 1-second intervals)
            annualized_vol = volatility * math.sqrt(365 * 24 * 3600 / seconds)
            return annualized_vol
        else:
            return None
    
    async def calculate_all_features(
        self,
        symbol: str,
        exchange: str
    ) -> FeatureSet:
        """
        Calculate all features for a symbol
        Returns FeatureSet with all features populated
        Includes feature caching (Opus optimization)
        """
        # Check cache first
        cache_key = self._get_cache_key(symbol, exchange)
        if self._is_cache_valid(cache_key):
            return self._feature_cache[cache_key]['features']
        
        # Run all calculations in parallel
        results = await asyncio.gather(
            self.calculate_spread_features(symbol, exchange),
            self.calculate_price_features(symbol, exchange),
            self.calculate_orderbook_features(symbol, exchange),
            self.calculate_volume_features(symbol, exchange),
            self.calculate_vpin(symbol, exchange, lookback_seconds=60),
            self.calculate_trade_flow_imbalance(symbol, exchange, seconds=60),
            self.calculate_volatility(symbol, exchange, seconds=60),
            self.calculate_volatility(symbol, exchange, seconds=300),
        )
        
        spread_features = results[0]
        price_features = results[1]
        book_features = results[2]
        volume_features = results[3]
        vpin = results[4]
        flow_imbalance = results[5]
        volatility_1m = results[6]
        volatility_5m = results[7]
        
        # Create FeatureSet
        features = FeatureSet(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            exchange=exchange,
            spread_abs=spread_features.get('spread_abs'),
            spread_bps=spread_features.get('spread_bps'),
            spread_zscore=spread_features.get('spread_zscore'),
            mid_price=price_features.get('mid_price'),
            fair_value=price_features.get('fair_value'),
            book_imbalance=book_features.get('book_imbalance'),
            bid_ask_ratio=book_features.get('bid_ask_ratio'),
            volume_5s=volume_features.get('volume_5s'),
            volume_1m=volume_features.get('volume_1m'),
            volume_zscore=volume_features.get('volume_zscore'),
            vpin=vpin,
            trade_flow_imbalance=flow_imbalance,
            volatility_1m=volatility_1m,
            volatility_5m=volatility_5m
        )
        
        # Cache the result
        self._feature_cache[cache_key] = {
            'features': features,
            'timestamp': time.time()
        }
        
        return features
        
        return features


async def example_usage():
    """Example usage of FeatureEngineer"""
    
    from tiered_market_data import TieredMarketData
    
    # Initialize tiered data
    tiered_data = TieredMarketData()
    await tiered_data.start()
    
    # Initialize feature engineer
    feature_eng = FeatureEngineer(tiered_data)
    
    try:
        # Calculate all features
        features = await feature_eng.calculate_all_features(
            symbol='BTCUSDT',
            exchange='binance_spot'
        )
        
        print("📊 Calculated Features:")
        print(f"   Timestamp: {features.timestamp}")
        print(f"   Symbol: {features.symbol}")
        print(f"   Spread (abs): {features.spread_abs}")
        print(f"   Spread (bps): {features.spread_bps}")
        print(f"   Mid Price: {features.mid_price}")
        print(f"   Book Imbalance: {features.book_imbalance}")
        print(f"   VPIN: {features.vpin}")
        print(f"   Trade Flow Imbalance: {features.trade_flow_imbalance}")
        print(f"   Volatility (1m): {features.volatility_1m}")
    
    finally:
        await tiered_data.stop()


if __name__ == '__main__':
    asyncio.run(example_usage())
