#!/usr/bin/env python3
"""
Test Cache Integration - Day 4 Validation
==========================================
Verify that tiered cache is being populated by batch writer.
"""

import asyncio
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, '/home/youssefbahloul/ai-trading-station/Services/QuestDB/Config')

from tiered_market_data import TieredMarketData
from feature_engineering import FeatureEngineer


async def test_cache_population():
    """Test that cache is populated with recent data."""
    
    print("=" * 80)
    print("🧪 Testing Tiered Cache Integration (Day 4)")
    print("=" * 80)
    print()
    
    # Initialize cache connection
    tiered_data = TieredMarketData()
    await tiered_data.start()
    
    print("✅ Connected to tiered cache")
    print()
    
    # Test symbols
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
    
    print("📊 Testing cache queries for recent data:")
    print("-" * 80)
    
    for symbol in test_symbols:
        # Query last 5 seconds of data
        ticks, tier, latency_ms = await tiered_data.get_recent_ticks(
            symbol=symbol,
            exchange='binance_spot',
            data_type='TRADE',
            seconds=5
        )
        
        print(f"{symbol:12} | {len(ticks):4} ticks | Tier: {tier:4} | Latency: {latency_ms:6.2f}ms | ", end='')
        
        if len(ticks) > 0:
            latest_tick = ticks[-1]
            age_ms = (datetime.now() - latest_tick.timestamp).total_seconds() * 1000
            print(f"Latest: {latest_tick.price:10.2f} | Age: {age_ms:6.0f}ms")
        else:
            print("NO DATA")
    
    print()
    
    # Get cache statistics
    print("📈 Cache Statistics:")
    print("-" * 80)
    stats = tiered_data.get_stats()
    
    cache_stats = stats['cache_stats']
    hot_stats = stats['hot_cache']
    
    print(f"Hot cache items: {hot_stats['total_items']:8,} ticks in {hot_stats['num_streams']} streams")
    print(f"Total queries:   {cache_stats['total_queries']:8,}")
    print(f"Hot hits:        {cache_stats['hot_hits']:8,} ({cache_stats['hot_hit_rate_pct']:5.2f}%)")
    print(f"Warm hits:       {cache_stats['warm_hits']:8,}")
    print(f"Cold queries:    {cache_stats['cold_hits']:8,}")
    print(f"Misses:          {cache_stats['misses']:8,}")
    
    overall_hit_rate = cache_stats['cache_hit_rate_pct']
    print(f"\n🎯 Overall hit rate: {overall_hit_rate:.2f}%")
    
    if overall_hit_rate >= 95.0:
        print("✅ PASS: Hit rate exceeds 95% target")
    elif overall_hit_rate >= 90.0:
        print("⚠️  WARN: Hit rate below 95% target but above 90%")
    else:
        print("❌ FAIL: Hit rate below 90%")
    
    print()
    
    # Test feature engineering with cache
    print("🔧 Testing Feature Engineering:")
    print("-" * 80)
    
    feature_eng = FeatureEngineer(tiered_data)
    
    # Query last 60 seconds for feature calculation
    ticks, tier, latency_ms = await tiered_data.get_recent_ticks(
        symbol='BTCUSDT',
        exchange='binance_spot',
        data_type='TRADE',
        seconds=60
    )
    
    if len(ticks) >= 10:
        features = await feature_eng.calculate_all_features(ticks)
        print(f"✅ Calculated {len(features)} features from {len(ticks)} ticks")
        print(f"   Query tier: {tier} | Latency: {latency_ms:.2f}ms")
        
        # Show sample features
        sample_features = ['vwap', 'bid_ask_spread', 'volume_imbalance', 'price_momentum']
        print("\nSample features:")
        for key in sample_features:
            if key in features:
                print(f"  {key:20}: {features[key]:12.6f}")
    else:
        print(f"⚠️  Not enough data for features ({len(ticks)} ticks)")
    
    print()
    print("=" * 80)
    print("✅ Cache Integration Test Complete")
    print("=" * 80)
    
    await tiered_data.close()


if __name__ == '__main__':
    asyncio.run(test_cache_population())
