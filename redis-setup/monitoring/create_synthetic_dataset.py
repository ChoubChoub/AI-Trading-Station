#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Redis jemalloc Testing
Based on GPT analysis - create realistic workload for fragmentation assessment

Purpose: Generate proper baseline with 5M keys and realistic churn patterns
Author: AI Trading Station Phase 3A
Date: September 28, 2025
"""

import redis
import random
import time
import json
import argparse
import sys
from typing import List, Tuple, Dict

class SyntheticDatasetGenerator:
    def __init__(self, host='127.0.0.1', port=6379, db=0):
        """Initialize Redis connection for dataset generation"""
        try:
            self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.redis.ping()
            print(f"✅ Connected to Redis at {host}:{port}")
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            sys.exit(1)
    
    def generate_key_sizes(self, count: int, sizes: List[int]) -> List[Tuple[str, bytes, int]]:
        """Generate keys with specified size distribution"""
        keys = []
        size_weights = [1, 2, 1]  # Favor 512B keys for realistic distribution
        
        print(f"🔧 Generating {count:,} keys with sizes {sizes}")
        
        for i in range(count):
            # Select size based on weights
            size = random.choices(sizes, weights=size_weights)[0]
            
            # Generate key and value
            key = f"synthetic:key:{i:08d}"
            value = b'x' * (size - len(key.encode()) - 10)  # Adjust for key overhead
            
            keys.append((key, value, size))
            
            if (i + 1) % 100000 == 0:
                print(f"  Generated {i+1:,}/{count:,} keys ({(i+1)/count*100:.1f}%)")
        
        return keys
    
    def load_dataset(self, keys: List[Tuple[str, bytes, int]], batch_size: int = 1000) -> Dict:
        """Load dataset into Redis with batching for performance"""
        print(f"📊 Loading {len(keys):,} keys into Redis (batch size: {batch_size})")
        
        start_time = time.time()
        total_size = 0
        
        # Get initial memory stats
        initial_memory = self.get_memory_stats()
        
        # Load in batches using pipeline
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            
            with self.redis.pipeline() as pipe:
                for key, value, size in batch:
                    pipe.set(key, value)
                    total_size += size
                pipe.execute()
            
            if (i + batch_size) % 50000 == 0:
                progress = min(i + batch_size, len(keys))
                print(f"  Loaded {progress:,}/{len(keys):,} keys ({progress/len(keys)*100:.1f}%)")
        
        load_time = time.time() - start_time
        final_memory = self.get_memory_stats()
        
        # Calculate statistics
        stats = {
            'keys_loaded': len(keys),
            'total_logical_size_mb': total_size / (1024 * 1024),
            'load_time_seconds': load_time,
            'keys_per_second': len(keys) / load_time,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_growth_mb': (final_memory['used_memory'] - initial_memory['used_memory']) / (1024 * 1024)
        }
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Keys: {stats['keys_loaded']:,}")
        print(f"   Logical size: {stats['total_logical_size_mb']:.1f} MB")
        print(f"   Load time: {stats['load_time_seconds']:.1f}s")
        print(f"   Memory growth: {stats['memory_growth_mb']:.1f} MB")
        
        return stats
    
    def run_churn_pattern(self, duration: int, rate: int, expire_ratio: float = 0.3) -> Dict:
        """Run realistic churn pattern: SET+EXPIRE, updates, deletes"""
        print(f"🔄 Running churn pattern for {duration}s at {rate} ops/sec")
        
        start_time = time.time()
        operations = 0
        sets = 0
        expires = 0
        updates = 0
        deletes = 0
        
        # Get key count for random access
        key_count = self.redis.dbsize()
        
        while time.time() - start_time < duration:
            loop_start = time.time()
            ops_this_second = 0
            
            while ops_this_second < rate and time.time() - start_time < duration:
                op_type = random.choice(['set', 'update', 'expire', 'delete'])
                
                if op_type == 'set':
                    # Create new temporary key
                    key = f"churn:temp:{operations}"
                    value = b'x' * random.choice([64, 512, 1024])
                    self.redis.set(key, value)
                    sets += 1
                
                elif op_type == 'update':
                    # Update existing key
                    key_id = random.randint(0, min(key_count - 1, 1000000))
                    key = f"synthetic:key:{key_id:08d}"
                    value = b'updated:' + b'x' * random.choice([64, 512])
                    self.redis.set(key, value)
                    updates += 1
                
                elif op_type == 'expire':
                    # Set expiration on temporary keys
                    key = f"churn:temp:{random.randint(max(0, operations - 1000), operations)}"
                    self.redis.expire(key, random.randint(1, 10))
                    expires += 1
                
                elif op_type == 'delete':
                    # Delete old temporary keys
                    key = f"churn:temp:{random.randint(max(0, operations - 2000), operations - 500)}"
                    self.redis.delete(key)
                    deletes += 1
                
                operations += 1
                ops_this_second += 1
            
            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
            
            if operations % (rate * 10) == 0:
                elapsed_total = time.time() - start_time
                print(f"  Churn progress: {elapsed_total:.0f}s, {operations:,} ops")
        
        total_time = time.time() - start_time
        final_memory = self.get_memory_stats()
        
        stats = {
            'duration': total_time,
            'total_operations': operations,
            'ops_per_second': operations / total_time,
            'sets': sets,
            'updates': updates,
            'expires': expires,
            'deletes': deletes,
            'final_memory': final_memory
        }
        
        print(f"✅ Churn pattern completed!")
        print(f"   Duration: {stats['duration']:.1f}s")
        print(f"   Operations: {stats['total_operations']:,}")
        print(f"   Rate: {stats['ops_per_second']:.0f} ops/sec")
        
        return stats
    
    def get_memory_stats(self) -> Dict:
        """Get comprehensive Redis memory statistics"""
        info = self.redis.info('memory')
        
        # Key memory metrics
        stats = {
            'used_memory': info.get('used_memory', 0),
            'used_memory_human': info.get('used_memory_human', '0B'),
            'used_memory_rss': info.get('used_memory_rss', 0),
            'used_memory_peak': info.get('used_memory_peak', 0),
            'mem_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0.0),
            'allocator_allocated': info.get('allocator_allocated', 0),
            'allocator_active': info.get('allocator_active', 0),
            'allocator_resident': info.get('allocator_resident', 0),
            'allocator_fragmentation_ratio': info.get('allocator_fragmentation_ratio', 0.0),
            'rss_overhead_ratio': info.get('rss_overhead_ratio', 0.0)
        }
        
        return stats
    
    def generate_report(self, load_stats: Dict, churn_stats: Dict = None) -> Dict:
        """Generate comprehensive fragmentation and performance report"""
        final_memory = self.get_memory_stats()
        key_count = self.redis.dbsize()
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': {
                'total_keys': key_count,
                'logical_size_mb': load_stats['total_logical_size_mb'],
                'load_performance': {
                    'keys_per_second': load_stats['keys_per_second'],
                    'load_time': load_stats['load_time_seconds']
                }
            },
            'memory_analysis': {
                'used_memory_mb': final_memory['used_memory'] / (1024 * 1024),
                'rss_mb': final_memory['used_memory_rss'] / (1024 * 1024),
                'fragmentation_ratio': final_memory['mem_fragmentation_ratio'],
                'allocator_fragmentation': final_memory['allocator_fragmentation_ratio'],
                'rss_overhead_ratio': final_memory['rss_overhead_ratio']
            },
            'fragmentation_assessment': self.assess_fragmentation(final_memory),
            'raw_memory_stats': final_memory
        }
        
        if churn_stats:
            report['churn_analysis'] = {
                'operations_per_second': churn_stats['ops_per_second'],
                'operation_breakdown': {
                    'sets': churn_stats['sets'],
                    'updates': churn_stats['updates'],
                    'expires': churn_stats['expires'],
                    'deletes': churn_stats['deletes']
                }
            }
        
        return report
    
    def assess_fragmentation(self, memory_stats: Dict) -> Dict:
        """Assess fragmentation levels with realistic thresholds"""
        frag_ratio = memory_stats['mem_fragmentation_ratio']
        used_mb = memory_stats['used_memory'] / (1024 * 1024)
        
        # GPT-informed thresholds based on dataset size
        if used_mb < 10:
            # Small dataset - fragmentation ratio less meaningful
            if frag_ratio > 10:
                level = "HIGH (but normal for small dataset)"
            elif frag_ratio > 5:
                level = "MEDIUM (normal for small dataset)" 
            else:
                level = "LOW"
        elif used_mb < 100:
            # Medium dataset - more reliable fragmentation assessment
            if frag_ratio > 3:
                level = "HIGH"
            elif frag_ratio > 2:
                level = "MEDIUM"
            else:
                level = "LOW"
        else:
            # Large dataset - fragmentation ratio most reliable
            if frag_ratio > 2:
                level = "HIGH"
            elif frag_ratio > 1.5:
                level = "MEDIUM"
            else:
                level = "LOW"
        
        return {
            'level': level,
            'fragmentation_ratio': frag_ratio,
            'dataset_size_mb': used_mb,
            'assessment_reliability': 'HIGH' if used_mb > 50 else 'MEDIUM' if used_mb > 10 else 'LOW'
        }

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic Redis dataset for jemalloc testing')
    parser.add_argument('--keys', type=int, default=5000000, help='Number of keys to generate')
    parser.add_argument('--sizes', type=str, default='64,512,4096', help='Key sizes (comma-separated)')
    parser.add_argument('--churn-rate', type=int, default=1000, help='Churn operations per second')
    parser.add_argument('--churn-duration', type=int, default=300, help='Churn duration in seconds')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--output', type=str, default='synthetic_baseline.json', help='Output report file')
    parser.add_argument('--skip-churn', action='store_true', help='Skip churn pattern (load only)')
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    print("🚀 Redis Synthetic Dataset Generator")
    print(f"   Target: {args.keys:,} keys")
    print(f"   Sizes: {sizes}")
    print(f"   Host: {args.host}:{args.port}")
    
    # Initialize generator
    generator = SyntheticDatasetGenerator(args.host, args.port)
    
    # Generate and load dataset
    keys = generator.generate_key_sizes(args.keys, sizes)
    load_stats = generator.load_dataset(keys)
    
    # Run churn pattern if requested
    churn_stats = None
    if not args.skip_churn:
        churn_stats = generator.run_churn_pattern(args.churn_duration, args.churn_rate)
    
    # Generate report
    report = generator.generate_report(load_stats, churn_stats)
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Report saved to {args.output}")
    print("\n=== FRAGMENTATION ASSESSMENT ===")
    frag = report['fragmentation_assessment']
    print(f"Fragmentation Level: {frag['level']}")
    print(f"Fragmentation Ratio: {frag['fragmentation_ratio']:.2f}")
    print(f"Dataset Size: {frag['dataset_size_mb']:.1f} MB")
    print(f"Assessment Reliability: {frag['assessment_reliability']}")
    
    if frag['assessment_reliability'] == 'LOW':
        print("\n⚠️  Dataset too small for reliable fragmentation assessment")
        print("   Consider using --keys 1000000 or higher for meaningful results")

if __name__ == '__main__':
    main()