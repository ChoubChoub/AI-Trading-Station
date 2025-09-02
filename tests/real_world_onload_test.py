#!/usr/bin/env python3

"""
Real-World OnLoad Testing Suite
===============================
Purpose: Simulate realistic trading scenarios with OnLoad acceleration
Scenarios: Market data processing, order execution, risk calculations
Performance: Target sub-10μs latency in production-like conditions
"""

import os
import sys
import time
import json
import socket
import struct
import threading
import multiprocessing
import queue
import random
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

@dataclass
class MarketData:
    """Market data message structure."""
    symbol: str
    price: float
    quantity: int
    timestamp: float
    sequence: int

@dataclass
class Order:
    """Trading order structure."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: int
    timestamp: float

@dataclass
class TradeExecution:
    """Trade execution result."""
    order_id: str
    executed_price: float
    executed_quantity: int
    execution_time: float
    latency_us: float

class MarketDataSimulator:
    """Simulates high-frequency market data feeds."""
    
    def __init__(self, symbols: List[str], update_rate_hz: int = 10000):
        self.symbols = symbols
        self.update_rate_hz = update_rate_hz
        self.sequence = 0
        self.prices = {symbol: 100.0 + random.uniform(-10, 10) for symbol in symbols}
        
    def generate_market_data(self) -> MarketData:
        """Generate realistic market data update."""
        symbol = random.choice(self.symbols)
        
        # Simulate price movement
        price_change = random.uniform(-0.01, 0.01)
        self.prices[symbol] = max(0.01, self.prices[symbol] + price_change)
        
        self.sequence += 1
        
        return MarketData(
            symbol=symbol,
            price=round(self.prices[symbol], 2),
            quantity=random.randint(100, 10000),
            timestamp=time.time_ns() / 1e9,
            sequence=self.sequence
        )

class TradingStrategy:
    """Simple momentum trading strategy."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.price_history = {symbol: [] for symbol in symbols}
        self.positions = {symbol: 0 for symbol in symbols}
        self.order_id_counter = 0
        
    def process_market_data(self, market_data: MarketData) -> Optional[Order]:
        """Process market data and potentially generate orders."""
        symbol = market_data.symbol
        price = market_data.price
        
        # Update price history
        self.price_history[symbol].append(price)
        if len(self.price_history[symbol]) > 10:
            self.price_history[symbol].pop(0)
        
        # Simple momentum strategy
        if len(self.price_history[symbol]) >= 5:
            recent_prices = self.price_history[symbol][-5:]
            if len(recent_prices) >= 2:
                price_momentum = recent_prices[-1] - recent_prices[0]
                
                # Generate buy signal
                if price_momentum > 0.05 and self.positions[symbol] < 10000:
                    self.order_id_counter += 1
                    return Order(
                        order_id=f"BUY_{self.order_id_counter}",
                        symbol=symbol,
                        side="BUY",
                        price=price + 0.01,  # Slightly above market
                        quantity=1000,
                        timestamp=time.time_ns() / 1e9
                    )
                
                # Generate sell signal
                elif price_momentum < -0.05 and self.positions[symbol] > -10000:
                    self.order_id_counter += 1
                    return Order(
                        order_id=f"SELL_{self.order_id_counter}",
                        symbol=symbol,
                        side="SELL",
                        price=price - 0.01,  # Slightly below market
                        quantity=1000,
                        timestamp=time.time_ns() / 1e9
                    )
        
        return None

class OrderExecutionSimulator:
    """Simulates order execution with realistic latency."""
    
    def __init__(self):
        self.execution_delay_us = random.uniform(0.5, 2.0)  # 0.5-2μs execution delay
        
    def execute_order(self, order: Order) -> TradeExecution:
        """Simulate order execution."""
        start_time = time.time_ns()
        
        # Simulate execution processing time
        time.sleep(self.execution_delay_us / 1e6)  # Convert to seconds
        
        end_time = time.time_ns()
        latency_us = (end_time - start_time) / 1000.0
        
        # Simulate partial fill with slippage
        slippage = random.uniform(-0.001, 0.001)
        executed_price = order.price + slippage
        executed_quantity = order.quantity  # Full fill for simplicity
        
        return TradeExecution(
            order_id=order.order_id,
            executed_price=executed_price,
            executed_quantity=executed_quantity,
            execution_time=end_time / 1e9,
            latency_us=latency_us
        )

class RealWorldOnLoadTest:
    """Real-world OnLoad performance testing framework."""
    
    def __init__(self, use_onload: bool = True):
        self.use_onload = use_onload
        self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "SPY"]
        self.test_duration = 60  # seconds
        self.target_latency_us = 4.37
        
        # Performance metrics
        self.market_data_latencies = []
        self.order_execution_latencies = []
        self.end_to_end_latencies = []
        
        # Components
        self.market_simulator = MarketDataSimulator(self.symbols)
        self.trading_strategy = TradingStrategy(self.symbols)
        self.execution_simulator = OrderExecutionSimulator()
        
    def setup_onload_environment(self):
        """Configure OnLoad environment variables."""
        if self.use_onload:
            os.environ.update({
                'EF_POLL_USEC': '0',
                'EF_INT_DRIVEN': '0',
                'EF_RX_TIMESTAMPING': '1',
                'EF_TX_TIMESTAMPING': '1',
                'EF_PRECISE_TIMESTAMPS': '1',
                'EF_NAME': 'trading-realworld-test'
            })
            print("OnLoad environment configured for real-world testing")
        else:
            print("Running baseline test without OnLoad")
    
    def create_market_data_server(self, host: str, port: int) -> None:
        """Market data server that broadcasts data to clients."""
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
            server.bind((host, port))
            server.listen(10)
            
            print(f"Market data server listening on {host}:{port}")
            
            clients = []
            start_time = time.time()
            
            # Accept connections
            server.settimeout(1.0)
            
            while time.time() - start_time < self.test_duration:
                try:
                    client, addr = server.accept()
                    client.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
                    clients.append(client)
                    print(f"Market data client connected: {addr}")
                except socket.timeout:
                    continue
                
                # Broadcast market data to all clients
                for client in clients[:]:
                    try:
                        market_data = self.market_simulator.generate_market_data()
                        
                        # Serialize market data
                        data_dict = asdict(market_data)
                        message = json.dumps(data_dict).encode('utf-8')
                        message_length = struct.pack('!I', len(message))
                        
                        send_start = time.time_ns()
                        client.send(message_length + message)
                        send_end = time.time_ns()
                        
                        send_latency = (send_end - send_start) / 1000.0  # microseconds
                        self.market_data_latencies.append(send_latency)
                        
                        # Control update rate
                        time.sleep(1.0 / self.market_simulator.update_rate_hz)
                        
                    except (BrokenPipeError, ConnectionResetError):
                        clients.remove(client)
                        client.close()
                        continue
            
            # Cleanup
            for client in clients:
                client.close()
            server.close()
            
        except Exception as e:
            print(f"Market data server error: {e}")
    
    def create_trading_client(self, host: str, port: int) -> List[float]:
        """Trading client that processes market data and generates orders."""
        client_latencies = []
        
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
            client.connect((host, port))
            
            print(f"Trading client connected to {host}:{port}")
            
            start_time = time.time()
            message_buffer = b''
            
            while time.time() - start_time < self.test_duration - 5:  # Stop 5 seconds early
                try:
                    # Receive message length
                    while len(message_buffer) < 4:
                        data = client.recv(4 - len(message_buffer))
                        if not data:
                            break
                        message_buffer += data
                    
                    if len(message_buffer) < 4:
                        break
                    
                    message_length = struct.unpack('!I', message_buffer[:4])[0]
                    message_buffer = message_buffer[4:]
                    
                    # Receive message data
                    while len(message_buffer) < message_length:
                        data = client.recv(message_length - len(message_buffer))
                        if not data:
                            break
                        message_buffer += data
                    
                    if len(message_buffer) < message_length:
                        break
                    
                    # Process received market data
                    receive_time = time.time_ns()
                    message_data = message_buffer[:message_length]
                    message_buffer = message_buffer[message_length:]
                    
                    # Deserialize market data
                    try:
                        data_dict = json.loads(message_data.decode('utf-8'))
                        market_data = MarketData(**data_dict)
                        
                        # Calculate market data latency
                        data_age_us = (receive_time / 1e9 - market_data.timestamp) * 1e6
                        
                        # Process with trading strategy
                        strategy_start = time.time_ns()
                        order = self.trading_strategy.process_market_data(market_data)
                        strategy_end = time.time_ns()
                        
                        strategy_latency = (strategy_end - strategy_start) / 1000.0
                        
                        if order:
                            # Simulate order execution
                            execution_start = time.time_ns()
                            execution = self.execution_simulator.execute_order(order)
                            execution_end = time.time_ns()
                            
                            execution_latency = (execution_end - execution_start) / 1000.0
                            
                            # Calculate end-to-end latency
                            end_to_end_latency = data_age_us + strategy_latency + execution_latency
                            
                            client_latencies.append(end_to_end_latency)
                            self.order_execution_latencies.append(execution.latency_us)
                            self.end_to_end_latencies.append(end_to_end_latency)
                    
                    except json.JSONDecodeError:
                        continue
                
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Client processing error: {e}")
                    break
            
            client.close()
            
        except Exception as e:
            print(f"Trading client error: {e}")
        
        return client_latencies
    
    def run_real_world_simulation(self) -> Dict[str, Any]:
        """Run complete real-world trading simulation."""
        print("=" * 60)
        print("REAL-WORLD ONLOAD TRADING SIMULATION")
        print("=" * 60)
        
        self.setup_onload_environment()
        
        # Test configuration
        host = "127.0.0.1"
        port = 12350
        
        results = {
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_duration': self.test_duration,
                'symbols': self.symbols,
                'onload_enabled': self.use_onload,
                'target_latency_us': self.target_latency_us
            },
            'performance_results': {},
            'trading_statistics': {}
        }
        
        try:
            # Start market data server in background
            server_process = multiprocessing.Process(
                target=self.create_market_data_server,
                args=(host, port)
            )
            server_process.start()
            
            time.sleep(2)  # Allow server to start
            
            # Run multiple trading clients
            num_clients = 3
            client_processes = []
            
            for i in range(num_clients):
                client_process = multiprocessing.Process(
                    target=self.create_trading_client,
                    args=(host, port)
                )
                client_processes.append(client_process)
                client_process.start()
            
            # Wait for test completion
            print(f"Running real-world simulation for {self.test_duration} seconds...")
            
            # Monitor progress
            for i in range(self.test_duration):
                time.sleep(1)
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{self.test_duration} seconds")
            
            # Stop all processes
            for client_process in client_processes:
                client_process.terminate()
                client_process.join(timeout=5)
            
            server_process.terminate()
            server_process.join(timeout=5)
            
            print("Real-world simulation completed")
            
            # Analyze results
            results['performance_results'] = self.analyze_performance_results()
            results['trading_statistics'] = self.calculate_trading_statistics()
            
        except Exception as e:
            print(f"Simulation error: {e}")
            results['error'] = str(e)
        
        return results
    
    def analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance metrics from the simulation."""
        performance = {}
        
        # Market data latency analysis
        if self.market_data_latencies:
            market_data_array = np.array(self.market_data_latencies)
            performance['market_data_latency'] = {
                'mean_us': float(np.mean(market_data_array)),
                'median_us': float(np.median(market_data_array)),
                'p95_us': float(np.percentile(market_data_array, 95)),
                'p99_us': float(np.percentile(market_data_array, 99)),
                'max_us': float(np.max(market_data_array)),
                'std_dev_us': float(np.std(market_data_array)),
                'sample_count': len(self.market_data_latencies)
            }
        
        # Order execution latency analysis
        if self.order_execution_latencies:
            execution_array = np.array(self.order_execution_latencies)
            performance['order_execution_latency'] = {
                'mean_us': float(np.mean(execution_array)),
                'median_us': float(np.median(execution_array)),
                'p95_us': float(np.percentile(execution_array, 95)),
                'p99_us': float(np.percentile(execution_array, 99)),
                'max_us': float(np.max(execution_array)),
                'std_dev_us': float(np.std(execution_array)),
                'sample_count': len(self.order_execution_latencies)
            }
        
        # End-to-end latency analysis
        if self.end_to_end_latencies:
            e2e_array = np.array(self.end_to_end_latencies)
            performance['end_to_end_latency'] = {
                'mean_us': float(np.mean(e2e_array)),
                'median_us': float(np.median(e2e_array)),
                'p95_us': float(np.percentile(e2e_array, 95)),
                'p99_us': float(np.percentile(e2e_array, 99)),
                'max_us': float(np.max(e2e_array)),
                'std_dev_us': float(np.std(e2e_array)),
                'sample_count': len(self.end_to_end_latencies)
            }
            
            # Performance evaluation
            mean_latency = float(np.mean(e2e_array))
            p95_latency = float(np.percentile(e2e_array, 95))
            p99_latency = float(np.percentile(e2e_array, 99))
            
            performance['target_evaluation'] = {
                'mean_target_met': mean_latency <= self.target_latency_us,
                'p95_below_5us': p95_latency <= 5.0,
                'p99_below_10us': p99_latency <= 10.0,
                'overall_performance': 'EXCELLENT' if mean_latency <= self.target_latency_us else 'GOOD' if mean_latency <= 10.0 else 'NEEDS_IMPROVEMENT'
            }
        
        return performance
    
    def calculate_trading_statistics(self) -> Dict[str, Any]:
        """Calculate trading-related statistics."""
        stats = {
            'total_market_updates': len(self.market_data_latencies),
            'total_orders_generated': len(self.order_execution_latencies),
            'total_executions': len(self.end_to_end_latencies),
            'order_generation_rate': len(self.order_execution_latencies) / self.test_duration if self.test_duration > 0 else 0,
            'execution_rate': len(self.end_to_end_latencies) / self.test_duration if self.test_duration > 0 else 0
        }
        
        # Calculate symbol distribution
        symbol_counts = {symbol: 0 for symbol in self.symbols}
        # This would be populated from actual order data in a real implementation
        
        stats['symbol_distribution'] = symbol_counts
        
        return stats
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print formatted results summary."""
        print("\n" + "=" * 60)
        print("REAL-WORLD SIMULATION RESULTS")
        print("=" * 60)
        
        metadata = results.get('test_metadata', {})
        print(f"Test Duration: {metadata.get('test_duration', 0)} seconds")
        print(f"OnLoad Enabled: {metadata.get('onload_enabled', False)}")
        print(f"Symbols Tested: {len(metadata.get('symbols', []))}")
        
        performance = results.get('performance_results', {})
        
        # End-to-end latency (most important)
        e2e = performance.get('end_to_end_latency', {})
        if e2e:
            print(f"\nEnd-to-End Trading Latency:")
            print(f"  Mean:     {e2e.get('mean_us', 0):.3f}μs (target: ≤{self.target_latency_us}μs)")
            print(f"  Median:   {e2e.get('median_us', 0):.3f}μs")
            print(f"  P95:      {e2e.get('p95_us', 0):.3f}μs")
            print(f"  P99:      {e2e.get('p99_us', 0):.3f}μs")
            print(f"  Max:      {e2e.get('max_us', 0):.3f}μs")
            print(f"  Samples:  {e2e.get('sample_count', 0)}")
        
        # Performance evaluation
        evaluation = performance.get('target_evaluation', {})
        if evaluation:
            print(f"\nPerformance Evaluation:")
            print(f"  Mean Target Met: {'✓' if evaluation.get('mean_target_met') else '✗'}")
            print(f"  P95 < 5μs: {'✓' if evaluation.get('p95_below_5us') else '✗'}")
            print(f"  P99 < 10μs: {'✓' if evaluation.get('p99_below_10us') else '✗'}")
            print(f"  Overall: {evaluation.get('overall_performance', 'UNKNOWN')}")
        
        # Trading statistics
        trading_stats = results.get('trading_statistics', {})
        if trading_stats:
            print(f"\nTrading Statistics:")
            print(f"  Market Updates: {trading_stats.get('total_market_updates', 0)}")
            print(f"  Orders Generated: {trading_stats.get('total_orders_generated', 0)}")
            print(f"  Order Rate: {trading_stats.get('order_generation_rate', 0):.1f} orders/sec")

def main():
    """Main execution function."""
    use_onload = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if '--no-onload' in sys.argv:
            use_onload = False
        if '--help' in sys.argv:
            print("Real-World OnLoad Trading Test")
            print("Usage: python3 real_world_onload_test.py [--no-onload] [--help]")
            print("Options:")
            print("  --no-onload    Run without OnLoad acceleration")
            print("  --help         Show this help message")
            return 0
    
    # Run the test
    test = RealWorldOnLoadTest(use_onload=use_onload)
    results = test.run_real_world_simulation()
    
    # Save results
    output_file = f"real_world_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    test.print_results_summary(results)
    
    # Return exit code based on performance
    performance = results.get('performance_results', {})
    evaluation = performance.get('target_evaluation', {})
    overall_perf = evaluation.get('overall_performance', 'UNKNOWN')
    
    return 0 if overall_perf in ['EXCELLENT', 'GOOD'] else 1

if __name__ == "__main__":
    exit(main())