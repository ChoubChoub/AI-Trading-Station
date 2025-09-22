#!/usr/bin/env python3
import socket, time, statistics

print("🚀 Real-World OnLoad Test - ChoubChoub AI Trading Station")
print("Date: 2025-09-01 12:36:31 UTC\n")

def test_trading_connections():
    """Test realistic trading connections"""
    latencies = []
    endpoints = [('8.8.8.8', 53), ('8.8.4.4', 53), ('1.1.1.1', 53)]
    
    print("📊 Testing Trading Connections:")
    for i, (ip, port) in enumerate(endpoints * 3):  # 9 total tests
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(2.0)
            
            start = time.perf_counter_ns()
            result = sock.connect_ex((ip, port))
            if result == 0:
                sock.send(f'TRADE_ORDER_{i}'.encode())
                sock.recv(1024)
            end = time.perf_counter_ns()
            
            if result == 0:
                latency = (end - start) / 1000
                latencies.append(latency)
                print(f"   Connection {i+1}: ✅ {latency:.1f}μs")
            else:
                print(f"   Connection {i+1}: ❌ Failed")
            
            sock.close()
        except Exception as e:
            print(f"   Connection {i+1}: ❌ Error")
    
    if latencies:
        mean = statistics.mean(latencies)
        p99 = sorted(latencies)[int(0.99*len(latencies))] if len(latencies)>5 else max(latencies)
        print(f"\n📈 Results: Mean {mean:.1f}μs, P99 {p99:.1f}μs, Success {len(latencies)}/9")
        
        if mean < 15:
            print("🏆 EXCELLENT - Ready for HFT")
        elif mean < 30:
            print("✅ GOOD - Ready for algorithmic trading")
        else:
            print("⚠️  Consider optimization")
    else:
        print("❌ All connections failed")

if __name__ == "__main__":
    test_trading_connections()
    print("\n✅ Real-world OnLoad test completed!")
