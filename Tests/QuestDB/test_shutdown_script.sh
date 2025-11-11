#!/bin/bash
###############################################################################
# Shutdown Script Verification Test
# Tests the complete graceful shutdown and restart cycle
###############################################################################

echo "🔧 Testing Graceful Shutdown Script"
echo "=====================================---"

echo "1️⃣ Checking service status before shutdown..."
systemctl is-active questdb.service binance-trades.service binance-bookticker.service batch-writer.service

echo "2️⃣ Running graceful shutdown..."
sudo /home/youssefbahloul/ai-trading-station/Services/System/shutdown-market-data.sh

echo "3️⃣ Verifying all services stopped..."
sleep 2
echo "QuestDB: $(systemctl is-active questdb.service)"
echo "Trades: $(systemctl is-active binance-trades.service)"  
echo "BookTicker: $(systemctl is-active binance-bookticker.service)"
echo "Batch Writer: $(systemctl is-active batch-writer.service)"
echo "Redis: $(systemctl is-active redis-hft.service)"

echo "4️⃣ Restarting services..."
sudo systemctl start questdb.service
sleep 8
sudo systemctl start binance-trades.service binance-bookticker.service batch-writer.service

echo "5️⃣ Verifying services restarted..."
sleep 5
systemctl is-active questdb.service binance-trades.service binance-bookticker.service batch-writer.service

echo "6️⃣ Checking shutdown service persistence..."
echo "Enabled: $(systemctl is-enabled shutdown-market-data.service)"
echo "Timeout: $(systemctl show shutdown-market-data.service --property=TimeoutStartUSec --value)"
echo "Targets: $(systemctl show shutdown-market-data.service --property=WantedBy --value)"

echo "✅ Shutdown script test complete!"
echo "📄 Check logs: tail /home/youssefbahloul/ai-trading-station/Services/Monitoring/logs/shutdown.log"