#!/bin/bash
echo 'Restoring original toggle script...'
cp /usr/local/bin/toggle_trading_mode.sh.backup /usr/local/bin/toggle_trading_mode.sh
rm -f /usr/local/bin/tm 2>/dev/null
echo 'Original system restored'
