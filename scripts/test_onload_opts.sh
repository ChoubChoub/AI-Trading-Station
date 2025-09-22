#!/bin/bash
test_option() {
    local opt="$1"
    local val="$2"
    export "$opt=$val"
    if onload /bin/true 2>&1 | grep -q "Unknown option \"$opt"; then
        echo "❌ $opt=$val - NOT SUPPORTED"
    else
        echo "✅ $opt=$val - SUPPORTED"
    fi
    unset "$opt"
}

echo "Testing Onload options..."
test_option "EF_POLL_USEC" "0"
test_option "EF_INT_DRIVEN" "0"
test_option "EF_SPIN_USEC" "1000000"
test_option "EF_CLUSTER_CORE_AFFINITY" "2,3"
test_option "EF_RXQ_SIZE" "2048"
test_option "EF_TXQ_SIZE" "1024"
test_option "EF_TCP_RECV_NONBLOCK" "1"
test_option "EF_TCP_SEND_NONBLOCK" "1"
test_option "EF_RX_MERGE" "1"
test_option "EF_TX_PUSH" "1"
test_option "EF_TCP_TMT_MODE" "1"
test_option "EF_CTPIO_MODE" "sf-np"
test_option "EF_TCP_SYN_OPTS" "3"
test_option "EF_TCP_RCVLOWAT" "1"
test_option "EF_PREFAULT_PACKETS" "1"
test_option "EF_EVENT_TIMER_QUANTUM" "1"
test_option "EF_PIO" "1"
test_option "EF_CTPIO_THRESH" "64"
test_option "EF_RX_TIMESTAMPING" "1"
test_option "EF_TX_TIMESTAMPING" "1"
