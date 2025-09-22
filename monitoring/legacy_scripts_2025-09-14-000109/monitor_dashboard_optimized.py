#!/usr/bin/env python3
"""
AI Trading Station - Optimized Dashboard (~4.5μs latency)
"""

import curses
import time
import threading
from datetime import datetime
from monitor_trading_system_optimized import TradingSystemMonitor

class TradingDashboard:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.monitor = TradingSystemMonitor()
        self.running = True
        self.update_count = 0
        
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
        curses.curs_set(0)
        stdscr.nodelay(1)
    
    def update_metrics_worker(self):
        """Background worker - updates every 10s (latency test takes ~5s)"""
        while self.running:
            try:
                self.monitor.collect_all_metrics()
                self.update_count += 1
                # Longer interval since each test takes time
                time.sleep(10)
            except:
                time.sleep(10)
    
    def run(self):
        """Main dashboard loop"""
        # Start updater
        updater = threading.Thread(target=self.update_metrics_worker, daemon=True)
        updater.start()
        
        # Initial collection
        self.monitor.collect_all_metrics()
        
        while self.running:
            try:
                self.stdscr.clear()
                
                # Header
                self.stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
                self.stdscr.addstr(0, 0, "="*80)
                self.stdscr.addstr(1, 18, "🚀 AI TRADING STATION - OPTIMIZED MONITOR")
                self.stdscr.addstr(2, 0, "="*80)
                self.stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
                
                metrics = self.monitor.metrics
                row = 4
                
                # Latency - should be ~4.5μs
                self.stdscr.addstr(row, 0, f"📊 TRADING LATENCY [Update #{self.update_count}]:", curses.A_BOLD)
                if metrics.get('latency'):
                    lat = metrics['latency']
                    color = 1 if lat['mean'] < 5 else (3 if lat['mean'] < 10 else 2)
                    self.stdscr.attron(curses.color_pair(color))
                    self.stdscr.addstr(row+1, 2, 
                        f"Mean: {lat['mean']:.2f}μs  "
                        f"P99: {lat['p99']:.2f}μs  "
                        f"Min: {lat['min']:.2f}μs  "
                        f"Max: {lat['max']:.2f}μs")
                    self.stdscr.attroff(curses.color_pair(color))
                    
                    if lat['mean'] < 5:
                        self.stdscr.addstr(row+2, 2, "🏆 WORLD-CLASS PERFORMANCE! Target: 4.5μs", curses.color_pair(1))
                
                row += 4
                
                # Rest of metrics...
                self.stdscr.addstr(row, 0, "🔧 CPU:", curses.A_BOLD)
                if metrics.get('cpu_isolation'):
                    status = metrics['cpu_isolation']['status']
                    self.stdscr.addstr(row, 10, f"Cores [2,3] {status}")
                
                self.stdscr.addstr(row+1, 0, "🔌 IRQ:", curses.A_BOLD)
                if metrics.get('irq_affinity'):
                    status = metrics['irq_affinity']['status']
                    self.stdscr.addstr(row+1, 10, f"{status}")
                
                row += 3
                
                # Footer
                self.stdscr.addstr(row, 0, "="*80)
                self.stdscr.addstr(row+1, 0, "Press 'q' to quit | Updates every 10s (full test cycle)")
                
                self.stdscr.refresh()
                
                key = self.stdscr.getch()
                if key == ord('q'):
                    self.running = False
                
                time.sleep(0.1)
                
            except:
                pass

def main(stdscr):
    dashboard = TradingDashboard(stdscr)
    dashboard.run()

if __name__ == '__main__':
    curses.wrapper(main)
