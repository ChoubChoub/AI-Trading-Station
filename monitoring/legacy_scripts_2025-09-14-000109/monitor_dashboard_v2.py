#!/usr/bin/env python3
"""
AI Trading Station - Real-time Dashboard V2
Shows persistent connection latency (the real trading metric)
"""

import curses
import time
import json
import threading
from monitor_trading_system_v2 import TradingSystemMonitor

class TradingDashboard:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.monitor = TradingSystemMonitor()
        self.running = True
        self.last_update = time.time()
        
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
        # Hide cursor
        curses.curs_set(0)
        stdscr.nodelay(1)
    
    def draw_header(self):
        """Draw dashboard header"""
        self.stdscr.attron(curses.color_pair(4))
        header = "🚀 AI TRADING STATION - REAL-TIME MONITOR"
        self.stdscr.addstr(0, 0, "="*80)
        self.stdscr.addstr(1, (80-len(header))//2, header)
        self.stdscr.addstr(2, 0, "="*80)
        self.stdscr.attroff(curses.color_pair(4))
    
    def draw_latency(self, row):
        """Draw latency metrics - THE MOST IMPORTANT"""
        self.stdscr.addstr(row, 0, "📊 TRADING LATENCY (Persistent Connection):", curses.A_BOLD)
        
        if self.monitor.metrics.get('latency'):
            lat = self.monitor.metrics['latency']
            color = 1 if lat['mean'] < 10 else (3 if lat['mean'] < 20 else 2)
            
            self.stdscr.attron(curses.color_pair(color))
            self.stdscr.addstr(row+1, 2, f"Mean: {lat['mean']:.2f}μs")
            self.stdscr.addstr(row+1, 20, f"P99: {lat['p99']:.2f}μs")
            self.stdscr.addstr(row+1, 36, f"Min: {lat['min']:.2f}μs")
            self.stdscr.addstr(row+1, 52, f"Max: {lat['max']:.2f}μs")
            self.stdscr.attroff(curses.color_pair(color))
            
            # Performance indicator
            if lat['mean'] < 5:
                self.stdscr.addstr(row+2, 2, "🏆 WORLD-CLASS PERFORMANCE!", curses.color_pair(1))
            elif lat['mean'] < 10:
                self.stdscr.addstr(row+2, 2, "✅ EXCELLENT - Ready for trading", curses.color_pair(1))
            else:
                self.stdscr.addstr(row+2, 2, "⚠️ Above target threshold", curses.color_pair(3))
        else:
            self.stdscr.addstr(row+1, 2, "Measuring...", curses.color_pair(3))
        
        return row + 3
    
    def draw_cpu_status(self, row):
        """Draw CPU isolation status"""
        self.stdscr.addstr(row, 0, "🔧 CPU ISOLATION:", curses.A_BOLD)
        
        if self.monitor.metrics.get('cpu_isolation'):
            cpu = self.monitor.metrics['cpu_isolation']
            color = 1 if cpu['status'] == 'OK' else 2
            
            self.stdscr.attron(curses.color_pair(color))
            self.stdscr.addstr(row+1, 2, f"Cores {cpu['isolated_cpus']}: {cpu['status']}")
            self.stdscr.attroff(curses.color_pair(color))
            
            if cpu['violations']:
                self.stdscr.addstr(row+2, 2, f"Violations: {', '.join(cpu['violations'])}", curses.color_pair(2))
                return row + 3
        
        return row + 2
    
    def draw_network_status(self, row):
        """Draw network status"""
        self.stdscr.addstr(row, 0, "🌐 NETWORK:", curses.A_BOLD)
        
        if self.monitor.metrics.get('network'):
            for i, (iface, stats) in enumerate(self.monitor.metrics['network'].items()):
                color = 1 if stats['status'] == 'OK' else (3 if stats['status'] == 'WARNING' else 2)
                self.stdscr.attron(curses.color_pair(color))
                self.stdscr.addstr(row+1+i, 2, f"{iface}: {stats['status']}")
                if 'dropin' in stats:
                    self.stdscr.addstr(row+1+i, 20, f"Drops: {stats['dropin']+stats['dropout']}")
                self.stdscr.attroff(curses.color_pair(color))
        
        return row + 3
    
    def draw_gpu_status(self, row):
        """Draw GPU status"""
        self.stdscr.addstr(row, 0, "🎮 GPU STATUS:", curses.A_BOLD)
        
        if self.monitor.metrics.get('gpu'):
            for i, gpu in enumerate(self.monitor.metrics['gpu']):
                self.stdscr.addstr(row+1+i, 2, f"GPU{i}: {gpu['temperature']:.0f}°C")
                self.stdscr.addstr(row+1+i, 20, f"Load: {gpu['utilization']:.0f}%")
                self.stdscr.addstr(row+1+i, 35, f"Mem: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f}MB")
        
        return row + 3
    
    def draw_alerts(self, row):
        """Draw recent alerts"""
        self.stdscr.addstr(row, 0, "⚠️ ALERTS:", curses.A_BOLD)
        
        if self.monitor.alerts:
            for i, alert in enumerate(list(self.monitor.alerts)[-3:]):
                self.stdscr.addstr(row+1+i, 2, f"• {alert}", curses.color_pair(3))
            return row + 4
        else:
            self.stdscr.addstr(row+1, 2, "No alerts", curses.color_pair(1))
            return row + 2
    
    def update_metrics(self):
        """Update metrics in background"""
        while self.running:
            self.monitor.collect_all_metrics()
            time.sleep(5)
    
    def run(self):
        """Main dashboard loop"""
        # Start metrics update thread
        update_thread = threading.Thread(target=self.update_metrics, daemon=True)
        update_thread.start()
        
        while self.running:
            self.stdscr.clear()
            
            # Draw components
            self.draw_header()
            row = 4
            row = self.draw_latency(row)
            row += 1
            row = self.draw_cpu_status(row)
            row += 1
            row = self.draw_network_status(row)
            row += 1
            row = self.draw_gpu_status(row)
            row += 1
            row = self.draw_alerts(row)
            
            # Footer
            self.stdscr.addstr(row + 2, 0, "="*80)
            self.stdscr.addstr(row + 3, 0, "Press 'q' to quit | Updates every 5s | Target latency: <10μs")
            self.stdscr.addstr(row + 4, 0, f"Last update: {time.strftime('%H:%M:%S')}")
            
            self.stdscr.refresh()
            
            # Check for quit
            key = self.stdscr.getch()
            if key == ord('q'):
                self.running = False
            
            time.sleep(0.1)

def main(stdscr):
    dashboard = TradingDashboard(stdscr)
    dashboard.run()

if __name__ == '__main__':
    curses.wrapper(main)
