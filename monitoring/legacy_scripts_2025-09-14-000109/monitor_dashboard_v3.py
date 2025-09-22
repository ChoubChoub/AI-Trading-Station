#!/usr/bin/env python3
"""
AI Trading Station - Real-time Dashboard V3
Complete dashboard with IRQ monitoring and all metrics
"""

import curses
import time
import json
import threading
import traceback
from datetime import datetime
from monitor_trading_system_v2 import TradingSystemMonitor

class TradingDashboard:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.monitor = TradingSystemMonitor()
        self.running = True
        self.last_update = None
        self.update_status = "Initializing..."
        self.latency_data = None
        
        # Setup colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Green - OK
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # Red - Error
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Yellow - Warning
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Cyan - Header
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)   # White - Normal
        
        # Hide cursor and set non-blocking
        curses.curs_set(0)
        stdscr.nodelay(1)
        stdscr.timeout(100)  # 100ms refresh
        
        # Initial metrics collection
        self.monitor.metrics = {}
    
    def draw_header(self):
        """Draw dashboard header"""
        try:
            max_y, max_x = self.stdscr.getmaxyx()
            self.stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
            header = "🚀 AI TRADING STATION - REAL-TIME MONITOR"
            self.stdscr.addstr(0, 0, "="*min(80, max_x-1))
            self.stdscr.addstr(1, (min(80, max_x)-len(header))//2, header[:max_x-1])
            self.stdscr.addstr(2, 0, "="*min(80, max_x-1))
            self.stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
        except:
            pass
    
    def draw_latency(self, row):
        """Draw latency metrics - THE MOST IMPORTANT"""
        try:
            self.stdscr.addstr(row, 0, "📊 TRADING LATENCY (Persistent Connection):", curses.A_BOLD)
            
            if self.latency_data:
                lat = self.latency_data
                # Determine color based on performance
                if lat['mean'] < 5:
                    color = 1  # Green - excellent
                elif lat['mean'] < 10:
                    color = 3  # Yellow - good
                else:
                    color = 2  # Red - needs attention
                
                # Display metrics
                self.stdscr.attron(curses.color_pair(color))
                self.stdscr.addstr(row+1, 2, f"Mean: {lat['mean']:.2f}μs")
                self.stdscr.addstr(row+1, 20, f"P99: {lat['p99']:.2f}μs")
                self.stdscr.addstr(row+1, 36, f"Min: {lat['min']:.2f}μs")
                self.stdscr.addstr(row+1, 52, f"Max: {lat['max']:.2f}μs")
                self.stdscr.attroff(curses.color_pair(color))
                
                # Performance message
                if lat['mean'] < 5:
                    self.stdscr.addstr(row+2, 2, "🏆 WORLD-CLASS PERFORMANCE! (<5μs)", curses.color_pair(1) | curses.A_BOLD)
                elif lat['mean'] < 10:
                    self.stdscr.addstr(row+2, 2, "✅ EXCELLENT - Ready for trading", curses.color_pair(1))
                else:
                    self.stdscr.addstr(row+2, 2, "⚠️ Above target threshold", curses.color_pair(3))
            else:
                self.stdscr.addstr(row+1, 2, f"Collecting measurements...", curses.color_pair(3))
                if self.last_update:
                    elapsed = (datetime.now() - self.last_update).total_seconds()
                    self.stdscr.addstr(row+2, 2, f"Next update in {max(0, 5-elapsed):.0f}s", curses.color_pair(5))
        except Exception as e:
            pass
        
        return row + 4
    
    def draw_cpu_and_irq(self, row):
        """Draw CPU isolation and IRQ affinity status"""
        try:
            # CPU Isolation
            self.stdscr.addstr(row, 0, "🔧 CPU ISOLATION:", curses.A_BOLD)
            cpu = self.monitor.metrics.get('cpu_isolation', {})
            if cpu:
                color = 1 if cpu.get('status') == 'OK' else 2
                self.stdscr.attron(curses.color_pair(color))
                self.stdscr.addstr(row+1, 2, f"Cores {cpu.get('isolated_cpus', [2,3])}: {cpu.get('status', 'N/A')}")
                self.stdscr.attroff(curses.color_pair(color))
                
                if cpu.get('violations'):
                    self.stdscr.addstr(row+2, 2, f"Violations: {', '.join(cpu['violations'][:50])}", curses.color_pair(2))
                    row += 1
            else:
                self.stdscr.addstr(row+1, 2, "Cores [2,3]: Checking...", curses.color_pair(5))
            
            # IRQ Affinity - ADDED BACK!
            self.stdscr.addstr(row+3, 0, "🔌 IRQ AFFINITY:", curses.A_BOLD)
            irq = self.monitor.metrics.get('irq_affinity', {})
            if irq:
                color = 1 if irq.get('status') == 'OK' else 2
                self.stdscr.attron(curses.color_pair(color))
                self.stdscr.addstr(row+4, 2, f"Status: {irq.get('status', 'N/A')}")
                self.stdscr.attroff(curses.color_pair(color))
                
                if irq.get('violations'):
                    violations_text = ', '.join(irq['violations'][:2])  # Show first 2 violations
                    if len(irq['violations']) > 2:
                        violations_text += f" (+{len(irq['violations'])-2} more)"
                    self.stdscr.addstr(row+5, 2, f"⚠️ {violations_text[:70]}", curses.color_pair(2))
                    row += 1
                else:
                    self.stdscr.addstr(row+5, 2, "✅ No IRQ violations on isolated CPUs", curses.color_pair(1))
            else:
                self.stdscr.addstr(row+4, 2, "Checking IRQ configuration...", curses.color_pair(5))
                
        except Exception as e:
            self.stdscr.addstr(row+1, 2, f"Error: {str(e)[:50]}", curses.color_pair(2))
        
        return row + 7
    
    def draw_network(self, row):
        """Draw network status"""
        try:
            self.stdscr.addstr(row, 0, "🌐 NETWORK INTERFACES:", curses.A_BOLD)
            net = self.monitor.metrics.get('network', {})
            
            if net:
                for i, (iface, stats) in enumerate(net.items()):
                    if i < 2:  # Show first 2 interfaces
                        if stats.get('status') == 'OK':
                            color = 1
                        elif stats.get('status') == 'WARNING':
                            color = 3
                        else:
                            color = 2
                        
                        self.stdscr.attron(curses.color_pair(color))
                        drops = stats.get('dropin', 0) + stats.get('dropout', 0)
                        self.stdscr.addstr(row+1+i, 2, f"{iface}: {stats.get('status', 'N/A')}")
                        self.stdscr.addstr(row+1+i, 25, f"Drops: {drops}")
                        
                        # Show packet counts if available
                        if 'packets_sent' in stats:
                            self.stdscr.addstr(row+1+i, 40, f"TX: {stats['packets_sent']:,}")
                        self.stdscr.attroff(curses.color_pair(color))
            else:
                self.stdscr.addstr(row+1, 2, "Collecting network stats...", curses.color_pair(5))
                
        except Exception as e:
            pass
        
        return row + 4
    
    def draw_gpu(self, row):
        """Draw GPU status"""
        try:
            self.stdscr.addstr(row, 0, "🎮 GPU STATUS:", curses.A_BOLD)
            gpus = self.monitor.metrics.get('gpu', [])
            
            if gpus:
                for i, gpu in enumerate(gpus[:2]):  # Show first 2 GPUs
                    temp = gpu.get('temperature', 0)
                    if temp < 70:
                        color = 1  # Green
                    elif temp < 85:
                        color = 3  # Yellow
                    else:
                        color = 2  # Red
                    
                    self.stdscr.attron(curses.color_pair(color))
                    self.stdscr.addstr(row+1+i, 2, f"GPU{i}: {temp:.0f}°C")
                    self.stdscr.attroff(curses.color_pair(color))
                    
                    self.stdscr.addstr(row+1+i, 15, f"Load: {gpu.get('utilization', 0):.0f}%")
                    
                    mem_used = gpu.get('memory_used', 0)
                    mem_total = gpu.get('memory_total', 1)
                    mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
                    self.stdscr.addstr(row+1+i, 30, f"Mem: {mem_used:.0f}/{mem_total:.0f}MB ({mem_pct:.0f}%)")
            else:
                self.stdscr.addstr(row+1, 2, "No GPU data available", curses.color_pair(5))
                
        except Exception as e:
            pass
        
        return row + 4
    
    def draw_alerts(self, row):
        """Draw recent alerts"""
        try:
            self.stdscr.addstr(row, 0, "⚠️ RECENT ALERTS:", curses.A_BOLD)
            
            if self.monitor.alerts:
                alerts = list(self.monitor.alerts)[-3:]  # Show last 3 alerts
                for i, alert in enumerate(alerts):
                    self.stdscr.addstr(row+1+i, 2, f"• {alert[:75]}", curses.color_pair(3))
            else:
                self.stdscr.addstr(row+1, 2, "✅ No alerts - System running smoothly", curses.color_pair(1))
                
        except Exception as e:
            pass
        
        return row + 5
    
    def draw_footer(self, row):
        """Draw footer with status"""
        try:
            max_y, max_x = self.stdscr.getmaxyx()
            self.stdscr.addstr(min(row, max_y-5), 0, "="*min(80, max_x-1))
            self.stdscr.addstr(min(row+1, max_y-4), 0, "Press 'q' to quit | 'r' to refresh | Updates every 5s", curses.color_pair(5))
            
            if self.last_update:
                update_time = self.last_update.strftime('%H:%M:%S UTC')
                self.stdscr.addstr(min(row+2, max_y-3), 0, f"Last update: {update_time}", curses.color_pair(5))
            
            # GitHub credit
            self.stdscr.addstr(min(row+3, max_y-2), 0, "GitHub: ChoubChoub/AI-Trading-Station | Target: <10μs", curses.color_pair(4))
            
        except Exception as e:
            pass
    
    def update_metrics(self):
        """Background thread to update metrics"""
        while self.running:
            try:
                self.update_status = "Collecting metrics..."
                metrics = self.monitor.collect_all_metrics()
                
                if metrics and metrics.get('latency'):
                    self.latency_data = metrics['latency']
                    self.last_update = datetime.now()
                    self.update_status = "Updated"
                
                # Sleep for 5 seconds between updates
                for _ in range(50):  # 50 * 0.1 = 5 seconds
                    if not self.running:
                        break
                    time.sleep(0.1)
                    
            except Exception as e:
                self.update_status = f"Error: {str(e)[:30]}"
                time.sleep(5)
    
    def run(self):
        """Main dashboard loop"""
        # Start background metrics update
        update_thread = threading.Thread(target=self.update_metrics, daemon=True)
        update_thread.start()
        
        # Give it a moment to collect initial metrics
        time.sleep(1)
        
        while self.running:
            try:
                self.stdscr.clear()
                max_y, max_x = self.stdscr.getmaxyx()
                
                # Draw all components
                self.draw_header()
                row = 4
                
                row = self.draw_latency(row)
                row = self.draw_cpu_and_irq(row)  # Now includes IRQ!
                row = self.draw_network(row)
                row = self.draw_gpu(row)
                row = self.draw_alerts(row)
                self.draw_footer(row)
                
                self.stdscr.refresh()
                
                # Handle keyboard input
                key = self.stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    self.running = False
                elif key == ord('r') or key == ord('R'):
                    # Force refresh
                    self.monitor.collect_all_metrics()
                    
            except curses.error:
                # Terminal resize or other curses errors
                pass
            except Exception as e:
                # Log error but keep running
                pass
            
            time.sleep(0.1)

def main(stdscr):
    """Main entry point"""
    try:
        dashboard = TradingDashboard(stdscr)
        dashboard.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Dashboard error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    curses.wrapper(main)
