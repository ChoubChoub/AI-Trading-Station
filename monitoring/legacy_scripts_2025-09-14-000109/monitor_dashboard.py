#!/usr/bin/env python3
"""
AI Trading Station Monitor Dashboard
Real-time monitoring dashboard with terminal UI
"""
import curses
import json
import time
import threading
from datetime import datetime
from pathlib import Path
import subprocess
import psutil
from collections import deque
class MonitorDashboard:
    def __init__(self, config_file='config/monitor_config.json'):
        """Initialize dashboard"""
        self.config = self.load_config(config_file)
        self.running = True
        self.data_lock = threading.Lock()
        # Data storage
        self.cpu_history = {i: deque(maxlen=60) for i in range(psutil.cpu_count())}
        self.isolated_cpu_history = {i: deque(maxlen=60) for i in self.config['cpu']['isolated_cores']}
        self.network_history = {}
        self.gpu_data = []
        self.alerts = deque(maxlen=20)
        self.last_update = None
    def load_config(self, config_file):
        """Load configuration"""
        config_path = Path(__file__).parent.parent / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "cpu": {"isolated_cores": [2, 3], "temp_threshold": 75},
                "network": {"trading_interfaces": ["enp130s0f0", "enp130s0f1"]},
                "gpu": {"temp_threshold": 80}
            }
    def update_data(self):
        """Update monitoring data in background"""
        while self.running:
            try:
                with self.data_lock:
                    # Update CPU data
                    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                    for i, usage in enumerate(cpu_percent):
                        if i in self.cpu_history:
                            self.cpu_history[i].append(usage)
                        if i in self.isolated_cpu_history:
                            self.isolated_cpu_history[i].append(usage)
                    # Update network data
                    for iface in self.config['network']['trading_interfaces']:
                        if iface not in self.network_history:
                            self.network_history[iface] = {
                                'bandwidth_in': deque(maxlen=60),
                                'bandwidth_out': deque(maxlen=60),
                                'errors': 0,
                                'drops': 0
                            }
                        stats = psutil.net_io_counters(pernic=True).get(iface)
                        if stats:
                            # Calculate bandwidth (simplified)
                            self.network_history[iface]['bandwidth_in'].append(stats.bytes_recv)
                            self.network_history[iface]['bandwidth_out'].append(stats.bytes_sent)
                            self.network_history[iface]['errors'] = stats.errin + stats.errout
                            self.network_history[iface]['drops'] = stats.dropin + stats.dropout
                    # Update GPU data
                    self.update_gpu_data()
                    # Check for alerts
                    self.check_alerts()
                    self.last_update = datetime.now()
            except Exception as e:
                self.alerts.append(f"Update error: {str(e)}")
            time.sleep(1)
    def update_gpu_data(self):
        """Update GPU information"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total', 
                 '--format=csv,noheader'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                self.gpu_data = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        self.gpu_data.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'temp': int(parts[2]),
                            'util': int(parts[3].rstrip(' %')),
                            'mem_used': parts[4],
                            'mem_total': parts[5]
                        })
        except:
            pass
    def check_alerts(self):
        """Check for system alerts"""
        # Check isolated CPU usage
        for core in self.config['cpu']['isolated_cores']:
            if core < len(self.cpu_history) and self.cpu_history[core]:
                usage = self.cpu_history[core][-1]
                if usage > 5:  # Alert if isolated core usage > 5%
                    self.alerts.append(f"High usage on isolated core {core}: {usage:.1f}%")
        # Check GPU temperature
        for gpu in self.gpu_data:
            if gpu['temp'] > self.config['gpu']['temp_threshold']:
                self.alerts.append(f"GPU {gpu['index']} temp high: {gpu['temp']}°C")
    def draw_header(self, stdscr, y, x, width):
        """Draw dashboard header"""
        title = "AI Trading Station Monitor Dashboard"
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(y, x + (width - len(title)) // 2, title)
        stdscr.attroff(curses.color_pair(2))
        if self.last_update:
            update_str = f"Last update: {self.last_update.strftime('%H:%M:%S')}"
            stdscr.addstr(y + 1, x + width - len(update_str) - 1, update_str)
        return y + 3
    def draw_cpu_panel(self, stdscr, y, x, width, height):
        """Draw CPU monitoring panel"""
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(y, x, "█" * width)
        stdscr.addstr(y + 1, x, "█ CPU MONITORING" + " " * (width - 17) + "█")
        stdscr.addstr(y + 2, x, "█" * width)
        stdscr.attroff(curses.color_pair(1))
        row = y + 3
        # Isolated cores status
        stdscr.addstr(row, x + 2, f"Isolated Cores: {self.config['cpu']['isolated_cores']}")
        row += 1
        with self.data_lock:
            for core in self.config['cpu']['isolated_cores']:
                if core in self.isolated_cpu_history and self.isolated_cpu_history[core]:
                    usage = self.isolated_cpu_history[core][-1]
                    bar_width = int((width - 20) * usage / 100)
                    color = curses.color_pair(3) if usage < 5 else curses.color_pair(4)
                    stdscr.attron(color)
                    stdscr.addstr(row, x + 2, f"Core {core:2d}: [{usage:5.1f}%] ")
                    stdscr.addstr("█" * bar_width)
                    stdscr.attroff(color)
                    row += 1
            # All cores summary
            row += 1
            stdscr.addstr(row, x + 2, "All Cores:")
            row += 1
            for i in range(min(8, len(self.cpu_history))):  # Show first 8 cores
                if self.cpu_history[i]:
                    usage = self.cpu_history[i][-1]
                    status = "ISOL" if i in self.config['cpu']['isolated_cores'] else "    "
                    stdscr.addstr(row, x + 2, f"Core {i}: {usage:5.1f}% {status}")
                    if (row - y) < height - 1:
                        row += 1
        return row + 1
    def draw_gpu_panel(self, stdscr, y, x, width, height):
        """Draw GPU monitoring panel"""
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(y, x, "█" * width)
        stdscr.addstr(y + 1, x, "█ GPU MONITORING" + " " * (width - 17) + "█")
        stdscr.addstr(y + 2, x, "█" * width)
        stdscr.attroff(curses.color_pair(1))
        row = y + 3
        with self.data_lock:
            for gpu in self.gpu_data:
                if (row - y) < height - 1:
                    # GPU info
                    stdscr.addstr(row, x + 2, f"GPU {gpu['index']}: {gpu['name'][:30]}")
                    row += 1
                    # Temperature
                    temp_color = curses.color_pair(3) if gpu['temp'] < self.config['gpu']['temp_threshold'] else curses.color_pair(4)
                    stdscr.attron(temp_color)
                    stdscr.addstr(row, x + 4, f"Temp: {gpu['temp']}°C")
                    stdscr.attroff(temp_color)
                    # Utilization
                    stdscr.addstr(row, x + 20, f"Util: {gpu['util']}%")
                    # Memory
                    stdscr.addstr(row, x + 35, f"Mem: {gpu['mem_used']}/{gpu['mem_total']}")
                    row += 2
        return row
    def draw_network_panel(self, stdscr, y, x, width, height):
        """Draw network monitoring panel"""
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(y, x, "█" * width)
        stdscr.addstr(y + 1, x, "█ NETWORK MONITORING" + " " * (width - 21) + "█")
        stdscr.addstr(y + 2, x, "█" * width)
        stdscr.attroff(curses.color_pair(1))
        row = y + 3
        with self.data_lock:
            for iface in self.config['network']['trading_interfaces']:
                if iface in self.network_history and (row - y) < height - 1:
                    data = self.network_history[iface]
                    stdscr.addstr(row, x + 2, f"Interface: {iface}")
                    row += 1
                    # Show errors/drops with color coding
                    err_color = curses.color_pair(3) if data['errors'] == 0 else curses.color_pair(4)
                    stdscr.attron(err_color)
                    stdscr.addstr(row, x + 4, f"Errors: {data['errors']}")
                    stdscr.attroff(err_color)
                    drop_color = curses.color_pair(3) if data['drops'] == 0 else curses.color_pair(4)
                    stdscr.attron(drop_color)
                    stdscr.addstr(row, x + 20, f"Drops: {data['drops']}")
                    stdscr.attroff(drop_color)
                    row += 2
        return row
    def draw_alerts_panel(self, stdscr, y, x, width, height):
        """Draw alerts panel"""
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(y, x, "█" * width)
        stdscr.addstr(y + 1, x, "█ ALERTS" + " " * (width - 9) + "█")
        stdscr.addstr(y + 2, x, "█" * width)
        stdscr.attroff(curses.color_pair(1))
        row = y + 3
        with self.data_lock:
            if not self.alerts:
                stdscr.attron(curses.color_pair(3))
                stdscr.addstr(row, x + 2, "✓ No alerts")
                stdscr.attroff(curses.color_pair(3))
            else:
                for alert in list(self.alerts)[-5:]:  # Show last 5 alerts
                    if (row - y) < height - 1:
                        stdscr.attron(curses.color_pair(4))
                        stdscr.addstr(row, x + 2, f"! {alert[:width-5]}")
                        stdscr.attroff(curses.color_pair(4))
                        row += 1
        return row
    def draw(self, stdscr):
        """Main drawing function"""
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        # Initialize colors
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Headers
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Title
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Good
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)     # Alert
        # Start data update thread
        update_thread = threading.Thread(target=self.update_data)
        update_thread.start()
        try:
            while self.running:
                stdscr.clear()
                height, width = stdscr.getmaxyx()
                # Draw header
                y = self.draw_header(stdscr, 0, 0, width)
                # Calculate panel dimensions
                panel_height = (height - y - 1) // 2
                left_width = width // 2 - 1
                right_width = width - left_width - 1
                # Draw panels
                self.draw_cpu_panel(stdscr, y, 0, left_width, panel_height)
                self.draw_gpu_panel(stdscr, y, left_width + 1, right_width, panel_height)
                y += panel_height
                self.draw_network_panel(stdscr, y, 0, left_width, panel_height)
                self.draw_alerts_panel(stdscr, y, left_width + 1, right_width, panel_height)
                # Footer
                stdscr.addstr(height - 1, 0, "Press 'q' to quit, 'r' to refresh")
                stdscr.refresh()
                # Handle input
                key = stdscr.getch()
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    with self.data_lock:
                        self.alerts.clear()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            update_thread.join()
def main():
    dashboard = MonitorDashboard()
    curses.wrapper(dashboard.draw)
if __name__ == '__main__':
    main()
