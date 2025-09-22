#!/usr/bin/env python3

# Read the broken file
with open('monitor_dashboard_complete.py', 'r') as f:
    content = f.read()

# Find and replace the broken draw_gpu function
broken_part = '''    def draw_gpu(self, row):
    """Draw GPU status"""
    try:
       self.stdscr.addstr(row, 0, "🎮 GPU STATUS:", curses.A_BOLD)
           metrics = self.monitor.metrics
           gpus = metrics.get('gpu', [])'''

fixed_part = '''    def draw_gpu(self, row):
        """Draw GPU status"""
        try:
            self.stdscr.addstr(row, 0, "🎮 GPU STATUS:", curses.A_BOLD)
            metrics = self.monitor.metrics
            gpus = metrics.get('gpu', [])'''

# Replace the broken part
content = content.replace(broken_part, fixed_part)

# Write the fixed content
with open('monitor_dashboard_complete_fixed.py', 'w') as f:
    f.write(content)

print("Fixed! Created monitor_dashboard_complete_fixed.py")
