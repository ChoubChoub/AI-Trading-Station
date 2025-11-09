#!/bin/bash
# ==============================================================================
# Desktop Mode Toggle Script for AI Trading Station
# ==============================================================================
# Purpose: Switch between server mode (multi-user) and desktop mode (graphical)
# 
# Industry Standard: Algo trading systems run in server mode by default
# Desktop mode is available on-demand for troubleshooting and configuration
# ==============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script must be run as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root or with sudo${NC}"
   exit 1
fi

# Function to display current mode
show_current_mode() {
    local current=$(systemctl get-default)
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Current System Mode:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [[ "$current" == "graphical.target" ]]; then
        echo -e "Mode: ${GREEN}Desktop Mode (graphical.target)${NC}"
        echo -e "Display Manager: $(systemctl is-active lightdm.service 2>/dev/null || echo 'inactive')"
    else
        echo -e "Mode: ${GREEN}Server Mode (multi-user.target)${NC} ${YELLOW}[Recommended]${NC}"
        echo -e "Display Manager: $(systemctl is-active lightdm.service 2>/dev/null || echo 'inactive')"
    fi
    echo ""
}

# Function to switch to server mode
switch_to_server() {
    echo -e "${YELLOW}Switching to Server Mode (multi-user.target)...${NC}"
    echo -e "${BLUE}This is the recommended mode for algo trading systems.${NC}"
    echo ""
    
    # Set default target
    systemctl set-default multi-user.target
    
    # Stop display manager if running
    if systemctl is-active --quiet lightdm.service; then
        echo "Stopping LightDM display manager..."
        systemctl stop lightdm.service
    fi
    
    echo -e "${GREEN}✓ System configured for Server Mode${NC}"
    echo ""
    echo "Changes will take full effect after reboot."
    echo "Desktop services are stopped immediately."
    echo ""
}

# Function to switch to desktop mode
switch_to_desktop() {
    echo -e "${YELLOW}Switching to Desktop Mode (graphical.target)...${NC}"
    echo -e "${BLUE}Desktop mode will be available on next boot or immediately if started.${NC}"
    echo ""
    
    # Set default target
    systemctl set-default graphical.target
    
    echo -e "${GREEN}✓ System configured for Desktop Mode${NC}"
    echo ""
    echo "To start desktop immediately: sudo systemctl start lightdm.service"
    echo "Or reboot to start desktop automatically."
    echo ""
}

# Function to start desktop temporarily (without changing default)
start_desktop_temp() {
    echo -e "${YELLOW}Starting Desktop Mode temporarily...${NC}"
    echo -e "${BLUE}This will not change the default boot mode.${NC}"
    echo ""
    
    if systemctl is-active --quiet lightdm.service; then
        echo -e "${YELLOW}Desktop is already running.${NC}"
        return 0
    fi
    
    systemctl start lightdm.service
    echo -e "${GREEN}✓ Desktop mode started${NC}"
    echo "Desktop is now accessible. Connect a monitor or use remote desktop."
    echo "To stop: sudo systemctl stop lightdm.service"
    echo ""
}

# Function to stop desktop temporarily
stop_desktop_temp() {
    echo -e "${YELLOW}Stopping Desktop Mode...${NC}"
    echo ""
    
    if ! systemctl is-active --quiet lightdm.service; then
        echo -e "${YELLOW}Desktop is not running.${NC}"
        return 0
    fi
    
    systemctl stop lightdm.service
    echo -e "${GREEN}✓ Desktop mode stopped${NC}"
    echo ""
}

# Display usage information
show_usage() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Desktop Mode Toggle - AI Trading Station${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Usage: sudo $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  status              Show current system mode"
    echo "  server              Set server mode as default (recommended)"
    echo "  desktop             Set desktop mode as default"
    echo "  start               Start desktop temporarily (keep server as default)"
    echo "  stop                Stop desktop (keep current default)"
    echo "  help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  sudo $0 status      # Check current mode"
    echo "  sudo $0 server      # Set server mode as default"
    echo "  sudo $0 start       # Start desktop now without changing default"
    echo ""
    echo -e "${YELLOW}Recommended Setup:${NC}"
    echo "  1. Keep server mode as default for optimal trading performance"
    echo "  2. Use 'start' command when desktop access is needed"
    echo "  3. Use 'stop' command when done with desktop work"
    echo ""
}

# Main script logic
main() {
    case "${1:-status}" in
        status)
            show_current_mode
            ;;
        server)
            show_current_mode
            switch_to_server
            show_current_mode
            ;;
        desktop)
            show_current_mode
            switch_to_desktop
            show_current_mode
            ;;
        start)
            show_current_mode
            start_desktop_temp
            ;;
        stop)
            show_current_mode
            stop_desktop_temp
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo -e "${RED}Error: Unknown command '$1'${NC}"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
