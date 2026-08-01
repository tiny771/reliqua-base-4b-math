#!/bin/bash
# Reliquary Checkpoint Manager - Bash Wrapper
# Simple command-line interface for checkpoint management

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/checkpoint_manager.py"
MODEL_DIR="${RELIQUARY_MODEL_DIR:-$HOME/reliquary-miner/model}"
STATE_ENDPOINT="${RELIQUARY_STATE_ENDPOINT:-http://localhost:8000/state}"
POLL_INTERVAL="${RELIQUARY_POLL_INTERVAL:-60}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=== Reliquary Checkpoint Manager ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    start          Start continuous monitoring (background)
    stop           Stop background monitoring process
    status         Check if monitoring is running
    once           Download latest checkpoint once and exit
    check          Check current checkpoint info
    monitor        Start continuous monitoring (foreground)
    logs           Show recent logs (if running as systemd service)
    help           Show this help message

Options:
    --endpoint URL     Override /state endpoint URL
    --model-dir PATH   Override model download directory
    --interval SECS    Override polling interval (seconds)

Examples:
    $0 start                           # Start background monitoring
    $0 monitor                         # Monitor in foreground
    $0 once                            # One-time download
    $0 check                           # Check current checkpoint
    $0 status                          # Check if running
    $0 logs                            # View service logs
    $0 stop                            # Stop background service

Environment Variables:
    RELIQUARY_STATE_ENDPOINT          /state endpoint URL
    RELIQUARY_MODEL_DIR               Model directory path
    RELIQUARY_POLL_INTERVAL           Polling interval in seconds

EOF
    exit 1
}

cmd_check() {
    print_header
    print_info "Fetching /state from $STATE_ENDPOINT"
    python3 "$PYTHON_SCRIPT" --once --endpoint "$STATE_ENDPOINT" --model-dir "$MODEL_DIR"
}

cmd_once() {
    print_header
    print_info "Downloading latest checkpoint (one-time)"
    python3 "$PYTHON_SCRIPT" \
        --once \
        --endpoint "$STATE_ENDPOINT" \
        --model-dir "$MODEL_DIR"
    print_success "Download complete"
}

cmd_monitor_fg() {
    print_header
    print_info "Starting continuous monitoring (foreground)"
    print_info "Endpoint: $STATE_ENDPOINT"
    print_info "Model dir: $MODEL_DIR"
    print_info "Poll interval: $POLL_INTERVAL seconds"
    print_info "Press Ctrl+C to stop"
    echo ""
    
    python3 "$PYTHON_SCRIPT" \
        --endpoint "$STATE_ENDPOINT" \
        --model-dir "$MODEL_DIR" \
        --poll-interval "$POLL_INTERVAL"
}

cmd_monitor_bg() {
    # Check if already running
    if cmd_status > /dev/null 2>&1; then
        print_error "Monitoring already running (PID: $?)"
        exit 1
    fi
    
    print_header
    print_info "Starting continuous monitoring (background)"
    print_info "Endpoint: $STATE_ENDPOINT"
    print_info "Model dir: $MODEL_DIR"
    print_info "Poll interval: $POLL_INTERVAL seconds"
    
    # Start in background with nohup
    nohup python3 "$PYTHON_SCRIPT" \
        --endpoint "$STATE_ENDPOINT" \
        --model-dir "$MODEL_DIR" \
        --poll-interval "$POLL_INTERVAL" \
        > "$SCRIPT_DIR/.checkpoint_manager.log" 2>&1 &
    
    local PID=$!
    sleep 1
    
    if kill -0 $PID 2>/dev/null; then
        print_success "Started with PID $PID"
        print_info "Logs: tail -f $SCRIPT_DIR/.checkpoint_manager.log"
    else
        print_error "Failed to start (check logs above)"
        exit 1
    fi
}

cmd_stop() {
    print_header
    
    # Try to stop via systemd first
    if systemctl is-active --quiet reliquary-checkpoint-manager.service 2>/dev/null; then
        print_info "Stopping systemd service..."
        sudo systemctl stop reliquary-checkpoint-manager.service
        print_success "Service stopped"
        return 0
    fi
    
    # Try to find and kill background process
    local pids=$(pgrep -f "checkpoint_manager.py" | grep -v "$$" || true)
    if [ -n "$pids" ]; then
        print_info "Killing processes: $pids"
        echo "$pids" | xargs kill -9
        print_success "Processes killed"
    else
        print_error "No running monitoring process found"
        exit 1
    fi
}

cmd_status() {
    print_header
    
    # Check systemd service
    if systemctl is-active --quiet reliquary-checkpoint-manager.service 2>/dev/null; then
        print_success "Systemd service is running"
        systemctl status reliquary-checkpoint-manager.service --no-pager
        return 0
    fi
    
    # Check background processes
    local pids=$(pgrep -f "checkpoint_manager.py" | grep -v "$$" || true)
    if [ -n "$pids" ]; then
        print_success "Background process is running (PIDs: $pids)"
        ps aux | grep checkpoint_manager.py | grep -v grep || true
        return 0
    fi
    
    print_error "No running monitoring process found"
    exit 1
}

cmd_logs() {
    print_header
    
    # Try systemd first
    if systemctl is-active --quiet reliquary-checkpoint-manager.service 2>/dev/null; then
        print_info "Showing systemd service logs (Ctrl+C to exit):"
        sudo journalctl -u reliquary-checkpoint-manager.service -f
        return 0
    fi
    
    # Try local log file
    local log_file="$SCRIPT_DIR/.checkpoint_manager.log"
    if [ -f "$log_file" ]; then
        print_info "Showing local logs (Ctrl+C to exit):"
        tail -f "$log_file"
        return 0
    fi
    
    print_error "No log file found"
    exit 1
}

# Parse arguments
if [ $# -eq 0 ]; then
    usage
fi

COMMAND="$1"
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --endpoint)
            STATE_ENDPOINT="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Execute command
case "$COMMAND" in
    start)
        cmd_monitor_bg
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    once)
        cmd_once
        ;;
    check)
        cmd_check
        ;;
    monitor)
        cmd_monitor_fg
        ;;
    logs)
        cmd_logs
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        usage
        ;;
esac
