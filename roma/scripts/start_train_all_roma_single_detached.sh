#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs/roma_single"
MAIN_LOG="$LOG_DIR/main.log"
PID_FILE="$LOG_DIR/main.pid"

mkdir -p "$LOG_DIR"

cd "$REPO_DIR"
nohup setsid bash "$REPO_DIR/scripts/train_all_roma_single.sh" > "$MAIN_LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

echo "started train_all_roma_single.sh"
echo "pid: $PID"
echo "log: $MAIN_LOG"
echo "pid file: $PID_FILE"
