#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p user_data/logs

# Generate timestamp for log filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="user_data/logs/strategy_run_${TIMESTAMP}.log"

echo "Starting Freqtrade strategy with logging..."
echo "Log file: $LOG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "----------------------------------------"

# Run freqtrade and save all output to log file
freqtrade trade \
    --config user_data/config.json \
    --strategy Alpha1Strategy \
    2>&1 | tee "$LOG_FILE"

echo "----------------------------------------"
echo "Strategy run completed."
echo "Log saved to: $LOG_FILE" 