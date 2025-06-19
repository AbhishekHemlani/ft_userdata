#!/bin/bash

echo "📥 Starting safe Freqtrade data download..."

PAIR_FILE="user_data/pairs_top30.json"
TIMERANGE="20190923-20220804"
FORMAT="feather"
EXCHANGE="binanceus"
MODE="spot"
DELAY=5  # seconds between each request

# Define timeframes to download in batches
timeframes=( "4h" "1h" "30m" "15m" "5m" )

# Read pairs from the file
pairs=$(jq -r '.[]' "$PAIR_FILE")

for tf in "${timeframes[@]}"; do
  echo "⏳ Downloading $tf timeframe for all pairs..."

  for pair in $pairs; do
    echo "🔄 Pair: $pair | Timeframe: $tf"
    docker compose run --rm freqtrade download-data \
      --exchange $EXCHANGE \
      --trading-mode $MODE \
      --pairs "$pair" \
      --timerange $TIMERANGE \
      --timeframes $tf \
      --data-format-ohlcv $FORMAT \
      --erase

    echo "✅ Done with $pair [$tf] — sleeping $DELAY sec..."
    sleep $DELAY
  done

  echo "✅ Completed timeframe: $tf"
done

echo "🎉 All data downloaded safely without hitting rate limits."
