# Freqtrade Alpha1 Strategy Repository

This repository contains a Freqtrade trading bot implementation with the Alpha1 strategy, designed for automated cryptocurrency trading on Binance US.

## Project Structure

```
ft_userdata/
├── user_data/
│   ├── strategies/              # Trading strategies
│   │   ├── Alpha1Strategy.py    # Main Alpha1 strategy with enhanced logging
│   │   ├── Alpha1Base.py        # Base template for Pareto strategies
│   │   ├── Alpha1Renko.py       # Renko-based strategy
│   │   └── sample_strategy.py   # Freqtrade sample strategy
│   ├── config.json              # Main configuration file
│   ├── config-usd-top30.json    # Configuration for top 30 USD pairs
│   ├── config-usd.json          # USD pairs configuration
│   ├── data/                    # Historical price data
│   │   ├── binanceus/           # Binance US data
│   │   └── binance/             # Binance data
│   ├── backtest_results/        # Backtest output files
│   ├── logs/                    # Strategy and bot logs
│   ├── notebooks/               # Jupyter notebooks for analysis
│   ├── hyperopts/               # Hyperopt scripts
│   └── hyperopt_results/        # Hyperopt output files
├── docker-compose.yml           # Docker configuration
├── run_strategy_with_logs.sh    # Script for running strategy with logging
└── README.md                    # This file
```

## Features

### 🚀 **Enhanced Alpha1Strategy**
- **Comprehensive Logging**: Detailed logging throughout the strategy lifecycle
- **Portfolio Derisk Logic**: Advanced risk management with automatic position monitoring
- **Multi-timeframe Analysis**: Uses 4H, 1H, 30T, and 15T timeframes
- **Technical Indicators**: DMI, RSI, SAR, Heikin Ashi, Linear Regression
- **Entry/Exit Signals**: Sophisticated signal generation with confirmation logic


## Quick Start

### Prerequisites
- [Freqtrade](https://www.freqtrade.io/) installed
- Python 3.8+
- Docker (optional)

### 1. Clone the Repository
```bash
git clone https://github.com/AbhishekHemlani/ft_userdata.git
cd ft_userdata
```

### 2. Configure Your Settings
Edit `user_data/config.json` with your exchange API keys:
```json
{
  "exchange": {
    "name": "binanceus",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_SECRET_KEY"
  }
}
```

### 3. Run the Strategy

#### Using Docker (Recommended)
```bash
docker compose up
```

#### Using the Logging Script
```bash
chmod +x run_strategy_with_logs.sh
./run_strategy_with_logs.sh
```

#### Direct Command
```bash
freqtrade trade --config user_data/config.json --strategy Alpha1Strategy
```

### 4. Run Backtesting
```bash
docker compose run --rm freqtrade backtesting \
  --config user_data/config-usd-top30.json \
  --strategy Alpha1Strategy \
  --timeframe 15m \
  --timerange 20190101-20240101 \
  --export trades 2>&1 | tee user_data/logs/backtest_$(date +%Y%m%d_%H%M%S).log
```

## Configuration Files

### Main Config (`user_data/config.json`)
- Basic trading settings
- Exchange configuration
- Risk management parameters
- Logging configuration

### Top 30 Config (`user_data/config-usd-top30.json`)
- Optimized for top 30 USD pairs
- Higher position limits
- Aggressive risk settings