<div align="center">
  <img src="src/assets/banner.png" alt="Nocturna Banner" width="420" style="margin-bottom: 20px;">

# NOCTURNA v2.0 - Advanced Trading Bot

## 🚀 The Most Advanced Automated Trading Bot Ever Created

NOCTURNA v2.0 is an enterprise‑level algorithmic trading system that implements a multi‑modal adaptive quantitative algorithm for trend following and range trading. It combines artificial intelligence, machine learning, and sentiment analysis to create an autonomous, highly profitable trading machine.

## ✨ Key Features

### 🧠 Core Trading Engine
- **NOCTURNA v2.0 Algorithm**: Complete implementation of the adaptive multi‑modal algorithm
- **4 Autonomous Trading Modes**:
  - **EVE**: Grid Trading for sideways markets
  - **LUCIFER**: Breakout Trading for key level breaks
  - **REAPER**: Reversal Trading for trend reversals
  - **SENTINEL**: Trend Following for strong trends
- **Automatic Market Regime Recognition**: 5 states (RANGING, TRENDING, REVERSING, BREAKOUT, VOLATILE)
- **Advanced Risk Management**: Dynamic stop losses, adaptive position sizing, drawdown control

### 🤖 Artificial Intelligence and Machine Learning
- **ML Optimizer**: Automatic parameter optimization using Random Forest and Gradient Boosting
- **Genetic Algorithms**: Continuous evolution of trading parameters
- **Advanced Backtesting**: Monte Carlo analysis, Walk-Forward testing
- **Sentiment Analysis**: News and social media sentiment analysis
- **Auto-Tuning**: Automatic adaptation to market conditions

### 📊 Professional Frontend
- **Real-time Dashboard**: Live performance monitoring
- **Advanced Controls**: Start/Stop/Pause/Emergency Stop
- **Interactive Visualizations**: Performance charts, equity curve, drawdown
- **Position Management**: Monitor active positions and orders
- **Parameter Configuration**: Real-time parameter tuning

### 🔗 Multi-Broker Integration
- **Alpaca Markets**: Stock and ETF trading
- **Polygon.io**: Real-time market data
- **Yahoo Finance**: Historical and fundamental data
- **Modular Architecture**: Easy addition of new brokers

### 🛡️ Security and Reliability
- **Multi-Level Risk Management**: Risk controls at order, position, and portfolio level
- **Emergency Stop**: Immediate halt of all operations
- **Comprehensive Logging**: Detailed tracking of all operations
- **Automatic Backup**: Automatic system state saving

## 🏗️ System Architecture

```

NOCTURNA v2.0/
├── src/
│   ├── core/                    # Core Trading Engine
│   │   ├── trading_engine.py    # Main engine
│   │   ├── strategy_manager.py  # Strategy management
│   │   ├── market_data.py       # Market data handling
│   │   ├── order_manager.py     # Order management
│   │   └── risk_manager.py      # Risk management
│   ├── advanced/                # Advanced Features
│   │   ├── ml_optimizer.py      # Machine Learning Optimizer
│   │   ├── backtester.py        # Backtesting System
│   │   └── sentiment_analyzer.py # Sentiment Analysis
│   ├── routes/                  # API Routes
│   │   └── trading.py           # REST API endpoints
│   └── main.py                  # Main Flask application
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── components/          # React components
│   │   └── services/            # API services
│   └── dist/                    # Production build
└── config/                      # Configuration files

```

## 🚀 Installation and Setup

### Prerequisites
- Python 3.11+
- Node.js 20+
- Alpaca Markets account (for live trading)
- Polygon.io API key (for real-time data)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd nocturna_trading_bot
```

2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

3. Frontend Setup

```bash
cd frontend
npm install
npm run build
```

4. Configuration

Create a .env file in the project root:

```env
# Alpaca Trading
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading
# ALPACA_BASE_URL=https://api.alpaca.markets  # Live trading

# Polygon.io
POLYGON_API_KEY=your_polygon_api_key

# Database
DATABASE_URL=sqlite:///nocturna.db

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379

# Trading Configuration
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
RISK_LEVEL=LOW
```

5. Start the System

```bash
# Start the backend
python src/main.py

# The frontend is automatically served at http://localhost:5000
```

📈 Usage

Web Dashboard

1. Open your browser at http://localhost:5000
2. Monitor real-time performance
3. Control the bot using Start/Stop/Pause buttons
4. Configure parameters in the Settings section

REST API

The system exposes a complete REST API for integration:

```bash
# System status
GET /api/status

# Start trading engine
POST /api/start

# Stop trading engine
POST /api/stop

# Get active positions
GET /api/positions

# Get orders
GET /api/orders

# Performance
GET /api/performance
```

Advanced Configuration

Trading Parameters

```python
TRADING_PARAMS = {
    'grid_spacing': 0.005,        # Grid spacing (0.5%)
    'atr_mult_sl': 2.0,          # ATR multiplier for stop loss
    'atr_mult_tp': 4.0,          # ATR multiplier for take profit
    'max_position_size': 0.1,     # Maximum position size (10%)
    'volatility_threshold': 2.0,  # Volatility threshold
    'trend_strength_threshold': 0.5, # Trend strength threshold
    'reversal_confirmation_bars': 3, # Reversal confirmation bars
    'breakout_volume_mult': 1.5   # Breakout volume multiplier
}
```

Machine Learning

```python
ML_CONFIG = {
    'optimization_frequency': 'weekly',  # Optimization frequency
    'n_iterations': 100,                 # Optimization iterations
    'validation_split': 0.2,            # Validation split
    'feature_selection': True,          # Automatic feature selection
    'ensemble_methods': ['rf', 'gb'],   # Ensemble methods
}
```

🧪 Backtesting

Simple Backtesting

```python
from src.advanced.backtester import AdvancedBacktester
from src.core.strategy_manager import StrategyManager

# Configure backtester
config = {
    'initial_capital': 100000,
    'commission_rate': 0.001,
    'slippage_rate': 0.0005
}

backtester = AdvancedBacktester(config)
strategy = StrategyManager(strategy_params)

# Run backtest
results = backtester.run_backtest(historical_data, strategy.generate_signals, strategy_params)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

Monte Carlo Analysis

```python
# Strategy robustness analysis
mc_results = backtester.monte_carlo_analysis(
    data=historical_data,
    strategy_function=strategy.generate_signals,
    strategy_params=params,
    n_simulations=1000
)

print(f"Probability of positive return: {mc_results['probability_positive']:.1%}")
print(f"95% VaR: {mc_results['return_stats']['percentile_5']:.2%}")
```

🤖 Machine Learning and Optimization

Automatic Optimization

```python
from src.advanced.ml_optimizer import MLOptimizer

# Configure optimizer
ml_optimizer = MLOptimizer(config)

# Optimize parameters
optimized_params = ml_optimizer.optimize_parameters(
    market_data=recent_data,
    current_params=current_params,
    backtest_function=backtester.run_backtest,
    n_iterations=50
)

print("Optimized parameters:", optimized_params)
```

Sentiment Analysis

```python
from src.advanced.sentiment_analyzer import SentimentAnalyzer

# Analyze sentiment
sentiment_analyzer = SentimentAnalyzer(config)

# Add sentiment data
sentiment_analyzer.add_sentiment_data(
    source='news',
    symbol='AAPL',
    text='Apple reports record quarterly earnings...'
)

# Get sentiment signal
signal = sentiment_analyzer.get_market_sentiment_signal('AAPL')
print(f"Sentiment Signal: {signal['signal']} (strength: {signal['strength']:.2f})")
```

📊 Performance Metrics

Key Metrics

· Total Return: Overall portfolio return
· Sharpe Ratio: Risk-adjusted return
· Sortino Ratio: Downside risk-adjusted return
· Calmar Ratio: Annualized return / Max Drawdown
· Win Rate: Percentage of winning trades
· Profit Factor: Gross profits / Gross losses
· Maximum Drawdown: Peak-to-trough decline

Risk Analysis

· VaR (Value at Risk): Maximum expected loss at 95% confidence
· CVaR (Conditional VaR): Average loss beyond VaR
· Beta: Correlation with the market
· Volatility: Standard deviation of returns

🔧 Advanced Configuration

Trading Modes

```python
# EVE Mode configuration (Grid Trading)
EVE_CONFIG = {
    'grid_levels': 10,
    'grid_spacing_pct': 0.5,
    'max_grid_positions': 5,
    'profit_target_pct': 1.0
}

# LUCIFER Mode configuration (Breakout)
LUCIFER_CONFIG = {
    'breakout_threshold': 2.0,
    'volume_confirmation': True,
    'momentum_filter': True,
    'max_breakout_age': 5
}

# REAPER Mode configuration (Reversal)
REAPER_CONFIG = {
    'reversal_signals': ['rsi_divergence', 'support_resistance'],
    'confirmation_bars': 3,
    'risk_reward_ratio': 2.0
}

# SENTINEL Mode configuration (Trend Following)
SENTINEL_CONFIG = {
    'trend_indicators': ['ema_cross', 'adx', 'macd'],
    'trend_strength_min': 0.6,
    'pullback_entry': True
}
```

Risk Management

```python
RISK_CONFIG = {
    'max_portfolio_risk': 0.02,      # 2% maximum portfolio risk
    'max_position_risk': 0.005,      # 0.5% maximum risk per position
    'correlation_limit': 0.7,        # Correlation limit between positions
    'sector_concentration': 0.3,     # Maximum concentration per sector
    'daily_loss_limit': 0.01,       # Daily loss limit (1%)
    'drawdown_stop': 0.1,           # Stop at 10% drawdown
}
```

🚀 Production Deployment

Docker Deployment

```dockerfile
# Included Dockerfile for containerized deployment
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN cd frontend && npm install && npm run build

EXPOSE 5000
CMD ["python", "src/main.py"]
```

Cloud Deployment

The system is optimized for deployment on:

· AWS EC2/ECS: With Auto Scaling support
· Google Cloud Run: Serverless deployment
· Azure Container Instances: Rapid deployment
· DigitalOcean Droplets: Cost-effective solution

Monitoring

· Prometheus: System metrics
· Grafana: Monitoring dashboards
· Sentry: Error tracking
· CloudWatch: Logging and alerting

📚 API Documentation

Main Endpoints

System

· GET /api/status - System status
· POST /api/start - Start trading engine
· POST /api/stop - Stop trading engine
· POST /api/pause - Pause trading engine
· POST /api/emergency-stop - Emergency stop

Trading

· GET /api/positions - Active positions
· GET /api/orders - Active orders
· POST /api/orders - Create new order
· DELETE /api/orders/{id} - Cancel order

Performance

· GET /api/performance - Performance metrics
· GET /api/equity-curve - Equity curve
· GET /api/trades - Trade history

Configuration

· GET /api/config - Current configuration
· PUT /api/config - Update configuration
· POST /api/optimize - Start ML optimization

🔒 Security

Security Measures

· API Authentication: JWT tokens for API access
· Encryption: Sensitive data encrypted
· Rate Limiting: Abuse protection
· Audit Logging: Complete operation logs
· Automatic Backup: Automatic system state backup

Best Practices

· Always use paper trading for initial tests
· Constantly monitor performance
· Always set appropriate risk limits
· Keep API keys up to date
· Perform regular configuration backups

🆘 Troubleshooting

Common Issues

Bot does not start

```bash
# Verify configuration
python -c "from src.core.trading_engine import TradingEngine; print('OK')"

# Check logs
tail -f logs/nocturna.log
```

API connection errors

```bash
# Test Alpaca connection
python -c "import alpaca_trade_api as tradeapi; api = tradeapi.REST(); print(api.get_account())"

# Test Polygon connection
python -c "from polygon import RESTClient; client = RESTClient(); print('OK')"
```

Poor performance

1. Verify trading parameters
2. Run backtesting on recent data
3. Optimize parameters with ML
4. Check market conditions

📞 Support

Documentation

· Wiki: Complete documentation in the project wiki
· Examples: Usage examples in the examples/ folder
· API Docs: Automatically generated API documentation

Community

· Discord: Real-time support on Discord server
· GitHub Issues: Bug reports and feature requests
· Forum: Discussions and strategy sharing

📄 License

This project is released under the MIT license. See the LICENSE file for details.

🙏 Acknowledgements

· FMZ Quant: For the original NOCTURNA algorithm
· Alpaca Markets: For trading APIs
· Polygon.io: For real-time market data
· Open Source Community: For the libraries used

---

⚡ Quick Start

```bash
# Clone and setup
git clone <repo-url> && cd nocturna_trading_bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure .env with your API keys
cp .env.example .env
# Edit .env with your credentials

# Start the system
python src/main.py

# Open browser at http://localhost:5000
```

🎯 NOCTURNA v2.0 - The Future of Algorithmic Trading is Here!

---

Disclaimer: Trading involves significant risk. This software is provided "as-is" without warranties. Always use paper trading for initial tests and only invest what you can afford to lose.

Yog-Sotho ❤️
---
