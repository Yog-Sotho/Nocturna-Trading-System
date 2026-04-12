<div align="center">
  <img src="src/assets/banner.png" alt="Nocturna Banner" width="420" style="margin-bottom: 20px;">
  
# NOCTURNA v2.0 - Advanced Trading Bot

## 🚀 Il Bot di Trading Automatico Più Avanzato Mai Creato

NOCTURNA v2.0 è un sistema di trading algoritmico di livello enterprise che implementa l'algoritmo quantitativo multi-modale adattivo per trend following e range trading. Combina intelligenza artificiale, machine learning e analisi del sentiment per creare una macchina da trading autonoma e altamente redditizia.

## ✨ Caratteristiche Principali

### 🧠 Core Trading Engine
- **Algoritmo NOCTURNA v2.0**: Implementazione completa dell'algoritmo multi-modale adattivo
- **4 Modalità di Trading Autonome**:
  - **EVE**: Grid Trading per mercati laterali
  - **LUCIFER**: Breakout Trading per rotture di livelli chiave
  - **REAPER**: Reversal Trading per inversioni di trend
  - **SENTINEL**: Trend Following per trend forti
- **Riconoscimento Automatico del Regime di Mercato**: 5 stati (RANGING, TRENDING, REVERSING, BREAKOUT, VOLATILE)
- **Gestione del Rischio Avanzata**: Stop loss dinamici, position sizing adattivo, controllo del drawdown

### 🤖 Intelligenza Artificiale e Machine Learning
- **ML Optimizer**: Ottimizzazione automatica dei parametri usando Random Forest e Gradient Boosting
- **Algoritmi Genetici**: Evoluzione continua dei parametri di trading
- **Backtesting Avanzato**: Monte Carlo analysis, Walk-Forward testing
- **Sentiment Analysis**: Analisi del sentiment da news e social media
- **Auto-Tuning**: Adattamento automatico alle condizioni di mercato

### 📊 Frontend Professionale
- **Dashboard Real-time**: Monitoraggio live delle performance
- **Controlli Avanzati**: Start/Stop/Pause/Emergency Stop
- **Visualizzazioni Interattive**: Grafici di performance, equity curve, drawdown
- **Gestione Posizioni**: Monitoraggio posizioni attive e ordini
- **Configurazione Parametri**: Tuning in tempo reale dei parametri

### 🔗 Integrazione Multi-Broker
- **Alpaca Markets**: Trading di azioni e ETF
- **Polygon.io**: Dati di mercato real-time
- **Yahoo Finance**: Dati storici e fondamentali
- **Architettura Modulare**: Facile aggiunta di nuovi broker

### 🛡️ Sicurezza e Affidabilità
- **Risk Management Multi-Livello**: Controlli di rischio a livello di ordine, posizione e portfolio
- **Emergency Stop**: Arresto immediato di tutte le operazioni
- **Logging Completo**: Tracciamento dettagliato di tutte le operazioni
- **Backup Automatico**: Salvataggio automatico dello stato del sistema

## 🏗️ Architettura del Sistema

```
NOCTURNA v2.0/
├── src/
│   ├── core/                    # Core Trading Engine
│   │   ├── trading_engine.py    # Motore principale
│   │   ├── strategy_manager.py  # Gestione strategie
│   │   ├── market_data.py       # Gestione dati di mercato
│   │   ├── order_manager.py     # Gestione ordini
│   │   └── risk_manager.py      # Gestione del rischio
│   ├── advanced/                # Funzionalità Avanzate
│   │   ├── ml_optimizer.py      # Machine Learning Optimizer
│   │   ├── backtester.py        # Sistema di Backtesting
│   │   └── sentiment_analyzer.py # Analisi del Sentiment
│   ├── routes/                  # API Routes
│   │   └── trading.py           # Endpoints REST API
│   └── main.py                  # Applicazione Flask principale
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── components/          # Componenti React
│   │   └── services/            # Servizi API
│   └── dist/                    # Build di produzione
└── config/                      # File di configurazione
```

## 🚀 Installazione e Setup

### Prerequisiti
- Python 3.11+
- Node.js 20+
- Account Alpaca Markets (per trading live)
- API Key Polygon.io (per dati real-time)

### 1. Clona il Repository
```bash
git clone <repository-url>
cd nocturna_trading_bot
```

### 2. Setup Backend
```bash
# Crea ambiente virtuale
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\\Scripts\\activate  # Windows

# Installa dipendenze
pip install -r requirements.txt
```

### 3. Setup Frontend
```bash
cd frontend
npm install
npm run build
```

### 4. Configurazione
Crea un file `.env` nella root del progetto:

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

# Redis (opzionale, per caching)
REDIS_URL=redis://localhost:6379

# Configurazione Trading
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1
RISK_LEVEL=LOW
```

### 5. Avvio del Sistema
```bash
# Avvia il backend
python src/main.py

# Il frontend è servito automaticamente su http://localhost:5000
```

## 📈 Utilizzo

### Dashboard Web
1. Apri il browser su `http://localhost:5000`
2. Monitora le performance in tempo reale
3. Controlla il bot con i pulsanti Start/Stop/Pause
4. Configura i parametri nella sezione Settings

### API REST
Il sistema espone API REST complete per l'integrazione:

```bash
# Status del sistema
GET /api/status

# Avvia il trading engine
POST /api/start

# Ferma il trading engine
POST /api/stop

# Ottieni posizioni attive
GET /api/positions

# Ottieni ordini
GET /api/orders

# Performance
GET /api/performance
```

### Configurazione Avanzata

#### Parametri di Trading
```python
TRADING_PARAMS = {
    'grid_spacing': 0.005,        # Spaziatura grid (0.5%)
    'atr_mult_sl': 2.0,          # Moltiplicatore ATR per stop loss
    'atr_mult_tp': 4.0,          # Moltiplicatore ATR per take profit
    'max_position_size': 0.1,     # Dimensione massima posizione (10%)
    'volatility_threshold': 2.0,  # Soglia volatilità
    'trend_strength_threshold': 0.5, # Soglia forza trend
    'reversal_confirmation_bars': 3, # Barre conferma inversione
    'breakout_volume_mult': 1.5   # Moltiplicatore volume breakout
}
```

#### Machine Learning
```python
ML_CONFIG = {
    'optimization_frequency': 'weekly',  # Frequenza ottimizzazione
    'n_iterations': 100,                 # Iterazioni per ottimizzazione
    'validation_split': 0.2,            # Split per validazione
    'feature_selection': True,          # Selezione automatica features
    'ensemble_methods': ['rf', 'gb'],   # Metodi ensemble
}
```

## 🧪 Backtesting

### Backtesting Semplice
```python
from src.advanced.backtester import AdvancedBacktester
from src.core.strategy_manager import StrategyManager

# Configura backtester
config = {
    'initial_capital': 100000,
    'commission_rate': 0.001,
    'slippage_rate': 0.0005
}

backtester = AdvancedBacktester(config)
strategy = StrategyManager(strategy_params)

# Esegui backtest
results = backtester.run_backtest(historical_data, strategy.generate_signals, strategy_params)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### Monte Carlo Analysis
```python
# Analisi robustezza strategia
mc_results = backtester.monte_carlo_analysis(
    data=historical_data,
    strategy_function=strategy.generate_signals,
    strategy_params=params,
    n_simulations=1000
)

print(f"Probabilità rendimento positivo: {mc_results['probability_positive']:.1%}")
print(f"VaR 95%: {mc_results['return_stats']['percentile_5']:.2%}")
```

## 🤖 Machine Learning e Ottimizzazione

### Ottimizzazione Automatica
```python
from src.advanced.ml_optimizer import MLOptimizer

# Configura optimizer
ml_optimizer = MLOptimizer(config)

# Ottimizza parametri
optimized_params = ml_optimizer.optimize_parameters(
    market_data=recent_data,
    current_params=current_params,
    backtest_function=backtester.run_backtest,
    n_iterations=50
)

print("Parametri ottimizzati:", optimized_params)
```

### Sentiment Analysis
```python
from src.advanced.sentiment_analyzer import SentimentAnalyzer

# Analizza sentiment
sentiment_analyzer = SentimentAnalyzer(config)

# Aggiungi dati di sentiment
sentiment_analyzer.add_sentiment_data(
    source='news',
    symbol='AAPL',
    text='Apple reports record quarterly earnings...'
)

# Ottieni segnale sentiment
signal = sentiment_analyzer.get_market_sentiment_signal('AAPL')
print(f"Sentiment Signal: {signal['signal']} (strength: {signal['strength']:.2f})")
```

## 📊 Metriche di Performance

### Metriche Principali
- **Total Return**: Rendimento totale del portfolio
- **Sharpe Ratio**: Rendimento aggiustato per il rischio
- **Sortino Ratio**: Rendimento aggiustato per downside risk
- **Calmar Ratio**: Rendimento annualizzato / Max Drawdown
- **Win Rate**: Percentuale di trade vincenti
- **Profit Factor**: Profitti / Perdite
- **Maximum Drawdown**: Massima perdita dal picco

### Analisi del Rischio
- **VaR (Value at Risk)**: Perdita massima attesa con confidenza 95%
- **CVaR (Conditional VaR)**: Perdita media oltre il VaR
- **Beta**: Correlazione con il mercato
- **Volatilità**: Deviazione standard dei rendimenti

## 🔧 Configurazione Avanzata

### Modalità di Trading
```python
# Configurazione modalità EVE (Grid Trading)
EVE_CONFIG = {
    'grid_levels': 10,
    'grid_spacing_pct': 0.5,
    'max_grid_positions': 5,
    'profit_target_pct': 1.0
}

# Configurazione modalità LUCIFER (Breakout)
LUCIFER_CONFIG = {
    'breakout_threshold': 2.0,
    'volume_confirmation': True,
    'momentum_filter': True,
    'max_breakout_age': 5
}

# Configurazione modalità REAPER (Reversal)
REAPER_CONFIG = {
    'reversal_signals': ['rsi_divergence', 'support_resistance'],
    'confirmation_bars': 3,
    'risk_reward_ratio': 2.0
}

# Configurazione modalità SENTINEL (Trend Following)
SENTINEL_CONFIG = {
    'trend_indicators': ['ema_cross', 'adx', 'macd'],
    'trend_strength_min': 0.6,
    'pullback_entry': True
}
```

### Risk Management
```python
RISK_CONFIG = {
    'max_portfolio_risk': 0.02,      # 2% rischio massimo portfolio
    'max_position_risk': 0.005,      # 0.5% rischio massimo per posizione
    'correlation_limit': 0.7,        # Limite correlazione tra posizioni
    'sector_concentration': 0.3,     # Concentrazione massima per settore
    'daily_loss_limit': 0.01,       # Limite perdita giornaliera (1%)
    'drawdown_stop': 0.1,           # Stop a 10% di drawdown
}
```

## 🚀 Deployment in Produzione

### Docker Deployment
```dockerfile
# Dockerfile incluso per deployment containerizzato
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt
RUN cd frontend && npm install && npm run build

EXPOSE 5000
CMD ["python", "src/main.py"]
```

### Cloud Deployment
Il sistema è ottimizzato per deployment su:
- **AWS EC2/ECS**: Con supporto per Auto Scaling
- **Google Cloud Run**: Deployment serverless
- **Azure Container Instances**: Deployment rapido
- **DigitalOcean Droplets**: Soluzione economica

### Monitoraggio
- **Prometheus**: Metriche di sistema
- **Grafana**: Dashboard di monitoraggio
- **Sentry**: Error tracking
- **CloudWatch**: Logging e alerting

## 📚 Documentazione API

### Endpoints Principali

#### Sistema
- `GET /api/status` - Status del sistema
- `POST /api/start` - Avvia trading engine
- `POST /api/stop` - Ferma trading engine
- `POST /api/pause` - Pausa trading engine
- `POST /api/emergency-stop` - Arresto di emergenza

#### Trading
- `GET /api/positions` - Posizioni attive
- `GET /api/orders` - Ordini attivi
- `POST /api/orders` - Crea nuovo ordine
- `DELETE /api/orders/{id}` - Cancella ordine

#### Performance
- `GET /api/performance` - Metriche di performance
- `GET /api/equity-curve` - Curva dell'equity
- `GET /api/trades` - Storico trade

#### Configurazione
- `GET /api/config` - Configurazione attuale
- `PUT /api/config` - Aggiorna configurazione
- `POST /api/optimize` - Avvia ottimizzazione ML

## 🔒 Sicurezza

### Misure di Sicurezza
- **Autenticazione API**: Token JWT per accesso API
- **Crittografia**: Dati sensibili crittografati
- **Rate Limiting**: Protezione contro abusi
- **Audit Logging**: Log completo delle operazioni
- **Backup Automatico**: Backup automatico dello stato

### Best Practices
- Usa sempre paper trading per test iniziali
- Monitora costantemente le performance
- Imposta sempre limiti di rischio appropriati
- Mantieni aggiornate le API keys
- Esegui backup regolari della configurazione

## 🆘 Troubleshooting

### Problemi Comuni

#### Bot non si avvia
```bash
# Verifica configurazione
python -c "from src.core.trading_engine import TradingEngine; print('OK')"

# Controlla log
tail -f logs/nocturna.log
```

#### Errori di connessione API
```bash
# Testa connessione Alpaca
python -c "import alpaca_trade_api as tradeapi; api = tradeapi.REST(); print(api.get_account())"

# Testa connessione Polygon
python -c "from polygon import RESTClient; client = RESTClient(); print('OK')"
```

#### Performance scarse
1. Verifica parametri di trading
2. Esegui backtesting su dati recenti
3. Ottimizza parametri con ML
4. Controlla condizioni di mercato

## 📞 Supporto

### Documentazione
- **Wiki**: Documentazione completa nel wiki del progetto
- **Examples**: Esempi di utilizzo nella cartella `examples/`
- **API Docs**: Documentazione API generata automaticamente

### Community
- **Discord**: Server Discord per supporto real-time
- **GitHub Issues**: Segnalazione bug e richieste feature
- **Forum**: Discussioni e condivisione strategie

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per i dettagli.

## 🙏 Ringraziamenti

- **FMZ Quant**: Per l'algoritmo NOCTURNA originale
- **Alpaca Markets**: Per le API di trading
- **Polygon.io**: Per i dati di mercato real-time
- **Community Open Source**: Per le librerie utilizzate

---

## ⚡ Quick Start

```bash
# Clone e setup
git clone <repo-url> && cd nocturna_trading_bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configura .env con le tue API keys
cp .env.example .env
# Modifica .env con le tue credenziali

# Avvia il sistema
python src/main.py

# Apri browser su http://localhost:5000
```

**🎯 NOCTURNA v2.0 - Il Futuro del Trading Algoritmico è Qui!**

Yog-Sotho ❤️
---

*Disclaimer: Il trading comporta rischi significativi. Questo software è fornito "as-is" senza garanzie. Usa sempre il paper trading per test iniziali e investi solo quello che puoi permetterti di perdere.*

