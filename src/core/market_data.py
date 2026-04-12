"""
NOCTURNA Trading System - Market Data Handler
Production-grade market data management with caching.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import deque
import threading

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class MarketDataHandler:
    """
    Production-grade market data handler.
    Manages data retrieval, caching, and technical indicator calculations.
    """

    # Cache settings
    CACHE_TTL = 60  # seconds
    MAX_CACHE_SIZE = 1000

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Data cache
        self.price_cache: Dict[str, Dict] = {}
        self.historical_cache: Dict[str, pd.DataFrame] = {}
        self.cache_lock = threading.RLock()

        # Price history for indicators
        self.price_history: Dict[str, deque] = {}

        # Real-time subscriptions
        self.subscriptions: Dict[str, List] = {}
        self.realtime_enabled = False

        # Data providers
        self.alpaca_client = None
        self.polygon_client = None
        self.yfinance_client = None

        self._initialize_providers()

        self.logger.info("Market Data Handler initialized")

    def _initialize_providers(self) -> None:
        """Initialize data provider clients."""
        # Alpaca for real-time data
        alpaca_key = os.environ.get('ALPACA_API_KEY')
        if alpaca_key:
            try:
                from alpaca_trade_api import REST as AlpacaREST
                self.alpaca_client = AlpacaREST(
                    key_id=alpaca_key,
                    secret_key=os.environ.get('ALPACA_SECRET_KEY'),
                    base_url=os.environ.get('ALPACA_BASE_URL')
                )
                self.logger.info("Alpaca data client initialized")
            except ImportError:
                self.logger.warning("Alpaca SDK not installed")
            except Exception as e:
                self.logger.error(f"Failed to initialize Alpaca client: {e}")

        # Polygon for additional market data
        polygon_key = os.environ.get('POLYGON_API_KEY')
        if polygon_key:
            try:
                from polygon import RESTClient
                self.polygon_client = RESTClient(polygon_key)
                self.logger.info("Polygon data client initialized")
            except ImportError:
                self.logger.warning("Polygon SDK not installed")
            except Exception as e:
                self.logger.error(f"Failed to initialize Polygon client: {e}")

        # Yahoo Finance as fallback
        try:
            import yfinance
            self.yfinance_client = yfinance
            self.logger.info("Yahoo Finance client available")
        except ImportError:
            self.logger.warning("yfinance not installed")

    def start_real_time_feed(self) -> None:
        """Start real-time market data feed."""
        if self.alpaca_client:
            self.realtime_enabled = True
            self.logger.info("Real-time feed started")
        else:
            self.logger.warning("No real-time data provider available")

    def stop_real_time_feed(self) -> None:
        """Stop real-time market data feed."""
        self.realtime_enabled = False
        self.subscriptions.clear()
        self.logger.info("Real-time feed stopped")

    def subscribe_to_symbol(self, symbol: str, callback) -> None:
        """Subscribe to real-time updates for a symbol."""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
            # Initialize price history
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=500)

        if callback not in self.subscriptions[symbol]:
            self.subscriptions[symbol].append(callback)

    def unsubscribe_from_symbol(self, symbol: str, callback) -> None:
        """Unsubscribe from symbol updates."""
        if symbol in self.subscriptions and callback in self.subscriptions[symbol]:
            self.subscriptions[symbol].remove(callback)

    def get_historical_data(self, symbol: str, timeframe: str = '1h',
                           limit: int = 500, start: datetime = None,
                           end: datetime = None) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Time interval (1m, 5m, 15m, 1h, 1d, 1w)
            limit: Number of bars to retrieve
            start: Start datetime (optional)
            end: End datetime (optional)

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"

        # Check cache
        with self.cache_lock:
            if cache_key in self.historical_cache:
                cached = self.historical_cache[cache_key]
                if (datetime.now(timezone.utc) - cached.get('timestamp', datetime.min)).total_seconds() < self.CACHE_TTL:
                    return cached.get('data', pd.DataFrame())

        # Fetch from provider
        df = self._fetch_historical_data(symbol, timeframe, limit, start, end)

        # Cache result
        with self.cache_lock:
            self.historical_cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now(timezone.utc)
            }

        return df

    def _fetch_historical_data(self, symbol: str, timeframe: str,
                              limit: int, start: datetime = None,
                              end: datetime = None) -> pd.DataFrame:
        """Fetch historical data from provider."""
        try:
            # Try Alpaca first
            if self.alpaca_client:
                df = self._fetch_from_alpaca(symbol, timeframe, limit, start, end)
                if not df.empty:
                    return df

            # Try Polygon
            if self.polygon_client:
                df = self._fetch_from_polygon(symbol, timeframe, limit, start, end)
                if not df.empty:
                    return df

            # Fallback to Yahoo Finance
            if self.yfinance_client:
                return self._fetch_from_yfinance(symbol, timeframe, limit, start, end)

            self.logger.error(f"No data provider available for {symbol}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_alpaca(self, symbol: str, timeframe: str,
                          limit: int, start: datetime = None,
                          end: datetime = None) -> pd.DataFrame:
        """Fetch data from Alpaca API."""
        try:
            timeframe_map = {
                '1m': '1Min', '5m': '5Min', '15m': '15Min',
                '1h': '1Hour', '1d': '1Day', '1w': '1Week'
            }
            tf = timeframe_map.get(timeframe, '1Hour')

            bars = self.alpaca_client.get_bars(
                symbol,
                tf,
                start=start.isoformat() if start else None,
                end=end.isoformat() if end else None,
                limit=limit
            ).df

            if bars.empty:
                return pd.DataFrame()

            return bars.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume'
            })

        except Exception as e:
            self.logger.error(f"Alpaca fetch error: {e}")
            return pd.DataFrame()

    def _fetch_from_polygon(self, symbol: str, timeframe: str,
                           limit: int, start: datetime = None,
                           end: datetime = None) -> pd.DataFrame:
        """Fetch data from Polygon.io API."""
        try:
            multiplier, timespan = self._parse_timeframe(timeframe)

            aggs = self.polygon_client.get_aggs(
                symbol, multiplier, timespan,
                start.isoformat() if start else None,
                end.isoformat() if end else None,
                limit=limit
            )

            if not aggs:
                return pd.DataFrame()

            data = [{
                'Open': a.open,
                'High': a.high,
                'Low': a.low,
                'Close': a.close,
                'Volume': a.volume,
                'timestamp': datetime.fromtimestamp(a.timestamp / 1000, tz=timezone.utc)
            } for a in aggs]

            return pd.DataFrame(data).set_index('timestamp')

        except Exception as e:
            self.logger.error(f"Polygon fetch error: {e}")
            return pd.DataFrame()

    def _fetch_from_yfinance(self, symbol: str, timeframe: str,
                            limit: int, start: datetime = None,
                            end: datetime = None) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '1h': '1h', '1d': '1d', '1w': '1wk'
            }
            interval = interval_map.get(timeframe, '1h')

            ticker = self.yfinance_client.Ticker(symbol)
            df = ticker.history(
                period='max' if not start else None,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True
            )

            if df.empty:
                return pd.DataFrame()

            return df[['Open', 'High', 'Low', 'Close', 'Volume']]

        except Exception as e:
            self.logger.error(f"Yahoo Finance fetch error: {e}")
            return pd.DataFrame()

    def _parse_timeframe(self, timeframe: str) -> tuple:
        """Parse timeframe string to multiplier and timespan."""
        timeframe_map = {
            '1m': (1, 'minute'), '5m': (5, 'minute'), '15m': (15, 'minute'),
            '1h': (1, 'hour'), '4h': (4, 'hour'), '1d': (1, 'day'),
            '1w': (1, 'week')
        }
        return timeframe_map.get(timeframe, (1, 'hour'))

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added indicator columns
        """
        if df.empty or len(df) < 50:
            return df

        try:
            # EMA calculations
            df['ema8'] = df['Close'].ewm(span=8, adjust=False).mean()
            df['ema34'] = df['Close'].ewm(span=34, adjust=False).mean()
            df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['ema200'] = df['Close'].ewm(span=200, adjust=False).mean()

            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd_line'] = exp1 - exp2
            df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']

            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Stochastic
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()

            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']

            return df

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return df

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            if self.alpaca_client:
                quote = self.alpaca_client.get_latest_quote(symbol)
                return (quote.bid_price + quote.ask_price) / 2

            if self.yfinance_client:
                ticker = self.yfinance_client.Ticker(symbol)
                return ticker.fast_info.last_price

            return None

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self.cache_lock:
            self.historical_cache.clear()
            self.price_cache.clear()
        self.logger.info("Market data cache cleared")
