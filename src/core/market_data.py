"""
Market Data Handler per NOCTURNA v2.0 Trading Bot
Gestisce l'acquisizione e la pre-elaborazione dei dati di mercato in tempo reale e storici.
"""

import asyncio
import websockets
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import yfinance as yf
from alpaca_trade_api import REST as AlpacaREST
from polygon import RESTClient as PolygonClient
import redis
import threading
import time

class MarketDataHandler:
    """
    Gestisce l'acquisizione e la distribuzione dei dati di mercato.
    Supporta dati in tempo reale e storici da multiple fonti.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Inizializzazione client API
        self.alpaca_client = None
        self.polygon_client = None
        self.redis_client = None
        
        # Cache per dati di mercato
        self.price_cache = {}
        self.candle_cache = {}
        
        # WebSocket connections
        self.ws_connections = {}
        self.subscribers = {}
        
        # Threading per aggiornamenti in tempo reale
        self.data_thread = None
        self.running = False
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Inizializza i client API per le diverse fonti di dati."""
        try:
            # Alpaca API
            if self.config.get('alpaca_api_key'):
                self.alpaca_client = AlpacaREST(
                    key_id=self.config['alpaca_api_key'],
                    secret_key=self.config['alpaca_secret_key'],
                    base_url=self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
                )
                self.logger.info("Alpaca client inizializzato")
            
            # Polygon API
            if self.config.get('polygon_api_key'):
                self.polygon_client = PolygonClient(self.config['polygon_api_key'])
                self.logger.info("Polygon client inizializzato")
            
            # Redis per caching
            if self.config.get('redis_host'):
                self.redis_client = redis.Redis(
                    host=self.config['redis_host'],
                    port=self.config.get('redis_port', 6379),
                    db=self.config.get('redis_db', 0)
                )
                self.logger.info("Redis client inizializzato")
                
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione dei client: {e}")
    
    def get_historical_data(self, symbol: str, timeframe: str = '1D', 
                          start_date: datetime = None, end_date: datetime = None,
                          limit: int = 1000) -> pd.DataFrame:
        """
        Recupera dati storici per un simbolo specifico.
        
        Args:
            symbol: Simbolo del titolo (es. 'AAPL', 'BTC/USD')
            timeframe: Timeframe dei dati ('1m', '5m', '1h', '1D')
            start_date: Data di inizio
            end_date: Data di fine
            limit: Numero massimo di candele
            
        Returns:
            DataFrame con colonne OHLCV
        """
        try:
            # Determina le date se non specificate
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=365)
            
            # Prova prima con Polygon se disponibile
            if self.polygon_client and '/' not in symbol:  # Solo per azioni
                return self._get_polygon_historical(symbol, timeframe, start_date, end_date)
            
            # Fallback su yfinance
            return self._get_yfinance_historical(symbol, timeframe, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero dati storici per {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_polygon_historical(self, symbol: str, timeframe: str, 
                               start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Recupera dati storici da Polygon API."""
        try:
            # Conversione timeframe
            timespan_map = {
                '1m': 'minute',
                '5m': 'minute',
                '15m': 'minute',
                '1h': 'hour',
                '1D': 'day'
            }
            
            multiplier_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '1h': 1,
                '1D': 1
            }
            
            timespan = timespan_map.get(timeframe, 'day')
            multiplier = multiplier_map.get(timeframe, 1)
            
            # Chiamata API Polygon
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            # Conversione in DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            self.logger.error(f"Errore Polygon API: {e}")
            return pd.DataFrame()
    
    def _get_yfinance_historical(self, symbol: str, timeframe: str,
                                start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Recupera dati storici da Yahoo Finance."""
        try:
            # Conversione timeframe per yfinance
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '1h': '1h',
                '1D': '1d'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Download dati
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Standardizzazione colonne
            df.columns = [col.lower() for col in df.columns]
            df.index.name = 'timestamp'
            
            return df
            
        except Exception as e:
            self.logger.error(f"Errore yfinance: {e}")
            return pd.DataFrame()
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Recupera il prezzo corrente di un simbolo.
        
        Args:
            symbol: Simbolo del titolo
            
        Returns:
            Prezzo corrente o None se non disponibile
        """
        try:
            # Controlla cache Redis
            if self.redis_client:
                cached_price = self.redis_client.get(f"price:{symbol}")
                if cached_price:
                    return float(cached_price)
            
            # Recupera da API
            if self.alpaca_client and '/' not in symbol:
                try:
                    quote = self.alpaca_client.get_latest_quote(symbol)
                    price = (quote.bid_price + quote.ask_price) / 2
                    
                    # Cache il risultato
                    if self.redis_client:
                        self.redis_client.setex(f"price:{symbol}", 5, price)
                    
                    return price
                except:
                    pass
            
            # Fallback su yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice') or info.get('previousClose')
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero prezzo per {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola indicatori tecnici sui dati OHLCV.
        
        Args:
            df: DataFrame con dati OHLCV
            
        Returns:
            DataFrame con indicatori aggiunti
        """
        if df.empty:
            return df
        
        try:
            # EMA (Exponential Moving Averages)
            df['ema8'] = df['close'].ewm(span=8).mean()
            df['ema34'] = df['close'].ewm(span=34).mean()
            df['ema50'] = df['close'].ewm(span=50).mean()
            df['ema200'] = df['close'].ewm(span=200).mean()
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd_line'] = exp1 - exp2
            df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd_line'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo degli indicatori: {e}")
            return df
    
    def subscribe_to_symbol(self, symbol: str, callback: Callable):
        """
        Sottoscrive agli aggiornamenti in tempo reale per un simbolo.
        
        Args:
            symbol: Simbolo da monitorare
            callback: Funzione da chiamare per ogni aggiornamento
        """
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        
        self.subscribers[symbol].append(callback)
        self.logger.info(f"Sottoscrizione aggiunta per {symbol}")
    
    def start_real_time_feed(self):
        """Avvia il feed di dati in tempo reale."""
        if self.running:
            return
        
        self.running = True
        self.data_thread = threading.Thread(target=self._real_time_worker)
        self.data_thread.daemon = True
        self.data_thread.start()
        self.logger.info("Feed dati in tempo reale avviato")
    
    def stop_real_time_feed(self):
        """Ferma il feed di dati in tempo reale."""
        self.running = False
        if self.data_thread:
            self.data_thread.join()
        self.logger.info("Feed dati in tempo reale fermato")
    
    def _real_time_worker(self):
        """Worker thread per aggiornamenti in tempo reale."""
        while self.running:
            try:
                for symbol in self.subscribers:
                    price = self.get_real_time_price(symbol)
                    if price:
                        # Notifica tutti i subscriber
                        for callback in self.subscribers[symbol]:
                            try:
                                callback(symbol, price, datetime.now())
                            except Exception as e:
                                self.logger.error(f"Errore callback per {symbol}: {e}")
                
                time.sleep(1)  # Aggiorna ogni secondo
                
            except Exception as e:
                self.logger.error(f"Errore nel worker dati: {e}")
                time.sleep(5)
    
    def get_market_status(self) -> Dict:
        """
        Recupera lo stato del mercato (aperto/chiuso).
        
        Returns:
            Dizionario con informazioni sullo stato del mercato
        """
        try:
            if self.alpaca_client:
                clock = self.alpaca_client.get_clock()
                return {
                    'is_open': clock.is_open,
                    'next_open': clock.next_open,
                    'next_close': clock.next_close,
                    'timestamp': datetime.now()
                }
            
            # Fallback: logica semplificata
            now = datetime.now()
            weekday = now.weekday()
            hour = now.hour
            
            # Mercato aperto lun-ven 9:30-16:00 EST (approssimativo)
            is_open = (weekday < 5 and 9 <= hour < 16)
            
            return {
                'is_open': is_open,
                'timestamp': now
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero stato mercato: {e}")
            return {'is_open': False, 'timestamp': datetime.now()}

