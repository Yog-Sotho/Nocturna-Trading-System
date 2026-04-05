"""
Strategy Manager per NOCTURNA v2.0 Trading Bot
Implementa la logica di identificazione dello stato di mercato e switching tra modalità.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

class TradingMode(Enum):
    """Modalità di trading disponibili."""
    EVE = "EVE"          # Grid Trading
    LUCIFER = "LUCIFER"  # Breakout Trading
    REAPER = "REAPER"    # Reversal Trading
    SENTINEL = "SENTINEL" # Trend Following

class MarketState(Enum):
    """Stati di mercato identificabili."""
    RANGING = "RANGING"
    TRENDING = "TRENDING"
    REVERSING = "REVERSING"
    BREAKOUT = "BREAKOUT"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"

class StrategyManager:
    """
    Gestisce l'identificazione dello stato di mercato e la selezione della strategia ottimale.
    Implementa la logica core dell'algoritmo NOCTURNA v2.0.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Stato corrente
        self.current_mode = TradingMode.SENTINEL
        self.current_market_state = MarketState.UNKNOWN
        self.last_analysis_time = None
        
        # Parametri della strategia
        self.parameters = self._load_parameters()
        
        # Cache per analisi
        self.analysis_cache = {}
        self.mode_history = []
        
        # Griglia EVE
        self.grid_levels = []
        self.grid_base_price = None
        
        self.logger.info("Strategy Manager inizializzato")
    
    def _load_parameters(self) -> Dict:
        """Carica i parametri della strategia dalla configurazione."""
        default_params = {
            # Parametri EMA
            'ema_fast': 8,
            'ema_medium': 34,
            'ema_slow': 50,
            'ema_trend': 200,
            
            # Parametri MACD
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Parametri ATR
            'atr_period': 14,
            'atr_mult_sl': 2.0,
            'atr_mult_tp': 3.0,
            
            # Parametri Grid (EVE)
            'grid_spacing': 0.005,  # 0.5%
            'grid_levels': 10,
            'grid_position_size': 0.1,
            
            # Parametri Risk Management
            'max_position_size': 0.2,  # 20% del capitale
            'max_daily_loss': 0.05,    # 5% perdita giornaliera massima
            'volatility_threshold': 2.0,
            
            # Parametri Trailing Stop
            'trail_trigger': 0.02,     # 2% profitto per attivare trailing
            'trail_offset': 0.01,      # 1% offset del trailing stop
            
            # Parametri Take Profit
            'tp_target': 0.025,        # 2.5% take profit
            
            # Soglie per identificazione stato mercato
            'ranging_threshold': 0.25,
            'trend_threshold': 1.0,
            'breakout_threshold': 1.5,
            'volatility_spike_threshold': 3.0
        }
        
        # Merge con parametri da config
        params = default_params.copy()
        params.update(self.config.get('strategy_parameters', {}))
        
        return params
    
    def analyze_market_state(self, df: pd.DataFrame, symbol: str) -> MarketState:
        """
        Analizza lo stato corrente del mercato basandosi sui dati OHLCV.
        
        Args:
            df: DataFrame con dati OHLCV e indicatori tecnici
            symbol: Simbolo analizzato
            
        Returns:
            Stato di mercato identificato
        """
        if df.empty or len(df) < 200:
            return MarketState.UNKNOWN
        
        try:
            latest = df.iloc[-1]
            prev_10 = df.iloc[-11:-1] if len(df) > 10 else df.iloc[:-1]
            
            # Calcola metriche per identificazione stato
            ema50 = latest['ema50']
            ema200 = latest['ema200']
            ema50_prev = df.iloc[-11]['ema50'] if len(df) > 10 else ema50
            atr = latest['atr']
            macd_line = latest['macd_line']
            macd_signal = latest['macd_signal']
            close = latest['close']
            
            # 1. Controllo volatilità estrema
            volatility_spike = self._detect_volatility_spike(df)
            if volatility_spike:
                self.logger.info(f"Volatilità estrema rilevata per {symbol}")
                return MarketState.VOLATILE
            
            # 2. Rilevamento mercato in range
            ema50_change = abs(ema50 - ema50_prev)
            ranging_condition = ema50_change < (atr * self.parameters['ranging_threshold'])
            
            if ranging_condition:
                self.logger.info(f"Mercato in range rilevato per {symbol}")
                return MarketState.RANGING
            
            # 3. Rilevamento breakout
            ema200_distance = abs(close - ema200)
            breakout_condition = ema200_distance > (atr * self.parameters['breakout_threshold'])
            
            if breakout_condition:
                self.logger.info(f"Breakout rilevato per {symbol}")
                return MarketState.BREAKOUT
            
            # 4. Rilevamento inversione
            ema8 = latest['ema8']
            ema34 = latest['ema34']
            ema8_prev = df.iloc[-2]['ema8'] if len(df) > 1 else ema8
            ema34_prev = df.iloc[-2]['ema34'] if len(df) > 1 else ema34
            
            # Crossover detection
            bullish_cross = (ema8 > ema34) and (ema8_prev <= ema34_prev)
            bearish_cross = (ema8 < ema34) and (ema8_prev >= ema34_prev)
            
            if bullish_cross or bearish_cross:
                self.logger.info(f"Inversione rilevata per {symbol}")
                return MarketState.REVERSING
            
            # 5. Rilevamento trend
            ema50_ema200_distance = abs(ema50 - ema200)
            trend_condition = (ema50_ema200_distance > atr and 
                             macd_line > macd_signal)
            
            if trend_condition:
                self.logger.info(f"Trend rilevato per {symbol}")
                return MarketState.TRENDING
            
            # Default: stato sconosciuto
            return MarketState.UNKNOWN
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi stato mercato per {symbol}: {e}")
            return MarketState.UNKNOWN
    
    def _detect_volatility_spike(self, df: pd.DataFrame) -> bool:
        """Rileva picchi di volatilità anomali."""
        try:
            if len(df) < 20:
                return False
            
            current_atr = df.iloc[-1]['atr']
            avg_atr = df['atr'].tail(20).mean()
            
            return current_atr > (avg_atr * self.parameters['volatility_spike_threshold'])
            
        except Exception:
            return False
    
    def select_trading_mode(self, market_state: MarketState) -> TradingMode:
        """
        Seleziona la modalità di trading ottimale basandosi sullo stato di mercato.
        
        Args:
            market_state: Stato corrente del mercato
            
        Returns:
            Modalità di trading selezionata
        """
        mode_mapping = {
            MarketState.RANGING: TradingMode.EVE,
            MarketState.REVERSING: TradingMode.REAPER,
            MarketState.TRENDING: TradingMode.SENTINEL,
            MarketState.BREAKOUT: TradingMode.LUCIFER,
            MarketState.VOLATILE: TradingMode.SENTINEL,  # Modalità conservativa
            MarketState.UNKNOWN: TradingMode.SENTINEL   # Default
        }
        
        selected_mode = mode_mapping.get(market_state, TradingMode.SENTINEL)
        
        # Log del cambio modalità
        if selected_mode != self.current_mode:
            self.logger.info(f"Cambio modalità: {self.current_mode.value} -> {selected_mode.value}")
            self.mode_history.append({
                'timestamp': datetime.now(),
                'from_mode': self.current_mode.value,
                'to_mode': selected_mode.value,
                'market_state': market_state.value
            })
        
        return selected_mode
    
    def generate_trading_signals(self, df: pd.DataFrame, symbol: str, 
                                mode: TradingMode) -> List[Dict]:
        """
        Genera segnali di trading basandosi sulla modalità attiva.
        
        Args:
            df: DataFrame con dati di mercato
            symbol: Simbolo analizzato
            mode: Modalità di trading attiva
            
        Returns:
            Lista di segnali di trading
        """
        signals = []
        
        try:
            if mode == TradingMode.EVE:
                signals = self._generate_eve_signals(df, symbol)
            elif mode == TradingMode.LUCIFER:
                signals = self._generate_lucifer_signals(df, symbol)
            elif mode == TradingMode.REAPER:
                signals = self._generate_reaper_signals(df, symbol)
            elif mode == TradingMode.SENTINEL:
                signals = self._generate_sentinel_signals(df, symbol)
            
            # Applica filtri di risk management
            signals = self._apply_risk_filters(signals, df, symbol)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione segnali per {symbol}: {e}")
            return []
    
    def _generate_eve_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Genera segnali per modalità EVE (Grid Trading)."""
        signals = []
        
        if df.empty:
            return signals
        
        try:
            latest = df.iloc[-1]
            current_price = latest['close']
            atr = latest['atr']
            
            # Inizializza griglia se necessario
            if not self.grid_levels or self.grid_base_price is None:
                self._initialize_grid(current_price, atr)
            
            # Controlla se il prezzo ha attraversato livelli di griglia
            for level in self.grid_levels:
                if level['side'] == 'buy' and current_price <= level['price']:
                    if not level['filled']:
                        signals.append({
                            'symbol': symbol,
                            'side': 'buy',
                            'type': 'limit',
                            'price': level['price'],
                            'quantity': self.parameters['grid_position_size'],
                            'mode': 'EVE',
                            'level_id': level['id'],
                            'timestamp': datetime.now()
                        })
                        level['filled'] = True
                
                elif level['side'] == 'sell' and current_price >= level['price']:
                    if not level['filled']:
                        signals.append({
                            'symbol': symbol,
                            'side': 'sell',
                            'type': 'limit',
                            'price': level['price'],
                            'quantity': self.parameters['grid_position_size'],
                            'mode': 'EVE',
                            'level_id': level['id'],
                            'timestamp': datetime.now()
                        })
                        level['filled'] = True
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Errore segnali EVE: {e}")
            return []
    
    def _generate_lucifer_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Genera segnali per modalità LUCIFER (Breakout Trading)."""
        signals = []
        
        if len(df) < 2:
            return signals
        
        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            current_price = latest['close']
            ema200 = latest['ema200']
            ema200_prev = previous['ema200']
            atr = latest['atr']
            
            # Breakout rialzista
            if (current_price > ema200 and 
                previous['close'] <= ema200_prev):
                
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price - (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price + (atr * self.parameters['atr_mult_tp']),
                    'mode': 'LUCIFER',
                    'signal_type': 'bullish_breakout',
                    'timestamp': datetime.now()
                })
            
            # Breakout ribassista
            elif (current_price < ema200 and 
                  previous['close'] >= ema200_prev):
                
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price + (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price - (atr * self.parameters['atr_mult_tp']),
                    'mode': 'LUCIFER',
                    'signal_type': 'bearish_breakout',
                    'timestamp': datetime.now()
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Errore segnali LUCIFER: {e}")
            return []
    
    def _generate_reaper_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Genera segnali per modalità REAPER (Reversal Trading)."""
        signals = []
        
        if len(df) < 2:
            return signals
        
        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            ema8 = latest['ema8']
            ema34 = latest['ema34']
            ema8_prev = previous['ema8']
            ema34_prev = previous['ema34']
            current_price = latest['close']
            atr = latest['atr']
            
            # Crossover rialzista
            if ema8 > ema34 and ema8_prev <= ema34_prev:
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price - (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price + (atr * self.parameters['atr_mult_tp']),
                    'mode': 'REAPER',
                    'signal_type': 'bullish_reversal',
                    'timestamp': datetime.now()
                })
            
            # Crossover ribassista
            elif ema8 < ema34 and ema8_prev >= ema34_prev:
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price + (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price - (atr * self.parameters['atr_mult_tp']),
                    'mode': 'REAPER',
                    'signal_type': 'bearish_reversal',
                    'timestamp': datetime.now()
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Errore segnali REAPER: {e}")
            return []
    
    def _generate_sentinel_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Genera segnali per modalità SENTINEL (Trend Following)."""
        signals = []
        
        if len(df) < 2:
            return signals
        
        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            ema50 = latest['ema50']
            ema200 = latest['ema200']
            macd_line = latest['macd_line']
            macd_signal = latest['macd_signal']
            macd_line_prev = previous['macd_line']
            macd_signal_prev = previous['macd_signal']
            current_price = latest['close']
            atr = latest['atr']
            
            # Segnale rialzista: EMA50 > EMA200 e MACD crossover
            if (ema50 > ema200 and 
                macd_line > macd_signal and 
                macd_line_prev <= macd_signal_prev):
                
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price - (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price + (atr * self.parameters['atr_mult_tp']),
                    'mode': 'SENTINEL',
                    'signal_type': 'bullish_trend',
                    'timestamp': datetime.now()
                })
            
            # Segnale ribassista: EMA50 < EMA200 e MACD crossover
            elif (ema50 < ema200 and 
                  macd_line < macd_signal and 
                  macd_line_prev >= macd_signal_prev):
                
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price + (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price - (atr * self.parameters['atr_mult_tp']),
                    'mode': 'SENTINEL',
                    'signal_type': 'bearish_trend',
                    'timestamp': datetime.now()
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Errore segnali SENTINEL: {e}")
            return []
    
    def _initialize_grid(self, base_price: float, atr: float):
        """Inizializza i livelli di griglia per modalità EVE."""
        self.grid_base_price = base_price
        self.grid_levels = []
        
        grid_spacing = self.parameters['grid_spacing']
        num_levels = self.parameters['grid_levels']
        
        # Crea livelli sopra e sotto il prezzo base
        for i in range(1, num_levels + 1):
            # Livelli di vendita (sopra)
            sell_price = base_price * (1 + grid_spacing * i)
            self.grid_levels.append({
                'id': f"sell_{i}",
                'price': sell_price,
                'side': 'sell',
                'filled': False
            })
            
            # Livelli di acquisto (sotto)
            buy_price = base_price * (1 - grid_spacing * i)
            self.grid_levels.append({
                'id': f"buy_{i}",
                'price': buy_price,
                'side': 'buy',
                'filled': False
            })
        
        self.logger.info(f"Griglia inizializzata con {len(self.grid_levels)} livelli")
    
    def _apply_risk_filters(self, signals: List[Dict], df: pd.DataFrame, 
                           symbol: str) -> List[Dict]:
        """Applica filtri di risk management ai segnali."""
        filtered_signals = []
        
        for signal in signals:
            # Filtro volatilità
            if self._detect_volatility_spike(df):
                self.logger.warning(f"Segnale bloccato per alta volatilità: {symbol}")
                continue
            
            # Filtro dimensione posizione
            if signal.get('quantity', 0) > self.parameters['max_position_size']:
                signal['quantity'] = self.parameters['max_position_size']
            
            # Aggiungi trailing stop se configurato
            if signal.get('side') == 'buy':
                signal['trail_trigger'] = signal.get('take_profit', 0) * (1 - self.parameters['trail_trigger'])
                signal['trail_offset'] = self.parameters['trail_offset']
            elif signal.get('side') == 'sell':
                signal['trail_trigger'] = signal.get('take_profit', 0) * (1 + self.parameters['trail_trigger'])
                signal['trail_offset'] = self.parameters['trail_offset']
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    def update_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Aggiorna la strategia basandosi sui nuovi dati di mercato.
        
        Args:
            df: DataFrame con dati di mercato aggiornati
            symbol: Simbolo analizzato
            
        Returns:
            Dizionario con stato aggiornato e segnali
        """
        try:
            # Analizza stato mercato
            market_state = self.analyze_market_state(df, symbol)
            
            # Seleziona modalità
            new_mode = self.select_trading_mode(market_state)
            
            # Genera segnali
            signals = self.generate_trading_signals(df, symbol, new_mode)
            
            # Aggiorna stato
            self.current_mode = new_mode
            self.current_market_state = market_state
            self.last_analysis_time = datetime.now()
            
            return {
                'symbol': symbol,
                'market_state': market_state.value,
                'trading_mode': new_mode.value,
                'signals': signals,
                'timestamp': self.last_analysis_time,
                'grid_levels': len(self.grid_levels) if self.grid_levels else 0
            }
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento strategia per {symbol}: {e}")
            return {
                'symbol': symbol,
                'market_state': MarketState.UNKNOWN.value,
                'trading_mode': self.current_mode.value,
                'signals': [],
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def get_strategy_status(self) -> Dict:
        """Restituisce lo stato corrente della strategia."""
        return {
            'current_mode': self.current_mode.value,
            'current_market_state': self.current_market_state.value,
            'last_analysis_time': self.last_analysis_time,
            'grid_levels_count': len(self.grid_levels) if self.grid_levels else 0,
            'grid_base_price': self.grid_base_price,
            'mode_history': self.mode_history[-10:],  # Ultimi 10 cambi
            'parameters': self.parameters
        }

