"""
NOCTURNA Trading System - Strategy Manager
Production-grade market analysis and trading strategy management.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TradingMode(Enum):
    """Trading mode enumeration."""
    EVE = "EVE"        # Grid Trading (sideways markets)
    LUCIFER = "LUCIFER"    # Breakout Trading (key level breaks)
    REAPER = "REAPER"      # Reversal Trading (trend reversals)
    SENTINEL = "SENTINEL"  # Trend Following (strong trends)


class MarketState(Enum):
    """Market state enumeration."""
    RANGING = "RANGING"
    TRENDING = "TRENDING"
    REVERSING = "REVERSING"
    BREAKOUT = "BREAKOUT"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class StrategyManager:
    """
    Production-grade strategy manager.
    Identifies market state and generates trading signals.
    """

    # Required minimum data points for reliable analysis
    MIN_DATA_POINTS = 200

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Current state
        self.current_mode = TradingMode.SENTINEL
        self.current_market_state = MarketState.UNKNOWN
        self.last_analysis_time: Optional[datetime] = None

        # Parameters
        self.parameters = self._load_parameters()

        # Analysis cache
        self.analysis_cache: Dict[str, Dict] = {}
        self.mode_history: List[Dict] = []

        # Grid levels for EVE mode
        self.grid_levels: List[Dict] = []
        self.grid_base_price: Optional[float] = None

        # Signal history for validation
        self.signal_history = deque(maxlen=100)

        self.logger.info("Strategy Manager initialized")

    def _load_parameters(self) -> Dict:
        """Load and validate strategy parameters."""
        default_params = {
            # EMA periods
            'ema_fast': 8,
            'ema_medium': 34,
            'ema_slow': 50,
            'ema_trend': 200,

            # MACD parameters
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,

            # ATR parameters
            'atr_period': 14,
            'atr_mult_sl': 2.0,
            'atr_mult_tp': 3.0,

            # Grid parameters (EVE mode)
            'grid_spacing': 0.005,  # 0.5%
            'grid_levels': 10,
            'grid_position_size': 0.1,

            # Risk parameters
            'max_position_size': 0.2,  # 20% of capital
            'max_daily_loss': 0.05,  # 5% max daily loss
            'volatility_threshold': 2.0,

            # Trailing stop parameters
            'trail_trigger': 0.02,  # 2% profit to activate
            'trail_offset': 0.01,  # 1% trailing offset

            # Take profit target
            'tp_target': 0.025,  # 2.5% take profit

            # Market state detection thresholds
            'ranging_threshold': 0.25,
            'trend_threshold': 1.0,
            'breakout_threshold': 1.5,
            'volatility_spike_threshold': 3.0,

            # Signal generation
            'min_signal_confidence': 0.6,
            'reversal_confirmation_bars': 3,
        }

        # Merge with config
        params = default_params.copy()
        strategy_config = self.config.get('strategy_parameters', {})
        params.update(strategy_config)

        # Validate ranges
        params['max_position_size'] = min(0.5, max(0.01, params['max_position_size']))
        params['atr_mult_sl'] = min(5.0, max(0.5, params['atr_mult_sl']))
        params['atr_mult_tp'] = min(10.0, max(0.5, params['atr_mult_tp']))
        params['grid_spacing'] = min(0.1, max(0.001, params['grid_spacing']))

        return params

    def analyze_market_state(self, df: pd.DataFrame, symbol: str) -> MarketState:
        """
        Analyze current market state based on price data.

        Args:
            df: DataFrame with OHLCV data and indicators
            symbol: Symbol being analyzed

        Returns:
            Identified market state
        """
        if df.empty or len(df) < self.MIN_DATA_POINTS:
            return MarketState.UNKNOWN

        try:
            latest = df.iloc[-1]
            previous_10 = df.iloc[-11:-1] if len(df) > 10 else df

            # Extract indicators
            ema50 = latest.get('ema50', 0)
            ema200 = latest.get('ema200', 0)
            ema50_prev = df.iloc[-11]['ema50'] if len(df) > 10 else ema50
            atr = latest.get('atr', 0)
            macd_line = latest.get('macd_line', 0)
            macd_signal = latest.get('macd_signal', 0)
            close = latest.get('close', 0)

            # 1. Check for extreme volatility
            if self._detect_volatility_spike(df):
                self.logger.info(f"Volatility spike detected for {symbol}")
                return MarketState.VOLATILE

            # 2. Check for ranging market
            ema50_change = abs(ema50 - ema50_prev)
            ranging_condition = ema50_change < (atr * self.parameters['ranging_threshold'])

            if ranging_condition:
                self.logger.debug(f"Ranging market detected for {symbol}")
                return MarketState.RANGING

            # 3. Check for breakout
            ema200_distance = abs(close - ema200)
            breakout_condition = ema200_distance > (atr * self.parameters['breakout_threshold'])

            if breakout_condition:
                self.logger.info(f"Breakout detected for {symbol}")
                return MarketState.BREAKOUT

            # 4. Check for reversal
            ema8 = latest.get('ema8', 0)
            ema34 = latest.get('ema34', 0)
            ema8_prev = df.iloc[-2]['ema8'] if len(df) > 1 else ema8
            ema34_prev = df.iloc[-2]['ema34'] if len(df) > 1 else ema34

            # Detect EMA crossover
            bullish_cross = (ema8 > ema34) and (ema8_prev <= ema34_prev)
            bearish_cross = (ema8 < ema34) and (ema8_prev >= ema34_prev)

            if bullish_cross or bearish_cross:
                self.logger.info(f"Reversal detected for {symbol}")
                return MarketState.REVERSING

            # 5. Check for trend
            ema_distance = abs(ema50 - ema200)
            trend_condition = (ema_distance > atr) and (macd_line > macd_signal)

            if trend_condition:
                self.logger.debug(f"Trending market detected for {symbol}")
                return MarketState.TRENDING

            return MarketState.UNKNOWN

        except Exception as e:
            self.logger.error(f"Error analyzing market state for {symbol}: {e}")
            return MarketState.UNKNOWN

    def _detect_volatility_spike(self, df: pd.DataFrame) -> bool:
        """Detect if volatility is abnormally high."""
        if len(df) < 20:
            return False

        try:
            current_atr = df.iloc[-1]['atr']
            avg_atr = df['atr'].tail(20).mean()

            if avg_atr == 0:
                return False

            return current_atr > (avg_atr * self.parameters['volatility_spike_threshold'])

        except Exception:
            return False

    def select_trading_mode(self, market_state: MarketState) -> TradingMode:
        """
        Select optimal trading mode based on market state.

        Args:
            market_state: Identified market state

        Returns:
            Selected trading mode
        """
        mode_mapping = {
            MarketState.RANGING: TradingMode.EVE,
            MarketState.REVERSING: TradingMode.REAPER,
            MarketState.TRENDING: TradingMode.SENTINEL,
            MarketState.BREAKOUT: TradingMode.LUCIFER,
            MarketState.VOLATILE: TradingMode.SENTINEL,  # Conservative mode
            MarketState.UNKNOWN: TradingMode.SENTINEL  # Default mode
        }

        selected_mode = mode_mapping.get(market_state, TradingMode.SENTINEL)

        # Log mode changes
        if selected_mode != self.current_mode:
            self.logger.info(f"Mode change: {self.current_mode.value} -> {selected_mode.value}")

            self.mode_history.append({
                'timestamp': datetime.now(timezone.utc),
                'from_mode': self.current_mode.value,
                'to_mode': selected_mode.value,
                'market_state': market_state.value
            })

            # Keep history manageable
            if len(self.mode_history) > 100:
                self.mode_history = self.mode_history[-100:]

        return selected_mode

    def generate_trading_signals(self, df: pd.DataFrame, symbol: str,
                                  mode: TradingMode) -> List[Dict]:
        """
        Generate trading signals based on active mode.

        Args:
            df: DataFrame with market data
            symbol: Symbol to analyze
            mode: Active trading mode

        Returns:
            List of trading signals
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

            # Apply risk filters
            signals = self._apply_risk_filters(signals, df, symbol)

            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    def _generate_eve_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Generate signals for EVE grid trading mode."""
        if df.empty:
            return []

        signals = []

        try:
            latest = df.iloc[-1]
            current_price = latest['close']
            atr = latest.get('atr', 0)

            # Initialize grid if needed
            if not self.grid_levels or self.grid_base_price is None:
                self._initialize_grid(current_price, atr)

            # Check if price crossed grid levels
            for level in self.grid_levels:
                if level['filled']:
                    continue

                if level['side'] == 'buy' and current_price <= level['price']:
                    signals.append({
                        'symbol': symbol,
                        'side': 'buy',
                        'type': 'limit',
                        'price': level['price'],
                        'quantity': self.parameters['grid_position_size'],
                        'mode': 'EVE',
                        'level_id': level['id'],
                        'timestamp': datetime.now(timezone.utc)
                    })
                    level['filled'] = True

                elif level['side'] == 'sell' and current_price >= level['price']:
                    signals.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'type': 'limit',
                        'price': level['price'],
                        'quantity': self.parameters['grid_position_size'],
                        'mode': 'EVE',
                        'level_id': level['id'],
                        'timestamp': datetime.now(timezone.utc)
                    })
                    level['filled'] = True

            return signals

        except Exception as e:
            self.logger.error(f"Error generating EVE signals: {e}")
            return []

    def _generate_lucifer_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Generate signals for LUCIFER breakout mode."""
        if len(df) < 2:
            return []

        signals = []

        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]

            current_price = latest['close']
            ema200 = latest.get('ema200', 0)
            ema200_prev = previous.get('ema200', 0)
            atr = latest.get('atr', 0)

            # Bullish breakout: price crosses above EMA200
            if (current_price > ema200 and previous['close'] <= ema200_prev):
                signals.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price - (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price + (atr * self.parameters['atr_mult_tp']),
                    'mode': 'LUCIFER',
                    'signal_type': 'bullish_breakout',
                    'timestamp': datetime.now(timezone.utc)
                })

            # Bearish breakout: price crosses below EMA200
            elif (current_price < ema200 and previous['close'] >= ema200_prev):
                signals.append({
                    'symbol': symbol,
                    'side': 'sell',
                    'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': current_price + (atr * self.parameters['atr_mult_sl']),
                    'take_profit': current_price - (atr * self.parameters['atr_mult_tp']),
                    'mode': 'LUCIFER',
                    'signal_type': 'bearish_breakout',
                    'timestamp': datetime.now(timezone.utc)
                })

            return signals

        except Exception as e:
            self.logger.error(f"Error generating LUCIFER signals: {e}")
            return []

    def _generate_reaper_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Generate signals for REAPER reversal mode."""
        if len(df) < 2:
            return []

        signals = []

        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]

            ema8 = latest.get('ema8', 0)
            ema34 = latest.get('ema34', 0)
            ema8_prev = previous.get('ema8', 0)
            ema34_prev = previous.get('ema34', 0)
            current_price = latest['close']
            atr = latest.get('atr', 0)

            # Bullish EMA crossover
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
                    'timestamp': datetime.now(timezone.utc)
                })

            # Bearish EMA crossover
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
                    'timestamp': datetime.now(timezone.utc)
                })

            return signals

        except Exception as e:
            self.logger.error(f"Error generating REAPER signals: {e}")
            return []

    def _generate_sentinel_signals(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Generate signals for SENTINEL trend-following mode."""
        if len(df) < 2:
            return []

        signals = []

        try:
            latest = df.iloc[-1]
            previous = df.iloc[-2]

            ema50 = latest.get('ema50', 0)
            ema200 = latest.get('ema200', 0)
            macd_line = latest.get('macd_line', 0)
            macd_signal = latest.get('macd_signal', 0)
            macd_line_prev = previous.get('macd_line', 0)
            macd_signal_prev = previous.get('macd_signal', 0)
            current_price = latest['close']
            atr = latest.get('atr', 0)

            # Bullish trend: EMA50 > EMA200 and MACD bullish crossover
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
                    'timestamp': datetime.now(timezone.utc)
                })

            # Bearish trend: EMA50 < EMA200 and MACD bearish crossover
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
                    'timestamp': datetime.now(timezone.utc)
                })

            return signals

        except Exception as e:
            self.logger.error(f"Error generating SENTINEL signals: {e}")
            return []

    def _initialize_grid(self, base_price: float, atr: float) -> None:
        """Initialize grid levels for EVE mode."""
        self.grid_base_price = base_price
        self.grid_levels = []

        grid_spacing = self.parameters['grid_spacing']
        num_levels = self.parameters['grid_levels']

        # Create levels above and below base price
        for i in range(1, num_levels + 1):
            # Sell levels (above)
            sell_price = base_price * (1 + grid_spacing * i)
            self.grid_levels.append({
                'id': f"sell_{i}",
                'price': sell_price,
                'side': 'sell',
                'filled': False
            })

            # Buy levels (below)
            buy_price = base_price * (1 - grid_spacing * i)
            self.grid_levels.append({
                'id': f"buy_{i}",
                'price': buy_price,
                'side': 'buy',
                'filled': False
            })

        self.logger.info(f"Grid initialized with {len(self.grid_levels)} levels")

    def _apply_risk_filters(self, signals: List[Dict], df: pd.DataFrame,
                            symbol: str) -> List[Dict]:
        """Apply risk management filters to signals."""
        filtered_signals = []

        for signal in signals:
            # Volatility filter
            if self._detect_volatility_spike(df):
                self.logger.warning(f"Signal blocked due to high volatility: {symbol}")
                continue

            # Adjust position size if too large
            if signal.get('quantity', 0) > self.parameters['max_position_size']:
                signal['quantity'] = self.parameters['max_position_size']

            # Add trailing stop parameters
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
        Update strategy based on new market data.

        Args:
            df: Updated market data
            symbol: Symbol being analyzed

        Returns:
            Dictionary with updated state and signals
        """
        try:
            # Analyze market state
            market_state = self.analyze_market_state(df, symbol)

            # Select trading mode
            new_mode = self.select_trading_mode(market_state)

            # Generate signals
            signals = self.generate_trading_signals(df, symbol, new_mode)

            # Update state
            self.current_mode = new_mode
            self.current_market_state = market_state
            self.last_analysis_time = datetime.now(timezone.utc)

            # Store signal history
            for signal in signals:
                self.signal_history.append({
                    'symbol': symbol,
                    'signal': signal,
                    'timestamp': datetime.now(timezone.utc)
                })

            return {
                'symbol': symbol,
                'market_state': market_state.value,
                'trading_mode': new_mode.value,
                'signals': signals,
                'timestamp': self.last_analysis_time,
                'grid_levels_count': len(self.grid_levels) if self.grid_levels else 0
            }

        except Exception as e:
            self.logger.error(f"Error updating strategy for {symbol}: {e}")
            return {
                'symbol': symbol,
                'market_state': MarketState.UNKNOWN.value,
                'trading_mode': self.current_mode.value,
                'signals': [],
                'timestamp': datetime.now(timezone.utc),
                'error': str(e)
            }

    def get_strategy_status(self) -> Dict:
        """Get current strategy status."""
        return {
            'current_mode': self.current_mode.value,
            'current_market_state': self.current_market_state.value,
            'last_analysis_time': self.last_analysis_time,
            'grid_levels_count': len(self.grid_levels) if self.grid_levels else 0,
            'grid_base_price': self.grid_base_price,
            'mode_history': self.mode_history[-10:],
            'parameters': self.parameters.copy()
        }
