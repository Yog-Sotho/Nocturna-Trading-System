"""
NOCTURNA Trading System - Enhanced Strategy Manager
Production-grade market analysis with multi-indicator confluence,
volume confirmation, sentiment integration, and 4 autonomous trading modes.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum
from collections import deque


class TradingMode(Enum):
    """Trading mode enumeration."""
    EVE = "EVE"            # Grid Trading (sideways markets)
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
    Production-grade strategy manager with multi-indicator confluence.
    Identifies market state, generates high-conviction trading signals,
    and integrates sentiment data for signal filtering.
    """

    MIN_DATA_POINTS = 200

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.current_mode = TradingMode.SENTINEL
        self.current_market_state = MarketState.UNKNOWN
        self.last_analysis_time: Optional[datetime] = None

        self.parameters = self._load_parameters()

        self.analysis_cache: Dict[str, Dict] = {}
        self.mode_history: List[Dict] = []

        # Grid levels per-symbol
        self.grid_levels: Dict[str, List[Dict]] = {}
        self.grid_base_prices: Dict[str, float] = {}

        # External data — set by engine each cycle
        self.current_positions: Dict[str, Dict] = {}
        self.current_sentiment: Dict[str, Dict] = {}

        self.signal_history = deque(maxlen=200)

        self.logger.info("Enhanced Strategy Manager initialized")

    def set_current_positions(self, positions: Dict[str, Dict]) -> None:
        """Update current positions from the engine."""
        self.current_positions = positions

    def set_sentiment_data(self, sentiment: Dict[str, Dict]) -> None:
        """Update sentiment data from the engine's SentimentAnalyzer."""
        self.current_sentiment = sentiment

    def _load_parameters(self) -> Dict:
        """Load and validate strategy parameters."""
        default_params = {
            'ema_fast': 8, 'ema_medium': 34, 'ema_slow': 50, 'ema_trend': 200,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'atr_period': 14, 'atr_mult_sl': 2.0, 'atr_mult_tp': 3.0,
            'grid_spacing': 0.005, 'grid_levels': 10, 'grid_position_size': 0.1,
            'max_position_size': 0.2, 'max_daily_loss': 0.05, 'volatility_threshold': 2.0,
            'trail_trigger': 0.02, 'trail_offset': 0.01, 'tp_target': 0.025,
            'ranging_threshold': 0.25, 'trend_threshold': 1.0,
            'breakout_threshold': 1.5, 'volatility_spike_threshold': 3.0,
            'min_signal_confidence': 0.6, 'reversal_confirmation_bars': 3,
            # Volume confirmation
            'volume_confirmation_mult': 1.5,
            'breakout_volume_mult': 2.0,
            'volume_climax_mult': 3.0,
            # RSI thresholds
            'rsi_overbought': 70, 'rsi_oversold': 30,
            'rsi_trend_bullish': 50, 'rsi_trend_bearish': 50,
            # BB squeeze
            'bb_squeeze_percentile': 20,
            # Sentiment
            'sentiment_block_threshold': -0.4,
            'sentiment_boost_threshold': 0.3,
        }
        params = default_params.copy()
        params.update(self.config.get('strategy_parameters', {}))
        params['max_position_size'] = min(0.5, max(0.01, params['max_position_size']))
        params['atr_mult_sl'] = min(5.0, max(0.5, params['atr_mult_sl']))
        params['atr_mult_tp'] = min(10.0, max(0.5, params['atr_mult_tp']))
        params['grid_spacing'] = min(0.1, max(0.001, params['grid_spacing']))
        return params

    # =========================================================================
    # VOLUME CONFIRMATION
    # =========================================================================

    def _has_volume_confirmation(self, df: pd.DataFrame, multiplier: float = None) -> bool:
        """Check if current bar has above-average volume."""
        if multiplier is None:
            multiplier = self.parameters['volume_confirmation_mult']
        try:
            return df.iloc[-1].get('volume_ratio', 1.0) >= multiplier
        except (IndexError, KeyError):
            return False

    def _detect_volume_climax(self, df: pd.DataFrame) -> bool:
        """Detect volume climax — extremely high volume signaling exhaustion."""
        try:
            return df.iloc[-1].get('volume_ratio', 1.0) >= self.parameters['volume_climax_mult']
        except (IndexError, KeyError):
            return False

    def _detect_volume_divergence(self, df: pd.DataFrame, direction: str) -> bool:
        """Detect price/volume divergence — price trending but volume declining."""
        if len(df) < 10:
            return False
        try:
            recent = df.tail(10)
            prices = recent['close'].values
            volumes = recent['volume'].values
            if direction == 'bullish':
                return prices[-1] > prices[0] and volumes[-1] < np.mean(volumes[:5])
            else:
                return prices[-1] < prices[0] and volumes[-1] < np.mean(volumes[:5])
        except Exception:
            return False

    # =========================================================================
    # INDICATOR HELPERS
    # =========================================================================

    def _get_rsi(self, df: pd.DataFrame) -> Dict:
        """Get RSI conditions."""
        try:
            latest = df.iloc[-1]
            rsi = latest.get('rsi', 50)
            prev_rsi = df.iloc[-2].get('rsi', 50) if len(df) > 1 else rsi
            return {
                'value': rsi,
                'overbought': rsi > self.parameters['rsi_overbought'],
                'oversold': rsi < self.parameters['rsi_oversold'],
                'bullish': rsi > self.parameters['rsi_trend_bullish'],
                'bearish': rsi < self.parameters['rsi_trend_bearish'],
                'rising': rsi > prev_rsi, 'falling': rsi < prev_rsi,
            }
        except Exception:
            return {'value': 50, 'overbought': False, 'oversold': False,
                    'bullish': False, 'bearish': False, 'rising': False, 'falling': False}

    def _get_bb(self, df: pd.DataFrame) -> Dict:
        """Get Bollinger Band conditions."""
        try:
            latest = df.iloc[-1]
            bb_pos = latest.get('bb_position', 0.5)
            bb_upper = latest.get('bb_upper', 0)
            bb_lower = latest.get('bb_lower', 0)
            bb_middle = latest.get('bb_middle', 0)
            squeeze = False
            if len(df) >= 20 and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                widths = (df['bb_upper'] - df['bb_lower']).tail(100)
                squeeze = (bb_upper - bb_lower) < widths.quantile(self.parameters['bb_squeeze_percentile'] / 100)
            return {'position': bb_pos, 'upper': bb_upper, 'lower': bb_lower,
                    'middle': bb_middle, 'squeeze': squeeze,
                    'above_upper': bb_pos > 1.0, 'below_lower': bb_pos < 0.0}
        except Exception:
            return {'position': 0.5, 'upper': 0, 'lower': 0, 'middle': 0,
                    'squeeze': False, 'above_upper': False, 'below_lower': False}

    def _get_stoch(self, df: pd.DataFrame) -> Dict:
        """Get Stochastic oscillator conditions."""
        try:
            latest = df.iloc[-1]
            k = latest.get('stoch_k', 50)
            d = latest.get('stoch_d', 50)
            prev_k = df.iloc[-2].get('stoch_k', 50) if len(df) > 1 else k
            prev_d = df.iloc[-2].get('stoch_d', 50) if len(df) > 1 else d
            return {
                'k': k, 'd': d,
                'overbought': k > 80, 'oversold': k < 20,
                'bull_cross': k > d and prev_k <= prev_d,
                'bear_cross': k < d and prev_k >= prev_d,
            }
        except Exception:
            return {'k': 50, 'd': 50, 'overbought': False, 'oversold': False,
                    'bull_cross': False, 'bear_cross': False}

    def _get_sentiment_filter(self, symbol: str, side: str) -> Dict:
        """Check sentiment alignment with proposed signal."""
        sentiment = self.current_sentiment.get(symbol, {})
        score = sentiment.get('sentiment_score', 0.0)
        conf = sentiment.get('confidence', 0.0)
        if conf < 0.3:
            return {'allowed': True, 'score': score, 'aligned': True}
        threshold = self.parameters['sentiment_block_threshold']
        if side == 'buy' and score < threshold:
            return {'allowed': False, 'score': score, 'aligned': False}
        return {'allowed': True, 'score': score,
                'aligned': score > 0 if side == 'buy' else score < 0}

    # =========================================================================
    # MARKET STATE ANALYSIS
    # =========================================================================

    def analyze_market_state(self, df: pd.DataFrame, symbol: str) -> MarketState:
        """Analyze market state with multi-indicator context."""
        if df.empty or len(df) < self.MIN_DATA_POINTS:
            return MarketState.UNKNOWN
        try:
            latest = df.iloc[-1]
            ema50 = latest.get('ema50', 0)
            ema200 = latest.get('ema200', 0)
            ema50_prev = df.iloc[-11]['ema50'] if len(df) > 10 else ema50
            atr = latest.get('atr', 0)
            macd_line = latest.get('macd_line', 0)
            macd_signal_val = latest.get('macd_signal', 0)
            close = latest.get('close', 0)

            if self._detect_volatility_spike(df):
                return MarketState.VOLATILE

            ema50_change = abs(ema50 - ema50_prev)
            bb = self._get_bb(df)
            if ema50_change < (atr * self.parameters['ranging_threshold']) and not bb['squeeze']:
                return MarketState.RANGING

            ema200_dist = abs(close - ema200)
            if (ema200_dist > atr * self.parameters['breakout_threshold'] or bb['squeeze']):
                if self._has_volume_confirmation(df, self.parameters['breakout_volume_mult']):
                    return MarketState.BREAKOUT

            ema8 = latest.get('ema8', 0)
            ema34 = latest.get('ema34', 0)
            ema8_prev = df.iloc[-2].get('ema8', ema8) if len(df) > 1 else ema8
            ema34_prev = df.iloc[-2].get('ema34', ema34) if len(df) > 1 else ema34
            rsi = self._get_rsi(df)
            bull_cross = ema8 > ema34 and ema8_prev <= ema34_prev
            bear_cross = ema8 < ema34 and ema8_prev >= ema34_prev
            if (bull_cross and rsi['oversold']) or (bear_cross and rsi['overbought']):
                return MarketState.REVERSING

            if abs(ema50 - ema200) > atr and (
                (ema50 > ema200 and macd_line > macd_signal_val) or
                (ema50 < ema200 and macd_line < macd_signal_val)
            ):
                return MarketState.TRENDING

            return MarketState.UNKNOWN
        except Exception as e:
            self.logger.error(f"Error analyzing market state for {symbol}: {e}")
            return MarketState.UNKNOWN

    def _detect_volatility_spike(self, df: pd.DataFrame) -> bool:
        """Detect abnormally high volatility."""
        if len(df) < 20:
            return False
        try:
            current_atr = df.iloc[-1]['atr']
            avg_atr = df['atr'].tail(20).mean()
            return avg_atr > 0 and current_atr > avg_atr * self.parameters['volatility_spike_threshold']
        except Exception:
            return False

    def select_trading_mode(self, market_state: MarketState) -> TradingMode:
        """Select optimal trading mode."""
        mapping = {
            MarketState.RANGING: TradingMode.EVE,
            MarketState.REVERSING: TradingMode.REAPER,
            MarketState.TRENDING: TradingMode.SENTINEL,
            MarketState.BREAKOUT: TradingMode.LUCIFER,
            MarketState.VOLATILE: TradingMode.SENTINEL,
            MarketState.UNKNOWN: TradingMode.SENTINEL,
        }
        selected = mapping.get(market_state, TradingMode.SENTINEL)
        if selected != self.current_mode:
            self.logger.info(f"Mode change: {self.current_mode.value} -> {selected.value}")
            self.mode_history.append({
                'timestamp': datetime.now(timezone.utc),
                'from_mode': self.current_mode.value,
                'to_mode': selected.value,
                'market_state': market_state.value,
            })
            if len(self.mode_history) > 100:
                self.mode_history = self.mode_history[-100:]
        return selected

    # =========================================================================
    # SIGNAL GENERATION — 4 ENHANCED MODES
    # =========================================================================

    def generate_trading_signals(self, df: pd.DataFrame, symbol: str,
                                  mode: TradingMode) -> List[Dict]:
        """Generate signals with multi-indicator confluence."""
        signals = []
        try:
            if mode == TradingMode.EVE:
                signals = self._gen_eve(df, symbol)
            elif mode == TradingMode.LUCIFER:
                signals = self._gen_lucifer(df, symbol)
            elif mode == TradingMode.REAPER:
                signals = self._gen_reaper(df, symbol)
            elif mode == TradingMode.SENTINEL:
                signals = self._gen_sentinel(df, symbol)

            signals = self._apply_risk_filters(signals, df, symbol)
            signals = self._apply_sentiment_filter(signals, symbol)
            signals = self._filter_position_conflicts(signals, symbol)
            return signals
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
            return []

    # --- SENTINEL: Trend Following with RSI + Volume + EMA alignment ---------

    def _gen_sentinel(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        SENTINEL confluence: EMA trend + MACD cross + RSI momentum + volume.
        Requires 4 of 5 confirmations for high-conviction entry.
        """
        if len(df) < 2:
            return []
        signals = []
        try:
            L = df.iloc[-1]
            P = df.iloc[-2]
            price = L['close']
            atr = L.get('atr', 0)
            ema50, ema200 = L.get('ema50', 0), L.get('ema200', 0)
            macd, macd_s = L.get('macd_line', 0), L.get('macd_signal', 0)
            macd_p, macd_sp = P.get('macd_line', 0), P.get('macd_signal', 0)
            rsi = self._get_rsi(df)
            vol = self._has_volume_confirmation(df)

            # BULLISH
            if (ema50 > ema200 and macd > macd_s and macd_p <= macd_sp
                    and rsi['bullish'] and vol):
                conf = sum([ema50 > ema200, macd > macd_s, rsi['bullish'],
                           vol, price > ema50]) / 5
                signals.append(self._build_signal(
                    symbol, 'buy', price, atr, 'SENTINEL',
                    'bullish_trend_confluence', conf))

            # BEARISH
            if (ema50 < ema200 and macd < macd_s and macd_p >= macd_sp
                    and rsi['bearish'] and vol):
                conf = sum([ema50 < ema200, macd < macd_s, rsi['bearish'],
                           vol, price < ema50]) / 5
                signals.append(self._build_signal(
                    symbol, 'sell', price, atr, 'SENTINEL',
                    'bearish_trend_confluence', conf))
            return signals
        except Exception as e:
            self.logger.error(f"Error generating SENTINEL signals: {e}")
            return []

    # --- LUCIFER: Breakout with squeeze + volume surge + 2-bar hold ----------

    def _gen_lucifer(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        LUCIFER confluence: EMA200 breakout + volume surge + BB squeeze + 2-bar hold.
        """
        if len(df) < 3:
            return []
        signals = []
        try:
            L, P, P2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
            price = L['close']
            atr = L.get('atr', 0)
            ema200 = L.get('ema200', 0)
            rsi = self._get_rsi(df)
            bb = self._get_bb(df)
            surge = self._has_volume_confirmation(df, self.parameters['breakout_volume_mult'])

            # BULLISH: above EMA200 for 2 bars, just broke out, volume surge
            above_2bars = price > ema200 and P['close'] > P.get('ema200', 0)
            just_broke = P2['close'] <= P2.get('ema200', ema200)
            if above_2bars and just_broke and surge and rsi['rising']:
                conf = sum([above_2bars, surge, bb['squeeze'] or bb['above_upper'],
                           rsi['bullish'], rsi['rising']]) / 5
                signals.append(self._build_signal(
                    symbol, 'buy', price, atr, 'LUCIFER',
                    'bullish_breakout_confluence', conf))

            # BEARISH
            below_2bars = price < ema200 and P['close'] < P.get('ema200', 0)
            just_broke_d = P2['close'] >= P2.get('ema200', ema200)
            if below_2bars and just_broke_d and surge and rsi['falling']:
                conf = sum([below_2bars, surge, bb['squeeze'] or bb['below_lower'],
                           rsi['bearish'], rsi['falling']]) / 5
                signals.append(self._build_signal(
                    symbol, 'sell', price, atr, 'LUCIFER',
                    'bearish_breakout_confluence', conf))
            return signals
        except Exception as e:
            self.logger.error(f"Error generating LUCIFER signals: {e}")
            return []

    # --- REAPER: Reversal with RSI/Stoch + divergence + volume ---------------

    def _gen_reaper(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        REAPER confluence: EMA cross + RSI extreme + Stochastic cross + volume.
        Requires 3+ of 5 confirmations (confidence >= 0.6).
        """
        if len(df) < 5:
            return []
        signals = []
        try:
            L, P = df.iloc[-1], df.iloc[-2]
            price = L['close']
            atr = L.get('atr', 0)
            ema8, ema34 = L.get('ema8', 0), L.get('ema34', 0)
            ema8p, ema34p = P.get('ema8', 0), P.get('ema34', 0)
            rsi = self._get_rsi(df)
            stoch = self._get_stoch(df)
            vol = self._has_volume_confirmation(df)

            # Was RSI extreme in last 5 bars?
            was_oversold = any(
                df.iloc[i].get('rsi', 50) < self.parameters['rsi_oversold']
                for i in range(-min(5, len(df)), -1)
            )
            was_overbought = any(
                df.iloc[i].get('rsi', 50) > self.parameters['rsi_overbought']
                for i in range(-min(5, len(df)), -1)
            )

            # BULLISH reversal
            if ema8 > ema34 and ema8p <= ema34p:
                checks = [True, was_oversold or rsi['oversold'],
                          stoch['bull_cross'] or stoch['oversold'], vol,
                          not self._detect_volume_divergence(df, 'bearish')]
                conf = sum(checks) / len(checks)
                if conf >= 0.6:
                    signals.append(self._build_signal(
                        symbol, 'buy', price, atr, 'REAPER',
                        'bullish_reversal_confluence', conf))

            # BEARISH reversal
            if ema8 < ema34 and ema8p >= ema34p:
                checks = [True, was_overbought or rsi['overbought'],
                          stoch['bear_cross'] or stoch['overbought'], vol,
                          not self._detect_volume_divergence(df, 'bullish')]
                conf = sum(checks) / len(checks)
                if conf >= 0.6:
                    signals.append(self._build_signal(
                        symbol, 'sell', price, atr, 'REAPER',
                        'bearish_reversal_confluence', conf))
            return signals
        except Exception as e:
            self.logger.error(f"Error generating REAPER signals: {e}")
            return []

    # --- EVE: Dynamic Grid with ATR spacing + BB boundaries ------------------

    def _gen_eve(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """EVE grid with ATR-based spacing and BB-bounded range. Auto-resets on breakout."""
        if df.empty:
            return []
        signals = []
        try:
            latest = df.iloc[-1]
            price = latest['close']
            atr = latest.get('atr', 0)
            bb = self._get_bb(df)

            # Auto-reset grid on price deviation > 5%
            if symbol in self.grid_levels:
                base = self.grid_base_prices.get(symbol, price)
                if base > 0 and abs(price - base) / base > 0.05:
                    self.logger.info(f"Grid reset for {symbol}: price deviated from base")
                    del self.grid_levels[symbol]

            # Initialize dynamic grid
            grids = self.grid_levels.get(symbol, [])
            if not grids or symbol not in self.grid_base_prices:
                self._init_dynamic_grid(price, atr, bb, symbol)
                grids = self.grid_levels.get(symbol, [])

            for level in grids:
                if level['filled']:
                    continue
                if level['side'] == 'buy' and price <= level['price']:
                    signals.append({
                        'symbol': symbol, 'side': 'buy', 'type': 'limit',
                        'price': level['price'],
                        'quantity': self.parameters['grid_position_size'],
                        'mode': 'EVE', 'level_id': level['id'],
                        'confidence': 0.7, 'timestamp': datetime.now(timezone.utc),
                    })
                    level['filled'] = True
                elif level['side'] == 'sell' and price >= level['price']:
                    signals.append({
                        'symbol': symbol, 'side': 'sell', 'type': 'limit',
                        'price': level['price'],
                        'quantity': self.parameters['grid_position_size'],
                        'mode': 'EVE', 'level_id': level['id'],
                        'confidence': 0.7, 'timestamp': datetime.now(timezone.utc),
                    })
                    level['filled'] = True
            return signals
        except Exception as e:
            self.logger.error(f"Error generating EVE signals: {e}")
            return []

    def _init_dynamic_grid(self, base_price: float, atr: float,
                            bb: Dict, symbol: str) -> None:
        """Initialize grid with ATR-based spacing bounded by Bollinger Bands."""
        self.grid_base_prices[symbol] = base_price
        levels = []
        atr_spacing = (atr / base_price) * 0.5 if base_price > 0 and atr > 0 else self.parameters['grid_spacing']
        spacing = max(atr_spacing, self.parameters['grid_spacing'])
        upper = bb.get('upper', base_price * 1.05)
        lower = bb.get('lower', base_price * 0.95)

        for i in range(1, self.parameters['grid_levels'] + 1):
            sp = base_price * (1 + spacing * i)
            if sp <= upper * 1.02:
                levels.append({'id': f"sell_{i}", 'price': sp, 'side': 'sell', 'filled': False})
            bp = base_price * (1 - spacing * i)
            if bp >= lower * 0.98:
                levels.append({'id': f"buy_{i}", 'price': bp, 'side': 'buy', 'filled': False})

        self.grid_levels[symbol] = levels
        self.logger.info(f"Dynamic grid for {symbol}: {len(levels)} levels, spacing={spacing:.4f}")

    # Legacy compat
    def _initialize_grid(self, base_price: float, atr: float, symbol: str) -> None:
        self._init_dynamic_grid(base_price, atr, self._get_bb(pd.DataFrame()), symbol)

    # =========================================================================
    # SIGNAL BUILDER
    # =========================================================================

    def _build_signal(self, symbol: str, side: str, price: float, atr: float,
                      mode: str, signal_type: str, confidence: float) -> Dict:
        """Build a standardized signal dict with SL/TP from ATR."""
        sl_mult = self.parameters['atr_mult_sl']
        tp_mult = self.parameters['atr_mult_tp']
        if side == 'buy':
            sl = price - atr * sl_mult
            tp = price + atr * tp_mult
        else:
            sl = price + atr * sl_mult
            tp = price - atr * tp_mult
        return {
            'symbol': symbol, 'side': side, 'type': 'market',
            'quantity': self.parameters['max_position_size'],
            'stop_loss': sl, 'take_profit': tp,
            'mode': mode, 'signal_type': signal_type,
            'confidence': confidence,
            'timestamp': datetime.now(timezone.utc),
        }

    # =========================================================================
    # FILTERS
    # =========================================================================

    def _apply_risk_filters(self, signals: List[Dict], df: pd.DataFrame,
                            symbol: str) -> List[Dict]:
        """Apply risk management and confidence filters."""
        filtered = []
        for signal in signals:
            if self._detect_volatility_spike(df):
                self.logger.warning(f"Signal blocked: high volatility for {symbol}")
                continue
            if signal.get('confidence', 0.5) < self.parameters['min_signal_confidence']:
                continue
            if signal.get('quantity', 0) > self.parameters['max_position_size']:
                signal['quantity'] = self.parameters['max_position_size']
            # Trailing stop
            cp = signal.get('price', df.iloc[-1].get('close', 0))
            tp = self.parameters['trail_trigger']
            if signal['side'] == 'buy' and cp > 0:
                signal['trail_trigger'] = cp * (1 + tp)
                signal['trail_offset'] = self.parameters['trail_offset']
            elif signal['side'] == 'sell' and cp > 0:
                signal['trail_trigger'] = cp * (1 - tp)
                signal['trail_offset'] = self.parameters['trail_offset']
            filtered.append(signal)
        return filtered

    def _apply_sentiment_filter(self, signals: List[Dict], symbol: str) -> List[Dict]:
        """Block signals that contradict strong sentiment."""
        if not self.current_sentiment:
            return signals
        filtered = []
        for signal in signals:
            sf = self._get_sentiment_filter(symbol, signal['side'])
            if not sf['allowed']:
                self.logger.info(f"Sentiment blocked {symbol} {signal['side']}: score={sf['score']:.2f}")
                continue
            if sf['aligned'] and abs(sf['score']) > self.parameters['sentiment_boost_threshold']:
                signal['confidence'] = min(signal.get('confidence', 0.5) + 0.1, 1.0)
                signal['sentiment_aligned'] = True
            filtered.append(signal)
        return filtered

    def _filter_position_conflicts(self, signals: List[Dict], symbol: str) -> List[Dict]:
        """Filter signals conflicting with existing positions."""
        pos = self.current_positions.get(symbol, {})
        qty = pos.get('quantity', 0)
        return [s for s in signals
                if not (s['side'] == 'buy' and qty > 0)
                and not (s['side'] == 'sell' and qty < 0)]

    # =========================================================================
    # ORCHESTRATION
    # =========================================================================

    def update_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Update strategy based on new market data."""
        try:
            state = self.analyze_market_state(df, symbol)
            mode = self.select_trading_mode(state)
            signals = self.generate_trading_signals(df, symbol, mode)

            self.current_mode = mode
            self.current_market_state = state
            self.last_analysis_time = datetime.now(timezone.utc)

            for s in signals:
                self.signal_history.append({
                    'symbol': symbol, 'signal': s,
                    'timestamp': datetime.now(timezone.utc),
                })

            return {
                'symbol': symbol, 'market_state': state.value,
                'trading_mode': mode.value, 'signals': signals,
                'timestamp': self.last_analysis_time,
                'grid_levels_count': len(self.grid_levels.get(symbol, [])),
            }
        except Exception as e:
            self.logger.error(f"Error updating strategy for {symbol}: {e}")
            return {
                'symbol': symbol, 'market_state': MarketState.UNKNOWN.value,
                'trading_mode': self.current_mode.value, 'signals': [],
                'timestamp': datetime.now(timezone.utc), 'error': str(e),
            }

    def get_strategy_status(self) -> Dict:
        """Get current strategy status."""
        return {
            'current_mode': self.current_mode.value,
            'current_market_state': self.current_market_state.value,
            'last_analysis_time': self.last_analysis_time,
            'grid_levels_count': sum(len(v) for v in self.grid_levels.values()),
            'grid_base_prices': self.grid_base_prices.copy(),
            'mode_history': self.mode_history[-10:],
            'sentiment_active': bool(self.current_sentiment),
            'parameters': self.parameters.copy(),
        }
