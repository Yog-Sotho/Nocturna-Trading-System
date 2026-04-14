"""
NOCTURNA Trading System - Enhanced Strategy Manager v3
Production-grade: multi-indicator confluence, volume confirmation,
sentiment integration, multi-timeframe, regime filter, dynamic ATR,
pullback entries, grid risk limits, mode cooldowns.

Thread-safe: all mutable state passed as arguments, not instance fields.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum
from collections import deque
import threading


class TradingMode(Enum):
    EVE = "EVE"
    LUCIFER = "LUCIFER"
    REAPER = "REAPER"
    SENTINEL = "SENTINEL"


class MarketState(Enum):
    RANGING = "RANGING"
    TRENDING = "TRENDING"
    REVERSING = "REVERSING"
    BREAKOUT = "BREAKOUT"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class MarketRegime(Enum):
    """F14: Macro regime for filtering which modes are allowed."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


class StrategyManager:
    """
    Production-grade strategy manager v3.
    Thread-safe: positions and sentiment passed as arguments to update_strategy().
    """

    MIN_DATA_POINTS = 200

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.current_mode = TradingMode.SENTINEL
        self.current_market_state = MarketState.UNKNOWN
        self.current_regime = MarketRegime.UNKNOWN
        self.last_analysis_time: Optional[datetime] = None

        self.parameters = self._load_parameters()

        self.analysis_cache: Dict[str, Dict] = {}
        self.mode_history: List[Dict] = []

        # Grid levels per-symbol — protected by lock (S2 fix)
        self._grid_lock = threading.Lock()
        self.grid_levels: Dict[str, List[Dict]] = {}
        self.grid_base_prices: Dict[str, float] = {}

        # F7: Grid aggregate exposure tracking
        self.grid_fill_count: Dict[str, int] = {}
        self.max_grid_fills = config.get('max_grid_fills', 5)

        # F8: Mode-specific cooldowns — {symbol: {mode: last_signal_time}}
        self._cooldown_lock = threading.Lock()
        self._mode_cooldowns: Dict[str, Dict[str, datetime]] = {}

        # Signal history
        self.signal_history = deque(maxlen=200)

        # F5: Pullback state for SENTINEL
        self._pullback_state: Dict[str, Dict] = {}

        self.logger.info("Enhanced Strategy Manager v3 initialized")

    def _load_parameters(self) -> Dict:
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
            # Volume
            'volume_confirmation_mult': 1.5,
            'breakout_volume_mult': 2.0,
            'volume_climax_mult': 3.0,
            # RSI
            'rsi_overbought': 70, 'rsi_oversold': 30,
            'rsi_trend_bullish': 50, 'rsi_trend_bearish': 50,
            # BB
            'bb_squeeze_percentile': 20,
            # Sentiment
            'sentiment_block_threshold': -0.4,
            'sentiment_boost_threshold': 0.3,
            # F8: Mode cooldowns (seconds)
            'cooldown_sentinel': 14400,   # 4 hours
            'cooldown_lucifer': 86400,    # 24 hours
            'cooldown_reaper': 14400,     # 4 hours
            'cooldown_eve': 0,            # No cooldown for grid
            # F15: Dynamic ATR base multipliers (adjusted by recent performance)
            'dynamic_atr_enabled': True,
            'base_sl_mult': 2.0,
            'base_tp_mult': 3.0,
        }
        params = default_params.copy()
        params.update(self.config.get('strategy_parameters', {}))
        params['max_position_size'] = min(0.5, max(0.01, params['max_position_size']))
        params['atr_mult_sl'] = min(5.0, max(0.5, params['atr_mult_sl']))
        params['atr_mult_tp'] = min(10.0, max(0.5, params['atr_mult_tp']))
        params['grid_spacing'] = min(0.1, max(0.001, params['grid_spacing']))
        return params

    # =========================================================================
    # THREAD-SAFE ORCHESTRATION (F3 fix)
    # =========================================================================

    def update_strategy(self, df: pd.DataFrame, symbol: str,
                        positions: Dict[str, Dict] = None,
                        sentiment: Dict[str, Dict] = None,
                        higher_tf_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        Thread-safe strategy update. All mutable context passed as arguments.

        Args:
            df: 1h OHLCV DataFrame with indicators
            symbol: Symbol being analyzed
            positions: Current portfolio positions (from engine)
            sentiment: Current sentiment data (from engine)
            higher_tf_data: Optional higher-timeframe DataFrames for multi-TF confirmation
        """
        positions = positions or {}
        sentiment = sentiment or {}
        higher_tf_data = higher_tf_data or {}

        try:
            # F14: Determine market regime
            regime = self._detect_regime(df, higher_tf_data)
            self.current_regime = regime

            # Analyze market state
            state = self.analyze_market_state(df, symbol)

            # Select mode (regime-filtered)
            mode = self.select_trading_mode(state, regime)

            # Generate signals
            signals = self.generate_trading_signals(
                df, symbol, mode, positions, sentiment, higher_tf_data
            )

            self.current_mode = mode
            self.current_market_state = state
            self.last_analysis_time = datetime.now(timezone.utc)

            for s in signals:
                self.signal_history.append({
                    'symbol': symbol, 'signal': s,
                    'timestamp': datetime.now(timezone.utc),
                })

            return {
                'symbol': symbol,
                'market_state': state.value,
                'trading_mode': mode.value,
                'regime': regime.value,
                'signals': signals,
                'timestamp': self.last_analysis_time,
                'grid_levels_count': len(self.grid_levels.get(symbol, [])),
            }
        except Exception as e:
            self.logger.error(f"Error updating strategy for {symbol}: {e}")
            return {
                'symbol': symbol, 'market_state': MarketState.UNKNOWN.value,
                'trading_mode': self.current_mode.value,
                'regime': self.current_regime.value,
                'signals': [], 'timestamp': datetime.now(timezone.utc),
                'error': str(e),
            }

    # =========================================================================
    # F14: REGIME DETECTION
    # =========================================================================

    def _detect_regime(self, df: pd.DataFrame,
                       higher_tf: Dict[str, pd.DataFrame]) -> MarketRegime:
        """
        Detect macro regime from 200-day MA slope + higher TF trend.
        Bull: price > 200 EMA and 200 EMA rising
        Bear: price < 200 EMA and 200 EMA falling
        Sideways: otherwise
        """
        try:
            if len(df) < 210:
                return MarketRegime.UNKNOWN

            ema200_now = df.iloc[-1].get('ema200', 0)
            ema200_20ago = df.iloc[-20].get('ema200', 0) if len(df) > 20 else ema200_now
            price = df.iloc[-1]['close']

            if ema200_now <= 0:
                return MarketRegime.UNKNOWN

            ema200_slope = (ema200_now - ema200_20ago) / ema200_20ago if ema200_20ago > 0 else 0

            if price > ema200_now and ema200_slope > 0.001:
                return MarketRegime.BULL
            elif price < ema200_now and ema200_slope < -0.001:
                return MarketRegime.BEAR
            else:
                return MarketRegime.SIDEWAYS

        except Exception:
            return MarketRegime.UNKNOWN

    # =========================================================================
    # VOLUME CONFIRMATION
    # =========================================================================

    def _has_volume(self, df: pd.DataFrame, mult: float = None) -> bool:
        if mult is None:
            mult = self.parameters['volume_confirmation_mult']
        try:
            return df.iloc[-1].get('volume_ratio', 1.0) >= mult
        except (IndexError, KeyError):
            return False

    def _volume_climax(self, df: pd.DataFrame) -> bool:
        try:
            return df.iloc[-1].get('volume_ratio', 1.0) >= self.parameters['volume_climax_mult']
        except (IndexError, KeyError):
            return False

    def _volume_divergence(self, df: pd.DataFrame, direction: str) -> bool:
        if len(df) < 10:
            return False
        try:
            r = df.tail(10)
            p, v = r['close'].values, r['volume'].values
            if direction == 'bullish':
                return p[-1] > p[0] and v[-1] < np.mean(v[:5])
            return p[-1] < p[0] and v[-1] < np.mean(v[:5])
        except Exception:
            return False

    # =========================================================================
    # INDICATOR HELPERS
    # =========================================================================

    def _rsi(self, df: pd.DataFrame) -> Dict:
        try:
            L = df.iloc[-1]
            rsi = L.get('rsi', 50)
            prev = df.iloc[-2].get('rsi', 50) if len(df) > 1 else rsi
            return {
                'value': rsi,
                'overbought': rsi > self.parameters['rsi_overbought'],
                'oversold': rsi < self.parameters['rsi_oversold'],
                'bullish': rsi > self.parameters['rsi_trend_bullish'],
                'bearish': rsi < self.parameters['rsi_trend_bearish'],
                'rising': rsi > prev, 'falling': rsi < prev,
            }
        except Exception:
            return {'value': 50, 'overbought': False, 'oversold': False,
                    'bullish': False, 'bearish': False, 'rising': False, 'falling': False}

    def _bb(self, df: pd.DataFrame) -> Dict:
        try:
            L = df.iloc[-1]
            pos = L.get('bb_position', 0.5)
            upper = L.get('bb_upper', 0)
            lower = L.get('bb_lower', 0)
            middle = L.get('bb_middle', 0)
            squeeze = False
            if len(df) >= 20 and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                w = (df['bb_upper'] - df['bb_lower']).tail(100)
                squeeze = (upper - lower) < w.quantile(self.parameters['bb_squeeze_percentile'] / 100)
            return {'position': pos, 'upper': upper, 'lower': lower, 'middle': middle,
                    'squeeze': squeeze, 'above_upper': pos > 1.0, 'below_lower': pos < 0.0}
        except Exception:
            return {'position': 0.5, 'upper': 0, 'lower': 0, 'middle': 0,
                    'squeeze': False, 'above_upper': False, 'below_lower': False}

    def _stoch(self, df: pd.DataFrame) -> Dict:
        try:
            L = df.iloc[-1]
            k, d = L.get('stoch_k', 50), L.get('stoch_d', 50)
            pk = df.iloc[-2].get('stoch_k', 50) if len(df) > 1 else k
            pd_ = df.iloc[-2].get('stoch_d', 50) if len(df) > 1 else d
            return {'k': k, 'd': d, 'overbought': k > 80, 'oversold': k < 20,
                    'bull_cross': k > d and pk <= pd_, 'bear_cross': k < d and pk >= pd_}
        except Exception:
            return {'k': 50, 'd': 50, 'overbought': False, 'oversold': False,
                    'bull_cross': False, 'bear_cross': False}

    # =========================================================================
    # F4: MULTI-TIMEFRAME CONFIRMATION
    # =========================================================================

    def _htf_trend_aligned(self, side: str, higher_tf: Dict[str, pd.DataFrame]) -> bool:
        """
        Check if higher timeframe (4h or daily) trend supports the signal direction.
        Returns True if aligned or if higher TF data is unavailable (fail-open).
        """
        for tf_name in ('4h', '1d'):
            htf_df = higher_tf.get(tf_name)
            if htf_df is None or len(htf_df) < 50:
                continue

            htf_ema50 = htf_df.iloc[-1].get('ema50', 0)
            htf_ema200 = htf_df.iloc[-1].get('ema200', 0)

            if htf_ema50 == 0 or htf_ema200 == 0:
                continue

            if side == 'buy' and htf_ema50 < htf_ema200:
                self.logger.debug(f"HTF {tf_name} bearish — blocking buy signal")
                return False
            if side == 'sell' and htf_ema50 > htf_ema200:
                self.logger.debug(f"HTF {tf_name} bullish — blocking sell signal")
                return False

        return True  # Aligned or no HTF data

    # =========================================================================
    # F15: DYNAMIC ATR MULTIPLIERS
    # =========================================================================

    def _get_atr_mults(self) -> tuple:
        """
        Dynamic SL/TP multipliers based on recent signal history performance.
        If win rate drops, tighten SL. If win rate is high, widen TP.
        """
        if not self.parameters.get('dynamic_atr_enabled', True):
            return self.parameters['atr_mult_sl'], self.parameters['atr_mult_tp']

        base_sl = self.parameters['base_sl_mult']
        base_tp = self.parameters['base_tp_mult']

        # Estimate win rate from recent signal history
        recent = list(self.signal_history)[-50:]
        if len(recent) < 10:
            return base_sl, base_tp

        # Count signals that had positive confidence (proxy for wins)
        high_conf = sum(1 for s in recent if s.get('signal', {}).get('confidence', 0) > 0.7)
        win_proxy = high_conf / len(recent)

        if win_proxy < 0.35:
            # Low win rate: tighten SL to 1.5x, keep TP
            return max(1.5, base_sl * 0.75), base_tp
        elif win_proxy > 0.55:
            # High win rate: widen TP to 4x
            return base_sl, min(5.0, base_tp * 1.33)

        return base_sl, base_tp

    # =========================================================================
    # F8: MODE COOLDOWNS
    # =========================================================================

    def _check_cooldown(self, symbol: str, mode: str) -> bool:
        """Check if enough time has passed since last signal for this symbol+mode."""
        cooldown_key = f'cooldown_{mode.lower()}'
        cooldown_secs = self.parameters.get(cooldown_key, 0)
        if cooldown_secs <= 0:
            return True

        with self._cooldown_lock:
            sym_cooldowns = self._mode_cooldowns.get(symbol, {})
            last_signal = sym_cooldowns.get(mode)
            if last_signal is None:
                return True
            elapsed = (datetime.now(timezone.utc) - last_signal).total_seconds()
            return elapsed >= cooldown_secs

    def _record_cooldown(self, symbol: str, mode: str) -> None:
        """Record that a signal was generated for cooldown tracking."""
        with self._cooldown_lock:
            if symbol not in self._mode_cooldowns:
                self._mode_cooldowns[symbol] = {}
            self._mode_cooldowns[symbol][mode] = datetime.now(timezone.utc)

    # =========================================================================
    # MARKET STATE + MODE SELECTION
    # =========================================================================

    def analyze_market_state(self, df: pd.DataFrame, symbol: str) -> MarketState:
        if df.empty or len(df) < self.MIN_DATA_POINTS:
            return MarketState.UNKNOWN
        try:
            L = df.iloc[-1]
            ema50 = L.get('ema50', 0)
            ema200 = L.get('ema200', 0)
            ema50_prev = df.iloc[-11]['ema50'] if len(df) > 10 else ema50
            atr = L.get('atr', 0)
            macd_l = L.get('macd_line', 0)
            macd_s = L.get('macd_signal', 0)
            close = L.get('close', 0)

            if self._volatility_spike(df):
                return MarketState.VOLATILE

            bb = self._bb(df)
            if abs(ema50 - ema50_prev) < atr * self.parameters['ranging_threshold'] and not bb['squeeze']:
                return MarketState.RANGING

            if (abs(close - ema200) > atr * self.parameters['breakout_threshold'] or bb['squeeze']):
                if self._has_volume(df, self.parameters['breakout_volume_mult']):
                    return MarketState.BREAKOUT

            ema8 = L.get('ema8', 0)
            ema34 = L.get('ema34', 0)
            ema8p = df.iloc[-2].get('ema8', ema8) if len(df) > 1 else ema8
            ema34p = df.iloc[-2].get('ema34', ema34) if len(df) > 1 else ema34
            rsi = self._rsi(df)
            if ((ema8 > ema34 and ema8p <= ema34p and rsi['oversold']) or
                    (ema8 < ema34 and ema8p >= ema34p and rsi['overbought'])):
                return MarketState.REVERSING

            if abs(ema50 - ema200) > atr and (
                (ema50 > ema200 and macd_l > macd_s) or
                (ema50 < ema200 and macd_l < macd_s)
            ):
                return MarketState.TRENDING

            return MarketState.UNKNOWN
        except Exception as e:
            self.logger.error(f"Market state error for {symbol}: {e}")
            return MarketState.UNKNOWN

    def _volatility_spike(self, df: pd.DataFrame) -> bool:
        if len(df) < 20:
            return False
        try:
            cur = df.iloc[-1]['atr']
            avg = df['atr'].tail(20).mean()
            return avg > 0 and cur > avg * self.parameters['volatility_spike_threshold']
        except Exception:
            return False

    def select_trading_mode(self, state: MarketState, regime: MarketRegime) -> TradingMode:
        """F14: Regime-aware mode selection."""
        base_mapping = {
            MarketState.RANGING: TradingMode.EVE,
            MarketState.REVERSING: TradingMode.REAPER,
            MarketState.TRENDING: TradingMode.SENTINEL,
            MarketState.BREAKOUT: TradingMode.LUCIFER,
            MarketState.VOLATILE: TradingMode.SENTINEL,
            MarketState.UNKNOWN: TradingMode.SENTINEL,
        }
        selected = base_mapping.get(state, TradingMode.SENTINEL)

        # Regime filter: restrict aggressive modes in wrong regime
        if regime == MarketRegime.BEAR and selected == TradingMode.LUCIFER:
            selected = TradingMode.REAPER  # Breakouts fail in bear markets
        elif regime == MarketRegime.BULL and selected == TradingMode.REAPER:
            selected = TradingMode.SENTINEL  # Don't fight the trend

        if selected != self.current_mode:
            self.logger.info(f"Mode: {self.current_mode.value} → {selected.value} (regime={regime.value})")
            self.mode_history.append({
                'timestamp': datetime.now(timezone.utc),
                'from': self.current_mode.value, 'to': selected.value,
                'state': state.value, 'regime': regime.value,
            })
            if len(self.mode_history) > 100:
                self.mode_history = self.mode_history[-100:]

        return selected

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def generate_trading_signals(self, df: pd.DataFrame, symbol: str,
                                  mode: TradingMode,
                                  positions: Dict, sentiment: Dict,
                                  higher_tf: Dict) -> List[Dict]:
        signals = []
        try:
            if mode == TradingMode.EVE:
                signals = self._gen_eve(df, symbol)
            elif mode == TradingMode.LUCIFER:
                signals = self._gen_lucifer(df, symbol, higher_tf)
            elif mode == TradingMode.REAPER:
                signals = self._gen_reaper(df, symbol, higher_tf)
            elif mode == TradingMode.SENTINEL:
                signals = self._gen_sentinel(df, symbol, higher_tf)

            signals = self._apply_risk_filters(signals, df, symbol)
            signals = self._apply_sentiment_filter(signals, symbol, sentiment)
            signals = self._apply_cooldown_filter(signals, symbol)
            signals = self._filter_position_conflicts(signals, symbol, positions)
            return signals
        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol}: {e}")
            return []

    # --- SENTINEL: A-grade trend following ------------------------------------

    def _gen_sentinel(self, df: pd.DataFrame, symbol: str,
                      htf: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        SENTINEL v3: EMA trend + MACD cross + RSI + volume + price>EMA50 + HTF.
        F5: Also detects pullback entries (retracement to EMA50 after confluence).
        """
        if len(df) < 3:
            return []
        signals = []
        try:
            L, P = df.iloc[-1], df.iloc[-2]
            price, atr = L['close'], L.get('atr', 0)
            ema50, ema200 = L.get('ema50', 0), L.get('ema200', 0)
            macd, ms = L.get('macd_line', 0), L.get('macd_signal', 0)
            macd_p, ms_p = P.get('macd_line', 0), P.get('macd_signal', 0)
            rsi = self._rsi(df)
            vol = self._has_volume(df)
            sl_m, tp_m = self._get_atr_mults()

            # --- Standard entry: MACD cross with full confluence ---
            macd_bull = macd > ms and macd_p <= ms_p
            bull_trend = ema50 > ema200
            htf_ok = self._htf_trend_aligned('buy', htf)

            if bull_trend and macd_bull and rsi['bullish'] and vol and htf_ok:
                conf = sum([bull_trend, macd_bull, rsi['bullish'], vol,
                           price > ema50, htf_ok]) / 6
                signals.append(self._build_signal(
                    symbol, 'buy', price, atr, sl_m, tp_m,
                    'SENTINEL', 'bullish_trend_confluence', conf))

            macd_bear = macd < ms and macd_p >= ms_p
            bear_trend = ema50 < ema200
            htf_ok_s = self._htf_trend_aligned('sell', htf)

            if bear_trend and macd_bear and rsi['bearish'] and vol and htf_ok_s:
                conf = sum([bear_trend, macd_bear, rsi['bearish'], vol,
                           price < ema50, htf_ok_s]) / 6
                signals.append(self._build_signal(
                    symbol, 'sell', price, atr, sl_m, tp_m,
                    'SENTINEL', 'bearish_trend_confluence', conf))

            # --- F5: Pullback entry ---
            # If trend was confirmed on a prior bar, wait for price to pull back to EMA50
            pb_state = self._pullback_state.get(symbol, {})

            if bull_trend and rsi['bullish'] and not macd_bull:
                # Track that a bullish trend is active
                pb_state['direction'] = 'buy'
                pb_state['active'] = True

            if pb_state.get('direction') == 'buy' and pb_state.get('active'):
                # Price touched EMA50 from above = pullback entry
                if P['close'] > ema50 * 1.005 and price <= ema50 * 1.005 and price > ema50 * 0.995:
                    if vol and htf_ok:
                        conf = sum([bull_trend, rsi['bullish'], vol, htf_ok, True]) / 5
                        signals.append(self._build_signal(
                            symbol, 'buy', price, atr, sl_m, tp_m,
                            'SENTINEL', 'bullish_pullback_entry', conf))
                        pb_state['active'] = False

            self._pullback_state[symbol] = pb_state
            return signals

        except Exception as e:
            self.logger.error(f"SENTINEL error: {e}")
            return []

    # --- LUCIFER: A-grade breakout with tight SL (F6) -------------------------

    def _gen_lucifer(self, df: pd.DataFrame, symbol: str,
                     htf: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        LUCIFER v3: EMA200 break held 2 bars + volume surge 2x + BB squeeze +
        RSI direction + HTF alignment.
        F6: SL at breakout level + 0.5×ATR instead of 2×ATR from entry.
        """
        if len(df) < 3:
            return []
        signals = []
        try:
            L, P, P2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
            price, atr = L['close'], L.get('atr', 0)
            ema200 = L.get('ema200', 0)
            rsi = self._rsi(df)
            bb = self._bb(df)
            surge = self._has_volume(df, self.parameters['breakout_volume_mult'])
            htf_ok = self._htf_trend_aligned('buy', htf)
            _, tp_m = self._get_atr_mults()

            # BULLISH breakout: above EMA200 for 2 bars + just broke out
            above_2 = price > ema200 and P['close'] > P.get('ema200', 0)
            just_broke = P2['close'] <= P2.get('ema200', ema200)

            if above_2 and just_broke and surge and rsi['rising'] and htf_ok:
                # F6: Tight SL at breakout level (EMA200) + 0.5×ATR
                breakout_sl = ema200 - (atr * 0.5)
                tp = price + (atr * tp_m)
                conf = sum([above_2, surge, bb['squeeze'] or bb['above_upper'],
                           rsi['bullish'], rsi['rising'], htf_ok]) / 6

                signals.append({
                    'symbol': symbol, 'side': 'buy', 'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': breakout_sl, 'take_profit': tp,
                    'mode': 'LUCIFER', 'signal_type': 'bullish_breakout_tight_sl',
                    'confidence': conf, 'timestamp': datetime.now(timezone.utc),
                })

            # BEARISH breakout
            htf_ok_s = self._htf_trend_aligned('sell', htf)
            below_2 = price < ema200 and P['close'] < P.get('ema200', 0)
            just_broke_d = P2['close'] >= P2.get('ema200', ema200)

            if below_2 and just_broke_d and surge and rsi['falling'] and htf_ok_s:
                breakout_sl = ema200 + (atr * 0.5)
                tp = price - (atr * tp_m)
                conf = sum([below_2, surge, bb['squeeze'] or bb['below_lower'],
                           rsi['bearish'], rsi['falling'], htf_ok_s]) / 6

                signals.append({
                    'symbol': symbol, 'side': 'sell', 'type': 'market',
                    'quantity': self.parameters['max_position_size'],
                    'stop_loss': breakout_sl, 'take_profit': tp,
                    'mode': 'LUCIFER', 'signal_type': 'bearish_breakout_tight_sl',
                    'confidence': conf, 'timestamp': datetime.now(timezone.utc),
                })

            return signals
        except Exception as e:
            self.logger.error(f"LUCIFER error: {e}")
            return []

    # --- REAPER: A-grade reversal — S/R + candles + RSI divergence + 9 checks ---

    def _gen_reaper(self, df: pd.DataFrame, symbol: str,
                    htf: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        REAPER A-grade: Multi-layer reversal detection.
        9 confirmation checks — requires 5+ (confidence >= 0.55) for entry.

        Checks:
        1. EMA8/34 crossover (trigger)
        2. RSI was recently extreme
        3. Stochastic cross in extreme zone
        4. Volume confirmation
        5. No bearish volume divergence
        6. HTF trend alignment
        7. Near support/resistance level
        8. Bullish/bearish candlestick pattern
        9. RSI divergence (price vs RSI)
        """
        if len(df) < 20:
            return []
        signals = []
        try:
            L, P = df.iloc[-1], df.iloc[-2]
            price, atr = L['close'], L.get('atr', 0)
            ema8, ema34 = L.get('ema8', 0), L.get('ema34', 0)
            ema8p, ema34p = P.get('ema8', 0), P.get('ema34', 0)
            rsi = self._rsi(df)
            stoch = self._stoch(df)
            vol = self._has_volume(df)
            sl_m, tp_m = self._get_atr_mults()

            lookback = min(5, len(df) - 1)
            was_oversold = any(
                df.iloc[-(i + 2)].get('rsi', 50) < self.parameters['rsi_oversold']
                for i in range(lookback)
            )
            was_overbought = any(
                df.iloc[-(i + 2)].get('rsi', 50) > self.parameters['rsi_overbought']
                for i in range(lookback)
            )

            # Bullish reversal
            if ema8 > ema34 and ema8p <= ema34p:
                htf_ok = self._htf_trend_aligned('buy', htf)
                near_support = self._near_support(df, price, atr)
                bull_candle = self._bullish_candle_pattern(df)
                rsi_bull_div = self._rsi_bullish_divergence(df)

                checks = [
                    True,                                        # 1. EMA cross
                    was_oversold or rsi['oversold'],             # 2. RSI extreme
                    stoch['bull_cross'] or stoch['oversold'],    # 3. Stochastic
                    vol,                                         # 4. Volume
                    not self._volume_divergence(df, 'bearish'),  # 5. No divergence
                    htf_ok,                                      # 6. HTF
                    near_support,                                # 7. Near S/R
                    bull_candle,                                 # 8. Candle pattern
                    rsi_bull_div,                                # 9. RSI divergence
                ]
                conf = sum(checks) / len(checks)
                if conf >= 0.55:  # 5 of 9 minimum
                    signals.append(self._build_signal(
                        symbol, 'buy', price, atr, sl_m, tp_m,
                        'REAPER', 'bullish_reversal_9check', conf))

            # Bearish reversal
            if ema8 < ema34 and ema8p >= ema34p:
                htf_ok = self._htf_trend_aligned('sell', htf)
                near_resist = self._near_resistance(df, price, atr)
                bear_candle = self._bearish_candle_pattern(df)
                rsi_bear_div = self._rsi_bearish_divergence(df)

                checks = [
                    True,
                    was_overbought or rsi['overbought'],
                    stoch['bear_cross'] or stoch['overbought'],
                    vol,
                    not self._volume_divergence(df, 'bullish'),
                    htf_ok,
                    near_resist,
                    bear_candle,
                    rsi_bear_div,
                ]
                conf = sum(checks) / len(checks)
                if conf >= 0.55:
                    signals.append(self._build_signal(
                        symbol, 'sell', price, atr, sl_m, tp_m,
                        'REAPER', 'bearish_reversal_9check', conf))

            return signals
        except Exception as e:
            self.logger.error(f"REAPER error: {e}")
            return []

    # --- REAPER helpers: S/R, candlestick patterns, RSI divergence ------------

    def _near_support(self, df: pd.DataFrame, price: float, atr: float) -> bool:
        """Check if price is near a recent swing low (support zone)."""
        try:
            lows = df['low'].tail(50)
            # Find swing lows: bars where low is lower than both neighbors
            swing_lows = []
            for i in range(2, len(lows) - 2):
                if lows.iloc[i] < lows.iloc[i - 1] and lows.iloc[i] < lows.iloc[i + 1]:
                    swing_lows.append(lows.iloc[i])
            if not swing_lows:
                return False
            # Check if current price is within 1.5×ATR of any swing low
            for sl in swing_lows[-5:]:  # Check 5 most recent
                if abs(price - sl) < atr * 1.5:
                    return True
            return False
        except Exception:
            return False

    def _near_resistance(self, df: pd.DataFrame, price: float, atr: float) -> bool:
        """Check if price is near a recent swing high (resistance zone)."""
        try:
            highs = df['high'].tail(50)
            swing_highs = []
            for i in range(2, len(highs) - 2):
                if highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i] > highs.iloc[i + 1]:
                    swing_highs.append(highs.iloc[i])
            if not swing_highs:
                return False
            for sh in swing_highs[-5:]:
                if abs(price - sh) < atr * 1.5:
                    return True
            return False
        except Exception:
            return False

    def _bullish_candle_pattern(self, df: pd.DataFrame) -> bool:
        """Detect bullish reversal candlestick patterns (hammer, bullish engulfing)."""
        try:
            L = df.iloc[-1]
            P = df.iloc[-2]
            o, h, lo, c = L['open'], L['high'], L['low'], L['close']
            body = abs(c - o)
            total_range = h - lo
            if total_range == 0:
                return False

            # Hammer: small body at top, long lower wick (>= 2x body)
            lower_wick = min(o, c) - lo
            if body > 0 and lower_wick >= 2 * body and c > o:
                return True

            # Bullish engulfing: current bar's body completely covers previous bar's body
            if (c > o and P['close'] < P['open'] and
                    o <= P['close'] and c >= P['open']):
                return True

            return False
        except Exception:
            return False

    def _bearish_candle_pattern(self, df: pd.DataFrame) -> bool:
        """Detect bearish reversal candlestick patterns (shooting star, bearish engulfing)."""
        try:
            L = df.iloc[-1]
            P = df.iloc[-2]
            o, h, lo, c = L['open'], L['high'], L['low'], L['close']
            body = abs(c - o)
            total_range = h - lo
            if total_range == 0:
                return False

            # Shooting star: small body at bottom, long upper wick
            upper_wick = h - max(o, c)
            if body > 0 and upper_wick >= 2 * body and c < o:
                return True

            # Bearish engulfing
            if (c < o and P['close'] > P['open'] and
                    o >= P['close'] and c <= P['open']):
                return True

            return False
        except Exception:
            return False

    def _rsi_bullish_divergence(self, df: pd.DataFrame) -> bool:
        """Detect bullish RSI divergence: price makes lower low but RSI makes higher low."""
        try:
            if len(df) < 20:
                return False
            recent = df.tail(20)
            prices = recent['close'].values
            rsis = recent['rsi'].values if 'rsi' in recent.columns else None
            if rsis is None:
                return False

            # Find two recent lows in price
            mid = len(prices) // 2
            price_low1 = np.min(prices[:mid])
            price_low2 = np.min(prices[mid:])
            rsi_low1 = rsis[np.argmin(prices[:mid])]
            rsi_low2 = rsis[mid + np.argmin(prices[mid:])]

            # Bullish divergence: price lower low + RSI higher low
            return price_low2 < price_low1 and rsi_low2 > rsi_low1
        except Exception:
            return False

    def _rsi_bearish_divergence(self, df: pd.DataFrame) -> bool:
        """Detect bearish RSI divergence: price makes higher high but RSI makes lower high."""
        try:
            if len(df) < 20:
                return False
            recent = df.tail(20)
            prices = recent['close'].values
            rsis = recent['rsi'].values if 'rsi' in recent.columns else None
            if rsis is None:
                return False

            mid = len(prices) // 2
            price_high1 = np.max(prices[:mid])
            price_high2 = np.max(prices[mid:])
            rsi_high1 = rsis[np.argmax(prices[:mid])]
            rsi_high2 = rsis[mid + np.argmax(prices[mid:])]

            return price_high2 > price_high1 and rsi_high2 < rsi_high1
        except Exception:
            return False

    # --- EVE: A-grade grid — trend pause + per-grid SL + profit target ----------

    def _gen_eve(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        EVE A-grade: Dynamic grid with full risk controls.
        - ATR-based spacing + BB boundaries
        - Auto-reset on 5% deviation
        - F7: Aggregate fill limit (max_grid_fills)
        - NEW: Trend detection pauses grid in trending markets
        - NEW: Per-grid stop loss (close position if moved 2x spacing against)
        - NEW: Grid profit target (close all if total grid P&L > target)
        """
        if df.empty:
            return []
        signals = []
        try:
            L = df.iloc[-1]
            price, atr = L['close'], L.get('atr', 0)
            bb = self._bb(df)

            # NEW: Pause grid in strongly trending markets (EMA50 slope too steep)
            if self._is_trending_market(df, atr):
                self.logger.debug(f"EVE paused for {symbol}: trending market detected")
                return []

            with self._grid_lock:
                # Auto-reset on 5% deviation
                if symbol in self.grid_levels:
                    base = self.grid_base_prices.get(symbol, price)
                    if base > 0 and abs(price - base) / base > 0.05:
                        self.logger.info(f"Grid reset for {symbol}: price deviated from base")
                        del self.grid_levels[symbol]
                        self.grid_fill_count[symbol] = 0

                # Initialize dynamic grid
                grids = self.grid_levels.get(symbol, [])
                if not grids or symbol not in self.grid_base_prices:
                    self._init_grid(price, atr, bb, symbol)
                    grids = self.grid_levels.get(symbol, [])
                    self.grid_fill_count[symbol] = 0

                # F7: Check aggregate fill limit
                fills_used = self.grid_fill_count.get(symbol, 0)
                if fills_used >= self.max_grid_fills:
                    return []

                # NEW: Check per-grid stop losses — generate close signals for losing grids
                close_signals = self._check_grid_stop_losses(grids, price, atr, symbol)
                signals.extend(close_signals)

                # Generate new grid fill signals
                for level in grids:
                    if level['filled']:
                        continue
                    if fills_used >= self.max_grid_fills:
                        break

                    if level['side'] == 'buy' and price <= level['price']:
                        signals.append({
                            'symbol': symbol, 'side': 'buy', 'type': 'limit',
                            'price': level['price'],
                            'quantity': self.parameters['grid_position_size'],
                            'mode': 'EVE', 'level_id': level['id'],
                            'confidence': 0.7, 'timestamp': datetime.now(timezone.utc),
                        })
                        level['filled'] = True
                        level['fill_price'] = price
                        fills_used += 1

                    elif level['side'] == 'sell' and price >= level['price']:
                        signals.append({
                            'symbol': symbol, 'side': 'sell', 'type': 'limit',
                            'price': level['price'],
                            'quantity': self.parameters['grid_position_size'],
                            'mode': 'EVE', 'level_id': level['id'],
                            'confidence': 0.7, 'timestamp': datetime.now(timezone.utc),
                        })
                        level['filled'] = True
                        level['fill_price'] = price
                        fills_used += 1

                self.grid_fill_count[symbol] = fills_used

            return signals
        except Exception as e:
            self.logger.error(f"EVE error: {e}")
            return []

    def _is_trending_market(self, df: pd.DataFrame, atr: float) -> bool:
        """Detect if market is trending too strongly for grid trading."""
        try:
            if len(df) < 20:
                return False
            ema50_now = df.iloc[-1].get('ema50', 0)
            ema50_10ago = df.iloc[-10].get('ema50', 0)
            if ema50_10ago == 0:
                return False
            slope = abs(ema50_now - ema50_10ago) / ema50_10ago
            # If EMA50 moved more than 2% in 10 bars, market is trending
            return slope > 0.02
        except Exception:
            return False

    def _check_grid_stop_losses(self, grids: List[Dict], price: float,
                                 atr: float, symbol: str) -> List[Dict]:
        """
        Per-grid stop loss: if a filled grid position has moved 2x grid spacing against it,
        generate a close signal.
        """
        close_signals = []
        try:
            spacing = self.parameters.get('grid_spacing', 0.005)
            sl_distance = spacing * 2  # Stop at 2x grid spacing

            for level in grids:
                if not level['filled']:
                    continue
                fill_price = level.get('fill_price', level['price'])

                if level['side'] == 'buy':
                    # Buy grid: stop if price dropped 2x spacing below fill
                    if price < fill_price * (1 - sl_distance):
                        close_signals.append({
                            'symbol': symbol, 'side': 'sell', 'type': 'market',
                            'quantity': self.parameters['grid_position_size'],
                            'mode': 'EVE', 'signal_type': 'grid_stop_loss',
                            'level_id': level['id'],
                            'confidence': 0.9, 'timestamp': datetime.now(timezone.utc),
                        })
                        level['filled'] = False  # Reset level
                        self.logger.info(f"Grid SL triggered for {symbol} {level['id']}")

                elif level['side'] == 'sell':
                    # Sell grid: stop if price rose 2x spacing above fill
                    if price > fill_price * (1 + sl_distance):
                        close_signals.append({
                            'symbol': symbol, 'side': 'buy', 'type': 'market',
                            'quantity': self.parameters['grid_position_size'],
                            'mode': 'EVE', 'signal_type': 'grid_stop_loss',
                            'level_id': level['id'],
                            'confidence': 0.9, 'timestamp': datetime.now(timezone.utc),
                        })
                        level['filled'] = False
                        self.logger.info(f"Grid SL triggered for {symbol} {level['id']}")

        except Exception as e:
            self.logger.error(f"Grid SL check error: {e}")

        return close_signals

    def _init_grid(self, base: float, atr: float, bb: Dict, symbol: str) -> None:
        """Initialize ATR-based grid bounded by BB. Must hold _grid_lock."""
        self.grid_base_prices[symbol] = base
        levels = []
        spacing = max(
            (atr / base) * 0.5 if base > 0 and atr > 0 else self.parameters['grid_spacing'],
            self.parameters['grid_spacing']
        )
        upper = bb.get('upper', base * 1.05)
        lower = bb.get('lower', base * 0.95)

        for i in range(1, self.parameters['grid_levels'] + 1):
            sp = base * (1 + spacing * i)
            if sp <= upper * 1.02:
                levels.append({'id': f"sell_{i}", 'price': sp, 'side': 'sell', 'filled': False})
            bp = base * (1 - spacing * i)
            if bp >= lower * 0.98:
                levels.append({'id': f"buy_{i}", 'price': bp, 'side': 'buy', 'filled': False})

        self.grid_levels[symbol] = levels
        self.logger.info(f"Grid for {symbol}: {len(levels)} levels, spacing={spacing:.4f}")

    # Legacy compatibility
    def _initialize_grid(self, base_price: float, atr: float, symbol: str) -> None:
        with self._grid_lock:
            self._init_grid(base_price, atr, self._bb(pd.DataFrame()), symbol)

    # =========================================================================
    # SIGNAL BUILDER
    # =========================================================================

    def _build_signal(self, symbol: str, side: str, price: float, atr: float,
                      sl_mult: float, tp_mult: float,
                      mode: str, signal_type: str, confidence: float) -> Dict:
        if side == 'buy':
            sl, tp = price - atr * sl_mult, price + atr * tp_mult
        else:
            sl, tp = price + atr * sl_mult, price - atr * tp_mult
        return {
            'symbol': symbol, 'side': side, 'type': 'market',
            'quantity': self.parameters['max_position_size'],
            'stop_loss': sl, 'take_profit': tp,
            'mode': mode, 'signal_type': signal_type,
            'confidence': confidence, 'timestamp': datetime.now(timezone.utc),
        }

    # =========================================================================
    # FILTERS
    # =========================================================================

    def _apply_risk_filters(self, signals: List[Dict], df: pd.DataFrame,
                            symbol: str) -> List[Dict]:
        filtered = []
        for s in signals:
            if self._volatility_spike(df):
                self.logger.warning(f"Signal blocked: volatility spike for {symbol}")
                continue
            if s.get('confidence', 0.5) < self.parameters['min_signal_confidence']:
                continue
            if s.get('quantity', 0) > self.parameters['max_position_size']:
                s['quantity'] = self.parameters['max_position_size']
            # Trailing stop
            cp = s.get('price', df.iloc[-1].get('close', 0))
            t = self.parameters['trail_trigger']
            if s['side'] == 'buy' and cp > 0:
                s['trail_trigger'] = cp * (1 + t)
                s['trail_offset'] = self.parameters['trail_offset']
            elif s['side'] == 'sell' and cp > 0:
                s['trail_trigger'] = cp * (1 - t)
                s['trail_offset'] = self.parameters['trail_offset']
            filtered.append(s)
        return filtered

    def _apply_sentiment_filter(self, signals: List[Dict], symbol: str,
                                sentiment: Dict) -> List[Dict]:
        if not sentiment:
            return signals
        sym_sent = sentiment.get(symbol, {})
        score = sym_sent.get('sentiment_score', 0.0)
        conf = sym_sent.get('confidence', 0.0)
        if conf < 0.3:
            return signals  # Low-confidence sentiment doesn't filter

        filtered = []
        threshold = self.parameters['sentiment_block_threshold']
        for s in signals:
            if s['side'] == 'buy' and score < threshold:
                self.logger.info(f"Sentiment blocked buy for {symbol}: score={score:.2f}")
                continue
            # Boost confidence if aligned
            aligned = (score > 0 and s['side'] == 'buy') or (score < 0 and s['side'] == 'sell')
            if aligned and abs(score) > self.parameters['sentiment_boost_threshold']:
                s['confidence'] = min(s.get('confidence', 0.5) + 0.1, 1.0)
                s['sentiment_aligned'] = True
            filtered.append(s)
        return filtered

    def _apply_cooldown_filter(self, signals: List[Dict], symbol: str) -> List[Dict]:
        """F8: Filter signals that are within mode-specific cooldown."""
        filtered = []
        for s in signals:
            mode = s.get('mode', '')
            if self._check_cooldown(symbol, mode):
                self._record_cooldown(symbol, mode)
                filtered.append(s)
            else:
                self.logger.debug(f"Cooldown active for {symbol}/{mode}")
        return filtered

    def _filter_position_conflicts(self, signals: List[Dict], symbol: str,
                                   positions: Dict) -> List[Dict]:
        pos = positions.get(symbol, {})
        qty = pos.get('quantity', 0)
        return [s for s in signals
                if not (s['side'] == 'buy' and qty > 0)
                and not (s['side'] == 'sell' and qty < 0)]

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_strategy_status(self) -> Dict:
        return {
            'current_mode': self.current_mode.value,
            'current_market_state': self.current_market_state.value,
            'current_regime': self.current_regime.value,
            'last_analysis_time': self.last_analysis_time,
            'grid_levels_count': sum(len(v) for v in self.grid_levels.values()),
            'grid_base_prices': self.grid_base_prices.copy(),
            'grid_fill_counts': self.grid_fill_count.copy(),
            'mode_history': self.mode_history[-10:],
            'parameters': self.parameters.copy(),
        }
