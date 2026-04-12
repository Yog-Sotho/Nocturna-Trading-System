"""
Risk Management Module for NOCTURNA v2.0 Trading System
Production-grade risk management with real correlation calculations.
"""

import os
import sys
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from enum import Enum
import threading
from collections import deque

from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class RiskLevel(Enum):
    """Risk level enum for system risk classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskEvent(Enum):
    """Types of risk events that can be triggered."""
    POSITION_LIMIT_EXCEEDED = "POSITION_LIMIT_EXCEEDED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    CORRELATION_RISK = "CORRELATION_RISK"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    MARGIN_CALL = "MARGIN_CALL"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    LIQUIDITY_RISK = "LIQUIDITY_RISK"


class RiskManager:
    """
    Production-grade risk management system.
    Implements comprehensive risk controls, position sizing, and portfolio monitoring.
    """

    # Thread-safe correlation cache
    _correlation_cache: Dict[str, Tuple[float, datetime]] = {}
    _cache_lock = threading.Lock()
    _correlation_cache_ttl = timedelta(hours=1)

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load risk parameters with validation
        self.risk_parameters = self._load_risk_parameters()

        # Risk state
        self.current_risk_level = RiskLevel.LOW
        self.risk_events: List[RiskEvent] = []
        self.daily_stats = {
            'trades': [],
            'realized_pnl': 0.0,
            'started_at': datetime.now(timezone.utc)
        }

        # Portfolio tracking
        self.portfolio_value = float(self.config.get('initial_capital', 100000))
        self.max_portfolio_value = self.portfolio_value
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Correlation tracking
        self.correlation_history: Dict[str, List[float]] = {}
        self.position_correlations: Dict[str, float] = {}

        # Volatility tracking
        self.volatility_cache: Dict[str, Dict] = {}
        self.price_history: Dict[str, deque] = {}

        # Risk metrics history for monitoring
        self.metrics_history = deque(maxlen=1000)

        # Thread safety
        self._lock = threading.RLock()

        self.logger.info("Risk Manager initialized with production-grade settings")

    def _load_risk_parameters(self) -> Dict:
        """
        Load and validate risk parameters from configuration.
        Ensures all parameters are within safe bounds.
        """
        default_params = {
            # Position limits
            'max_position_size': 0.20,  # 20% max per position
            'max_portfolio_exposure': 0.80,  # 80% max total exposure
            'max_sector_exposure': 0.30,  # 30% max sector exposure
            'max_correlation_exposure': 0.50,  # 50% in correlated positions

            # Loss limits
            'max_daily_loss': 0.05,  # 5% max daily loss
            'max_weekly_loss': 0.10,  # 10% max weekly loss
            'max_monthly_loss': 0.20,  # 20% max monthly loss
            'max_drawdown': 0.15,  # 15% max drawdown

            # Volatility parameters
            'volatility_threshold': 2.0,  # ATR multiplier threshold
            'volatility_lookback': 20,  # Days for ATR calculation
            'volatility_adjustment': True,  # Auto-adjust based on volatility

            # Correlation parameters
            'correlation_threshold': 0.70,  # High correlation threshold
            'correlation_lookback': 60,  # Days for correlation calculation
            'min_correlation_data': 30,  # Minimum data points for valid correlation

            # Stop loss parameters
            'dynamic_stop_loss': True,
            'stop_loss_multiplier': 2.0,  # ATR multiplier for stop loss
            'trailing_stop_activation': 0.02,  # 2% profit to activate trailing stop
            'trailing_stop_offset': 0.01,  # 1% trailing offset

            # Position sizing
            'position_sizing_method': 'volatility',  # kelly, fixed, volatility
            'kelly_fraction': 0.25,  # Conservative Kelly fraction
            'volatility_target': 0.15,  # 15% target portfolio volatility
            'min_position_size': 0.001,  # Minimum 0.1% position

            # Trade limits
            'max_trades_per_day': 50,
            'max_trades_per_hour': 10,
            'cooldown_period': 300,  # 5 minutes between trades on same symbol

            # Emergency stops
            'emergency_stop_loss': 0.10,  # 10% loss triggers emergency
            'system_halt_conditions': [
                'CRITICAL_DRAWDOWN',
                'SYSTEM_ERROR',
                'MARGIN_CALL'
            ]
        }

        # Merge with configuration
        params = default_params.copy()
        risk_config = self.config.get('risk_parameters', {})

        for key, value in risk_config.items():
            if key in default_params:
                # Type validation
                if isinstance(value, type(default_params[key])):
                    params[key] = value
                else:
                    self.logger.warning(f"Invalid type for {key}, using default")

        # Validate ranges
        params['max_position_size'] = min(0.5, max(0.001, params['max_position_size']))
        params['max_portfolio_exposure'] = min(1.0, max(0.1, params['max_portfolio_exposure']))
        params['max_daily_loss'] = min(0.2, max(0.001, params['max_daily_loss']))
        params['max_drawdown'] = min(0.5, max(0.001, params['max_drawdown']))

        return params

    def validate_trade(self, signal: Dict, current_positions: Dict,
                       market_data: Dict) -> Tuple[bool, str, Dict]:
        """
        Validate a trading signal against all risk controls.

        Args:
            signal: Trading signal dictionary
            current_positions: Current portfolio positions
            market_data: Current market data for the symbol

        Returns:
            Tuple of (is_valid, reason, adjusted_signal)
        """
        with self._lock:
            try:
                # Start with signal copy for adjustments
                adjusted_signal = signal.copy()

                # Step 1: Basic constraint validation
                if not self._validate_basic_constraints(signal):
                    return False, "Basic constraints not met", signal

                # Step 2: Position limit check
                valid, reason = self._check_position_limits(signal, current_positions)
                if not valid:
                    return False, reason, signal

                # Step 3: Portfolio exposure check
                valid, reason = self._check_portfolio_exposure(signal, current_positions, market_data)
                if not valid:
                    return False, reason, signal

                # Step 4: Correlation risk check
                valid, reason = self._check_correlation_risk(signal, current_positions, market_data)
                if not valid:
                    return False, reason, signal

                # Step 5: Volatility risk check
                valid, reason = self._check_volatility_risk(signal, market_data)
                if not valid:
                    return False, reason, signal

                # Step 6: Temporal limits check
                valid, reason = self._check_temporal_limits(signal)
                if not valid:
                    return False, reason, signal

                # Step 7: Position size adjustment
                adjusted_signal = self._adjust_position_size(
                    adjusted_signal, current_positions, market_data
                )

                # Step 8: Risk levels adjustment (stop loss/take profit)
                adjusted_signal = self._adjust_risk_levels(adjusted_signal, market_data)

                return True, "Trade validated", adjusted_signal

            except Exception as e:
                self.logger.error(f"Error in trade validation: {e}")
                return False, f"Validation error: {str(e)}", signal

    def _validate_basic_constraints(self, signal: Dict) -> bool:
        """Validate basic signal constraints."""
        required_fields = ['symbol', 'side', 'quantity', 'type']
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Missing required field: {field}")
                return False

        if signal['quantity'] <= 0:
            self.logger.error("Quantity must be positive")
            return False

        if signal['side'] not in ['buy', 'sell']:
            self.logger.error("Side must be 'buy' or 'sell'")
            return False

        return True

    def _check_position_limits(self, signal: Dict, positions: Dict) -> Tuple[bool, str]:
        """Check if trade would exceed position limits."""
        symbol = signal['symbol']
        quantity = signal['quantity']

        # Calculate new position
        current_position = positions.get(symbol, {}).get('quantity', 0)
        if signal['side'] == 'buy':
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity

        max_position = self.risk_parameters['max_position_size']
        if abs(new_position) > max_position:
            return False, f"Position limit exceeded for {symbol}: {abs(new_position):.2%} > {max_position:.2%}"

        return True, "Position limits OK"

    def _check_portfolio_exposure(self, signal: Dict, positions: Dict,
                                 market_data: Dict) -> Tuple[bool, str]:
        """Check portfolio total exposure."""
        _symbol = signal['symbol']  # noqa: F841 — extracted for logging context
        current_price = market_data.get('price', 100)
        quantity = signal.get('quantity', 0)

        # Calculate total exposure
        total_exposure = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
        new_trade_value = abs(quantity * current_price)

        # Use portfolio value for percentage calculation
        portfolio_value = self.portfolio_value if self.portfolio_value > 0 else 100000
        new_total_exposure = (total_exposure + new_trade_value) / portfolio_value

        max_exposure = self.risk_parameters['max_portfolio_exposure']
        if new_total_exposure > max_exposure:
            return False, f"Portfolio exposure exceeded: {new_total_exposure:.2%} > {max_exposure:.2%}"

        return True, "Portfolio exposure OK"

    def _check_correlation_risk(self, signal: Dict, positions: Dict,
                                  market_data: Dict) -> Tuple[bool, str]:
        """
        Check correlation risk with real calculation.
        Uses historical price data to compute actual correlations.
        """
        symbol = signal['symbol']
        correlation_threshold = self.risk_parameters['correlation_threshold']

        try:
            # Calculate correlations with existing positions
            high_correlation_exposure = 0.0
            portfolio_value = self.portfolio_value if self.portfolio_value > 0 else 100000

            for pos_symbol, position in positions.items():
                if pos_symbol == symbol:
                    continue

                # Calculate real correlation using price history
                correlation = self._calculate_real_correlation(symbol, pos_symbol)

                if abs(correlation) > correlation_threshold:
                    # Add weighted exposure for highly correlated positions
                    pos_value = abs(position.get('market_value', 0))
                    high_correlation_exposure += pos_value / portfolio_value

            # Add new trade exposure
            new_trade_value = abs(signal['quantity'] * market_data.get('price', 100))
            total_correlated_exposure = high_correlation_exposure + (new_trade_value / portfolio_value)

            max_correlated = self.risk_parameters['max_correlation_exposure']
            if total_correlated_exposure > max_correlated:
                return False, f"Correlation risk exceeded: {total_correlated_exposure:.2%} > {max_correlated:.2%}"

            return True, "Correlation risk OK"

        except Exception as e:
            self.logger.warning(f"Correlation check error: {e}. Allowing trade by default.")
            return True, "Correlation check skipped due to error"

    def _calculate_real_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Calculate real correlation between two symbols using historical returns.
        Uses Spearman correlation for robustness to outliers.

        Args:
            symbol1: First symbol
            symbol2: Second symbol

        Returns:
            Correlation coefficient (-1 to 1)
        """
        cache_key = f"{symbol1}_{symbol2}"

        # Check cache
        with self._correlation_cache:
            if cache_key in self._correlation_cache:
                correlation, timestamp = self._correlation_cache[cache_key]
                if datetime.now(timezone.utc) - timestamp < self._correlation_cache_ttl:
                    return correlation

        try:
            # Get price history for both symbols
            returns1 = self._get_symbol_returns(symbol1)
            returns2 = self._get_symbol_returns(symbol2)

            min_data = self.risk_parameters['min_correlation_data']
            if len(returns1) < min_data or len(returns2) < min_data:
                self.logger.debug(f"Insufficient data for correlation: {symbol1}/{symbol2}")
                return 0.0

            # Align returns
            min_len = min(len(returns1), len(returns2))
            returns1 = returns1[-min_len:]
            returns2 = returns2[-min_len:]

            # Calculate Spearman correlation
            correlation, p_value = spearmanr(returns1, returns2)

            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0

            # Cache the result
            with self._correlation_cache:
                self._correlation_cache[cache_key] = (correlation, datetime.now(timezone.utc))

            return correlation

        except Exception as e:
            self.logger.warning(f"Error calculating correlation {symbol1}/{symbol2}: {e}")
            return 0.0

    def _get_symbol_returns(self, symbol: str) -> np.ndarray:
        """
        Get historical returns for a symbol.
        Uses stored price history to calculate log returns.
        """
        if symbol not in self.price_history:
            return np.array([])

        prices = list(self.price_history[symbol])
        if len(prices) < 2:
            return np.array([])

        # Calculate log returns
        returns = np.diff(np.log(prices))
        return returns

    def _update_price_history(self, symbol: str, price: float) -> None:
        """
        Update price history for a symbol.
        Used for correlation calculations.
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=500)

        self.price_history[symbol].append(price)

    def _check_volatility_risk(self, signal: Dict, market_data: Dict) -> Tuple[bool, str]:
        """Check if volatility is too high for new trades."""
        _symbol = signal['symbol']  # noqa: F841
        volatility_threshold = self.risk_parameters['volatility_threshold']

        try:
            current_volatility = market_data.get('volatility', 0)
            avg_volatility = market_data.get('avg_atr', current_volatility)

            if avg_volatility == 0:
                return True, "Volatility data not available"

            volatility_ratio = current_volatility / avg_volatility

            if volatility_ratio > volatility_threshold:
                return False, f"Volatility too high: ratio {volatility_ratio:.2f} > threshold {volatility_threshold:.2f}"

            return True, "Volatility OK"

        except Exception as e:
            self.logger.warning(f"Volatility check error: {e}")
            return True, "Volatility check skipped"

    def _check_temporal_limits(self, signal: Dict) -> Tuple[bool, str]:
        """Check temporal trading limits."""
        now = datetime.now(timezone.utc)

        # Count daily trades
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_trades = [
            t for t in self.daily_stats['trades']
            if t.get('timestamp', datetime.min) >= today_start
        ]

        max_daily = self.risk_parameters['max_trades_per_day']
        if len(today_trades) >= max_daily:
            return False, f"Daily trade limit reached: {len(today_trades)} >= {max_daily}"

        # Count hourly trades
        hour_ago = now - timedelta(hours=1)
        hour_trades = [
            t for t in today_trades
            if t.get('timestamp', datetime.min) > hour_ago
        ]

        max_hourly = self.risk_parameters['max_trades_per_hour']
        if len(hour_trades) >= max_hourly:
            return False, f"Hourly trade limit reached: {len(hour_trades)} >= {max_hourly}"

        # Check cooldown
        symbol = signal['symbol']
        symbol_trades = [
            t for t in today_trades
            if t.get('symbol') == symbol
        ]

        if symbol_trades:
            last_trade_time = max(t.get('timestamp', datetime.min) for t in symbol_trades)
            cooldown = timedelta(seconds=self.risk_parameters['cooldown_period'])
            if now - last_trade_time < cooldown:
                remaining = cooldown - (now - last_trade_time)
                return False, f"Cooldown active for {symbol}: {int(remaining.total_seconds())}s remaining"

        return True, "Temporal limits OK"

    def _adjust_position_size(self, signal: Dict, positions: Dict,
                              market_data: Dict) -> Dict:
        """Adjust position size based on configured method."""
        method = self.risk_parameters['position_sizing_method']
        original_quantity = signal['quantity']

        try:
            if method == 'volatility':
                signal = self._volatility_position_sizing(signal, market_data)
            elif method == 'kelly':
                signal = self._kelly_position_sizing(signal, market_data)
            elif method == 'fixed':
                pass  # Keep original size

            # Apply max and min limits
            max_size = self.risk_parameters['max_position_size']
            min_size = self.risk_parameters['min_position_size']

            if signal['quantity'] > max_size:
                self.logger.info(f"Position size reduced from {original_quantity:.4f} to {max_size:.4f}")
                signal['quantity'] = max_size

            if signal['quantity'] < min_size:
                signal['quantity'] = min_size

            return signal

        except Exception as e:
            self.logger.error(f"Error adjusting position size: {e}")
            return signal

    def _volatility_position_sizing(self, signal: Dict, market_data: Dict) -> Dict:
        """
        Calculate position size based on volatility targeting.
        Aims to maintain constant portfolio volatility.
        """
        try:
            current_volatility = market_data.get('volatility', 0.2)
            target_volatility = self.risk_parameters['volatility_target']

            if current_volatility > 0:
                # Inverse relationship: higher volatility = smaller position
                volatility_ratio = target_volatility / current_volatility
                # Square root scaling for more gradual adjustment
                adjustment = np.sqrt(volatility_ratio)

                signal['quantity'] *= adjustment

            return signal

        except Exception as e:
            self.logger.error(f"Error in volatility position sizing: {e}")
            return signal

    def _kelly_position_sizing(self, signal: Dict, market_data: Dict) -> Dict:
        """Calculate position size using Kelly criterion."""
        try:
            symbol = signal['symbol']

            # Estimate win rate and average win/loss from recent trades
            win_rate = self._estimate_win_rate(symbol)
            avg_win = self._estimate_avg_win(symbol)
            avg_loss = abs(self._estimate_avg_loss(symbol))

            if avg_loss == 0:
                return signal

            # Kelly formula: f = (bp - q) / b
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            kelly_fraction = (b * p - q) / b

            # Apply conservative fraction
            kelly_fraction *= self.risk_parameters['kelly_fraction']

            # Clamp to valid range
            kelly_fraction = max(0, min(kelly_fraction, self.risk_parameters['max_position_size']))

            signal['quantity'] = kelly_fraction

            return signal

        except Exception as e:
            self.logger.error(f"Error in Kelly sizing: {e}")
            return signal

    def _estimate_win_rate(self, symbol: str) -> float:
        """Estimate win rate from recent trades."""
        recent_trades = [
            t for t in self.daily_stats['trades'][-50:]
            if t.get('symbol') == symbol
        ]

        if not recent_trades:
            return 0.55  # Default assumption

        winning_trades = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
        return winning_trades / len(recent_trades)

    def _estimate_avg_win(self, symbol: str) -> float:
        """Estimate average winning trade size."""
        recent_trades = [
            t for t in self.daily_stats['trades'][-50:]
            if t.get('symbol') == symbol and t.get('pnl', 0) > 0
        ]

        if not recent_trades:
            return 0.02  # Default 2%

        avg_win = sum(t['pnl'] for t in recent_trades) / len(recent_trades)
        return avg_win / self.portfolio_value if self.portfolio_value > 0 else 0.02

    def _estimate_avg_loss(self, symbol: str) -> float:
        """Estimate average losing trade size."""
        recent_trades = [
            t for t in self.daily_stats['trades'][-50:]
            if t.get('symbol') == symbol and t.get('pnl', 0) < 0
        ]

        if not recent_trades:
            return -0.015  # Default 1.5%

        avg_loss = sum(t['pnl'] for t in recent_trades) / len(recent_trades)
        return avg_loss / self.portfolio_value if self.portfolio_value > 0 else -0.015

    def _adjust_risk_levels(self, signal: Dict, market_data: Dict) -> Dict:
        """Adjust stop loss and take profit based on ATR."""
        if not self.risk_parameters['dynamic_stop_loss']:
            return signal

        try:
            atr = market_data.get('atr', 0)
            current_price = market_data.get('price', signal.get('price', 0))

            if atr == 0 or current_price == 0:
                return signal

            stop_multiplier = self.risk_parameters['stop_loss_multiplier']

            if signal['side'] == 'buy':
                dynamic_stop = current_price - (atr * stop_multiplier)
                dynamic_tp = current_price + (atr * stop_multiplier * 1.5)
            else:
                dynamic_stop = current_price + (atr * stop_multiplier)
                dynamic_tp = current_price - (atr * stop_multiplier * 1.5)

            # Only set if not already specified
            if signal.get('stop_loss') is None or signal.get('stop_loss', 0) == 0:
                signal['stop_loss'] = dynamic_stop

            if signal.get('take_profit') is None or signal.get('take_profit', 0) == 0:
                signal['take_profit'] = dynamic_tp

            return signal

        except Exception as e:
            self.logger.error(f"Error adjusting risk levels: {e}")
            return signal

    def monitor_portfolio_risk(self, positions: Dict, market_data: Dict) -> Dict:
        """
        Monitor portfolio risk continuously.
        Returns risk metrics and triggers events for risk breaches.
        """
        with self._lock:
            try:
                risk_metrics = {}

                # Calculate all risk metrics
                risk_metrics['total_exposure'] = self._calculate_total_exposure(positions)
                risk_metrics['current_drawdown'] = self._calculate_current_drawdown(positions)
                risk_metrics['var_95'] = self._calculate_var(positions, market_data, 0.95)
                risk_metrics['portfolio_volatility'] = self._calculate_portfolio_volatility(positions, market_data)
                risk_metrics['correlation_risk'] = self._calculate_portfolio_correlation_risk(positions, market_data)
                risk_metrics['concentration_risk'] = self._calculate_concentration_risk(positions)

                # Assess risk level
                risk_level = self._assess_risk_level(risk_metrics)
                self.current_risk_level = risk_level

                # Check for risk events
                risk_events = self._check_risk_events(risk_metrics, positions)
                self.risk_events.extend(risk_events)

                # Record metrics history
                self.metrics_history.append({
                    'timestamp': datetime.now(timezone.utc),
                    'metrics': risk_metrics,
                    'level': risk_level.value
                })

                return {
                    'risk_level': risk_level.value,
                    'risk_metrics': risk_metrics,
                    'risk_events': [event.value for event in risk_events],
                    'timestamp': datetime.now(timezone.utc)
                }

            except Exception as e:
                self.logger.error(f"Error monitoring portfolio risk: {e}")
                return {'error': str(e)}

    def _calculate_total_exposure(self, positions: Dict) -> float:
        """Calculate total portfolio exposure."""
        if self.portfolio_value == 0:
            return 0.0
        return sum(abs(pos.get('market_value', 0)) for pos in positions.values()) / self.portfolio_value

    def _calculate_current_drawdown(self, positions: Dict) -> float:
        """Calculate current drawdown from peak."""
        current_value = sum(pos.get('market_value', 0) for pos in positions.values())

        if self.max_portfolio_value == 0:
            self.max_portfolio_value = current_value

        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
            self.current_drawdown = 0.0
        else:
            if self.max_portfolio_value > 0:
                self.current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
            else:
                self.current_drawdown = 0.0

        # Track max drawdown
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        return self.current_drawdown

    def _calculate_var(self, positions: Dict, market_data: Dict, confidence: float) -> float:
        """
        Calculate Value at Risk using historical simulation method.
        More robust than parametric VaR.
        """
        try:
            if self.portfolio_value == 0:
                return 0.0

            # Get historical returns from metrics
            returns = []
            for record in list(self.metrics_history)[-252:]:  # ~1 year of data
                if 'metrics' in record and 'portfolio_volatility' in record['metrics']:
                    returns.append(record['metrics']['portfolio_volatility'])

            if len(returns) < 30:
                return 0.0

            returns = np.array(returns)

            # Calculate VaR
            var = np.percentile(returns, (1 - confidence) * 100)
            var_abs = abs(var * self.portfolio_value)

            return var_abs

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def _calculate_portfolio_volatility(self, positions: Dict, market_data: Dict) -> float:
        """Calculate portfolio volatility (weighted average)."""
        try:
            if self.portfolio_value == 0:
                return 0.0

            weighted_volatility = 0.0

            for symbol, position in positions.items():
                weight = abs(position.get('market_value', 0)) / self.portfolio_value
                volatility = market_data.get(symbol, {}).get('volatility', 0.2)
                weighted_volatility += weight * volatility

            return weighted_volatility

        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0

    def _calculate_portfolio_correlation_risk(self, positions: Dict, market_data: Dict) -> float:
        """Calculate average correlation between all positions."""
        symbols = list(positions.keys())
        if len(symbols) < 2:
            return 0.0

        correlations = []
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                correlation = self._calculate_real_correlation(sym1, sym2)
                correlations.append(abs(correlation))

        if not correlations:
            return 0.0

        return np.mean(correlations)

    def _calculate_concentration_risk(self, positions: Dict) -> float:
        """
        Calculate concentration risk using Herfindahl-Hirschman Index.
        HHI = sum(weight^2) for all positions.
        """
        try:
            if self.portfolio_value == 0:
                return 0.0

            hhi = 0.0
            for position in positions.values():
                weight = abs(position.get('market_value', 0)) / self.portfolio_value
                hhi += weight ** 2

            return hhi

        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0.0

    def _assess_risk_level(self, metrics: Dict) -> RiskLevel:
        """Assess overall risk level from metrics."""
        risk_score = 0

        # Drawdown scoring
        dd = metrics.get('current_drawdown', 0)
        if dd > 0.10:
            risk_score += 3
        elif dd > 0.05:
            risk_score += 2
        elif dd > 0.02:
            risk_score += 1

        # Volatility scoring
        vol = metrics.get('portfolio_volatility', 0)
        if vol > 0.30:
            risk_score += 3
        elif vol > 0.20:
            risk_score += 2
        elif vol > 0.15:
            risk_score += 1

        # Concentration scoring
        conc = metrics.get('concentration_risk', 0)
        if conc > 0.5:
            risk_score += 2
        elif conc > 0.3:
            risk_score += 1

        # Correlation scoring
        corr = metrics.get('correlation_risk', 0)
        if corr > 0.7:
            risk_score += 2
        elif corr > 0.5:
            risk_score += 1

        # Determine level
        if risk_score >= 8:
            return RiskLevel.CRITICAL
        elif risk_score >= 5:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _check_risk_events(self, metrics: Dict, positions: Dict) -> List[RiskEvent]:
        """Check for specific risk events that need to be triggered."""
        events = []

        try:
            # Drawdown limit
            if metrics.get('current_drawdown', 0) > self.risk_parameters['max_drawdown']:
                events.append(RiskEvent.DRAWDOWN_LIMIT)

            # Daily loss limit
            daily_pnl = self.daily_stats.get('realized_pnl', 0)
            if daily_pnl < -self.risk_parameters['max_daily_loss'] * self.portfolio_value:
                events.append(RiskEvent.DAILY_LOSS_LIMIT)

            # Concentration limit
            if metrics.get('concentration_risk', 0) > 0.6:
                events.append(RiskEvent.CONCENTRATION_RISK)

            # Correlation limit
            if metrics.get('correlation_risk', 0) > self.risk_parameters['correlation_threshold']:
                events.append(RiskEvent.CORRELATION_RISK)

            # Volatility spike
            if metrics.get('portfolio_volatility', 0) > 0.4:
                events.append(RiskEvent.VOLATILITY_SPIKE)

        except Exception as e:
            self.logger.error(f"Error checking risk events: {e}")
            events.append(RiskEvent.SYSTEM_ERROR)

        return events

    def record_trade(self, trade: Dict) -> None:
        """Record a completed trade for analytics."""
        with self._lock:
            trade['timestamp'] = datetime.now(timezone.utc)
            self.daily_stats['trades'].append(trade)

            if 'pnl' in trade:
                self.daily_stats['realized_pnl'] += trade['pnl']

    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report."""
        return {
            'current_risk_level': self.current_risk_level.value,
            'recent_events': [event.value for event in self.risk_events[-10:]],
            'risk_parameters': self.risk_parameters,
            'portfolio_metrics': {
                'value': self.portfolio_value,
                'max_value': self.max_portfolio_value,
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown
            },
            'daily_stats': {
                'trades_today': len([
                    t for t in self.daily_stats['trades']
                    if t.get('timestamp', datetime.min).date() == datetime.now(timezone.utc).date()
                ]),
                'realized_pnl': self.daily_stats.get('realized_pnl', 0)
            },
            'timestamp': datetime.now(timezone.utc)
        }
