# FILE LOCATION: src/advanced/ml_optimizer.py
"""
Machine Learning Optimizer for NOCTURNA v2.0
Uses ML algorithms to automatically optimize trading parameters.
"""

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class MLOptimizer:
    """
    Machine Learning system for automatic parameter optimization.
    Uses ensemble methods and time series cross-validation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Modelli ML
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }

        # Scaler for normalization
        self.scaler = StandardScaler()

        # Parameters to optimize
        self.parameter_ranges = {
            'grid_spacing': (0.001, 0.02),
            'atr_mult_sl': (1.0, 5.0),
            'atr_mult_tp': (1.0, 8.0),
            'max_position_size': (0.05, 0.5),
            'volatility_threshold': (1.0, 3.0),
            'trend_strength_threshold': (0.3, 0.8),
            'reversal_confirmation_bars': (2, 10),
            'breakout_volume_mult': (1.2, 3.0)
        }

        # Optimization history
        self.optimization_history = []

        # Best parameters found
        self.best_parameters = None
        self.best_score = -np.inf

        self.logger.info("ML Optimizer initialized")

    def generate_features(self, market_data: pd.DataFrame,
                         current_params: dict) -> np.ndarray:
        """
        Generate features for the ML model from market data.

        Args:
            market_data: DataFrame con dati OHLCV
            current_params: Current system parameters

        Returns:
            Array di features normalizzate
        """
        try:
            features = []

            # Market features
            if len(market_data) > 0:
                # Volatility
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized
                features.append(volatility)

                # Trend strength
                sma_20 = market_data['close'].rolling(20).mean()
                sma_50 = market_data['close'].rolling(50).mean()
                sma_50_last = sma_50.iloc[-1]
                trend_strength = (sma_20.iloc[-1] - sma_50_last) / sma_50_last if sma_50_last != 0 else 0.0
                features.append(trend_strength)

                # Volume profile
                avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
                current_volume = market_data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                features.append(volume_ratio)

                # Normalized ATR
                high_low = market_data['high'] - market_data['low']
                high_close = np.abs(market_data['high'] - market_data['close'].shift())
                low_close = np.abs(market_data['low'] - market_data['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(14).mean().iloc[-1]
                close_price = market_data['close'].iloc[-1]
                atr_normalized = atr / close_price if close_price > 0 else 0.0
                features.append(atr_normalized)

                # RSI — with division-by-zero guard
                delta = market_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                rsi_value = rsi.iloc[-1]
                features.append(rsi_value / 100.0 if not np.isnan(rsi_value) else 0.5)

                # Bollinger Bands position — with zero-width guard
                bb_middle = market_data['close'].rolling(20).mean()
                bb_std = market_data['close'].rolling(20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
                if bb_width > 0:
                    bb_position = (close_price - bb_lower.iloc[-1]) / bb_width
                else:
                    bb_position = 0.5  # Midpoint when bands are flat
                features.append(bb_position)

            else:
                # Default values when no data available
                features.extend([0.2, 0.0, 1.0, 0.02, 0.5, 0.5])

            # Current parameter features (normalized to 0-1)
            for param_name in self.parameter_ranges:
                if param_name in current_params:
                    param_value = current_params[param_name]
                    param_min, param_max = self.parameter_ranges[param_name]
                    param_range = param_max - param_min
                    normalized_value = (param_value - param_min) / param_range if param_range > 0 else 0.5
                    features.append(normalized_value)
                else:
                    features.append(0.5)

            # NOTE: Time-of-optimization features (hour, day_of_week) intentionally
            # removed — they caused data leakage by correlating wall-clock time with
            # performance, which is pure noise that leads to overfitting.

            return np.array(features).reshape(1, -1)

        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            # Return default features (6 market + N params)
            n_features = 6 + len(self.parameter_ranges)
            return np.zeros((1, n_features)).reshape(1, -1)

    def calculate_performance_score(self, backtest_results: dict) -> float:
        """
        Calculate performance score from backtest results.
        All components normalized to [0,1] before weighting to prevent scale bias.

        Returns:
            Score (higher = better)
        """
        try:
            total_return = backtest_results.get('total_return', 0.0)
            sharpe_ratio = backtest_results.get('sharpe_ratio', 0.0)
            max_drawdown = backtest_results.get('max_drawdown', 0.0)
            win_rate = backtest_results.get('win_rate', 0.0)
            profit_factor = backtest_results.get('profit_factor', 1.0)

            # Normalize each component to [0, 1]
            # Return: clip to [-1, 2] then map to [0, 1]
            norm_return = np.clip((total_return + 1.0) / 3.0, 0, 1)
            # Sharpe: clip to [-2, 4] then map to [0, 1]
            norm_sharpe = np.clip((sharpe_ratio + 2.0) / 6.0, 0, 1)
            # Drawdown: 0% = perfect (1.0), 50% = terrible (0.0)
            norm_drawdown = np.clip(1.0 - abs(max_drawdown) * 2.0, 0, 1)
            # Win rate: already 0-1
            norm_win_rate = np.clip(win_rate, 0, 1)
            # Profit factor: clip to [0, 5] then map to [0, 1]
            norm_pf = np.clip(profit_factor / 5.0, 0, 1)

            # Weighted composite score (all inputs now on same scale)
            score = (
                norm_return * 0.30 +
                norm_sharpe * 0.25 +
                norm_drawdown * 0.20 +
                norm_win_rate * 0.15 +
                norm_pf * 0.10
            )

            return score

        except Exception as e:
            self.logger.error(f"Error calculating performance score: {e}")
            return -1.0

    def optimize_parameters(self, market_data: pd.DataFrame,
                          current_params: dict,
                          backtest_function: callable,
                          n_iterations: int = 50) -> dict:
        """
        Optimize parameters using genetic algorithms and ML.

        Args:
            market_data: Market data for optimization
            current_params: Current parameters
            backtest_function: Function to run backtests
            n_iterations: Number of iterations

        Returns:
            Optimized parameters
        """
        try:
            self.logger.info(f"Starting parameter optimization ({n_iterations} iterations)")

            best_params = current_params.copy()
            best_score = -np.inf

            # History for ML training
            training_data = []

            for iteration in range(n_iterations):
                # Generate candidate parameters
                if iteration < n_iterations // 2:
                    # First half: random exploration
                    candidate_params = self._generate_random_parameters()
                else:
                    # Seconda metà: sfruttamento con ML
                    if len(training_data) > 10:
                        candidate_params = self._generate_ml_parameters(
                            market_data, training_data
                        )
                    else:
                        candidate_params = self._generate_random_parameters()

                # Run backtest
                try:
                    backtest_results = backtest_function(candidate_params)
                    score = self.calculate_performance_score(backtest_results)

                    # Add to training data
                    features = self.generate_features(market_data, candidate_params)
                    training_data.append({
                        'features': features.flatten(),
                        'score': score,
                        'params': candidate_params.copy()
                    })

                    # Update best
                    if score > best_score:
                        best_score = score
                        best_params = candidate_params.copy()
                        self.logger.info(f"Nuovo best score: {score:.4f} (iter {iteration})")

                except Exception as e:
                    self.logger.warning(f"Backtest error at iteration {iteration}: {e}")
                    continue

            # Save results
            self.best_parameters = best_params
            self.best_score = best_score

            # Add to history
            self.optimization_history.append({
                'timestamp': datetime.now(UTC),
                'best_params': best_params,
                'best_score': best_score,
                'iterations': n_iterations
            })

            self.logger.info(f"Optimization complete. Best score: {best_score:.4f}")

            return best_params

        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return current_params

    def _generate_random_parameters(self) -> dict:
        """Generate random parameters within defined ranges."""
        params = {}

        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            params[param_name] = np.random.uniform(min_val, max_val)

        return params

    def _generate_ml_parameters(self, market_data: pd.DataFrame,
                               training_data: list[dict]) -> dict:
        """
        Generate parameters using ML predictions.

        Uses a window-local scaler to avoid lookahead bias: the scaler
        is fit only on the training observations available at this point.

        Args:
            market_data: Current market data
            training_data: Accumulated training data

        Returns:
            Model-predicted parameters
        """
        try:
            if len(training_data) < 10:
                return self._generate_random_parameters()

            # Prepare training data
            X = np.array([data['features'] for data in training_data])
            y = np.array([data['score'] for data in training_data])

            # FIX CQ-05: Use a fresh scaler per call to prevent distributional
            # leakage from future time windows into historical scaling.
            window_scaler = StandardScaler()
            X_scaled = window_scaler.fit_transform(X)

            # Train model
            model = self.models['random_forest']
            model.fit(X_scaled, y)

            # Generate candidates and predict
            best_predicted_params = None
            best_predicted_score = -np.inf

            for _ in range(20):  # Try 20 candidates
                candidate_params = self._generate_random_parameters()
                features = self.generate_features(market_data, candidate_params)
                features_scaled = window_scaler.transform(features)

                predicted_score = model.predict(features_scaled)[0]

                if predicted_score > best_predicted_score:
                    best_predicted_score = predicted_score
                    best_predicted_params = candidate_params

            return best_predicted_params or self._generate_random_parameters()

        except Exception as e:
            self.logger.error(f"Error generating ML parameters: {e}")
            return self._generate_random_parameters()

    def adaptive_parameter_tuning(self, market_data: pd.DataFrame,
                                 current_params: dict,
                                 recent_performance: dict) -> dict:
        """
        Adaptive parameter tuning based on recent performance.

        Args:
            market_data: Dati di mercato recenti
            current_params: Current parameters
            recent_performance: Recent performance metrics

        Returns:
            Adjusted parameters
        """
        try:
            adjusted_params = current_params.copy()

            # Analyze performance
            win_rate = recent_performance.get('win_rate', 0.5)
            avg_return = recent_performance.get('avg_return', 0.0)
            volatility = recent_performance.get('volatility', 0.0)

            # Adaptive adjustments based on win rate
            if win_rate < 0.4:
                # Low win rate — increase conservatism
                adjusted_params['max_position_size'] *= 0.9
                adjusted_params['atr_mult_sl'] *= 1.1
            elif win_rate > 0.7:
                # High win rate — increase aggressiveness slightly
                adjusted_params['max_position_size'] *= 1.05
                adjusted_params['atr_mult_tp'] *= 1.1

            # Adjust based on average return
            if avg_return < -0.02:
                # Negative returns — tighten risk controls
                adjusted_params['max_position_size'] *= 0.85
            elif avg_return > 0.05:
                # Strong positive returns — widen take profit
                adjusted_params['atr_mult_tp'] *= 1.05

            if volatility > 0.3:  # High volatility
                # More conservative parameters
                adjusted_params['grid_spacing'] *= 1.2
                adjusted_params['volatility_threshold'] *= 1.1

            elif volatility < 0.1:  # Bassa volatilità
                # More aggressive parameters
                adjusted_params['grid_spacing'] *= 0.9
                adjusted_params['volatility_threshold'] *= 0.9

            # Assicura che i parametri rimangano nei range
            for param_name, value in adjusted_params.items():
                if param_name in self.parameter_ranges:
                    min_val, max_val = self.parameter_ranges[param_name]
                    adjusted_params[param_name] = np.clip(value, min_val, max_val)

            self.logger.info("Adjusted parameters adattivamente")

            return adjusted_params

        except Exception as e:
            self.logger.error(f"Adaptive tuning error: {e}")
            return current_params

    def get_parameter_importance(self, training_data: list[dict]) -> dict:
        """
        Calculate parameter importance using feature importance.

        Args:
            training_data: Dati di training

        Returns:
            Dictionary with parameter importance
        """
        try:
            if len(training_data) < 10:
                return {}

            # Prepare data
            X = np.array([data['features'] for data in training_data])
            y = np.array([data['score'] for data in training_data])

            # Train Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Feature importance
            importance = model.feature_importances_

            # Map to feature names (6 market features + N param features, no time features)
            feature_names = [
                'volatility', 'trend_strength', 'volume_ratio',
                'atr_normalized', 'rsi', 'bb_position'
            ] + list(self.parameter_ranges.keys())

            importance_dict = {}
            for i, name in enumerate(feature_names[:len(importance)]):
                importance_dict[name] = importance[i]

            return importance_dict

        except Exception as e:
            self.logger.error(f"Error calculating importance: {e}")
            return {}

    def save_optimization_results(self, filepath: str):
        """Save optimization results to file."""
        try:
            results = {
                'best_parameters': self.best_parameters,
                'best_score': self.best_score,
                'optimization_history': [
                    {
                        'timestamp': hist['timestamp'].isoformat(),
                        'best_params': hist['best_params'],
                        'best_score': hist['best_score'],
                        'iterations': hist['iterations']
                    }
                    for hist in self.optimization_history
                ],
                'parameter_ranges': self.parameter_ranges
            }

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Results saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving: {e}")

    def load_optimization_results(self, filepath: str):
        """Load optimization results from file."""
        try:
            with open(filepath) as f:
                results = json.load(f)

            self.best_parameters = results.get('best_parameters')
            self.best_score = results.get('best_score', -np.inf)

            # Ricostruisci history
            self.optimization_history = []
            for hist in results.get('optimization_history', []):
                hist_item = hist.copy()
                hist_item['timestamp'] = datetime.fromisoformat(hist['timestamp'])
                self.optimization_history.append(hist_item)

            self.logger.info(f"Results loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Loading error: {e}")

    def get_optimization_report(self) -> dict:
        """Generate a complete optimization report."""
        try:
            if not self.optimization_history:
                return {'status': 'No optimization history available'}

            latest = self.optimization_history[-1]

            report = {
                'status': 'active',
                'latest_optimization': {
                    'timestamp': latest['timestamp'].isoformat(),
                    'best_score': latest['best_score'],
                    'iterations': latest['iterations']
                },
                'best_parameters': self.best_parameters,
                'total_optimizations': len(self.optimization_history),
                'score_improvement': 0.0
            }

            # Calculate improvement
            if len(self.optimization_history) > 1:
                first_score = self.optimization_history[0]['best_score']
                latest_score = latest['best_score']
                improvement = ((latest_score - first_score) / abs(first_score)) * 100
                report['score_improvement'] = improvement

            return report

        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return {'status': 'error', 'message': str(e)}

