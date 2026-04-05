"""
Machine Learning Optimizer per NOCTURNA v2.0
Utilizza algoritmi di ML per ottimizzare automaticamente i parametri di trading.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

class MLOptimizer:
    """
    Sistema di Machine Learning per l'ottimizzazione automatica dei parametri.
    Utilizza ensemble methods e time series cross-validation.
    """
    
    def __init__(self, config: Dict):
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
        
        # Scaler per normalizzazione
        self.scaler = StandardScaler()
        
        # Parametri da ottimizzare
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
        
        # Storico ottimizzazioni
        self.optimization_history = []
        
        # Best parameters trovati
        self.best_parameters = None
        self.best_score = -np.inf
        
        self.logger.info("ML Optimizer inizializzato")
    
    def generate_features(self, market_data: pd.DataFrame, 
                         current_params: Dict) -> np.ndarray:
        """
        Genera features per il modello ML dai dati di mercato.
        
        Args:
            market_data: DataFrame con dati OHLCV
            current_params: Parametri attuali del sistema
            
        Returns:
            Array di features normalizzate
        """
        try:
            features = []
            
            # Features di mercato
            if len(market_data) > 0:
                # Volatilità
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualizzata
                features.append(volatility)
                
                # Trend strength
                sma_20 = market_data['close'].rolling(20).mean()
                sma_50 = market_data['close'].rolling(50).mean()
                trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
                features.append(trend_strength)
                
                # Volume profile
                avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
                current_volume = market_data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                features.append(volume_ratio)
                
                # ATR normalizzato
                high_low = market_data['high'] - market_data['low']
                high_close = np.abs(market_data['high'] - market_data['close'].shift())
                low_close = np.abs(market_data['low'] - market_data['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = true_range.rolling(14).mean().iloc[-1]
                atr_normalized = atr / market_data['close'].iloc[-1]
                features.append(atr_normalized)
                
                # RSI
                delta = market_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.iloc[-1] / 100.0)  # Normalizzato 0-1
                
                # Bollinger Bands position
                bb_middle = market_data['close'].rolling(20).mean()
                bb_std = market_data['close'].rolling(20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)
                bb_position = (market_data['close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                features.append(bb_position)
                
            else:
                # Valori di default se non ci sono dati
                features.extend([0.2, 0.0, 1.0, 0.02, 0.5, 0.5])
            
            # Features dei parametri attuali
            for param_name in self.parameter_ranges:
                if param_name in current_params:
                    param_value = current_params[param_name]
                    param_min, param_max = self.parameter_ranges[param_name]
                    normalized_value = (param_value - param_min) / (param_max - param_min)
                    features.append(normalized_value)
                else:
                    features.append(0.5)  # Valore medio se parametro mancante
            
            # Features temporali
            now = datetime.now()
            hour_of_day = now.hour / 24.0
            day_of_week = now.weekday() / 6.0
            features.extend([hour_of_day, day_of_week])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione features: {e}")
            # Ritorna features di default
            return np.zeros((1, len(self.parameter_ranges) + 8)).reshape(1, -1)
    
    def calculate_performance_score(self, backtest_results: Dict) -> float:
        """
        Calcola uno score di performance dai risultati di backtest.
        
        Args:
            backtest_results: Risultati del backtest
            
        Returns:
            Score di performance (più alto = migliore)
        """
        try:
            # Componenti dello score
            total_return = backtest_results.get('total_return', 0.0)
            sharpe_ratio = backtest_results.get('sharpe_ratio', 0.0)
            max_drawdown = backtest_results.get('max_drawdown', 0.0)
            win_rate = backtest_results.get('win_rate', 0.0)
            profit_factor = backtest_results.get('profit_factor', 1.0)
            
            # Score composito con pesi
            score = (
                total_return * 0.3 +
                sharpe_ratio * 0.25 +
                (1.0 - abs(max_drawdown)) * 0.2 +  # Penalizza drawdown
                win_rate * 0.15 +
                (profit_factor - 1.0) * 0.1
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo score: {e}")
            return -1.0
    
    def optimize_parameters(self, market_data: pd.DataFrame, 
                          current_params: Dict,
                          backtest_function: callable,
                          n_iterations: int = 50) -> Dict:
        """
        Ottimizza i parametri usando algoritmi genetici e ML.
        
        Args:
            market_data: Dati di mercato per l'ottimizzazione
            current_params: Parametri attuali
            backtest_function: Funzione per eseguire backtest
            n_iterations: Numero di iterazioni
            
        Returns:
            Parametri ottimizzati
        """
        try:
            self.logger.info(f"Inizio ottimizzazione parametri ({n_iterations} iterazioni)")
            
            best_params = current_params.copy()
            best_score = -np.inf
            
            # Storico per training ML
            training_data = []
            
            for iteration in range(n_iterations):
                # Genera parametri candidati
                if iteration < n_iterations // 2:
                    # Prima metà: esplorazione casuale
                    candidate_params = self._generate_random_parameters()
                else:
                    # Seconda metà: sfruttamento con ML
                    if len(training_data) > 10:
                        candidate_params = self._generate_ml_parameters(
                            market_data, training_data
                        )
                    else:
                        candidate_params = self._generate_random_parameters()
                
                # Esegui backtest
                try:
                    backtest_results = backtest_function(candidate_params)
                    score = self.calculate_performance_score(backtest_results)
                    
                    # Aggiungi ai dati di training
                    features = self.generate_features(market_data, candidate_params)
                    training_data.append({
                        'features': features.flatten(),
                        'score': score,
                        'params': candidate_params.copy()
                    })
                    
                    # Aggiorna best
                    if score > best_score:
                        best_score = score
                        best_params = candidate_params.copy()
                        self.logger.info(f"Nuovo best score: {score:.4f} (iter {iteration})")
                    
                except Exception as e:
                    self.logger.warning(f"Errore nel backtest iter {iteration}: {e}")
                    continue
            
            # Salva risultati
            self.best_parameters = best_params
            self.best_score = best_score
            
            # Aggiungi alla storia
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'best_params': best_params,
                'best_score': best_score,
                'iterations': n_iterations
            })
            
            self.logger.info(f"Ottimizzazione completata. Best score: {best_score:.4f}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Errore nell'ottimizzazione: {e}")
            return current_params
    
    def _generate_random_parameters(self) -> Dict:
        """Genera parametri casuali nei range definiti."""
        params = {}
        
        for param_name, (min_val, max_val) in self.parameter_ranges.items():
            params[param_name] = np.random.uniform(min_val, max_val)
        
        return params
    
    def _generate_ml_parameters(self, market_data: pd.DataFrame,
                               training_data: List[Dict]) -> Dict:
        """
        Genera parametri usando predizioni ML.
        
        Args:
            market_data: Dati di mercato attuali
            training_data: Dati di training accumulati
            
        Returns:
            Parametri predetti dal modello
        """
        try:
            if len(training_data) < 10:
                return self._generate_random_parameters()
            
            # Prepara dati di training
            X = np.array([data['features'] for data in training_data])
            y = np.array([data['score'] for data in training_data])
            
            # Normalizza features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train modello
            model = self.models['random_forest']
            model.fit(X_scaled, y)
            
            # Genera candidati e predici
            best_predicted_params = None
            best_predicted_score = -np.inf
            
            for _ in range(20):  # Prova 20 candidati
                candidate_params = self._generate_random_parameters()
                features = self.generate_features(market_data, candidate_params)
                features_scaled = self.scaler.transform(features)
                
                predicted_score = model.predict(features_scaled)[0]
                
                if predicted_score > best_predicted_score:
                    best_predicted_score = predicted_score
                    best_predicted_params = candidate_params
            
            return best_predicted_params or self._generate_random_parameters()
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione ML parametri: {e}")
            return self._generate_random_parameters()
    
    def adaptive_parameter_tuning(self, market_data: pd.DataFrame,
                                 current_params: Dict,
                                 recent_performance: Dict) -> Dict:
        """
        Tuning adattivo dei parametri basato su performance recenti.
        
        Args:
            market_data: Dati di mercato recenti
            current_params: Parametri attuali
            recent_performance: Performance recenti
            
        Returns:
            Parametri aggiustati
        """
        try:
            adjusted_params = current_params.copy()
            
            # Analizza performance
            win_rate = recent_performance.get('win_rate', 0.5)
            avg_return = recent_performance.get('avg_return', 0.0)
            volatility = recent_performance.get('volatility', 0.0)
            
            # Aggiustamenti adattivi
            if win_rate < 0.4:  # Win rate basso
                # Aumenta conservatività
                adjusted_params['max_position_size'] *= 0.9
                adjusted_params['atr_mult_sl'] *= 1.1  # Stop loss più stretto
                
            elif win_rate > 0.7:  # Win rate alto
                # Aumenta aggressività
                adjusted_params['max_position_size'] *= 1.05
                adjusted_params['atr_mult_tp'] *= 1.1  # Take profit più ampio
            
            if volatility > 0.3:  # Alta volatilità
                # Parametri più conservativi
                adjusted_params['grid_spacing'] *= 1.2
                adjusted_params['volatility_threshold'] *= 1.1
                
            elif volatility < 0.1:  # Bassa volatilità
                # Parametri più aggressivi
                adjusted_params['grid_spacing'] *= 0.9
                adjusted_params['volatility_threshold'] *= 0.9
            
            # Assicura che i parametri rimangano nei range
            for param_name, value in adjusted_params.items():
                if param_name in self.parameter_ranges:
                    min_val, max_val = self.parameter_ranges[param_name]
                    adjusted_params[param_name] = np.clip(value, min_val, max_val)
            
            self.logger.info("Parametri aggiustati adattivamente")
            
            return adjusted_params
            
        except Exception as e:
            self.logger.error(f"Errore nel tuning adattivo: {e}")
            return current_params
    
    def get_parameter_importance(self, training_data: List[Dict]) -> Dict:
        """
        Calcola l'importanza dei parametri usando feature importance.
        
        Args:
            training_data: Dati di training
            
        Returns:
            Dizionario con importanza dei parametri
        """
        try:
            if len(training_data) < 10:
                return {}
            
            # Prepara dati
            X = np.array([data['features'] for data in training_data])
            y = np.array([data['score'] for data in training_data])
            
            # Train Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Feature importance
            importance = model.feature_importances_
            
            # Mappa alle feature names
            feature_names = list(self.parameter_ranges.keys()) + [
                'volatility', 'trend_strength', 'volume_ratio', 
                'atr_normalized', 'rsi', 'bb_position',
                'hour_of_day', 'day_of_week'
            ]
            
            importance_dict = {}
            for i, name in enumerate(feature_names[:len(importance)]):
                importance_dict[name] = importance[i]
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo importanza: {e}")
            return {}
    
    def save_optimization_results(self, filepath: str):
        """Salva i risultati dell'ottimizzazione su file."""
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
            
            self.logger.info(f"Risultati salvati in {filepath}")
            
        except Exception as e:
            self.logger.error(f"Errore nel salvataggio: {e}")
    
    def load_optimization_results(self, filepath: str):
        """Carica i risultati dell'ottimizzazione da file."""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.best_parameters = results.get('best_parameters')
            self.best_score = results.get('best_score', -np.inf)
            
            # Ricostruisci history
            self.optimization_history = []
            for hist in results.get('optimization_history', []):
                hist_item = hist.copy()
                hist_item['timestamp'] = datetime.fromisoformat(hist['timestamp'])
                self.optimization_history.append(hist_item)
            
            self.logger.info(f"Risultati caricati da {filepath}")
            
        except Exception as e:
            self.logger.error(f"Errore nel caricamento: {e}")
    
    def get_optimization_report(self) -> Dict:
        """Genera un report completo dell'ottimizzazione."""
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
            
            # Calcola miglioramento
            if len(self.optimization_history) > 1:
                first_score = self.optimization_history[0]['best_score']
                latest_score = latest['best_score']
                improvement = ((latest_score - first_score) / abs(first_score)) * 100
                report['score_improvement'] = improvement
            
            return report
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione report: {e}")
            return {'status': 'error', 'message': str(e)}

