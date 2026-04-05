"""
Risk Management Module per NOCTURNA v2.0 Trading Bot
Implementa controlli avanzati di gestione del rischio e protezione del capitale.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

class RiskLevel(Enum):
    """Livelli di rischio del sistema."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskEvent(Enum):
    """Tipi di eventi di rischio."""
    POSITION_LIMIT_EXCEEDED = "POSITION_LIMIT_EXCEEDED"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    CORRELATION_RISK = "CORRELATION_RISK"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    MARGIN_CALL = "MARGIN_CALL"
    SYSTEM_ERROR = "SYSTEM_ERROR"

class RiskManager:
    """
    Gestisce tutti gli aspetti del risk management per il trading bot.
    Implementa controlli pre-trade, monitoraggio continuo e azioni correttive.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Parametri di rischio
        self.risk_parameters = self._load_risk_parameters()
        
        # Stato del rischio
        self.current_risk_level = RiskLevel.LOW
        self.risk_events = []
        self.daily_stats = {}
        
        # Limiti dinamici
        self.dynamic_limits = {}
        
        # Portfolio tracking
        self.portfolio_value = 0.0
        self.max_portfolio_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Correlazioni
        self.correlation_matrix = {}
        self.position_correlations = {}
        
        # Volatilità
        self.volatility_cache = {}
        
        self.logger.info("Risk Manager inizializzato")
    
    def _load_risk_parameters(self) -> Dict:
        """Carica i parametri di risk management."""
        default_params = {
            # Limiti di posizione
            'max_position_size': 0.20,          # 20% del capitale per posizione
            'max_portfolio_exposure': 0.80,     # 80% esposizione massima
            'max_sector_exposure': 0.30,        # 30% per settore
            'max_correlation_exposure': 0.50,   # 50% in posizioni correlate
            
            # Limiti di perdita
            'max_daily_loss': 0.05,             # 5% perdita giornaliera
            'max_weekly_loss': 0.10,            # 10% perdita settimanale
            'max_monthly_loss': 0.20,           # 20% perdita mensile
            'max_drawdown': 0.15,               # 15% drawdown massimo
            
            # Volatilità
            'volatility_threshold': 2.0,        # Soglia volatilità (multipli ATR)
            'volatility_lookback': 20,          # Giorni per calcolo volatilità
            'volatility_adjustment': True,      # Aggiustamento automatico
            
            # Correlazione
            'correlation_threshold': 0.7,       # Soglia correlazione
            'correlation_lookback': 60,         # Giorni per calcolo correlazione
            
            # Stop loss dinamici
            'dynamic_stop_loss': True,
            'stop_loss_multiplier': 2.0,        # Multiplo ATR per stop loss
            'trailing_stop_activation': 0.02,   # 2% profitto per attivare trailing
            
            # Dimensionamento posizioni
            'position_sizing_method': 'kelly',  # kelly, fixed, volatility
            'kelly_fraction': 0.25,             # Frazione Kelly conservativa
            'volatility_target': 0.15,          # Target volatilità portfolio
            
            # Limiti temporali
            'max_trades_per_day': 50,
            'max_trades_per_hour': 10,
            'cooldown_period': 300,             # 5 minuti tra trade simili
            
            # Emergency stops
            'emergency_stop_loss': 0.10,        # 10% perdita per stop emergenza
            'system_halt_conditions': [
                'CRITICAL_DRAWDOWN',
                'SYSTEM_ERROR',
                'MARGIN_CALL'
            ]
        }
        
        # Merge con configurazione
        params = default_params.copy()
        params.update(self.config.get('risk_parameters', {}))
        
        return params
    
    def validate_trade(self, signal: Dict, current_positions: Dict, 
                      market_data: Dict) -> Tuple[bool, str, Dict]:
        """
        Valida un segnale di trading contro tutti i controlli di rischio.
        
        Args:
            signal: Segnale di trading da validare
            current_positions: Posizioni attuali del portfolio
            market_data: Dati di mercato correnti
            
        Returns:
            Tuple (is_valid, reason, adjusted_signal)
        """
        try:
            # Copia del segnale per modifiche
            adjusted_signal = signal.copy()
            
            # 1. Controlli base
            if not self._validate_basic_constraints(signal):
                return False, "Vincoli base non rispettati", signal
            
            # 2. Controllo limiti di posizione
            valid, reason = self._check_position_limits(signal, current_positions)
            if not valid:
                return False, reason, signal
            
            # 3. Controllo esposizione portfolio
            valid, reason = self._check_portfolio_exposure(signal, current_positions)
            if not valid:
                return False, reason, signal
            
            # 4. Controllo correlazioni
            valid, reason = self._check_correlation_risk(signal, current_positions, market_data)
            if not valid:
                return False, reason, signal
            
            # 5. Controllo volatilità
            valid, reason = self._check_volatility_risk(signal, market_data)
            if not valid:
                return False, reason, signal
            
            # 6. Controllo limiti temporali
            valid, reason = self._check_temporal_limits(signal)
            if not valid:
                return False, reason, signal
            
            # 7. Aggiustamento dimensione posizione
            adjusted_signal = self._adjust_position_size(adjusted_signal, current_positions, market_data)
            
            # 8. Aggiustamento stop loss e take profit
            adjusted_signal = self._adjust_risk_levels(adjusted_signal, market_data)
            
            return True, "Trade validato", adjusted_signal
            
        except Exception as e:
            self.logger.error(f"Errore nella validazione trade: {e}")
            return False, f"Errore validazione: {str(e)}", signal
    
    def _validate_basic_constraints(self, signal: Dict) -> bool:
        """Valida vincoli base del segnale."""
        try:
            # Controllo campi obbligatori
            required_fields = ['symbol', 'side', 'quantity', 'type']
            for field in required_fields:
                if field not in signal:
                    self.logger.error(f"Campo mancante: {field}")
                    return False
            
            # Controllo valori
            if signal['quantity'] <= 0:
                self.logger.error("Quantità deve essere positiva")
                return False
            
            if signal['side'] not in ['buy', 'sell']:
                self.logger.error("Side deve essere buy o sell")
                return False
            
            # Controllo dimensione minima/massima
            min_quantity = self.risk_parameters.get('min_trade_size', 0.001)
            max_quantity = self.risk_parameters.get('max_trade_size', 1.0)
            
            if signal['quantity'] < min_quantity or signal['quantity'] > max_quantity:
                self.logger.error(f"Quantità fuori range: {signal['quantity']}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore validazione base: {e}")
            return False
    
    def _check_position_limits(self, signal: Dict, positions: Dict) -> Tuple[bool, str]:
        """Controlla i limiti di posizione."""
        try:
            symbol = signal['symbol']
            quantity = signal['quantity']
            side = signal['side']
            
            # Posizione corrente
            current_position = positions.get(symbol, {}).get('quantity', 0)
            
            # Calcola nuova posizione
            if side == 'buy':
                new_position = current_position + quantity
            else:
                new_position = current_position - quantity
            
            # Controllo limite per simbolo
            max_position = self.risk_parameters['max_position_size']
            if abs(new_position) > max_position:
                return False, f"Limite posizione superato per {symbol}: {abs(new_position)} > {max_position}"
            
            return True, "Limiti posizione OK"
            
        except Exception as e:
            self.logger.error(f"Errore controllo limiti posizione: {e}")
            return False, "Errore controllo limiti"
    
    def _check_portfolio_exposure(self, signal: Dict, positions: Dict) -> Tuple[bool, str]:
        """Controlla l'esposizione totale del portfolio."""
        try:
            # Calcola esposizione corrente
            total_exposure = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
            
            # Stima valore del nuovo trade
            estimated_value = signal['quantity'] * signal.get('price', 0)
            if not estimated_value:
                # Usa prezzo di mercato approssimativo
                estimated_value = signal['quantity'] * 100  # Placeholder
            
            new_total_exposure = total_exposure + estimated_value
            
            # Controllo limite esposizione
            max_exposure = self.portfolio_value * self.risk_parameters['max_portfolio_exposure']
            
            if new_total_exposure > max_exposure:
                return False, f"Limite esposizione portfolio superato: {new_total_exposure} > {max_exposure}"
            
            return True, "Esposizione portfolio OK"
            
        except Exception as e:
            self.logger.error(f"Errore controllo esposizione: {e}")
            return False, "Errore controllo esposizione"
    
    def _check_correlation_risk(self, signal: Dict, positions: Dict, 
                               market_data: Dict) -> Tuple[bool, str]:
        """Controlla il rischio di correlazione."""
        try:
            symbol = signal['symbol']
            
            # Calcola correlazioni con posizioni esistenti
            high_correlation_exposure = 0.0
            
            for pos_symbol, position in positions.items():
                if pos_symbol == symbol:
                    continue
                
                # Recupera correlazione
                correlation = self._get_correlation(symbol, pos_symbol, market_data)
                
                if abs(correlation) > self.risk_parameters['correlation_threshold']:
                    high_correlation_exposure += abs(position.get('market_value', 0))
            
            # Aggiungi valore del nuovo trade
            estimated_value = signal['quantity'] * signal.get('price', 100)
            total_correlated_exposure = high_correlation_exposure + estimated_value
            
            # Controllo limite
            max_correlated = self.portfolio_value * self.risk_parameters['max_correlation_exposure']
            
            if total_correlated_exposure > max_correlated:
                return False, f"Limite correlazione superato: {total_correlated_exposure} > {max_correlated}"
            
            return True, "Rischio correlazione OK"
            
        except Exception as e:
            self.logger.error(f"Errore controllo correlazione: {e}")
            return True, "Controllo correlazione saltato"  # Non bloccare per errori
    
    def _check_volatility_risk(self, signal: Dict, market_data: Dict) -> Tuple[bool, str]:
        """Controlla il rischio di volatilità."""
        try:
            symbol = signal['symbol']
            
            # Recupera dati di volatilità
            volatility_data = market_data.get(symbol, {})
            current_atr = volatility_data.get('atr', 0)
            avg_atr = volatility_data.get('avg_atr', current_atr)
            
            if current_atr == 0 or avg_atr == 0:
                return True, "Dati volatilità non disponibili"
            
            # Calcola spike di volatilità
            volatility_ratio = current_atr / avg_atr
            threshold = self.risk_parameters['volatility_threshold']
            
            if volatility_ratio > threshold:
                return False, f"Volatilità troppo alta: {volatility_ratio:.2f} > {threshold}"
            
            return True, "Volatilità OK"
            
        except Exception as e:
            self.logger.error(f"Errore controllo volatilità: {e}")
            return True, "Controllo volatilità saltato"
    
    def _check_temporal_limits(self, signal: Dict) -> Tuple[bool, str]:
        """Controlla i limiti temporali di trading."""
        try:
            now = datetime.now()
            
            # Controllo numero trade giornalieri
            today_trades = len([t for t in self.daily_stats.get('trades', [])
                              if t.get('timestamp', datetime.min).date() == now.date()])
            
            max_daily = self.risk_parameters['max_trades_per_day']
            if today_trades >= max_daily:
                return False, f"Limite trade giornalieri raggiunto: {today_trades} >= {max_daily}"
            
            # Controllo numero trade orari
            hour_ago = now - timedelta(hours=1)
            hour_trades = len([t for t in self.daily_stats.get('trades', [])
                             if t.get('timestamp', datetime.min) > hour_ago])
            
            max_hourly = self.risk_parameters['max_trades_per_hour']
            if hour_trades >= max_hourly:
                return False, f"Limite trade orari raggiunto: {hour_trades} >= {max_hourly}"
            
            # Controllo cooldown
            symbol = signal['symbol']
            last_trade = self._get_last_trade_time(symbol)
            cooldown = self.risk_parameters['cooldown_period']
            
            if last_trade and (now - last_trade).total_seconds() < cooldown:
                return False, f"Periodo cooldown attivo per {symbol}"
            
            return True, "Limiti temporali OK"
            
        except Exception as e:
            self.logger.error(f"Errore controllo temporale: {e}")
            return True, "Controllo temporale saltato"
    
    def _adjust_position_size(self, signal: Dict, positions: Dict, 
                             market_data: Dict) -> Dict:
        """Aggiusta la dimensione della posizione basandosi sul metodo configurato."""
        try:
            method = self.risk_parameters['position_sizing_method']
            
            if method == 'kelly':
                signal = self._kelly_position_sizing(signal, market_data)
            elif method == 'volatility':
                signal = self._volatility_position_sizing(signal, market_data)
            elif method == 'fixed':
                # Mantieni dimensione originale
                pass
            
            # Applica limiti massimi
            max_size = self.risk_parameters['max_position_size']
            if signal['quantity'] > max_size:
                signal['quantity'] = max_size
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Errore aggiustamento posizione: {e}")
            return signal
    
    def _kelly_position_sizing(self, signal: Dict, market_data: Dict) -> Dict:
        """Calcola dimensione posizione usando criterio di Kelly."""
        try:
            symbol = signal['symbol']
            
            # Parametri per Kelly
            win_rate = self._estimate_win_rate(symbol)
            avg_win = self._estimate_avg_win(symbol)
            avg_loss = self._estimate_avg_loss(symbol)
            
            if avg_loss == 0:
                return signal
            
            # Formula Kelly: f = (bp - q) / b
            # dove b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Applica frazione conservativa
            conservative_fraction = kelly_fraction * self.risk_parameters['kelly_fraction']
            conservative_fraction = max(0, min(conservative_fraction, self.risk_parameters['max_position_size']))
            
            signal['quantity'] = conservative_fraction
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Errore Kelly sizing: {e}")
            return signal
    
    def _volatility_position_sizing(self, signal: Dict, market_data: Dict) -> Dict:
        """Calcola dimensione posizione basandosi sulla volatilità."""
        try:
            symbol = signal['symbol']
            volatility_data = market_data.get(symbol, {})
            current_volatility = volatility_data.get('volatility', 0.2)  # Default 20%
            
            target_volatility = self.risk_parameters['volatility_target']
            
            # Aggiusta dimensione inversamente alla volatilità
            volatility_adjustment = target_volatility / current_volatility
            adjusted_quantity = signal['quantity'] * volatility_adjustment
            
            # Applica limiti
            max_size = self.risk_parameters['max_position_size']
            adjusted_quantity = min(adjusted_quantity, max_size)
            
            signal['quantity'] = adjusted_quantity
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Errore volatility sizing: {e}")
            return signal
    
    def _adjust_risk_levels(self, signal: Dict, market_data: Dict) -> Dict:
        """Aggiusta stop loss e take profit dinamicamente."""
        try:
            if not self.risk_parameters['dynamic_stop_loss']:
                return signal
            
            symbol = signal['symbol']
            volatility_data = market_data.get(symbol, {})
            atr = volatility_data.get('atr', 0)
            current_price = volatility_data.get('price', signal.get('price', 0))
            
            if atr == 0 or current_price == 0:
                return signal
            
            # Calcola stop loss dinamico
            stop_multiplier = self.risk_parameters['stop_loss_multiplier']
            
            if signal['side'] == 'buy':
                dynamic_stop = current_price - (atr * stop_multiplier)
                dynamic_tp = current_price + (atr * stop_multiplier * 1.5)  # R:R 1:1.5
            else:
                dynamic_stop = current_price + (atr * stop_multiplier)
                dynamic_tp = current_price - (atr * stop_multiplier * 1.5)
            
            # Aggiorna solo se non già specificati o se migliori
            if 'stop_loss' not in signal or signal['stop_loss'] == 0:
                signal['stop_loss'] = dynamic_stop
            
            if 'take_profit' not in signal or signal['take_profit'] == 0:
                signal['take_profit'] = dynamic_tp
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Errore aggiustamento risk levels: {e}")
            return signal
    
    def monitor_portfolio_risk(self, positions: Dict, market_data: Dict) -> Dict:
        """Monitora continuamente il rischio del portfolio."""
        try:
            risk_metrics = {}
            
            # Calcola metriche di rischio
            risk_metrics['total_exposure'] = self._calculate_total_exposure(positions)
            risk_metrics['current_drawdown'] = self._calculate_current_drawdown(positions)
            risk_metrics['var_95'] = self._calculate_var(positions, market_data, 0.95)
            risk_metrics['portfolio_volatility'] = self._calculate_portfolio_volatility(positions, market_data)
            risk_metrics['correlation_risk'] = self._calculate_correlation_risk(positions, market_data)
            risk_metrics['concentration_risk'] = self._calculate_concentration_risk(positions)
            
            # Determina livello di rischio
            risk_level = self._assess_risk_level(risk_metrics)
            
            # Genera eventi di rischio se necessario
            risk_events = self._check_risk_events(risk_metrics, positions)
            
            # Aggiorna stato
            self.current_risk_level = risk_level
            self.risk_events.extend(risk_events)
            
            return {
                'risk_level': risk_level.value,
                'risk_metrics': risk_metrics,
                'risk_events': [event.value for event in risk_events],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Errore monitoraggio rischio: {e}")
            return {'error': str(e)}
    
    def _calculate_total_exposure(self, positions: Dict) -> float:
        """Calcola l'esposizione totale del portfolio."""
        return sum(abs(pos.get('market_value', 0)) for pos in positions.values())
    
    def _calculate_current_drawdown(self, positions: Dict) -> float:
        """Calcola il drawdown corrente."""
        current_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if self.max_portfolio_value == 0:
            self.max_portfolio_value = current_value
        
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        return self.current_drawdown
    
    def _calculate_var(self, positions: Dict, market_data: Dict, confidence: float) -> float:
        """Calcola Value at Risk del portfolio."""
        try:
            # Implementazione semplificata del VaR
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            if total_value == 0:
                return 0.0
            
            # Usa volatilità media ponderata
            weighted_volatility = 0.0
            for symbol, position in positions.items():
                weight = abs(position.get('market_value', 0)) / total_value
                volatility = market_data.get(symbol, {}).get('volatility', 0.2)
                weighted_volatility += weight * volatility
            
            # VaR parametrico (assumendo distribuzione normale)
            from scipy.stats import norm
            z_score = norm.ppf(1 - confidence)
            var = total_value * weighted_volatility * z_score
            
            return abs(var)
            
        except Exception as e:
            self.logger.error(f"Errore calcolo VaR: {e}")
            return 0.0
    
    def _calculate_portfolio_volatility(self, positions: Dict, market_data: Dict) -> float:
        """Calcola la volatilità del portfolio."""
        try:
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            if total_value == 0:
                return 0.0
            
            # Volatilità ponderata semplificata
            weighted_volatility = 0.0
            for symbol, position in positions.items():
                weight = abs(position.get('market_value', 0)) / total_value
                volatility = market_data.get(symbol, {}).get('volatility', 0.2)
                weighted_volatility += weight * volatility
            
            return weighted_volatility
            
        except Exception as e:
            self.logger.error(f"Errore calcolo volatilità portfolio: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self, positions: Dict, market_data: Dict) -> float:
        """Calcola il rischio di correlazione del portfolio."""
        try:
            symbols = list(positions.keys())
            if len(symbols) < 2:
                return 0.0
            
            # Calcola correlazioni medie
            total_correlations = 0
            count = 0
            
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    correlation = self._get_correlation(symbol1, symbol2, market_data)
                    total_correlations += abs(correlation)
                    count += 1
            
            return total_correlations / count if count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Errore calcolo rischio correlazione: {e}")
            return 0.0
    
    def _calculate_concentration_risk(self, positions: Dict) -> float:
        """Calcola il rischio di concentrazione."""
        try:
            total_value = sum(abs(pos.get('market_value', 0)) for pos in positions.values())
            
            if total_value == 0:
                return 0.0
            
            # Calcola indice Herfindahl
            herfindahl_index = sum((abs(pos.get('market_value', 0)) / total_value) ** 2 
                                 for pos in positions.values())
            
            return herfindahl_index
            
        except Exception as e:
            self.logger.error(f"Errore calcolo concentrazione: {e}")
            return 0.0
    
    def _assess_risk_level(self, metrics: Dict) -> RiskLevel:
        """Valuta il livello di rischio complessivo."""
        try:
            risk_score = 0
            
            # Drawdown
            if metrics.get('current_drawdown', 0) > 0.10:
                risk_score += 3
            elif metrics.get('current_drawdown', 0) > 0.05:
                risk_score += 2
            elif metrics.get('current_drawdown', 0) > 0.02:
                risk_score += 1
            
            # Volatilità
            if metrics.get('portfolio_volatility', 0) > 0.30:
                risk_score += 3
            elif metrics.get('portfolio_volatility', 0) > 0.20:
                risk_score += 2
            elif metrics.get('portfolio_volatility', 0) > 0.15:
                risk_score += 1
            
            # Concentrazione
            if metrics.get('concentration_risk', 0) > 0.5:
                risk_score += 2
            elif metrics.get('concentration_risk', 0) > 0.3:
                risk_score += 1
            
            # Correlazione
            if metrics.get('correlation_risk', 0) > 0.7:
                risk_score += 2
            elif metrics.get('correlation_risk', 0) > 0.5:
                risk_score += 1
            
            # Determina livello
            if risk_score >= 8:
                return RiskLevel.CRITICAL
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Errore valutazione rischio: {e}")
            return RiskLevel.MEDIUM
    
    def _check_risk_events(self, metrics: Dict, positions: Dict) -> List[RiskEvent]:
        """Controlla eventi di rischio specifici."""
        events = []
        
        try:
            # Drawdown critico
            if metrics.get('current_drawdown', 0) > self.risk_parameters['max_drawdown']:
                events.append(RiskEvent.DRAWDOWN_LIMIT)
            
            # Perdita giornaliera
            daily_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
            if daily_pnl < -self.risk_parameters['max_daily_loss'] * self.portfolio_value:
                events.append(RiskEvent.DAILY_LOSS_LIMIT)
            
            # Concentrazione eccessiva
            if metrics.get('concentration_risk', 0) > 0.6:
                events.append(RiskEvent.POSITION_LIMIT_EXCEEDED)
            
            # Correlazione alta
            if metrics.get('correlation_risk', 0) > self.risk_parameters['correlation_threshold']:
                events.append(RiskEvent.CORRELATION_RISK)
            
            # Volatilità estrema
            if metrics.get('portfolio_volatility', 0) > 0.4:
                events.append(RiskEvent.VOLATILITY_SPIKE)
            
        except Exception as e:
            self.logger.error(f"Errore controllo eventi rischio: {e}")
            events.append(RiskEvent.SYSTEM_ERROR)
        
        return events
    
    def _get_correlation(self, symbol1: str, symbol2: str, market_data: Dict) -> float:
        """Recupera o calcola la correlazione tra due simboli."""
        try:
            # Controlla cache
            key = f"{symbol1}_{symbol2}"
            if key in self.correlation_matrix:
                return self.correlation_matrix[key]
            
            # Calcolo semplificato (in produzione usare dati storici)
            # Per ora restituisce correlazione casuale
            import random
            correlation = random.uniform(-0.5, 0.8)
            
            # Cache risultato
            self.correlation_matrix[key] = correlation
            self.correlation_matrix[f"{symbol2}_{symbol1}"] = correlation
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Errore calcolo correlazione: {e}")
            return 0.0
    
    def _estimate_win_rate(self, symbol: str) -> float:
        """Stima il win rate per un simbolo."""
        # Implementazione semplificata
        return 0.55  # 55% win rate di default
    
    def _estimate_avg_win(self, symbol: str) -> float:
        """Stima il guadagno medio per un simbolo."""
        return 0.02  # 2% guadagno medio
    
    def _estimate_avg_loss(self, symbol: str) -> float:
        """Stima la perdita media per un simbolo."""
        return 0.015  # 1.5% perdita media
    
    def _get_last_trade_time(self, symbol: str) -> Optional[datetime]:
        """Recupera l'ora dell'ultimo trade per un simbolo."""
        trades = self.daily_stats.get('trades', [])
        symbol_trades = [t for t in trades if t.get('symbol') == symbol]
        
        if symbol_trades:
            return max(t.get('timestamp', datetime.min) for t in symbol_trades)
        
        return None
    
    def get_risk_report(self) -> Dict:
        """Genera un report completo del rischio."""
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
            'timestamp': datetime.now()
        }

