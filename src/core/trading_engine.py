"""
Core Trading Engine per NOCTURNA v2.0 Trading Bot
Coordina tutti i moduli e implementa la logica principale di trading.
"""

import asyncio
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from .market_data import MarketDataHandler
from .strategy_manager import StrategyManager, TradingMode, MarketState
from .order_manager import OrderExecutionManager
from .risk_manager import RiskManager, RiskLevel

class TradingEngineState:
    """Stati del trading engine."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class TradingEngine:
    """
    Core Trading Engine che coordina tutti i componenti del sistema NOCTURNA v2.0.
    Implementa la logica principale di trading e gestisce il ciclo di vita del bot.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Stato del sistema
        self.state = TradingEngineState.STOPPED
        self.start_time = None
        self.last_update = None
        
        # Componenti principali
        self.market_data = MarketDataHandler(config.get('market_data', {}))
        self.strategy_manager = StrategyManager(config.get('strategy', {}))
        self.order_manager = OrderExecutionManager(config.get('trading', {}))
        self.risk_manager = RiskManager(config.get('risk', {}))
        
        # Configurazione trading
        self.symbols = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
        self.update_interval = config.get('update_interval', 60)  # secondi
        self.max_concurrent_analysis = config.get('max_concurrent_analysis', 5)
        
        # Threading
        self.main_thread = None
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_analysis)
        self.running = False
        
        # Callbacks e eventi
        self.event_callbacks = {}
        self.performance_metrics = {}
        
        # Cache e stato
        self.symbol_data = {}
        self.active_signals = {}
        self.performance_history = []
        
        # Statistiche
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'start_time': None,
            'uptime': 0
        }
        
        self._setup_callbacks()
        self.logger.info("Trading Engine inizializzato")
    
    def _setup_callbacks(self):
        """Configura i callback tra i componenti."""
        # Callback per ordini completati
        self.order_manager.add_order_callback(self._on_order_filled)
        
        # Callback per aggiornamenti posizioni
        self.order_manager.add_position_callback(self._on_position_update)
    
    def start(self) -> bool:
        """
        Avvia il trading engine.
        
        Returns:
            True se avviato con successo
        """
        try:
            if self.state != TradingEngineState.STOPPED:
                self.logger.warning(f"Engine già in stato: {self.state}")
                return False
            
            self.logger.info("Avvio Trading Engine...")
            self.state = TradingEngineState.STARTING
            
            # Avvia componenti
            self._start_components()
            
            # Avvia thread principale
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop)
            self.main_thread.daemon = True
            self.main_thread.start()
            
            # Aggiorna stato
            self.state = TradingEngineState.RUNNING
            self.start_time = datetime.now()
            self.stats['start_time'] = self.start_time
            
            self.logger.info("Trading Engine avviato con successo")
            self._notify_event('engine_started', {'timestamp': self.start_time})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'avvio del Trading Engine: {e}")
            self.state = TradingEngineState.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Ferma il trading engine.
        
        Returns:
            True se fermato con successo
        """
        try:
            self.logger.info("Arresto Trading Engine...")
            
            # Ferma il loop principale
            self.running = False
            
            # Aspetta che il thread principale termini
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=30)
            
            # Ferma componenti
            self._stop_components()
            
            # Aggiorna stato
            self.state = TradingEngineState.STOPPED
            
            self.logger.info("Trading Engine fermato")
            self._notify_event('engine_stopped', {'timestamp': datetime.now()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'arresto del Trading Engine: {e}")
            return False
    
    def pause(self) -> bool:
        """Mette in pausa il trading engine."""
        try:
            if self.state != TradingEngineState.RUNNING:
                return False
            
            self.state = TradingEngineState.PAUSED
            self.logger.info("Trading Engine in pausa")
            self._notify_event('engine_paused', {'timestamp': datetime.now()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella pausa: {e}")
            return False
    
    def resume(self) -> bool:
        """Riprende il trading engine dalla pausa."""
        try:
            if self.state != TradingEngineState.PAUSED:
                return False
            
            self.state = TradingEngineState.RUNNING
            self.logger.info("Trading Engine ripreso")
            self._notify_event('engine_resumed', {'timestamp': datetime.now()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella ripresa: {e}")
            return False
    
    def emergency_stop(self, reason: str = "Emergency stop triggered"):
        """Arresto di emergenza del sistema."""
        try:
            self.logger.critical(f"ARRESTO DI EMERGENZA: {reason}")
            
            # Cancella tutti gli ordini attivi
            self._cancel_all_orders()
            
            # Chiudi tutte le posizioni (opzionale, configurabile)
            if self.config.get('emergency_close_positions', False):
                self._close_all_positions()
            
            # Ferma il sistema
            self.state = TradingEngineState.EMERGENCY_STOP
            self.running = False
            
            self._notify_event('emergency_stop', {
                'reason': reason,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Errore nell'arresto di emergenza: {e}")
    
    def _start_components(self):
        """Avvia tutti i componenti del sistema."""
        try:
            # Avvia market data feed
            self.market_data.start_real_time_feed()
            
            # Avvia monitoraggio ordini
            self.order_manager.start_monitoring()
            
            # Sottoscrivi ai simboli
            for symbol in self.symbols:
                self.market_data.subscribe_to_symbol(symbol, self._on_price_update)
            
            self.logger.info("Componenti avviati")
            
        except Exception as e:
            self.logger.error(f"Errore nell'avvio componenti: {e}")
            raise
    
    def _stop_components(self):
        """Ferma tutti i componenti del sistema."""
        try:
            # Ferma market data feed
            self.market_data.stop_real_time_feed()
            
            # Ferma monitoraggio ordini
            self.order_manager.stop_monitoring()
            
            # Chiudi executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("Componenti fermati")
            
        except Exception as e:
            self.logger.error(f"Errore nell'arresto componenti: {e}")
    
    def _main_loop(self):
        """Loop principale del trading engine."""
        self.logger.info("Loop principale avviato")
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Controlla stato
                if self.state == TradingEngineState.PAUSED:
                    time.sleep(1)
                    continue
                
                if self.state == TradingEngineState.EMERGENCY_STOP:
                    break
                
                # Aggiorna dati e analizza mercato
                self._update_market_analysis()
                
                # Monitora rischio
                self._monitor_risk()
                
                # Aggiorna statistiche
                self._update_statistics()
                
                # Aggiorna timestamp
                self.last_update = datetime.now()
                
                # Calcola tempo di sleep
                loop_time = time.time() - loop_start
                sleep_time = max(0, self.update_interval - loop_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Errore nel loop principale: {e}")
                time.sleep(10)  # Pausa prima di riprovare
        
        self.logger.info("Loop principale terminato")
    
    def _update_market_analysis(self):
        """Aggiorna l'analisi di mercato per tutti i simboli."""
        try:
            # Analizza simboli in parallelo
            futures = []
            
            for symbol in self.symbols:
                future = self.executor.submit(self._analyze_symbol, symbol)
                futures.append((symbol, future))
            
            # Raccoglie risultati
            for symbol, future in futures:
                try:
                    result = future.result(timeout=30)
                    if result:
                        self._process_analysis_result(symbol, result)
                        
                except Exception as e:
                    self.logger.error(f"Errore nell'analisi di {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento analisi: {e}")
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analizza un singolo simbolo."""
        try:
            # Recupera dati storici
            df = self.market_data.get_historical_data(
                symbol=symbol,
                timeframe='1h',
                limit=500
            )
            
            if df.empty:
                return None
            
            # Calcola indicatori tecnici
            df = self.market_data.calculate_technical_indicators(df)
            
            # Aggiorna strategia
            strategy_result = self.strategy_manager.update_strategy(df, symbol)
            
            # Cache dati
            self.symbol_data[symbol] = {
                'data': df,
                'strategy_result': strategy_result,
                'last_update': datetime.now()
            }
            
            return strategy_result
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi di {symbol}: {e}")
            return None
    
    def _process_analysis_result(self, symbol: str, result: Dict):
        """Processa il risultato dell'analisi di un simbolo."""
        try:
            signals = result.get('signals', [])
            
            for signal in signals:
                self._process_trading_signal(signal)
            
            # Aggiorna metriche
            self._update_symbol_metrics(symbol, result)
            
        except Exception as e:
            self.logger.error(f"Errore nel processing risultato per {symbol}: {e}")
    
    def _process_trading_signal(self, signal: Dict):
        """Processa un segnale di trading."""
        try:
            symbol = signal['symbol']
            
            # Recupera posizioni correnti
            positions = self.order_manager.get_positions()
            
            # Recupera dati di mercato
            market_data = self._get_market_data_for_risk(symbol)
            
            # Valida con risk manager
            is_valid, reason, adjusted_signal = self.risk_manager.validate_trade(
                signal, positions, market_data
            )
            
            if not is_valid:
                self.logger.info(f"Segnale rifiutato per {symbol}: {reason}")
                return
            
            # Invia ordine
            order_id = self.order_manager.submit_order(adjusted_signal)
            
            if order_id:
                self.active_signals[order_id] = adjusted_signal
                self.logger.info(f"Ordine inviato: {order_id} per {symbol}")
                
                self._notify_event('signal_executed', {
                    'signal': adjusted_signal,
                    'order_id': order_id,
                    'timestamp': datetime.now()
                })
            
        except Exception as e:
            self.logger.error(f"Errore nel processing segnale: {e}")
    
    def _monitor_risk(self):
        """Monitora il rischio del portfolio."""
        try:
            positions = self.order_manager.get_positions()
            market_data = {symbol: self._get_market_data_for_risk(symbol) 
                          for symbol in self.symbols}
            
            risk_report = self.risk_manager.monitor_portfolio_risk(positions, market_data)
            
            # Controlla eventi critici
            risk_events = risk_report.get('risk_events', [])
            
            if 'DRAWDOWN_LIMIT' in risk_events or 'DAILY_LOSS_LIMIT' in risk_events:
                self.emergency_stop("Limite di rischio superato")
            
            elif risk_report.get('risk_level') == 'CRITICAL':
                self.logger.warning("Livello di rischio CRITICO rilevato")
                # Implementa azioni correttive
                self._handle_critical_risk()
            
        except Exception as e:
            self.logger.error(f"Errore nel monitoraggio rischio: {e}")
    
    def _handle_critical_risk(self):
        """Gestisce situazioni di rischio critico."""
        try:
            # Cancella ordini pendenti
            self._cancel_pending_orders()
            
            # Riduce esposizione
            self._reduce_exposure()
            
            # Notifica
            self._notify_event('critical_risk', {
                'action': 'risk_reduction',
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Errore nella gestione rischio critico: {e}")
    
    def _get_market_data_for_risk(self, symbol: str) -> Dict:
        """Recupera dati di mercato per analisi del rischio."""
        try:
            symbol_cache = self.symbol_data.get(symbol, {})
            df = symbol_cache.get('data', pd.DataFrame())
            
            if df.empty:
                return {}
            
            latest = df.iloc[-1]
            
            return {
                'price': latest.get('close', 0),
                'atr': latest.get('atr', 0),
                'volatility': latest.get('atr', 0) / latest.get('close', 1),
                'avg_atr': df['atr'].tail(20).mean() if 'atr' in df.columns else 0
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero dati per rischio: {e}")
            return {}
    
    def _on_price_update(self, symbol: str, price: float, timestamp: datetime):
        """Callback per aggiornamenti di prezzo."""
        try:
            # Aggiorna cache prezzi
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = {}
            
            self.symbol_data[symbol]['last_price'] = price
            self.symbol_data[symbol]['last_price_update'] = timestamp
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento prezzo {symbol}: {e}")
    
    def _on_order_filled(self, order: Dict):
        """Callback per ordini completati."""
        try:
            order_id = order['id']
            symbol = order['symbol']
            
            # Aggiorna statistiche
            self.stats['total_trades'] += 1
            
            # Rimuovi da segnali attivi
            if order_id in self.active_signals:
                del self.active_signals[order_id]
            
            self.logger.info(f"Ordine completato: {order_id} per {symbol}")
            
            self._notify_event('order_filled', {
                'order': order,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Errore nel callback ordine: {e}")
    
    def _on_position_update(self, position: Dict):
        """Callback per aggiornamenti posizioni."""
        try:
            symbol = position['symbol']
            pnl = position.get('unrealized_pnl', 0)
            
            # Aggiorna statistiche P&L
            if pnl > 0:
                self.stats['winning_trades'] += 1
            elif pnl < 0:
                self.stats['losing_trades'] += 1
            
            self.stats['total_pnl'] += pnl
            
        except Exception as e:
            self.logger.error(f"Errore nel callback posizione: {e}")
    
    def _cancel_all_orders(self):
        """Cancella tutti gli ordini attivi."""
        try:
            active_orders = self.order_manager.active_orders
            
            for order_id in list(active_orders.keys()):
                self.order_manager.cancel_order(order_id)
            
            self.logger.info(f"Cancellati {len(active_orders)} ordini")
            
        except Exception as e:
            self.logger.error(f"Errore nella cancellazione ordini: {e}")
    
    def _cancel_pending_orders(self):
        """Cancella solo gli ordini pendenti."""
        try:
            active_orders = self.order_manager.active_orders
            cancelled = 0
            
            for order_id, order in active_orders.items():
                if order['status'].value in ['PENDING', 'SUBMITTED']:
                    self.order_manager.cancel_order(order_id)
                    cancelled += 1
            
            self.logger.info(f"Cancellati {cancelled} ordini pendenti")
            
        except Exception as e:
            self.logger.error(f"Errore nella cancellazione ordini pendenti: {e}")
    
    def _close_all_positions(self):
        """Chiude tutte le posizioni aperte."""
        try:
            positions = self.order_manager.get_positions()
            
            for symbol, position in positions.items():
                quantity = abs(position['quantity'])
                side = 'sell' if position['quantity'] > 0 else 'buy'
                
                close_signal = {
                    'symbol': symbol,
                    'side': side,
                    'type': 'market',
                    'quantity': quantity
                }
                
                self.order_manager.submit_order(close_signal)
            
            self.logger.info(f"Chiuse {len(positions)} posizioni")
            
        except Exception as e:
            self.logger.error(f"Errore nella chiusura posizioni: {e}")
    
    def _reduce_exposure(self):
        """Riduce l'esposizione del portfolio."""
        try:
            positions = self.order_manager.get_positions()
            
            for symbol, position in positions.items():
                # Riduce del 50% ogni posizione
                reduce_quantity = abs(position['quantity']) * 0.5
                side = 'sell' if position['quantity'] > 0 else 'buy'
                
                if reduce_quantity > 0:
                    reduce_signal = {
                        'symbol': symbol,
                        'side': side,
                        'type': 'market',
                        'quantity': reduce_quantity
                    }
                    
                    self.order_manager.submit_order(reduce_signal)
            
            self.logger.info("Esposizione ridotta")
            
        except Exception as e:
            self.logger.error(f"Errore nella riduzione esposizione: {e}")
    
    def _update_statistics(self):
        """Aggiorna le statistiche del sistema."""
        try:
            if self.start_time:
                self.stats['uptime'] = (datetime.now() - self.start_time).total_seconds()
            
            # Calcola win rate
            total_closed = self.stats['winning_trades'] + self.stats['losing_trades']
            if total_closed > 0:
                self.stats['win_rate'] = self.stats['winning_trades'] / total_closed
            else:
                self.stats['win_rate'] = 0.0
            
            # Aggiorna drawdown
            positions = self.order_manager.get_positions()
            current_drawdown = self.risk_manager._calculate_current_drawdown(positions)
            self.stats['max_drawdown'] = max(self.stats['max_drawdown'], current_drawdown)
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento statistiche: {e}")
    
    def _update_symbol_metrics(self, symbol: str, result: Dict):
        """Aggiorna le metriche per un simbolo."""
        try:
            if symbol not in self.performance_metrics:
                self.performance_metrics[symbol] = {
                    'signals_generated': 0,
                    'trades_executed': 0,
                    'last_mode': None,
                    'mode_changes': 0
                }
            
            metrics = self.performance_metrics[symbol]
            
            # Aggiorna contatori
            metrics['signals_generated'] += len(result.get('signals', []))
            
            # Controlla cambio modalità
            current_mode = result.get('trading_mode')
            if metrics['last_mode'] and metrics['last_mode'] != current_mode:
                metrics['mode_changes'] += 1
            
            metrics['last_mode'] = current_mode
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento metriche {symbol}: {e}")
    
    def _notify_event(self, event_type: str, data: Dict):
        """Notifica un evento ai callback registrati."""
        try:
            callbacks = self.event_callbacks.get(event_type, [])
            
            for callback in callbacks:
                try:
                    callback(event_type, data)
                except Exception as e:
                    self.logger.error(f"Errore callback evento {event_type}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Errore nella notifica evento: {e}")
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """Aggiunge un callback per un tipo di evento."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        
        self.event_callbacks[event_type].append(callback)
    
    def get_status(self) -> Dict:
        """Restituisce lo stato completo del sistema."""
        try:
            return {
                'engine_state': self.state,
                'start_time': self.start_time,
                'last_update': self.last_update,
                'uptime': self.stats.get('uptime', 0),
                'symbols': self.symbols,
                'active_orders': len(self.order_manager.active_orders),
                'active_positions': len(self.order_manager.get_positions()),
                'current_mode': self.strategy_manager.current_mode.value,
                'current_market_state': self.strategy_manager.current_market_state.value,
                'risk_level': self.risk_manager.current_risk_level.value,
                'statistics': self.stats,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero stato: {e}")
            return {'error': str(e)}
    
    def get_detailed_status(self) -> Dict:
        """Restituisce uno stato dettagliato del sistema."""
        try:
            basic_status = self.get_status()
            
            # Aggiungi dettagli aggiuntivi
            detailed = basic_status.copy()
            detailed.update({
                'positions': self.order_manager.get_positions(),
                'active_orders': self.order_manager.active_orders,
                'strategy_status': self.strategy_manager.get_strategy_status(),
                'risk_report': self.risk_manager.get_risk_report(),
                'trading_summary': self.order_manager.get_trading_summary(),
                'symbol_data': {symbol: {
                    'last_price': data.get('last_price'),
                    'last_update': data.get('last_price_update'),
                    'strategy_result': data.get('strategy_result', {})
                } for symbol, data in self.symbol_data.items()}
            })
            
            return detailed
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero stato dettagliato: {e}")
            return {'error': str(e)}
    
    def update_config(self, new_config: Dict) -> bool:
        """Aggiorna la configurazione del sistema."""
        try:
            # Aggiorna configurazione
            self.config.update(new_config)
            
            # Aggiorna simboli se cambiati
            if 'symbols' in new_config:
                old_symbols = set(self.symbols)
                new_symbols = set(new_config['symbols'])
                
                # Rimuovi sottoscrizioni vecchie
                for symbol in old_symbols - new_symbols:
                    # Implementa rimozione sottoscrizione
                    pass
                
                # Aggiungi nuove sottoscrizioni
                for symbol in new_symbols - old_symbols:
                    self.market_data.subscribe_to_symbol(symbol, self._on_price_update)
                
                self.symbols = new_config['symbols']
            
            # Aggiorna parametri componenti
            if 'strategy' in new_config:
                self.strategy_manager.parameters.update(new_config['strategy'])
            
            if 'risk' in new_config:
                self.risk_manager.risk_parameters.update(new_config['risk'])
            
            self.logger.info("Configurazione aggiornata")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento configurazione: {e}")
            return False

