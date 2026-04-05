"""
Order Execution Manager per NOCTURNA v2.0 Trading Bot
Gestisce l'invio e il monitoraggio degli ordini ai broker di trading.
"""

import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum
import json
import threading
from alpaca_trade_api import REST as AlpacaREST
from alpaca_trade_api.entity import Order as AlpacaOrder
import uuid

class OrderStatus(Enum):
    """Stati possibili degli ordini."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    """Tipi di ordini supportati."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Lati degli ordini."""
    BUY = "buy"
    SELL = "sell"

class OrderExecutionManager:
    """
    Gestisce l'esecuzione degli ordini e il monitoraggio delle posizioni.
    Supporta multiple broker API e gestione avanzata degli ordini.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Client broker
        self.alpaca_client = None
        
        # Gestione ordini
        self.active_orders = {}
        self.order_history = []
        self.positions = {}
        
        # Trailing stops
        self.trailing_stops = {}
        
        # Threading per monitoraggio
        self.monitor_thread = None
        self.running = False
        
        # Callbacks
        self.order_callbacks = []
        self.position_callbacks = []
        
        # Risk management
        self.daily_pnl = 0.0
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.position_limits = config.get('position_limits', {})
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Inizializza i client dei broker."""
        try:
            # Alpaca API
            if self.config.get('alpaca_api_key'):
                self.alpaca_client = AlpacaREST(
                    key_id=self.config['alpaca_api_key'],
                    secret_key=self.config['alpaca_secret_key'],
                    base_url=self.config.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
                )
                self.logger.info("Alpaca trading client inizializzato")
                
                # Carica posizioni esistenti
                self._load_existing_positions()
            
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione client trading: {e}")
    
    def _load_existing_positions(self):
        """Carica le posizioni esistenti dal broker."""
        try:
            if self.alpaca_client:
                positions = self.alpaca_client.list_positions()
                for pos in positions:
                    self.positions[pos.symbol] = {
                        'symbol': pos.symbol,
                        'quantity': float(pos.qty),
                        'side': 'long' if float(pos.qty) > 0 else 'short',
                        'avg_price': float(pos.avg_entry_price),
                        'market_value': float(pos.market_value),
                        'unrealized_pnl': float(pos.unrealized_pl),
                        'last_update': datetime.now()
                    }
                
                self.logger.info(f"Caricate {len(self.positions)} posizioni esistenti")
                
        except Exception as e:
            self.logger.error(f"Errore nel caricamento posizioni: {e}")
    
    def submit_order(self, signal: Dict) -> Optional[str]:
        """
        Invia un ordine al broker basandosi su un segnale di trading.
        
        Args:
            signal: Dizionario con dettagli del segnale
            
        Returns:
            ID dell'ordine se inviato con successo, None altrimenti
        """
        try:
            # Validazione segnale
            if not self._validate_signal(signal):
                return None
            
            # Controlli di risk management
            if not self._check_risk_limits(signal):
                return None
            
            # Preparazione ordine
            order_data = self._prepare_order(signal)
            if not order_data:
                return None
            
            # Invio ordine
            order_id = self._send_order_to_broker(order_data)
            if order_id:
                # Registra ordine
                self._register_order(order_id, order_data, signal)
                
                # Setup trailing stop se necessario
                if signal.get('trail_trigger'):
                    self._setup_trailing_stop(order_id, signal)
                
                self.logger.info(f"Ordine inviato: {order_id} per {signal['symbol']}")
                return order_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nell'invio ordine: {e}")
            return None
    
    def _validate_signal(self, signal: Dict) -> bool:
        """Valida un segnale di trading."""
        required_fields = ['symbol', 'side', 'type', 'quantity']
        
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Campo mancante nel segnale: {field}")
                return False
        
        # Validazione valori
        if signal['quantity'] <= 0:
            self.logger.error("Quantità deve essere positiva")
            return False
        
        if signal['side'] not in ['buy', 'sell']:
            self.logger.error("Side deve essere 'buy' o 'sell'")
            return False
        
        if signal['type'] not in ['market', 'limit', 'stop', 'stop_limit']:
            self.logger.error("Tipo ordine non supportato")
            return False
        
        return True
    
    def _check_risk_limits(self, signal: Dict) -> bool:
        """Controlla i limiti di rischio prima di inviare un ordine."""
        try:
            symbol = signal['symbol']
            quantity = signal['quantity']
            
            # Controllo perdita giornaliera massima
            if self.daily_pnl < -self.max_daily_loss:
                self.logger.warning("Limite perdita giornaliera raggiunto")
                return False
            
            # Controllo limite posizione per simbolo
            current_position = self.positions.get(symbol, {}).get('quantity', 0)
            symbol_limit = self.position_limits.get(symbol, 1.0)
            
            if signal['side'] == 'buy':
                new_position = current_position + quantity
            else:
                new_position = current_position - quantity
            
            if abs(new_position) > symbol_limit:
                self.logger.warning(f"Limite posizione superato per {symbol}")
                return False
            
            # Controllo orari di trading
            if not self._is_market_open():
                self.logger.warning("Mercato chiuso, ordine bloccato")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel controllo limiti: {e}")
            return False
    
    def _is_market_open(self) -> bool:
        """Controlla se il mercato è aperto."""
        try:
            if self.alpaca_client:
                clock = self.alpaca_client.get_clock()
                return clock.is_open
            
            # Fallback: controllo semplificato
            now = datetime.now()
            weekday = now.weekday()
            hour = now.hour
            
            return weekday < 5 and 9 <= hour < 16
            
        except Exception:
            return True  # Default: assume mercato aperto
    
    def _prepare_order(self, signal: Dict) -> Optional[Dict]:
        """Prepara i dati dell'ordine per l'invio al broker."""
        try:
            order_data = {
                'symbol': signal['symbol'],
                'qty': signal['quantity'],
                'side': signal['side'],
                'type': signal['type'],
                'time_in_force': signal.get('time_in_force', 'day'),
                'client_order_id': str(uuid.uuid4())
            }
            
            # Aggiungi prezzo per ordini limit
            if signal['type'] in ['limit', 'stop_limit']:
                if 'price' not in signal:
                    self.logger.error("Prezzo richiesto per ordine limit")
                    return None
                order_data['limit_price'] = signal['price']
            
            # Aggiungi stop price per ordini stop
            if signal['type'] in ['stop', 'stop_limit']:
                if 'stop_price' not in signal:
                    self.logger.error("Stop price richiesto per ordine stop")
                    return None
                order_data['stop_price'] = signal['stop_price']
            
            # Aggiungi ordini bracket (stop loss e take profit)
            if signal.get('stop_loss') or signal.get('take_profit'):
                order_data['order_class'] = 'bracket'
                
                if signal.get('stop_loss'):
                    order_data['stop_loss'] = {
                        'stop_price': signal['stop_loss']
                    }
                
                if signal.get('take_profit'):
                    order_data['take_profit'] = {
                        'limit_price': signal['take_profit']
                    }
            
            return order_data
            
        except Exception as e:
            self.logger.error(f"Errore nella preparazione ordine: {e}")
            return None
    
    def _send_order_to_broker(self, order_data: Dict) -> Optional[str]:
        """Invia l'ordine al broker."""
        try:
            if self.alpaca_client:
                order = self.alpaca_client.submit_order(**order_data)
                return order.id
            
            # Simulazione per testing
            if self.config.get('simulation_mode', False):
                order_id = f"SIM_{uuid.uuid4().hex[:8]}"
                self.logger.info(f"Ordine simulato: {order_id}")
                return order_id
            
            self.logger.error("Nessun client broker disponibile")
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nell'invio ordine al broker: {e}")
            return None
    
    def _register_order(self, order_id: str, order_data: Dict, signal: Dict):
        """Registra un ordine nel sistema di tracking."""
        order_record = {
            'id': order_id,
            'symbol': order_data['symbol'],
            'side': order_data['side'],
            'type': order_data['type'],
            'quantity': order_data['qty'],
            'status': OrderStatus.SUBMITTED,
            'submitted_at': datetime.now(),
            'signal': signal,
            'order_data': order_data,
            'fills': [],
            'last_update': datetime.now()
        }
        
        self.active_orders[order_id] = order_record
        self.order_history.append(order_record.copy())
    
    def _setup_trailing_stop(self, order_id: str, signal: Dict):
        """Configura un trailing stop per un ordine."""
        if not signal.get('trail_trigger'):
            return
        
        trailing_config = {
            'order_id': order_id,
            'symbol': signal['symbol'],
            'side': signal['side'],
            'trigger_price': signal['trail_trigger'],
            'offset': signal.get('trail_offset', 0.01),
            'active': False,
            'highest_price': None,
            'lowest_price': None,
            'created_at': datetime.now()
        }
        
        self.trailing_stops[order_id] = trailing_config
        self.logger.info(f"Trailing stop configurato per ordine {order_id}")
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancella un ordine attivo.
        
        Args:
            order_id: ID dell'ordine da cancellare
            
        Returns:
            True se cancellato con successo
        """
        try:
            if order_id not in self.active_orders:
                self.logger.warning(f"Ordine {order_id} non trovato")
                return False
            
            # Cancella presso il broker
            if self.alpaca_client:
                self.alpaca_client.cancel_order(order_id)
            
            # Aggiorna stato locale
            self.active_orders[order_id]['status'] = OrderStatus.CANCELLED
            self.active_orders[order_id]['cancelled_at'] = datetime.now()
            
            # Rimuovi trailing stop se presente
            if order_id in self.trailing_stops:
                del self.trailing_stops[order_id]
            
            self.logger.info(f"Ordine {order_id} cancellato")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nella cancellazione ordine {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Recupera lo stato di un ordine."""
        try:
            if order_id in self.active_orders:
                return self.active_orders[order_id].copy()
            
            # Cerca nella cronologia
            for order in self.order_history:
                if order['id'] == order_id:
                    return order.copy()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nel recupero stato ordine {order_id}: {e}")
            return None
    
    def get_positions(self) -> Dict:
        """Recupera tutte le posizioni attive."""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Recupera la posizione per un simbolo specifico."""
        return self.positions.get(symbol)
    
    def start_monitoring(self):
        """Avvia il monitoraggio degli ordini e posizioni."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_worker)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Monitoraggio ordini avviato")
    
    def stop_monitoring(self):
        """Ferma il monitoraggio degli ordini e posizioni."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Monitoraggio ordini fermato")
    
    def _monitoring_worker(self):
        """Worker thread per il monitoraggio continuo."""
        while self.running:
            try:
                # Aggiorna stato ordini
                self._update_orders_status()
                
                # Aggiorna posizioni
                self._update_positions()
                
                # Gestisci trailing stops
                self._process_trailing_stops()
                
                # Calcola P&L giornaliero
                self._update_daily_pnl()
                
                time.sleep(5)  # Aggiorna ogni 5 secondi
                
            except Exception as e:
                self.logger.error(f"Errore nel monitoraggio: {e}")
                time.sleep(10)
    
    def _update_orders_status(self):
        """Aggiorna lo stato degli ordini attivi."""
        try:
            if not self.alpaca_client:
                return
            
            for order_id in list(self.active_orders.keys()):
                try:
                    # Recupera stato dal broker
                    broker_order = self.alpaca_client.get_order(order_id)
                    
                    # Aggiorna stato locale
                    local_order = self.active_orders[order_id]
                    local_order['status'] = OrderStatus(broker_order.status.lower())
                    local_order['filled_qty'] = float(broker_order.filled_qty or 0)
                    local_order['filled_avg_price'] = float(broker_order.filled_avg_price or 0)
                    local_order['last_update'] = datetime.now()
                    
                    # Se ordine completato, aggiorna posizione
                    if local_order['status'] == OrderStatus.FILLED:
                        self._update_position_from_fill(local_order)
                        
                        # Rimuovi dagli ordini attivi
                        del self.active_orders[order_id]
                        
                        # Notifica callbacks
                        self._notify_order_callbacks(local_order)
                    
                except Exception as e:
                    self.logger.error(f"Errore aggiornamento ordine {order_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento ordini: {e}")
    
    def _update_positions(self):
        """Aggiorna le posizioni dal broker."""
        try:
            if not self.alpaca_client:
                return
            
            broker_positions = self.alpaca_client.list_positions()
            
            # Aggiorna posizioni esistenti
            current_symbols = set()
            for pos in broker_positions:
                symbol = pos.symbol
                current_symbols.add(symbol)
                
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': float(pos.qty),
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'avg_price': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'last_update': datetime.now()
                }
            
            # Rimuovi posizioni chiuse
            for symbol in list(self.positions.keys()):
                if symbol not in current_symbols:
                    del self.positions[symbol]
                    
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento posizioni: {e}")
    
    def _update_position_from_fill(self, order: Dict):
        """Aggiorna la posizione basandosi su un ordine eseguito."""
        try:
            symbol = order['symbol']
            quantity = order['filled_qty']
            price = order['filled_avg_price']
            side = order['side']
            
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'symbol': symbol,
                    'quantity': 0,
                    'avg_price': 0,
                    'total_cost': 0
                }
            
            position = self.positions[symbol]
            
            if side == 'buy':
                new_quantity = position['quantity'] + quantity
                new_total_cost = (position['quantity'] * position['avg_price']) + (quantity * price)
                position['avg_price'] = new_total_cost / new_quantity if new_quantity != 0 else 0
                position['quantity'] = new_quantity
            else:  # sell
                position['quantity'] -= quantity
                if position['quantity'] <= 0:
                    position['quantity'] = 0
                    position['avg_price'] = 0
            
            position['last_update'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento posizione: {e}")
    
    def _process_trailing_stops(self):
        """Processa i trailing stops attivi."""
        try:
            for order_id, config in list(self.trailing_stops.items()):
                symbol = config['symbol']
                
                # Recupera prezzo corrente
                current_price = self._get_current_price(symbol)
                if not current_price:
                    continue
                
                # Controlla se il trailing stop deve essere attivato
                if not config['active']:
                    if ((config['side'] == 'buy' and current_price >= config['trigger_price']) or
                        (config['side'] == 'sell' and current_price <= config['trigger_price'])):
                        config['active'] = True
                        config['highest_price'] = current_price
                        config['lowest_price'] = current_price
                        self.logger.info(f"Trailing stop attivato per {symbol}")
                        continue
                
                # Aggiorna prezzi estremi
                if config['side'] == 'buy':
                    if current_price > config['highest_price']:
                        config['highest_price'] = current_price
                    
                    # Controlla se deve essere eseguito
                    stop_price = config['highest_price'] * (1 - config['offset'])
                    if current_price <= stop_price:
                        self._execute_trailing_stop(order_id, config, current_price)
                
                else:  # sell
                    if current_price < config['lowest_price']:
                        config['lowest_price'] = current_price
                    
                    # Controlla se deve essere eseguito
                    stop_price = config['lowest_price'] * (1 + config['offset'])
                    if current_price >= stop_price:
                        self._execute_trailing_stop(order_id, config, current_price)
                        
        except Exception as e:
            self.logger.error(f"Errore nel processing trailing stops: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Recupera il prezzo corrente di un simbolo."""
        try:
            if self.alpaca_client:
                quote = self.alpaca_client.get_latest_quote(symbol)
                return (quote.bid_price + quote.ask_price) / 2
            return None
        except:
            return None
    
    def _execute_trailing_stop(self, order_id: str, config: Dict, current_price: float):
        """Esegue un trailing stop."""
        try:
            symbol = config['symbol']
            
            # Crea ordine di stop
            stop_signal = {
                'symbol': symbol,
                'side': 'sell' if config['side'] == 'buy' else 'buy',
                'type': 'market',
                'quantity': self.positions.get(symbol, {}).get('quantity', 0)
            }
            
            if stop_signal['quantity'] > 0:
                self.submit_order(stop_signal)
                self.logger.info(f"Trailing stop eseguito per {symbol} a {current_price}")
            
            # Rimuovi trailing stop
            del self.trailing_stops[order_id]
            
        except Exception as e:
            self.logger.error(f"Errore nell'esecuzione trailing stop: {e}")
    
    def _update_daily_pnl(self):
        """Aggiorna il P&L giornaliero."""
        try:
            total_unrealized = sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
            # Qui dovresti aggiungere anche il P&L realizzato del giorno
            self.daily_pnl = total_unrealized
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo P&L giornaliero: {e}")
    
    def _notify_order_callbacks(self, order: Dict):
        """Notifica i callback registrati per gli ordini."""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                self.logger.error(f"Errore callback ordine: {e}")
    
    def add_order_callback(self, callback: Callable):
        """Aggiunge un callback per eventi ordini."""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable):
        """Aggiunge un callback per eventi posizioni."""
        self.position_callbacks.append(callback)
    
    def get_trading_summary(self) -> Dict:
        """Restituisce un riassunto dell'attività di trading."""
        return {
            'active_orders': len(self.active_orders),
            'active_positions': len(self.positions),
            'daily_pnl': self.daily_pnl,
            'trailing_stops': len(self.trailing_stops),
            'total_orders_today': len([o for o in self.order_history 
                                     if o.get('submitted_at', datetime.min).date() == datetime.now().date()]),
            'last_update': datetime.now()
        }

