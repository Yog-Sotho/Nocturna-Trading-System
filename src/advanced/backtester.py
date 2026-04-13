"""
Advanced Backtesting Engine per NOCTURNA v2.0
Sistema completo per il backtesting delle strategie di trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class BacktestOrder:
    """Rappresenta un ordine nel backtest."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: Optional[datetime] = None
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None

@dataclass
class BacktestPosition:
    """Rappresenta una posizione nel backtest."""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class BacktestTrade:
    """Rappresenta un trade completato."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    commission: float
    duration: timedelta

class AdvancedBacktester:
    """
    Sistema di backtesting avanzato con supporto per:
    - Slippage realistico
    - Commissioni variabili
    - Gestione del rischio
    - Analisi delle performance dettagliate
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configurazione backtesting
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 0.05%
        self.min_trade_size = config.get('min_trade_size', 100.0)
        self.annualization_factor = config.get('trading_days_per_year', 252)  # 252 for equities, 365 for crypto
        
        # Stato del backtest
        self.reset()
        
        self.logger.info("Advanced Backtester inizializzato")
    
    def reset(self):
        """Reset dello stato del backtester."""
        self.current_capital = self.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.orders: Dict[str, BacktestOrder] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self.current_time = None
        self.order_counter = 0
        
        # Metriche
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
    
    def run_backtest(self, data: pd.DataFrame, strategy_function: callable,
                    strategy_params: Dict) -> Dict:
        """
        Esegue un backtest completo.
        
        Args:
            data: DataFrame con dati OHLCV
            strategy_function: Funzione della strategia
            strategy_params: Parametri della strategia
            
        Returns:
            Risultati del backtest
        """
        try:
            self.reset()
            self.logger.info(f"Inizio backtest su {len(data)} barre")
            
            # Assicura che i dati siano ordinati per timestamp
            data = data.sort_index()
            
            # Itera attraverso i dati
            for timestamp, row in data.iterrows():
                self.current_time = timestamp
                
                # Aggiorna posizioni con prezzi correnti
                self._update_positions(row)
                
                # Processa ordini pendenti
                self._process_pending_orders(row)
                
                # Esegui strategia
                signals = strategy_function(data.loc[:timestamp], strategy_params)
                
                # Processa segnali
                if signals:
                    self._process_signals(signals, row)
                
                # Aggiorna equity curve
                self._update_equity_curve(row)
            
            # Calcola risultati finali
            results = self._calculate_results()
            
            self.logger.info(f"Backtest completato. Return: {results['total_return']:.2%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Errore nel backtest: {e}")
            return {'error': str(e)}
    
    def _update_positions(self, market_data: pd.Series):
        """Aggiorna le posizioni con i prezzi correnti."""
        current_price = market_data['close']
        
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                # Calcola PnL non realizzato
                if position.quantity > 0:  # Long position
                    position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                else:  # Short position
                    position.unrealized_pnl = (position.avg_price - current_price) * abs(position.quantity)
    
    def _process_pending_orders(self, market_data: pd.Series):
        """Processa gli ordini pendenti."""
        filled_orders = []
        
        for order_id, order in self.orders.items():
            if order.status != OrderStatus.PENDING:
                continue
            
            fill_price = self._check_order_fill(order, market_data)
            
            if fill_price is not None:
                # Applica slippage
                if order.side == OrderSide.BUY:
                    fill_price *= (1 + self.slippage_rate)
                else:
                    fill_price *= (1 - self.slippage_rate)
                
                # Esegui fill
                self._fill_order(order, fill_price)
                filled_orders.append(order_id)
        
        # Rimuovi ordini riempiti
        for order_id in filled_orders:
            del self.orders[order_id]
    
    def _check_order_fill(self, order: BacktestOrder, 
                         market_data: pd.Series) -> Optional[float]:
        """
        Controlla se un ordine può essere riempito.
        
        Args:
            order: Ordine da controllare
            market_data: Dati di mercato correnti
            
        Returns:
            Prezzo di riempimento o None
        """
        high = market_data['high']
        low = market_data['low']
        open_price = market_data['open']

        if order.type == OrderType.MARKET:
            return open_price  # Market orders si riempiono al prezzo di apertura
        
        elif order.type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and low <= order.price:
                return min(order.price, open_price)
            elif order.side == OrderSide.SELL and high >= order.price:
                return max(order.price, open_price)
        
        elif order.type == OrderType.STOP:
            if order.side == OrderSide.BUY and high >= order.stop_price:
                return max(order.stop_price, open_price)
            elif order.side == OrderSide.SELL and low <= order.stop_price:
                return min(order.stop_price, open_price)
        
        return None
    
    def _fill_order(self, order: BacktestOrder, fill_price: float):
        """Riempie un ordine."""
        try:
            # Calcola commissione
            commission = abs(order.quantity * fill_price * self.commission_rate)
            self.total_commission += commission
            
            # Aggiorna capitale
            if order.side == OrderSide.BUY:
                cost = order.quantity * fill_price + commission
                if cost > self.current_capital:
                    # Ordine rifiutato per capitale insufficiente
                    order.status = OrderStatus.REJECTED
                    return
                
                self.current_capital -= cost
            else:
                proceeds = order.quantity * fill_price - commission
                self.current_capital += proceeds
            
            # Aggiorna posizione
            self._update_position(order.symbol, order.side, order.quantity, fill_price)
            
            # Aggiorna ordine
            order.status = OrderStatus.FILLED
            order.fill_price = fill_price
            order.fill_timestamp = self.current_time
            
            self.logger.debug(f"Ordine riempito: {order.symbol} {order.side.value} "
                            f"{order.quantity} @ {fill_price:.4f}")
            
        except Exception as e:
            self.logger.error(f"Errore nel riempimento ordine: {e}")
            order.status = OrderStatus.REJECTED
    
    def _update_position(self, symbol: str, side: OrderSide, 
                        quantity: float, price: float):
        """Aggiorna una posizione."""
        if symbol not in self.positions:
            self.positions[symbol] = BacktestPosition(symbol, 0, 0)
        
        position = self.positions[symbol]
        
        if side == OrderSide.BUY:
            new_quantity = position.quantity + quantity
        else:
            new_quantity = position.quantity - quantity
        
        # Calculate new average price
        if new_quantity == 0:
            # Position closed — calculate realized P&L
            entry_price = position.avg_price  # Save BEFORE zeroing
            if position.quantity > 0:  # Was long
                realized_pnl = (price - entry_price) * quantity
            else:  # Was short
                realized_pnl = (entry_price - price) * quantity

            position.realized_pnl += realized_pnl

            # Record trade with correct entry price
            self._record_trade(symbol, side, quantity, entry_price,
                             price, realized_pnl)

            position.avg_price = 0  # Zero AFTER recording
            
        elif (position.quantity > 0 and new_quantity > 0) or \
             (position.quantity < 0 and new_quantity < 0):
            # Stessa direzione - aggiorna prezzo medio
            total_cost = position.quantity * position.avg_price + quantity * price
            position.avg_price = total_cost / new_quantity
        else:
            # Cambio direzione o riduzione posizione
            if abs(new_quantity) < abs(position.quantity):
                # Riduzione posizione - calcola PnL parziale
                closed_quantity = quantity
                if position.quantity > 0:
                    realized_pnl = (price - position.avg_price) * closed_quantity
                else:
                    realized_pnl = (position.avg_price - price) * closed_quantity
                
                position.realized_pnl += realized_pnl
                self._record_trade(symbol, side, closed_quantity, 
                                 position.avg_price, price, realized_pnl)
            else:
                # Inversione posizione
                position.avg_price = price
        
        position.quantity = new_quantity
    
    def _record_trade(self, symbol: str, side: OrderSide, quantity: float,
                     entry_price: float, exit_price: float, pnl: float):
        """Registra un trade completato."""
        trade = BacktestTrade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=self.current_time,  # Semplificato
            exit_time=self.current_time,
            pnl=pnl,
            commission=quantity * exit_price * self.commission_rate,
            duration=timedelta(0)  # Semplificato
        )
        
        self.trades.append(trade)
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    def _process_signals(self, signals: List[Dict], market_data: pd.Series):
        """Processa i segnali di trading."""
        for signal in signals:
            try:
                order = self._create_order_from_signal(signal, market_data)
                if order:
                    self.orders[order.id] = order
                    
            except Exception as e:
                self.logger.error(f"Errore nel processing segnale: {e}")
    
    def _create_order_from_signal(self, signal: Dict, 
                                 market_data: pd.Series) -> Optional[BacktestOrder]:
        """Crea un ordine da un segnale."""
        try:
            symbol = signal['symbol']
            side = OrderSide(signal['side'])
            order_type = OrderType(signal.get('type', 'market'))
            quantity = signal['quantity']
            
            # Validazioni
            if quantity < self.min_trade_size:
                return None
            
            # Genera ID ordine
            self.order_counter += 1
            order_id = f"order_{self.order_counter}"
            
            order = BacktestOrder(
                id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                timestamp=self.current_time
            )
            
            # Imposta prezzi per ordini limit/stop
            if order_type == OrderType.LIMIT:
                order.price = signal.get('price', market_data['close'])
            elif order_type == OrderType.STOP:
                order.stop_price = signal.get('stop_price', market_data['close'])
            
            return order
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione ordine: {e}")
            return None
    
    def _update_equity_curve(self, market_data: pd.Series):
        """Update equity curve with correct portfolio valuation."""
        # Total portfolio value = cash + sum(qty * current_price)
        total_value = self.current_capital

        for position in self.positions.values():
            if position.quantity != 0:
                # Market value IS the position value — do not add unrealized_pnl separately
                total_value += position.quantity * market_data['close']

        # Update drawdown
        if total_value > self.peak_equity:
            self.peak_equity = total_value

        current_drawdown = (self.peak_equity - total_value) / self.peak_equity if self.peak_equity > 0 else 0

        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Add point to curve
        self.equity_curve.append({
            'timestamp': self.current_time,
            'equity': total_value,
            'drawdown': current_drawdown,
            'cash': self.current_capital
        })
    
    def _calculate_results(self) -> Dict:
        """Calcola i risultati finali del backtest."""
        try:
            if not self.equity_curve:
                return {'error': 'No equity data available'}
            
            # Valori finali
            final_equity = self.equity_curve[-1]['equity']
            total_return = (final_equity - self.initial_capital) / self.initial_capital
            
            # Calcola metriche di performance
            equity_series = pd.Series([point['equity'] for point in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            
            # Sharpe ratio (assuming risk-free rate = 0)
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(self.annualization_factor)
            else:
                sharpe_ratio = 0.0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1 and negative_returns.std() > 0:
                sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(self.annualization_factor)
            else:
                sortino_ratio = 0.0
            
            # Win rate
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
            
            # Profit factor
            winning_pnl = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
            losing_pnl = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
            profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
            
            # Average trade
            avg_trade = sum(trade.pnl for trade in self.trades) / len(self.trades) if self.trades else 0.0
            
            # Calmar ratio
            calmar_ratio = total_return / self.max_drawdown if self.max_drawdown > 0 else 0.0
            
            results = {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': self.max_drawdown,
                'total_commission': self.total_commission,
                'avg_trade': avg_trade,
                'equity_curve': self.equity_curve,
                'trades': [
                    {
                        'symbol': trade.symbol,
                        'side': trade.side.value,
                        'quantity': trade.quantity,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'pnl': trade.pnl,
                        'commission': trade.commission
                    }
                    for trade in self.trades
                ]
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo risultati: {e}")
            return {'error': str(e)}
    
    def monte_carlo_analysis(self, data: pd.DataFrame, strategy_function: callable,
                           strategy_params: Dict, n_simulations: int = 1000) -> Dict:
        """
        Esegue analisi Monte Carlo per valutare la robustezza della strategia.
        
        Args:
            data: Dati storici
            strategy_function: Funzione della strategia
            strategy_params: Parametri della strategia
            n_simulations: Numero di simulazioni
            
        Returns:
            Risultati dell'analisi Monte Carlo
        """
        try:
            self.logger.info(f"Inizio analisi Monte Carlo ({n_simulations} simulazioni)")
            
            results = []
            
            for i in range(n_simulations):
                # Shuffle dei dati mantenendo la struttura temporale locale
                shuffled_data = self._shuffle_data_blocks(data)
                
                # Esegui backtest
                result = self.run_backtest(shuffled_data, strategy_function, strategy_params)
                
                if 'error' not in result:
                    results.append(result)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Completate {i + 1} simulazioni")
            
            # Analizza risultati
            if not results:
                return {'error': 'No valid simulations completed'}
            
            returns = [r['total_return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            max_drawdowns = [r['max_drawdown'] for r in results]
            
            analysis = {
                'n_simulations': len(results),
                'return_stats': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'percentile_5': np.percentile(returns, 5),
                    'percentile_95': np.percentile(returns, 95)
                },
                'sharpe_stats': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios)
                },
                'drawdown_stats': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'min': np.min(max_drawdowns),
                    'max': np.max(max_drawdowns),
                    'percentile_95': np.percentile(max_drawdowns, 95)
                },
                'probability_positive': sum(1 for r in returns if r > 0) / len(returns)
            }
            
            self.logger.info("Analisi Monte Carlo completata")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi Monte Carlo: {e}")
            return {'error': str(e)}
    
    def _shuffle_data_blocks(self, data: pd.DataFrame, block_size: int = 20) -> pd.DataFrame:
        """
        Shuffle data in blocks to maintain local temporal correlations
        while creating genuinely different return sequences for Monte Carlo.

        Args:
            data: Original data
            block_size: Block size

        Returns:
            Shuffled data with synthetic sequential index
        """
        try:
            # Split into blocks
            blocks = []
            for i in range(0, len(data), block_size):
                block = data.iloc[i:i+block_size].copy()
                blocks.append(block)

            # Shuffle blocks
            np.random.shuffle(blocks)

            # Reconstruct DataFrame with new sequential index
            # Do NOT sort_index — that would undo the shuffle
            shuffled_data = pd.concat(blocks, ignore_index=True)

            # Create synthetic sequential timestamps for the backtester
            if len(data.index) > 0 and hasattr(data.index[0], 'timestamp'):
                # Preserve the original time spacing but in new block order
                original_freq = pd.infer_freq(data.index)
                if original_freq:
                    shuffled_data.index = pd.date_range(
                        start=data.index[0],
                        periods=len(shuffled_data),
                        freq=original_freq
                    )

            return shuffled_data

        except Exception as e:
            self.logger.error(f"Error in data block shuffle: {e}")
            return data
    
    def walk_forward_analysis(self, data: pd.DataFrame, strategy_function: callable,
                            strategy_params: Dict, train_period: int = 252,
                            test_period: int = 63) -> Dict:
        """
        Esegue walk-forward analysis per validare la strategia.
        
        Args:
            data: Dati storici
            strategy_function: Funzione della strategia
            strategy_params: Parametri base della strategia
            train_period: Periodo di training (giorni)
            test_period: Periodo di test (giorni)
            
        Returns:
            Risultati del walk-forward
        """
        try:
            self.logger.info("Inizio walk-forward analysis")
            
            results = []
            start_idx = 0
            
            while start_idx + train_period + test_period <= len(data):
                # Training data (reserved for parameter optimization in future)
                train_end = start_idx + train_period
                _train_data = data.iloc[start_idx:train_end]  # noqa: F841 — needed for optimization
                
                # Dati di test
                test_end = train_end + test_period
                test_data = data.iloc[train_end:test_end]
                
                # Ottimizza parametri sui dati di training (semplificato)
                optimized_params = strategy_params.copy()
                
                # Test sui dati out-of-sample
                test_result = self.run_backtest(test_data, strategy_function, optimized_params)
                
                if 'error' not in test_result:
                    test_result['period_start'] = test_data.index[0]
                    test_result['period_end'] = test_data.index[-1]
                    results.append(test_result)
                
                # Avanza finestra
                start_idx += test_period
            
            # Analizza risultati aggregati
            if not results:
                return {'error': 'No valid periods completed'}
            
            total_returns = [r['total_return'] for r in results]
            
            analysis = {
                'n_periods': len(results),
                'period_results': results,
                'aggregate_stats': {
                    'mean_return': np.mean(total_returns),
                    'std_return': np.std(total_returns),
                    'positive_periods': sum(1 for r in total_returns if r > 0),
                    'negative_periods': sum(1 for r in total_returns if r < 0),
                    'consistency_ratio': sum(1 for r in total_returns if r > 0) / len(total_returns)
                }
            }
            
            self.logger.info("Walk-forward analysis completata")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Errore nel walk-forward: {e}")
            return {'error': str(e)}

