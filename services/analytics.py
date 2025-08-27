"""
Financial analytics and performance calculations.
"""

import numpy as np
import pandas as pd
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..models import AccountInfo, Position, Order, PortfolioHistory

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    total_return: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    max_drawdown: Decimal
    volatility: Decimal
    var_5: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    expectancy: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int

class FinancialAnalytics:
    """Professional financial analytics calculator."""
    
    def __init__(self, initial_balance: Decimal = Decimal('100000')):
        self.initial_balance = initial_balance
        self.risk_free_rate = Decimal('0.02')  # 2% risk-free rate
    
    def calculate_returns(self, equity_values: List[Decimal]) -> List[Decimal]:
        """Calculate daily returns from equity values."""
        if len(equity_values) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                returns.append(daily_return)
        
        return returns
    
    def total_return(self, initial: Decimal, final: Decimal) -> Decimal:
        """Calculate total return percentage."""
        if initial <= 0:
            return Decimal('0')
        return ((final - initial) / initial) * 100
    
    def annualized_return(self, initial: Decimal, final: Decimal, days: int) -> Decimal:
        """Calculate annualized return (CAGR)."""
        if initial <= 0 or days <= 0:
            return Decimal('0')
        
        years = Decimal(str(days)) / Decimal('365.25')
        if years <= 0:
            return Decimal('0')
        
        return (pow(final / initial, 1 / float(years)) - 1) * 100
    
    def sharpe_ratio(self, returns: List[Decimal], periods_per_year: int = 252) -> Decimal:
        """Calculate Sharpe ratio."""
        if not returns:
            return Decimal('0')
        
        returns_array = np.array([float(r) for r in returns])
        
        if len(returns_array) == 0:
            return Decimal('0')
        
        excess_returns = returns_array - (float(self.risk_free_rate) / periods_per_year)
        
        if np.std(excess_returns) == 0:
            return Decimal('0')
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        return Decimal(str(round(sharpe, 4)))
    
    def sortino_ratio(self, returns: List[Decimal], periods_per_year: int = 252) -> Decimal:
        """Calculate Sortino ratio (downside deviation)."""
        if not returns:
            return Decimal('0')
        
        returns_array = np.array([float(r) for r in returns])
        excess_returns = returns_array - (float(self.risk_free_rate) / periods_per_year)
        
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return Decimal('0')
        
        downside_deviation = np.std(downside_returns)
        
        if downside_deviation == 0:
            return Decimal('0')
        
        sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(periods_per_year)
        return Decimal(str(round(sortino, 4)))
    
    def max_drawdown(self, equity_values: List[Decimal]) -> Tuple[Decimal, int]:
        """Calculate maximum drawdown and duration."""
        if not equity_values:
            return Decimal('0'), 0
        
        peak = equity_values[0]
        max_dd = Decimal('0')
        max_duration = 0
        current_duration = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
                current_duration = 0
            else:
                current_duration += 1
                drawdown = (peak - value) / peak * 100
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_duration = current_duration
        
        return max_dd, max_duration
    
    def calmar_ratio(self, annualized_ret: Decimal, max_dd: Decimal) -> Decimal:
        """Calculate Calmar ratio."""
        if max_dd == 0:
            return Decimal('0')
        return annualized_ret / max_dd
    
    def value_at_risk(self, returns: List[Decimal], confidence: float = 0.05) -> Decimal:
        """Calculate Value at Risk (VaR)."""
        if not returns:
            return Decimal('0')
        
        returns_array = np.array([float(r) for r in returns])
        var = np.percentile(returns_array, confidence * 100)
        return Decimal(str(round(var, 6)))
    
    def analyze_trades(self, orders: List[Order]) -> Dict:
        """Analyze trading performance from orders."""
        filled_orders = [order for order in orders if order.is_filled]
        
        if not filled_orders:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': Decimal('0'),
                'avg_win': Decimal('0'),
                'avg_loss': Decimal('0'),
                'profit_factor': Decimal('0'),
                'expectancy': Decimal('0')
            }
        
        # Group trades by symbol to calculate P&L
        trades_by_symbol = {}
        
        for order in filled_orders:
            symbol = order.symbol
            side = order.side
            qty = order.filled_qty
            price = Decimal(str(order.limit_price)) if order.limit_price else Decimal('0')
            
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = {'buys': [], 'sells': []}
            
            trade_value = qty * price
            trades_by_symbol[symbol][side + 's'].append({
                'qty': qty,
                'price': price,
                'value': trade_value,
                'timestamp': order.filled_at or order.updated_at
            })
        
        # Calculate P&L for each symbol
        trade_pnls = []
        
        for symbol, trades in trades_by_symbol.items():
            buys = trades['buys']
            sells = trades['sells']
            
            # Simple FIFO matching
            buy_queue = buys.copy()
            
            for sell in sells:
                sell_qty = sell['qty']
                sell_price = sell['price']
                
                while sell_qty > 0 and buy_queue:
                    buy = buy_queue[0]
                    buy_qty = buy['qty']
                    buy_price = buy['price']
                    
                    matched_qty = min(sell_qty, buy_qty)
                    pnl = matched_qty * (sell_price - buy_price)
                    trade_pnls.append(pnl)
                    
                    sell_qty -= matched_qty
                    buy['qty'] -= matched_qty
                    
                    if buy['qty'] == 0:
                        buy_queue.pop(0)
        
        if not trade_pnls:
            return {
                'total_trades': len(filled_orders),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': Decimal('0'),
                'avg_win': Decimal('0'),
                'avg_loss': Decimal('0'),
                'profit_factor': Decimal('0'),
                'expectancy': Decimal('0')
            }
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        total_trades = len(trade_pnls)
        num_winning = len(winning_trades)
        num_losing = len(losing_trades)
        
        win_rate = Decimal(str(num_winning / total_trades * 100)) if total_trades > 0 else Decimal('0')
        avg_win = Decimal(str(sum(winning_trades) / num_winning)) if num_winning > 0 else Decimal('0')
        avg_loss = Decimal(str(sum(losing_trades) / num_losing)) if num_losing > 0 else Decimal('0')
        
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        
        profit_factor = Decimal(str(gross_profit / gross_loss)) if gross_loss > 0 else Decimal('0')
        expectancy = Decimal(str(sum(trade_pnls) / total_trades)) if total_trades > 0 else Decimal('0')
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    def calculate_comprehensive_metrics(self, portfolio_history: PortfolioHistory, 
                                      orders: List[Order]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        # Handle empty or invalid data
        if not portfolio_history.equity or len(portfolio_history.equity) < 2:
            return PerformanceMetrics(
                total_return=Decimal('0'),
                annualized_return=Decimal('0'),
                sharpe_ratio=Decimal('0'),
                sortino_ratio=Decimal('0'),
                calmar_ratio=Decimal('0'),
                max_drawdown=Decimal('0'),
                volatility=Decimal('0'),
                var_5=Decimal('0'),
                win_rate=Decimal('0'),
                profit_factor=Decimal('0'),
                avg_win=Decimal('0'),
                avg_loss=Decimal('0'),
                expectancy=Decimal('0'),
                total_trades=0,
                winning_trades=0,
                losing_trades=0
            )
        
        # Get initial and final values
        initial_value = portfolio_history.equity[0]
        if initial_value == 0:
            initial_value = self.initial_balance
        final_value = portfolio_history.equity[-1]
        
        # Calculate time period
        days = (portfolio_history.timestamp[-1] - portfolio_history.timestamp[0]).days
        if days == 0:
            days = 1
        
        # Calculate returns
        returns = self.calculate_returns(portfolio_history.equity)
        
        # Calculate metrics
        total_ret = self.total_return(initial_value, final_value)
        annual_ret = self.annualized_return(initial_value, final_value, days)
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        max_dd, _ = self.max_drawdown(portfolio_history.equity)
        calmar = self.calmar_ratio(annual_ret, max_dd)
        
        # Volatility
        volatility = Decimal('0')
        if returns:
            volatility = Decimal(str(np.std([float(r) for r in returns]) * np.sqrt(252) * 100))
        
        # VaR
        var_5 = self.value_at_risk(returns) * final_value
        
        # Trade analysis
        trade_metrics = self.analyze_trades(orders)
        
        return PerformanceMetrics(
            total_return=total_ret,
            annualized_return=annual_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            volatility=volatility,
            var_5=var_5,
            win_rate=trade_metrics['win_rate'],
            profit_factor=trade_metrics['profit_factor'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            expectancy=trade_metrics['expectancy'],
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades']
        )
