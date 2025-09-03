"""
Financial analytics and performance calculations.
"""

import numpy as np
import pandas as pd
import decimal
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from models import AccountInfo, Position, Order, PortfolioHistory

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
    # Additional detailed trading metrics
    avg_trade_size: Decimal
    avg_winning_trade_size: Decimal
    avg_losing_trade_size: Decimal
    expected_win_per_trade: Decimal
    cost_per_loss: Decimal
    total_pnl: Decimal

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
            try:
                if equity_values[i-1] > 0:
                    daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                    returns.append(daily_return)
            except (ZeroDivisionError, decimal.DivisionUndefined, decimal.InvalidOperation):
                # Skip this return calculation if division fails
                continue
        
        return returns
    
    def total_return(self, initial: Decimal, final: Decimal) -> Decimal:
        """Calculate total return percentage."""
        try:
            if initial <= 0:
                return Decimal('0')
            return ((final - initial) / initial) * 100
        except (ZeroDivisionError, decimal.DivisionUndefined, decimal.InvalidOperation):
            return Decimal('0')
    
    def annualized_return(self, initial: Decimal, final: Decimal, days: int) -> Decimal:
        """Calculate annualized return (CAGR)."""
        try:
            if initial <= 0 or days <= 0:
                return Decimal('0')
            
            years = Decimal(str(days)) / Decimal('365.25')
            if years <= 0:
                return Decimal('0')
            
            return Decimal(str((pow(float(final / initial), 1 / float(years)) - 1) * 100))
        except (ZeroDivisionError, decimal.DivisionUndefined, decimal.InvalidOperation, ValueError, OverflowError):
            return Decimal('0')
    
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
                if peak > 0:
                    drawdown = (peak - value) / peak * 100
                else:
                    drawdown = Decimal('0')
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_duration = current_duration
        
        return max_dd, max_duration
    
    def calmar_ratio(self, annualized_ret: Decimal, max_dd: Decimal) -> Decimal:
        """Calculate Calmar ratio."""
        try:
            if max_dd == 0:
                return Decimal('0')
            return annualized_ret / max_dd
        except (ZeroDivisionError, decimal.DivisionUndefined, decimal.InvalidOperation):
            return Decimal('0')
    
    def value_at_risk(self, returns: List[Decimal], confidence: float = 0.05) -> Decimal:
        """Calculate Value at Risk (VaR)."""
        if not returns:
            return Decimal('0')
        
        returns_array = np.array([float(r) for r in returns])
        var = np.percentile(returns_array, confidence * 100)
        return Decimal(str(round(var, 6)))
    
    def score_strategy(self, metrics: 'PerformanceMetrics') -> Dict:
        """Score the trading strategy out of 10 with detailed reasoning."""
        
        score = 0.0
        reasoning = []
        
        # 1. Total Return (0-2 points) - Absolute performance
        total_return = float(metrics.total_return)
        if total_return > 10:
            score += 2.0
            reasoning.append(f"Excellent return ({total_return:.1f}%): +2.0pts")
        elif total_return > 5:
            score += 1.5
            reasoning.append(f"Good return ({total_return:.1f}%): +1.5pts")
        elif total_return > 2:
            score += 1.0
            reasoning.append(f"Moderate return ({total_return:.1f}%): +1.0pts")
        elif total_return > 0:
            score += 0.5
            reasoning.append(f"Positive return ({total_return:.1f}%): +0.5pts")
        else:
            reasoning.append(f"Negative return ({total_return:.1f}%): +0.0pts")
        
        # 2. Sharpe Ratio (0-2 points) - Risk-adjusted returns
        sharpe = float(metrics.sharpe_ratio)
        if sharpe > 3:
            score += 2.0
            reasoning.append(f"Outstanding Sharpe ({sharpe:.2f}): +2.0pts")
        elif sharpe > 2:
            score += 1.5
            reasoning.append(f"Excellent Sharpe ({sharpe:.2f}): +1.5pts")
        elif sharpe > 1:
            score += 1.0
            reasoning.append(f"Good Sharpe ({sharpe:.2f}): +1.0pts")
        elif sharpe > 0.5:
            score += 0.5
            reasoning.append(f"Fair Sharpe ({sharpe:.2f}): +0.5pts")
        else:
            reasoning.append(f"Poor Sharpe ({sharpe:.2f}): +0.0pts")
        
        # 3. Max Drawdown (0-2 points) - Risk control
        max_dd = float(metrics.max_drawdown)
        if max_dd < 2:
            score += 2.0
            reasoning.append(f"Excellent risk control ({max_dd:.2f}% DD): +2.0pts")
        elif max_dd < 5:
            score += 1.5
            reasoning.append(f"Good risk control ({max_dd:.2f}% DD): +1.5pts")
        elif max_dd < 10:
            score += 1.0
            reasoning.append(f"Moderate risk control ({max_dd:.2f}% DD): +1.0pts")
        elif max_dd < 20:
            score += 0.5
            reasoning.append(f"High drawdown ({max_dd:.2f}% DD): +0.5pts")
        else:
            reasoning.append(f"Excessive drawdown ({max_dd:.2f}% DD): +0.0pts")
        
        # 4. Win Rate (0-1.5 points) - Consistency
        win_rate = float(metrics.win_rate)
        if win_rate > 60:
            score += 1.5
            reasoning.append(f"Excellent win rate ({win_rate:.1f}%): +1.5pts")
        elif win_rate > 50:
            score += 1.2
            reasoning.append(f"Good win rate ({win_rate:.1f}%): +1.2pts")
        elif win_rate > 40:
            score += 0.8
            reasoning.append(f"Moderate win rate ({win_rate:.1f}%): +0.8pts")
        elif win_rate > 30:
            score += 0.4
            reasoning.append(f"Low win rate ({win_rate:.1f}%): +0.4pts")
        else:
            reasoning.append(f"Poor win rate ({win_rate:.1f}%): +0.0pts")
        
        # 5. Expectancy (0-1.5 points) - Profitability per trade
        expectancy = float(metrics.expectancy)
        if expectancy > 100:
            score += 1.5
            reasoning.append(f"Excellent expectancy (${expectancy:.0f}): +1.5pts")
        elif expectancy > 50:
            score += 1.2
            reasoning.append(f"Good expectancy (${expectancy:.0f}): +1.2pts")
        elif expectancy > 20:
            score += 0.8
            reasoning.append(f"Moderate expectancy (${expectancy:.0f}): +0.8pts")
        elif expectancy > 0:
            score += 0.4
            reasoning.append(f"Low expectancy (${expectancy:.0f}): +0.4pts")
        else:
            reasoning.append(f"Negative expectancy (${expectancy:.0f}): +0.0pts")
        
        # 6. Trading Activity (0-1 point) - Sufficient sample size
        total_trades = metrics.total_trades
        if total_trades > 100:
            score += 1.0
            reasoning.append(f"Excellent sample size ({total_trades} trades): +1.0pts")
        elif total_trades > 50:
            score += 0.7
            reasoning.append(f"Good sample size ({total_trades} trades): +0.7pts")
        elif total_trades > 20:
            score += 0.4
            reasoning.append(f"Moderate sample size ({total_trades} trades): +0.4pts")
        else:
            reasoning.append(f"Small sample size ({total_trades} trades): +0.0pts")
        
        # Calculate grade
        final_score = min(10.0, score)  # Cap at 10
        
        if final_score >= 9:
            grade = "A+"
            assessment = "Exceptional Strategy"
        elif final_score >= 8:
            grade = "A"
            assessment = "Excellent Strategy"
        elif final_score >= 7:
            grade = "B+"
            assessment = "Very Good Strategy"
        elif final_score >= 6:
            grade = "B"
            assessment = "Good Strategy"
        elif final_score >= 5:
            grade = "C+"
            assessment = "Average Strategy"
        elif final_score >= 4:
            grade = "C"
            assessment = "Below Average Strategy"
        elif final_score >= 3:
            grade = "D"
            assessment = "Poor Strategy"
        else:
            grade = "F"
            assessment = "Failing Strategy"
        
        return {
            'score': final_score,
            'grade': grade,
            'assessment': assessment,
            'reasoning': reasoning,
            'max_possible': 10.0
        }
    
    
    def analyze_trades(self, orders: List[Order], positions: List[Position] = None) -> Dict:
        """Analyze trading performance from orders."""
        
        # Debug: Check order statuses
        status_counts = {}
        for order in orders:
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info(f"Order statuses found: {status_counts}")
        
        # Include partially filled orders too
        filled_orders = [order for order in orders if order.status.value in ['filled', 'partially_filled'] and order.filled_qty > 0]
        
        logger.info(f"Found {len(filled_orders)} filled/partially filled orders out of {len(orders)} total")
        
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
        
        # Debug logging
        logger.info(f"Trade P&L Analysis: {len(trade_pnls)} total P&L records")
        logger.info(f"Winning trades: {num_winning}, Losing trades: {num_losing}")
        if trade_pnls:
            logger.info(f"Sample P&L values: {[float(pnl) for pnl in trade_pnls[:5]]}")
            logger.info(f"P&L range: ${min(trade_pnls):.2f} to ${max(trade_pnls):.2f}")
        
        # If positions data is available, use that for more accurate win/loss analysis
        if positions:
            profitable_positions = [pos for pos in positions if pos.unrealized_pl > 0]
            losing_positions = [pos for pos in positions if pos.unrealized_pl < 0]
            
            total_positions = len(positions)
            num_winning = len(profitable_positions)
            num_losing = len(losing_positions)
            
            # Calculate position-based metrics
            total_unrealized_pnl = sum(pos.unrealized_pl for pos in positions)
            avg_winning_pnl = sum(pos.unrealized_pl for pos in profitable_positions) / num_winning if num_winning > 0 else Decimal('0')
            avg_losing_pnl = sum(pos.unrealized_pl for pos in losing_positions) / num_losing if num_losing > 0 else Decimal('0')
            
            try:
                win_rate = Decimal(str(num_winning / total_positions * 100)) if total_positions > 0 else Decimal('0')
            except (ZeroDivisionError, decimal.DivisionUndefined):
                win_rate = Decimal('0')
            
            total_trades = total_positions
            avg_win = avg_winning_pnl
            avg_loss = avg_losing_pnl
            
            logger.info(f"Position-based analysis: {total_positions} positions, {num_winning} profitable ({win_rate:.1f}%), {num_losing} losing")
            logger.info(f"Avg winning position: ${avg_winning_pnl:.2f}, Avg losing position: ${avg_losing_pnl:.2f}")
            
        else:
            # Fallback to original order-based analysis
            try:
                win_rate = Decimal(str(num_winning / total_trades * 100)) if total_trades > 0 else Decimal('0')
            except (ZeroDivisionError, decimal.DivisionUndefined):
                win_rate = Decimal('0')
            
            logger.info(f"Order-based analysis: {total_trades} matched trades, {num_winning} winning, {num_losing} losing")
            
        try:
            avg_win = Decimal(str(sum(winning_trades) / num_winning)) if num_winning > 0 else Decimal('0')
        except (ZeroDivisionError, decimal.DivisionUndefined):
            avg_win = Decimal('0')
            
        try:
            avg_loss = Decimal(str(sum(losing_trades) / num_losing)) if num_losing > 0 else Decimal('0')
        except (ZeroDivisionError, decimal.DivisionUndefined):
            avg_loss = Decimal('0')
        
        gross_profit = sum(winning_trades) if winning_trades else Decimal('0')
        gross_loss = abs(sum(losing_trades)) if losing_trades else Decimal('0')
        
        try:
            profit_factor = Decimal(str(gross_profit / gross_loss)) if gross_loss > 0 else Decimal('0')
        except (ZeroDivisionError, decimal.DivisionUndefined):
            profit_factor = Decimal('0')
            
        try:
            expectancy = Decimal(str(sum(trade_pnls) / total_trades)) if total_trades > 0 else Decimal('0')
        except (ZeroDivisionError, decimal.DivisionUndefined):
            expectancy = Decimal('0')
        
        # Calculate additional trade size statistics
        trade_sizes = []
        winning_trade_sizes = []
        losing_trade_sizes = []
        
        # Analyze filled orders for trade sizes
        for order in filled_orders:
            if order.filled_qty > 0 and order.limit_price:
                trade_size = order.filled_qty * Decimal(str(order.limit_price))
                trade_sizes.append(trade_size)
                
                # Categorize by side (approximation - buy orders contribute to winning, sell to losing)
                if order.side == 'buy':
                    winning_trade_sizes.append(trade_size)
                else:
                    losing_trade_sizes.append(trade_size)
        
        # Calculate average trade sizes
        try:
            avg_trade_size = Decimal(str(sum(trade_sizes) / len(trade_sizes))) if trade_sizes else Decimal('0')
        except (ZeroDivisionError, decimal.DivisionUndefined):
            avg_trade_size = Decimal('0')
            
        try:
            avg_winning_trade_size = Decimal(str(sum(winning_trade_sizes) / len(winning_trade_sizes))) if winning_trade_sizes else Decimal('0')
        except (ZeroDivisionError, decimal.DivisionUndefined):
            avg_winning_trade_size = Decimal('0')
            
        try:
            avg_losing_trade_size = Decimal(str(sum(losing_trade_sizes) / len(losing_trade_sizes))) if losing_trade_sizes else Decimal('0')
        except (ZeroDivisionError, decimal.DivisionUndefined):
            avg_losing_trade_size = Decimal('0')
        
        # Expected win per trade (more detailed)
        total_pnl = sum(trade_pnls) if trade_pnls else Decimal('0')
        expected_win_per_trade = expectancy  # This is already calculated above
        
        # Cost per loss trade (average loss amount)
        cost_per_loss = abs(avg_loss) if avg_loss < 0 else Decimal('0')
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winning,
            'losing_trades': num_losing,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade_size': avg_trade_size,
            'avg_winning_trade_size': avg_winning_trade_size,
            'avg_losing_trade_size': avg_losing_trade_size,
            'expected_win_per_trade': expected_win_per_trade,
            'cost_per_loss': cost_per_loss,
            'total_pnl': total_pnl
        }
    
    def calculate_comprehensive_metrics(self, portfolio_history: PortfolioHistory, 
                                      orders: List[Order], positions: List[Position] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from first trade date."""
        
        try:
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
                losing_trades=0,
                avg_trade_size=Decimal('0'),
                avg_winning_trade_size=Decimal('0'),
                avg_losing_trade_size=Decimal('0'),
                expected_win_per_trade=Decimal('0'),
                cost_per_loss=Decimal('0'),
                total_pnl=Decimal('0')
            )
            
            # Find the first trade date to start return calculations from actual trading
            # Manual override for exact date range (August 5th to August 29th, 2025 = 24 days)
            from datetime import datetime, timezone
            manual_start_date = datetime(2025, 8, 5, tzinfo=timezone.utc)
            manual_end_date = datetime(2025, 8, 29, tzinfo=timezone.utc)
            
            # Use manual date range for precise calculation
            logger.info(f"Using manual date range: {manual_start_date.date()} to {manual_end_date.date()} (24 days)")
            
            # Still check API data for validation
            if orders:
                filled_orders = [order for order in orders if order.status.value in ['filled', 'partially_filled'] and order.filled_at]
                if filled_orders:
                    api_first_trade = min(order.filled_at for order in filled_orders)
                    logger.info(f"First trade found in API data: {api_first_trade.date()} (for reference only)")
            
            # Use full portfolio history for calculations (better data quality)
            equity_values = portfolio_history.equity
            timestamps = portfolio_history.timestamp
            
            # Find portfolio values for August 5th (start) and August 29th (end)
            start_idx = 0
            end_idx = -1
            
            # Find closest data point to August 5th
            for i, timestamp in enumerate(timestamps):
                if timestamp.date() >= manual_start_date.date():
                    start_idx = max(0, i - 1)  # Go one data point back to get pre-trading value
                    break
            
            # Find closest data point to August 29th
            for i, timestamp in enumerate(timestamps):
                if timestamp.date() >= manual_end_date.date():
                    end_idx = i
                    break
            
            if end_idx == -1:  # If August 29th not found, use latest data
                end_idx = len(equity_values) - 1
            
            initial_value = equity_values[start_idx]
            final_value = equity_values[end_idx]
            start_date = timestamps[start_idx] 
            end_date = timestamps[end_idx]
            
            logger.info(f"Found portfolio data: Aug 5th ~= {start_date.date()} (${initial_value:,}), Aug 29th ~= {end_date.date()} (${final_value:,})")
            
            if initial_value == 0:
                initial_value = self.initial_balance
            
            # Use exactly 24 days for precise calculation
            days = 24
            logger.info(f"Using exact 24-day period for annualized calculation")
                
            # Also show the full portfolio history range for comparison
            full_initial = equity_values[0]
            if full_initial == 0:
                full_initial = self.initial_balance
            logger.info(f"Full portfolio range: ${full_initial:,} to ${final_value:,}")
            if full_initial > 0:
                logger.info(f"Full range return: {((final_value - full_initial) / full_initial * 100):.2f}%")
            
            # Log the final calculation details
            logger.info(f"Calculating returns from {start_date.date()} to {end_date.date()} ({days} days)")
            logger.info(f"Aug 5th-29th range: ${initial_value:,} to ${final_value:,}")
            total_return_pct = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
            logger.info(f"24-day return: {total_return_pct:.2f}%")
            
            # Calculate returns from filtered equity data
            returns = self.calculate_returns(equity_values)
            
            # Calculate metrics
            total_ret = self.total_return(initial_value, final_value)
            annual_ret = self.annualized_return(initial_value, final_value, days)
            sharpe = self.sharpe_ratio(returns)
            sortino = self.sortino_ratio(returns)
            max_dd, _ = self.max_drawdown(equity_values)
            calmar = self.calmar_ratio(annual_ret, max_dd)
            
            # Volatility
            volatility = Decimal('0')
            if returns:
                volatility = Decimal(str(np.std([float(r) for r in returns]) * np.sqrt(252) * 100))
            
            # VaR
            var_5 = self.value_at_risk(returns) * final_value
            
            # Trade analysis
            trade_metrics = self.analyze_trades(orders, positions)
            
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
            losing_trades=trade_metrics['losing_trades'],
            avg_trade_size=trade_metrics['avg_trade_size'],
            avg_winning_trade_size=trade_metrics['avg_winning_trade_size'],
            avg_losing_trade_size=trade_metrics['avg_losing_trade_size'],
            expected_win_per_trade=trade_metrics['expected_win_per_trade'],
            cost_per_loss=trade_metrics['cost_per_loss'],
            total_pnl=trade_metrics['total_pnl']
        )
        
        except Exception as e:
            import traceback
            logger.error(f"Error calculating performance metrics: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return default metrics on any error
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
                losing_trades=0,
                avg_trade_size=Decimal('0'),
                avg_winning_trade_size=Decimal('0'),
                avg_losing_trade_size=Decimal('0'),
                expected_win_per_trade=Decimal('0'),
                cost_per_loss=Decimal('0'),
                total_pnl=Decimal('0')
            )
