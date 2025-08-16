#!/usr/bin/env python3
"""
Alpaca Trading API Script
Fetches and displays information about the last trades made.
"""

import requests
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys

# Alpaca API Configuration
# For security, replace these with your actual API keys or use environment variables
ALPACA_API_KEY = "YOUR_API_KEY_HERE"
ALPACA_SECRET_KEY = "YOUR_SECRET_KEY_HERE"

# Alternative: Use environment variables (recommended for production)
# import os
# ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_API_KEY_HERE')
# ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_SECRET_KEY_HERE')

# Try both paper and live endpoints
ENDPOINTS = [
    "https://paper-api.alpaca.markets",  # Paper trading
    "https://api.alpaca.markets"         # Live trading
]

# API Headers
HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# Global variable to store the working base URL
BASE_URL = None


def find_working_endpoint() -> Optional[str]:
    """Find which endpoint works with the provided credentials."""
    global BASE_URL
    
    for endpoint in ENDPOINTS:
        try:
            print(f"🔍 Testing endpoint: {endpoint}")
            response = requests.get(f"{endpoint}/v2/account", headers=HEADERS, timeout=10)
            if response.status_code == 200:
                print(f"✅ Successfully connected to: {endpoint}")
                BASE_URL = endpoint
                return endpoint
            elif response.status_code == 401:
                print(f"❌ Unauthorized for: {endpoint}")
            else:
                print(f"❌ Error {response.status_code} for: {endpoint}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection failed for {endpoint}: {e}")
    
    return None


def get_account_info() -> Optional[Dict]:
    """Get account information."""
    global BASE_URL
    
    if BASE_URL is None:
        print("🔍 Finding working API endpoint...")
        if find_working_endpoint() is None:
            print("❌ No working endpoint found. Please check your credentials.")
            return None
    
    try:
        response = requests.get(f"{BASE_URL}/v2/account", headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching account info: {e}")
        return None


def get_orders(status: str = "all", limit: int = 50) -> Optional[List[Dict]]:
    """Get orders with specified status."""
    try:
        params = {
            "status": status,
            "limit": limit,
            "direction": "desc"  # Most recent first
        }
        response = requests.get(f"{BASE_URL}/v2/orders", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching orders: {e}")
        return None


def get_positions() -> Optional[List[Dict]]:
    """Get current positions."""
    try:
        response = requests.get(f"{BASE_URL}/v2/positions", headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching positions: {e}")
        return None


def get_portfolio_history(period: str = "1D") -> Optional[Dict]:
    """Get portfolio history."""
    try:
        params = {"period": period}
        response = requests.get(f"{BASE_URL}/v2/account/portfolio/history", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching portfolio history: {e}")
        return None


def get_portfolio_history_from_date(start_date: str = "2024-04-11") -> Optional[Dict]:
    """Get portfolio history from a specific date."""
    try:
        # Calculate end date (today) and format properly with time
        end_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        start_date_formatted = f"{start_date}T00:00:00Z"
        
        params = {
            "timeframe": "1Day",
            "start": start_date_formatted,
            "end": end_date
        }
        
        print(f"🔍 Fetching portfolio history from {start_date_formatted} to {end_date}")
        response = requests.get(f"{BASE_URL}/v2/account/portfolio/history", headers=HEADERS, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching portfolio history from date: {e}")
        return None


def get_extended_portfolio_history() -> Optional[Dict]:
    """Get extended portfolio history for better analysis."""
    # First try to get data from April 11th
    april_data = get_portfolio_history_from_date("2024-04-11")
    if april_data:
        return april_data
    
    # If that fails, try different periods until we find one that works
    periods = ["1M", "3M", "6M", "1Y"]
    
    for period in periods:
        try:
            params = {"period": period}
            response = requests.get(f"{BASE_URL}/v2/account/portfolio/history", headers=HEADERS, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 422:
                # Try shorter period
                continue
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            if period == periods[-1]:  # Last attempt
                print(f"Error fetching extended portfolio history: {e}")
                return None
            continue
    
    return None


def get_closed_orders_from_date(start_date: str = "2024-04-11") -> Optional[List[Dict]]:
    """Get all closed/filled orders from a specific date for trade analysis."""
    try:
        # Convert start_date to datetime for filtering
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        
        params = {
            "status": "closed",
            "limit": 500,  # Get more orders for better analysis
            "direction": "desc"
        }
        response = requests.get(f"{BASE_URL}/v2/orders", headers=HEADERS, params=params)
        response.raise_for_status()
        
        all_orders = response.json()
        
        # Filter orders from the start date
        filtered_orders = []
        for order in all_orders:
            order_date_str = order.get('filled_at') or order.get('updated_at') or order.get('created_at')
            if order_date_str:
                try:
                    # Parse the order date
                    order_date = datetime.fromisoformat(order_date_str.replace('Z', '+00:00'))
                    if order_date.replace(tzinfo=None) >= start_dt:
                        filtered_orders.append(order)
                except:
                    # If date parsing fails, include the order to be safe
                    filtered_orders.append(order)
        
        print(f"📊 Filtered {len(filtered_orders)} orders from {start_date} onwards (out of {len(all_orders)} total)")
        return filtered_orders
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching closed orders: {e}")
        return None


def get_closed_orders() -> Optional[List[Dict]]:
    """Get all closed/filled orders for trade analysis."""
    try:
        params = {
            "status": "closed",
            "limit": 500,  # Get more orders for better analysis
            "direction": "desc"
        }
        response = requests.get(f"{BASE_URL}/v2/orders", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching closed orders: {e}")
        return None


def format_currency(amount: float) -> str:
    """Format currency amount."""
    return f"${amount:,.2f}"


def format_datetime(dt_string: str) -> str:
    """Format datetime string for display."""
    try:
        dt = datetime.fromisoformat(dt_string.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except:
        return dt_string


class PerformanceCalculator:
    """Calculate comprehensive trading performance metrics."""
    
    def __init__(self, portfolio_history: Dict, orders: List[Dict], account: Dict):
        self.portfolio_history = portfolio_history
        self.orders = orders
        self.account = account
        self.equity_values = portfolio_history.get('equity', [])
        self.timestamps = portfolio_history.get('timestamp', [])
        
        # Convert timestamps to datetime objects
        self.dates = []
        for ts in self.timestamps:
            try:
                self.dates.append(datetime.fromtimestamp(ts))
            except:
                continue
    
    def calculate_returns(self) -> List[float]:
        """Calculate daily returns from equity curve."""
        if len(self.equity_values) < 2:
            return []
        
        returns = []
        for i in range(1, len(self.equity_values)):
            if self.equity_values[i-1] > 0:
                daily_return = (self.equity_values[i] - self.equity_values[i-1]) / self.equity_values[i-1]
                returns.append(daily_return)
        return returns
    
    def total_return(self) -> float:
        """Calculate total return percentage."""
        if len(self.equity_values) < 2:
            return 0.0
        
        initial = self.equity_values[0]
        final = self.equity_values[-1]
        
        # Debug output
        print(f"🔍 Debug Total Return: Initial=${initial:,.2f}, Final=${final:,.2f}")
        
        if initial > 0:
            total_ret = ((final - initial) / initial) * 100
            print(f"🔍 Debug Total Return: {total_ret:+.2f}%")
            return total_ret
        return 0.0
    
    def annualized_return(self) -> float:
        """Calculate CAGR (Compound Annual Growth Rate)."""
        if len(self.equity_values) < 2 or len(self.dates) < 2:
            return 0.0
        
        initial = self.equity_values[0]
        final = self.equity_values[-1]
        
        if initial <= 0:
            return 0.0
        
        # Calculate number of years
        time_diff = self.dates[-1] - self.dates[0]
        years = time_diff.days / 365.25
        
        if years <= 0:
            return 0.0
        
        cagr = (pow(final / initial, 1 / years) - 1) * 100
        return cagr
    
    def analyze_trades(self) -> Dict:
        """Analyze individual trades for win/loss metrics."""
        filled_orders = [order for order in self.orders if order.get('status') == 'filled']
        
        if not filled_orders:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
        
        # Group trades by symbol to calculate P&L
        trades_by_symbol = {}
        
        for order in filled_orders:
            symbol = order.get('symbol')
            side = order.get('side')
            qty = float(order.get('filled_qty', 0))
            price = float(order.get('filled_avg_price', 0))
            
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = {'buys': [], 'sells': []}
            
            trade_value = qty * price
            trades_by_symbol[symbol][side + 's'].append({
                'qty': qty,
                'price': price,
                'value': trade_value,
                'timestamp': order.get('filled_at', order.get('updated_at'))
            })
        
        # Calculate P&L for each symbol
        trade_pnls = []
        
        for symbol, trades in trades_by_symbol.items():
            buys = trades['buys']
            sells = trades['sells']
            
            # Simple FIFO matching for P&L calculation
            remaining_qty = 0
            avg_cost = 0
            
            # Process buys first
            for buy in buys:
                if remaining_qty == 0:
                    avg_cost = buy['price']
                    remaining_qty = buy['qty']
                else:
                    # Update average cost
                    total_cost = (avg_cost * remaining_qty) + buy['value']
                    remaining_qty += buy['qty']
                    avg_cost = total_cost / remaining_qty
            
            # Process sells
            for sell in sells:
                if remaining_qty > 0:
                    sold_qty = min(sell['qty'], remaining_qty)
                    pnl = sold_qty * (sell['price'] - avg_cost)
                    trade_pnls.append(pnl)
                    remaining_qty -= sold_qty
        
        if not trade_pnls:
            return {
                'total_trades': len(filled_orders),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        total_trades = len(trade_pnls)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        avg_win = statistics.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(statistics.mean(losing_trades)) if losing_trades else 0
        
        total_profit = sum(winning_trades)
        total_loss = abs(sum(losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        expectancy = statistics.mean(trade_pnls) if trade_pnls else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': win_count,
            'losing_trades': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'trade_pnls': trade_pnls
        }
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        returns = self.calculate_returns()
        if not returns:
            return 0.0
        
        avg_return = statistics.mean(returns)
        return_std = statistics.stdev(returns) if len(returns) > 1 else 0
        
        if return_std == 0:
            return 0.0
        
        # Annualize the calculation
        daily_rf_rate = risk_free_rate / 252  # 252 trading days
        annualized_return = avg_return * 252
        annualized_std = return_std * math.sqrt(252)
        
        return (annualized_return - risk_free_rate) / annualized_std
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)."""
        returns = self.calculate_returns()
        if not returns:
            return 0.0
        
        avg_return = statistics.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return float('inf') if avg_return > 0 else 0.0
        
        downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0
        
        if downside_std == 0:
            return float('inf') if avg_return > 0 else 0.0
        
        # Annualize
        daily_rf_rate = risk_free_rate / 252
        annualized_return = avg_return * 252
        annualized_downside_std = downside_std * math.sqrt(252)
        
        return (annualized_return - risk_free_rate) / annualized_downside_std
    
    def calculate_max_drawdown(self) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and recovery time."""
        if len(self.equity_values) < 2:
            return 0.0, 0, 0
        
        peak = self.equity_values[0]
        max_dd = 0.0
        peak_idx = 0
        trough_idx = 0
        recovery_days = 0
        
        for i, value in enumerate(self.equity_values):
            if value > peak:
                peak = value
                peak_idx = i
            
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            
            if drawdown > max_dd:
                max_dd = drawdown
                trough_idx = i
        
        # Calculate recovery time
        if max_dd > 0:
            for i in range(trough_idx, len(self.equity_values)):
                if self.equity_values[i] >= peak:
                    recovery_days = i - trough_idx
                    break
        
        return max_dd, recovery_days, trough_idx - peak_idx
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        cagr = self.annualized_return()
        max_dd, _, _ = self.calculate_max_drawdown()
        
        if max_dd == 0:
            return float('inf') if cagr > 0 else 0.0
        
        return cagr / max_dd
    
    def calculate_volatility(self) -> float:
        """Calculate annualized volatility."""
        returns = self.calculate_returns()
        if len(returns) < 2:
            return 0.0
        
        daily_std = statistics.stdev(returns)
        return daily_std * math.sqrt(252) * 100  # Annualized percentage
    
    def calculate_var(self, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        returns = self.calculate_returns()
        if not returns:
            return 0.0
        
        returns_sorted = sorted(returns)
        var_index = int(len(returns_sorted) * confidence_level)
        
        if var_index < len(returns_sorted):
            var_return = returns_sorted[var_index]
            current_value = self.equity_values[-1] if self.equity_values else 0
            return abs(var_return * current_value)
        
        return 0.0
    
    def analyze_equity_curve(self) -> Dict:
        """Analyze equity curve characteristics."""
        if len(self.equity_values) < 2:
            return {}
        
        returns = self.calculate_returns()
        
        # Calculate trend (slope)
        if len(self.equity_values) > 1:
            x = list(range(len(self.equity_values)))
            y = self.equity_values
            
            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        else:
            slope = 0
        
        # Calculate smoothness (consistency)
        smoothness = 1 / (statistics.stdev(returns) + 0.0001) if returns else 0
        
        return {
            'slope': slope,
            'smoothness': smoothness,
            'trend_strength': abs(slope) * smoothness
        }


def display_account_info(account: Dict) -> None:
    """Display account information."""
    print("=" * 60)
    print("🏦 ACCOUNT INFORMATION")
    print("=" * 60)
    
    print(f"Account ID: {account.get('id', 'N/A')}")
    print(f"Account Status: {account.get('status', 'N/A')}")
    print(f"Trading Blocked: {account.get('trade_suspended_by_user', 'N/A')}")
    print(f"Pattern Day Trader: {account.get('pattern_day_trader', 'N/A')}")
    
    print("\n💰 BALANCE INFORMATION:")
    print(f"  Equity: {format_currency(float(account.get('equity', 0)))}")
    print(f"  Cash: {format_currency(float(account.get('cash', 0)))}")
    print(f"  Buying Power: {format_currency(float(account.get('buying_power', 0)))}")
    print(f"  Portfolio Value: {format_currency(float(account.get('portfolio_value', 0)))}")
    
    day_trade_count = account.get('day_trade_count', 0)
    print(f"  Day Trade Count: {day_trade_count}")
    
    if 'last_equity' in account:
        last_equity = float(account['last_equity'])
        current_equity = float(account.get('equity', 0))
        change = current_equity - last_equity
        change_pct = (change / last_equity * 100) if last_equity != 0 else 0
        print(f"  Daily P&L: {format_currency(change)} ({change_pct:+.2f}%)")


def display_recent_orders(orders: List[Dict]) -> None:
    """Display recent orders."""
    print("\n" + "=" * 60)
    print("📋 RECENT ORDERS")
    print("=" * 60)
    
    if not orders:
        print("No orders found.")
        return
    
    for i, order in enumerate(orders[:10], 1):  # Show last 10 orders
        symbol = order.get('symbol', 'N/A')
        side = order.get('side', 'N/A').upper()
        qty = order.get('qty', 'N/A')
        status = order.get('status', 'N/A')
        order_type = order.get('order_type', 'N/A')
        
        print(f"\n{i}. {symbol} - {side} {qty} shares")
        print(f"   Status: {status}")
        print(f"   Type: {order_type}")
        
        if 'limit_price' in order and order['limit_price']:
            print(f"   Limit Price: {format_currency(float(order['limit_price']))}")
        
        if 'filled_avg_price' in order and order['filled_avg_price']:
            filled_price = float(order['filled_avg_price'])
            filled_qty = float(order.get('filled_qty', 0))
            total_value = filled_price * filled_qty
            print(f"   Filled: {order.get('filled_qty', 0)} @ {format_currency(filled_price)}")
            print(f"   Total Value: {format_currency(total_value)}")
        
        if 'created_at' in order:
            print(f"   Created: {format_datetime(order['created_at'])}")
        
        if 'updated_at' in order:
            print(f"   Updated: {format_datetime(order['updated_at'])}")


def display_positions(positions: List[Dict]) -> None:
    """Display current positions."""
    print("\n" + "=" * 60)
    print("📊 CURRENT POSITIONS")
    print("=" * 60)
    
    if not positions:
        print("No open positions.")
        return
    
    total_market_value = 0
    
    for position in positions:
        symbol = position.get('symbol', 'N/A')
        qty = float(position.get('qty', 0))
        market_value = float(position.get('market_value', 0))
        cost_basis = float(position.get('cost_basis', 0))
        unrealized_pl = float(position.get('unrealized_pl', 0))
        unrealized_plpc = float(position.get('unrealized_plpc', 0)) * 100
        
        total_market_value += market_value
        
        print(f"\n📈 {symbol}")
        print(f"   Quantity: {qty:,.0f} shares")
        print(f"   Market Value: {format_currency(market_value)}")
        print(f"   Cost Basis: {format_currency(cost_basis)}")
        print(f"   Unrealized P&L: {format_currency(unrealized_pl)} ({unrealized_plpc:+.2f}%)")
        
        if 'avg_entry_price' in position:
            avg_price = float(position['avg_entry_price'])
            print(f"   Avg Entry Price: {format_currency(avg_price)}")
    
    print(f"\n💼 Total Position Value: {format_currency(total_market_value)}")


def display_portfolio_summary(portfolio: Dict) -> None:
    """Display portfolio performance summary."""
    print("\n" + "=" * 60)
    print("📈 PORTFOLIO PERFORMANCE")
    print("=" * 60)
    
    equity = portfolio.get('equity', [])
    if len(equity) >= 2:
        start_value = equity[0]
        end_value = equity[-1]
        change = end_value - start_value
        change_pct = (change / start_value * 100) if start_value != 0 else 0
        
        print(f"Period Change: {format_currency(change)} ({change_pct:+.2f}%)")
        print(f"Start Value: {format_currency(start_value)}")
        print(f"End Value: {format_currency(end_value)}")


def display_comprehensive_metrics(calculator: PerformanceCalculator) -> None:
    """Display comprehensive performance metrics."""
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS (FROM APRIL 11th, 2024)")
    print("=" * 80)
    
    # Performance Metrics
    print("\n🎯 PERFORMANCE METRICS")
    print("-" * 40)
    total_return = calculator.total_return()
    cagr = calculator.annualized_return()
    
    print(f"Total Return: {total_return:+.2f}%")
    print(f"Annualized Return (CAGR): {cagr:+.2f}%")
    
    # Trade Analysis
    trade_analysis = calculator.analyze_trades()
    if trade_analysis['total_trades'] > 0:
        print(f"Win Rate: {trade_analysis['win_rate']:.1f}%")
        print(f"Profit Factor: {trade_analysis['profit_factor']:.2f}")
        print(f"Average Win: {format_currency(trade_analysis['avg_win'])}")
        print(f"Average Loss: {format_currency(trade_analysis['avg_loss'])}")
        print(f"Expectancy: {format_currency(trade_analysis['expectancy'])}")
        
        # Win/Loss Ratio
        if trade_analysis['avg_loss'] > 0:
            win_loss_ratio = trade_analysis['avg_win'] / trade_analysis['avg_loss']
            print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
    else:
        print("No completed trades found for analysis")
    
    # Risk-Adjusted Metrics
    print("\n⚖️ RISK-ADJUSTED METRICS")
    print("-" * 40)
    sharpe = calculator.calculate_sharpe_ratio()
    sortino = calculator.calculate_sortino_ratio()
    calmar = calculator.calculate_calmar_ratio()
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Calmar Ratio: {calmar:.2f}")
    
    # Risk Metrics
    print("\n⚠️ RISK METRICS")
    print("-" * 40)
    max_dd, recovery_days, drawdown_days = calculator.calculate_max_drawdown()
    volatility = calculator.calculate_volatility()
    var_5 = calculator.calculate_var(0.05)
    
    print(f"Maximum Drawdown: {max_dd:.2f}%")
    print(f"Drawdown Duration: {drawdown_days} days")
    print(f"Recovery Time: {recovery_days} days" if recovery_days > 0 else "Recovery Time: Not yet recovered")
    print(f"Volatility (Annualized): {volatility:.2f}%")
    print(f"Value at Risk (5%): {format_currency(var_5)}")
    
    # Trade Efficiency Metrics
    if trade_analysis['total_trades'] > 0:
        print("\n⚡ TRADE EFFICIENCY METRICS")
        print("-" * 40)
        print(f"Total Trades: {trade_analysis['total_trades']}")
        print(f"Winning Trades: {trade_analysis['winning_trades']}")
        print(f"Losing Trades: {trade_analysis['losing_trades']}")
        
        # Calculate additional metrics if we have equity curve data
        if len(calculator.equity_values) > 1:
            trading_days = len(calculator.equity_values)
            turnover = trade_analysis['total_trades'] / (trading_days / 252) if trading_days > 0 else 0
            print(f"Annual Turnover: {turnover:.1f} trades/year")
    
    # Equity Curve Diagnostics
    equity_analysis = calculator.analyze_equity_curve()
    if equity_analysis:
        print("\n📈 EQUITY CURVE DIAGNOSTICS")
        print("-" * 40)
        print(f"Equity Curve Slope: {equity_analysis['slope']:.2f}")
        print(f"Equity Curve Smoothness: {equity_analysis['smoothness']:.2f}")
        print(f"Trend Strength: {equity_analysis['trend_strength']:.2f}")
    
    # Performance Summary
    print("\n📋 PERFORMANCE SUMMARY")
    print("-" * 40)
    
    # Risk-adjusted performance rating
    risk_score = 0
    if sharpe > 1:
        risk_score += 1
    if sortino > 1:
        risk_score += 1
    if max_dd < 10:
        risk_score += 1
    if trade_analysis.get('win_rate', 0) > 50:
        risk_score += 1
    if trade_analysis.get('profit_factor', 0) > 1.5:
        risk_score += 1
    
    performance_rating = ["Poor", "Below Average", "Average", "Good", "Excellent"][risk_score]
    
    print(f"Performance Rating: {performance_rating} ({risk_score}/5)")
    
    if risk_score >= 4:
        print("🎉 Excellent performance! Strong risk-adjusted returns.")
    elif risk_score >= 3:
        print("✅ Good performance with reasonable risk management.")
    elif risk_score >= 2:
        print("⚠️ Average performance. Consider improving risk management.")
    else:
        print("❌ Performance needs improvement. Review strategy and risk controls.")
    
    print("\n" + "=" * 80)


def main():
    """Main function to fetch and display trading information."""
    print("🚀 Alpaca Paper Trading Information")
    print(f"API Key: {ALPACA_API_KEY}")
    print("Fetching data from Alpaca API...")
    
    # Get account information
    account = get_account_info()
    if account:
        display_account_info(account)
    else:
        print("❌ Failed to fetch account information")
        print("\n🔧 Troubleshooting tips:")
        print("1. Verify your API credentials are correct")
        print("2. Check if your account is activated")
        print("3. Ensure you have the correct permissions")
        print("4. Try generating new API keys from your Alpaca dashboard")
        return
    
    # Get recent orders
    print("\nFetching recent orders...")
    orders = get_orders(status="all", limit=20)
    if orders is not None:
        display_recent_orders(orders)
    else:
        print("❌ Failed to fetch orders")
    
    # Get current positions
    print("\nFetching current positions...")
    positions = get_positions()
    if positions is not None:
        display_positions(positions)
    else:
        print("❌ Failed to fetch positions")
    
    # Get portfolio history
    print("\nFetching portfolio performance...")
    portfolio = get_portfolio_history("1D")
    if portfolio:
        display_portfolio_summary(portfolio)
    else:
        print("❌ Failed to fetch portfolio history")
    
    # Get extended data for comprehensive analysis starting from April 11th
    print("\nFetching extended data for comprehensive analysis from April 11th, 2024...")
    extended_portfolio = get_portfolio_history_from_date("2024-04-11")
    closed_orders = get_closed_orders_from_date("2024-04-11")
    
    if extended_portfolio and account:
        try:
            equity_data = extended_portfolio.get('equity', [])
            print(f"📊 Analyzing {len(equity_data)} data points...")
            
            # Debug: Show equity curve range
            if equity_data:
                print(f"🔍 Debug: Equity range ${equity_data[0]:,.2f} → ${equity_data[-1]:,.2f}")
                print(f"🔍 Debug: Current account value: ${float(account.get('portfolio_value', 0)):,.2f}")
            
            calculator = PerformanceCalculator(extended_portfolio, closed_orders or [], account)
            display_comprehensive_metrics(calculator)
        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            print("This might be due to insufficient trading history.")
            
            # Try with basic portfolio data as fallback
            print("\n🔄 Attempting basic analysis with available data...")
            try:
                if portfolio:
                    basic_calculator = PerformanceCalculator(portfolio, closed_orders or [], account)
                    display_comprehensive_metrics(basic_calculator)
                else:
                    print("❌ No portfolio data available for analysis")
            except Exception as e2:
                print(f"❌ Basic analysis also failed: {e2}")
    else:
        print("❌ Insufficient data for comprehensive performance analysis")
        print("📝 Note: Extended analysis requires historical trading data")
        
        # Try with basic data if available
        if portfolio and account:
            print("\n🔄 Attempting basic analysis with daily data...")
            try:
                basic_calculator = PerformanceCalculator(portfolio, closed_orders or [], account)
                display_comprehensive_metrics(basic_calculator)
            except Exception as e:
                print(f"❌ Basic analysis failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Data fetch complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        sys.exit(1)
