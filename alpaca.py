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
import os
import subprocess
import platform

# Chart generation imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from scipy import stats
    CHARTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Chart dependencies not available: {e}")
    print("📦 Install with: pip install matplotlib seaborn pandas numpy scipy")
    CHARTS_AVAILABLE = False

# Alpaca API Configuration
# Try to import from secure config file first, then fall back to environment variables
try:
    from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, DEFAULT_ANALYSIS_PERIODS, MAX_ORDERS_TO_FETCH, ANALYSIS_START_DATE
    print("🔑 Using configuration from config.py")
except ImportError:
    # Fallback to environment variables and defaults
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_API_KEY_HERE')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'YOUR_SECRET_KEY_HERE')
    DEFAULT_ANALYSIS_PERIODS = ["1M", "3M", "1Y", "1W", "1D"]
    MAX_ORDERS_TO_FETCH = 100
    ANALYSIS_START_DATE = None
    if ALPACA_API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠️  No API keys found. Please:")
        print("   1. Edit config.py with your Alpaca API keys, OR")
        print("   2. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")

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
    print("🔍 Attempting to fetch full portfolio history...")
    
    # Try to get maximum available history with different approaches
    strategies = [
        # Strategy 1: Try maximum periods first
        {"periods": ["1Y", "6M", "3M", "1M"], "description": "Maximum period"},
        
        # Strategy 2: Try date-based fetching from account creation
        {"dates": ["2024-01-01", "2024-04-01", "2024-07-01"], "description": "Date-based"},
        
        # Strategy 3: Try all available periods
        {"periods": ["2Y", "1Y", "6M", "3M", "1M", "1W"], "description": "All periods"}
    ]
    
    for strategy in strategies:
        if "periods" in strategy:
            print(f"🔍 Trying {strategy['description']} approach...")
            for period in strategy["periods"]:
                try:
                    print(f"  📅 Attempting {period} period...")
                    params = {"period": period}
                    response = requests.get(f"{BASE_URL}/v2/account/portfolio/history", headers=HEADERS, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        equity_data = data.get('equity', [])
                        if equity_data and len(equity_data) > 1:
                            print(f"✅ Success! Got {len(equity_data)} data points from {period} period")
                            return data
                    elif response.status_code == 422:
                        print(f"  ❌ {period} not available")
                        continue
                    else:
                        print(f"  ❌ Error {response.status_code} for {period}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"  ❌ Request failed for {period}: {e}")
                    continue
        
        elif "dates" in strategy:
            print(f"🔍 Trying {strategy['description']} approach...")
            for start_date in strategy["dates"]:
                try:
                    print(f"  📅 Attempting from {start_date}...")
                    data = get_portfolio_history_from_date(start_date)
                    if data and data.get('equity') and len(data.get('equity', [])) > 1:
                        print(f"✅ Success! Got {len(data.get('equity', []))} data points from {start_date}")
                        return data
                except Exception as e:
                    print(f"  ❌ Failed for {start_date}: {e}")
                    continue
    
    print("⚠️ Could not fetch extended history, will use basic data")
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
        # Alpaca API limit is 500 per request, so we'll fetch maximum
        params = {
            "status": "closed",
            "limit": 500,  # Maximum allowed by API
            "direction": "desc"
        }
        response = requests.get(f"{BASE_URL}/v2/orders", headers=HEADERS, params=params)
        response.raise_for_status()
        
        orders = response.json()
        print(f"📋 Fetched {len(orders)} closed orders (API limit: 500)")
        
        # Note: For accounts with >500 orders, would need pagination
        # This gets the 500 most recent closed orders
        return orders
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


class ChartGenerator:
    """Generate comprehensive trading analytics charts."""
    
    def __init__(self, calculator: PerformanceCalculator):
        self.calculator = calculator
        self.charts_dir = "charts"
        self.setup_style()
        self.ensure_charts_directory()
    
    def setup_style(self):
        """Setup matplotlib and seaborn styling."""
        if not CHARTS_AVAILABLE:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'use') else 'default')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def ensure_charts_directory(self):
        """Create charts directory if it doesn't exist."""
        if not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)
    
    def open_file_in_explorer(self, filepath: str):
        """Open file in default application or explorer."""
        try:
            system = platform.system().lower()
            abs_path = os.path.abspath(filepath)
            
            if system == "windows":
                # Use os.startfile for Windows
                os.startfile(abs_path)
            elif system == "darwin":  # macOS
                subprocess.run(["open", abs_path], check=False)
            else:  # Linux and others
                subprocess.run(["xdg-open", abs_path], check=False)
                
            print(f"📁 Chart opened: {abs_path}")
        except Exception as e:
            print(f"⚠️ Could not open file automatically: {e}")
            print(f"📁 Chart saved to: {os.path.abspath(filepath)}")
    
    def create_equity_curve_chart(self, save_and_open: bool = True) -> Optional[str]:
        """Create equity curve chart with key metrics annotations."""
        if not CHARTS_AVAILABLE or not self.calculator.equity_values:
            print("⚠️ Cannot create equity curve chart - no data or dependencies missing")
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
            
            # Prepare data
            dates = self.calculator.dates
            equity = self.calculator.equity_values
            
            if not dates or len(dates) != len(equity):
                dates = list(range(len(equity)))
        
            # Main equity curve
            ax1.plot(dates, equity, linewidth=2, color='#2E86AB', label='Portfolio Value')
            ax1.fill_between(dates, equity, alpha=0.3, color='#2E86AB')
            
            # Add peak and trough annotations
            peak_idx = np.argmax(equity)
            trough_idx = np.argmin(equity)
            
            ax1.scatter(dates[peak_idx], equity[peak_idx], color='green', s=100, zorder=5)
            ax1.annotate(f'Peak: ${equity[peak_idx]:,.0f}', 
                        xy=(dates[peak_idx], equity[peak_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax1.scatter(dates[trough_idx], equity[trough_idx], color='red', s=100, zorder=5)
            ax1.annotate(f'Trough: ${equity[trough_idx]:,.0f}', 
                        xy=(dates[trough_idx], equity[trough_idx]),
                        xytext=(10, -20), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Format axes
            ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Returns subplot
            returns = self.calculator.calculate_returns()
            if returns and len(returns) == len(dates) - 1:
                returns_dates = dates[1:]
                ax2.bar(returns_dates, [r * 100 for r in returns], 
                       color=['green' if r > 0 else 'red' for r in returns], 
                       alpha=0.7, width=1.0)
            
            ax2.set_title('Daily Returns (%)', fontsize=12)
            ax2.set_ylabel('Return (%)', fontsize=10)
            ax2.set_xlabel('Time Period', fontsize=10)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis if we have actual dates
            if isinstance(dates[0], datetime):
                for ax in [ax1, ax2]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
        
            if save_and_open:
                filename = f"{self.charts_dir}/equity_curve.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"📈 Equity curve chart saved: {filename}")
                self.open_file_in_explorer(filename)
                return filename
            
            return None
            
        except Exception as e:
            print(f"❌ Error creating equity curve chart: {e}")
            plt.close('all')
            return None
    
    def create_drawdown_chart(self, save_and_open: bool = True) -> Optional[str]:
        """Create underwater (drawdown) chart."""
        if not CHARTS_AVAILABLE or not self.calculator.equity_values:
            print("⚠️ Cannot create drawdown chart - no data or dependencies missing")
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Calculate drawdown
            equity = self.calculator.equity_values
            dates = self.calculator.dates
            
            if not dates or len(dates) != len(equity):
                dates = list(range(len(equity)))
            
            peak = equity[0]
            drawdowns = []
            
            for value in equity:
                if value > peak:
                    peak = value
                drawdown = (value - peak) / peak * 100
                drawdowns.append(drawdown)
        
            # Create underwater chart
            ax.fill_between(dates, drawdowns, 0, color='red', alpha=0.3, label='Drawdown')
            ax.plot(dates, drawdowns, color='darkred', linewidth=2)
            
            # Add maximum drawdown annotation
            max_dd_idx = np.argmin(drawdowns)
            max_dd_value = drawdowns[max_dd_idx]
            
            ax.scatter(dates[max_dd_idx], max_dd_value, color='red', s=150, zorder=5)
            ax.annotate(f'Max DD: {max_dd_value:.1f}%', 
                       xy=(dates[max_dd_idx], max_dd_value),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.set_title('Portfolio Drawdown (Underwater Chart)', fontsize=16, fontweight='bold')
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.set_xlabel('Time Period', fontsize=12)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis if we have actual dates
            if isinstance(dates[0], datetime):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            if save_and_open:
                filename = f"{self.charts_dir}/drawdown_chart.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"📉 Drawdown chart saved: {filename}")
                self.open_file_in_explorer(filename)
                return filename
            
            return None
            
        except Exception as e:
            print(f"❌ Error creating drawdown chart: {e}")
            plt.close('all')
            return None
    
    def create_returns_analysis_chart(self, save_and_open: bool = True) -> Optional[str]:
        """Create returns distribution and rolling metrics chart."""
        if not CHARTS_AVAILABLE:
            print("⚠️ Cannot create returns analysis chart - dependencies missing")
            return None
        
        returns = self.calculator.calculate_returns()
        if not returns:
            print("⚠️ Cannot create returns analysis chart - no returns data")
            return None
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Returns distribution histogram
            ax1.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.4f}')
            ax1.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.4f}')
            ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Daily Return')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
            # Q-Q plot for normality check
            try:
                stats.probplot(returns, dist="norm", plot=ax2)
                ax2.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            except Exception as e:
                print(f"⚠️ Could not create Q-Q plot: {e}")
                ax2.text(0.5, 0.5, 'Q-Q Plot\nNot Available', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
            
            # Rolling Sharpe ratio
            if len(returns) > 30:
                rolling_window = min(30, len(returns) // 3)
                rolling_returns = pd.Series(returns)
                rolling_sharpe = rolling_returns.rolling(rolling_window).mean() / rolling_returns.rolling(rolling_window).std() * np.sqrt(252)
                
                ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='purple')
                ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sharpe = 1')
                ax3.axhline(y=2, color='green', linestyle='--', alpha=0.7, label='Sharpe = 2')
                ax3.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day)', fontsize=14, fontweight='bold')
                ax3.set_ylabel('Sharpe Ratio')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Rolling volatility
            if len(returns) > 30:
                rolling_vol = rolling_returns.rolling(rolling_window).std() * np.sqrt(252) * 100
                ax4.plot(rolling_vol.index, rolling_vol.values, linewidth=2, color='orange')
                ax4.set_title(f'Rolling Volatility ({rolling_window}-day)', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Annualized Volatility (%)')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_and_open:
                filename = f"{self.charts_dir}/returns_analysis.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"📊 Returns analysis chart saved: {filename}")
                self.open_file_in_explorer(filename)
                return filename
            
            return None
            
        except Exception as e:
            print(f"❌ Error creating returns analysis chart: {e}")
            plt.close('all')
            return None
    
    def create_trade_analysis_chart(self, save_and_open: bool = True) -> Optional[str]:
        """Create trade P&L analysis charts."""
        if not CHARTS_AVAILABLE:
            return None
        
        trade_analysis = self.calculator.analyze_trades()
        if trade_analysis['total_trades'] == 0:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Trade P&L scatter plot
        trade_pnls = trade_analysis.get('trade_pnls', [])
        if trade_pnls:
            trade_numbers = list(range(1, len(trade_pnls) + 1))
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
            
            ax1.scatter(trade_numbers, trade_pnls, c=colors, alpha=0.7, s=50)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax1.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Trade Number')
            ax1.set_ylabel('P&L ($)')
            ax1.grid(True, alpha=0.3)
            
            # Cumulative P&L
            cumulative_pnl = np.cumsum(trade_pnls)
            ax2.plot(trade_numbers, cumulative_pnl, linewidth=2, color='blue')
            ax2.fill_between(trade_numbers, cumulative_pnl, alpha=0.3, color='blue')
            ax2.set_title('Cumulative Trade P&L', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Cumulative P&L ($)')
            ax2.grid(True, alpha=0.3)
        
        # Win/Loss distribution pie chart
        win_count = trade_analysis['winning_trades']
        loss_count = trade_analysis['losing_trades']
        
        if win_count > 0 or loss_count > 0:
            labels = ['Winning Trades', 'Losing Trades']
            sizes = [win_count, loss_count]
            colors = ['lightgreen', 'lightcoral']
            explode = (0.05, 0.05)
            
            ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            ax3.set_title(f'Win/Loss Ratio\n(Win Rate: {trade_analysis["win_rate"]:.1f}%)', 
                         fontsize=14, fontweight='bold')
        
        # Performance metrics bar chart
        metrics = {
            'Win Rate (%)': trade_analysis['win_rate'],
            'Profit Factor': trade_analysis['profit_factor'],
            'Avg Win ($)': trade_analysis['avg_win'],
            'Avg Loss ($)': abs(trade_analysis['avg_loss'])
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax4.bar(metric_names, metric_values, color=['green', 'blue', 'lightgreen', 'lightcoral'])
        ax4.set_title('Key Trading Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_and_open:
            filename = f"{self.charts_dir}/trade_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💰 Trade analysis chart saved: {filename}")
            self.open_file_in_explorer(filename)
            return filename
        
        return None
    
    def create_risk_metrics_dashboard(self, save_and_open: bool = True) -> Optional[str]:
        """Create comprehensive risk metrics dashboard."""
        if not CHARTS_AVAILABLE:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Risk metrics gauge charts
        sharpe = self.calculator.calculate_sharpe_ratio()
        sortino = self.calculator.calculate_sortino_ratio()
        max_dd, _, _ = self.calculator.calculate_max_drawdown()
        volatility = self.calculator.calculate_volatility()
        
        # Sharpe ratio gauge
        self._create_gauge_chart(ax1, sharpe, "Sharpe Ratio", 
                               thresholds=[0, 1, 2, 3], 
                               colors=['red', 'yellow', 'lightgreen', 'green'])
        
        # Sortino ratio gauge
        self._create_gauge_chart(ax2, sortino, "Sortino Ratio", 
                               thresholds=[0, 1, 2, 3], 
                               colors=['red', 'yellow', 'lightgreen', 'green'])
        
        # Maximum drawdown gauge
        self._create_gauge_chart(ax3, max_dd, "Max Drawdown (%)", 
                               thresholds=[0, 5, 10, 20], 
                               colors=['green', 'lightgreen', 'yellow', 'red'],
                               reverse=True)
        
        # Volatility gauge
        self._create_gauge_chart(ax4, volatility, "Volatility (%)", 
                               thresholds=[0, 10, 20, 40], 
                               colors=['green', 'lightgreen', 'yellow', 'red'],
                               reverse=True)
        
        plt.suptitle('Risk Metrics Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_and_open:
            filename = f"{self.charts_dir}/risk_dashboard.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"⚠️ Risk dashboard saved: {filename}")
            self.open_file_in_explorer(filename)
            return filename
        
        return None
    
    def _create_gauge_chart(self, ax, value, title, thresholds, colors, reverse=False):
        """Create a gauge chart for risk metrics."""
        # Determine color based on value and thresholds
        if reverse:
            color_idx = 0
            for i, threshold in enumerate(thresholds[1:], 1):
                if value >= threshold:
                    color_idx = i
        else:
            color_idx = len(colors) - 1
            for i, threshold in enumerate(thresholds[1:], 1):
                if value < threshold:
                    color_idx = i - 1
                    break
        
        color = colors[min(color_idx, len(colors) - 1)]
        
        # Create bar chart as gauge
        ax.bar(0, value, color=color, alpha=0.7, width=0.5)
        ax.set_title(f'{title}\n{value:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        
        # Add threshold lines
        for threshold in thresholds[1:]:
            ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
    
    def create_performance_summary_dashboard(self, save_and_open: bool = True) -> Optional[str]:
        """Create comprehensive performance summary dashboard."""
        if not CHARTS_AVAILABLE:
            return None
        
        fig = plt.figure(figsize=(20, 14))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main equity curve (top row, spans 3 columns)
        ax_equity = fig.add_subplot(gs[0, :3])
        dates = self.calculator.dates
        equity = self.calculator.equity_values
        
        if not dates or len(dates) != len(equity):
            dates = list(range(len(equity)))
        
        ax_equity.plot(dates, equity, linewidth=3, color='#2E86AB')
        ax_equity.fill_between(dates, equity, alpha=0.3, color='#2E86AB')
        ax_equity.set_title('Portfolio Performance Overview', fontsize=16, fontweight='bold')
        ax_equity.set_ylabel('Portfolio Value ($)')
        ax_equity.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax_equity.grid(True, alpha=0.3)
        
        # Key metrics panel (top right)
        ax_metrics = fig.add_subplot(gs[0, 3])
        ax_metrics.axis('off')
        
        total_return = self.calculator.total_return()
        cagr = self.calculator.annualized_return()
        sharpe = self.calculator.calculate_sharpe_ratio()
        max_dd, _, _ = self.calculator.calculate_max_drawdown()
        
        metrics_text = f"""
KEY METRICS

Total Return: {total_return:+.1f}%
CAGR: {cagr:+.1f}%
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.1f}%

Portfolio Value:
${equity[-1] if equity else 0:,.0f}
        """
        
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Drawdown chart (middle left)
        ax_dd = fig.add_subplot(gs[1, :2])
        peak = equity[0] if equity else 0
        drawdowns = []
        for value in equity:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak * 100 if peak > 0 else 0
            drawdowns.append(drawdown)
        
        ax_dd.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
        ax_dd.plot(dates, drawdowns, color='darkred', linewidth=2)
        ax_dd.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.grid(True, alpha=0.3)
        
        # Returns distribution (middle right)
        ax_returns = fig.add_subplot(gs[1, 2:])
        returns = self.calculator.calculate_returns()
        if returns:
            ax_returns.hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax_returns.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2)
            ax_returns.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax_returns.set_xlabel('Daily Return')
            ax_returns.set_ylabel('Frequency')
            ax_returns.grid(True, alpha=0.3)
        
        # Trade analysis (bottom left)
        trade_analysis = self.calculator.analyze_trades()
        if trade_analysis['total_trades'] > 0:
            ax_trades = fig.add_subplot(gs[2, :2])
            
            win_count = trade_analysis['winning_trades']
            loss_count = trade_analysis['losing_trades']
            
            if win_count > 0 or loss_count > 0:
                labels = ['Wins', 'Losses']
                sizes = [win_count, loss_count]
                colors = ['lightgreen', 'lightcoral']
                
                ax_trades.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                             startangle=90)
                ax_trades.set_title(f'Trade Analysis\nWin Rate: {trade_analysis["win_rate"]:.1f}%', 
                                   fontsize=14, fontweight='bold')
        
        # Risk metrics summary (bottom right)
        ax_risk = fig.add_subplot(gs[2, 2:])
        ax_risk.axis('off')
        
        volatility = self.calculator.calculate_volatility()
        sortino = self.calculator.calculate_sortino_ratio()
        calmar = self.calculator.calculate_calmar_ratio()
        
        risk_text = f"""
RISK ANALYSIS

Volatility: {volatility:.1f}%
Sortino Ratio: {sortino:.2f}
Calmar Ratio: {calmar:.2f}

Max Drawdown: {max_dd:.1f}%
Sharpe Ratio: {sharpe:.2f}
        """
        
        ax_risk.text(0.1, 0.9, risk_text, transform=ax_risk.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        plt.suptitle('Portfolio Performance Dashboard', fontsize=18, fontweight='bold')
        
        if save_and_open:
            filename = f"{self.charts_dir}/performance_dashboard.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Performance dashboard saved: {filename}")
            self.open_file_in_explorer(filename)
            return filename
        
        return None
    
    def generate_all_charts(self, save_and_open: bool = True) -> List[str]:
        """Generate all available charts."""
        if not CHARTS_AVAILABLE:
            print("❌ Chart generation not available. Please install required dependencies:")
            print("   pip install matplotlib seaborn pandas numpy scipy")
            return []
        
        generated_files = []
        
        print("\n🎨 Generating comprehensive analytics charts...")
        
        # Validate data before generating charts
        if not self.calculator.equity_values:
            print("❌ No equity data available for chart generation")
            return []
        
        if len(self.calculator.equity_values) < 2:
            print("❌ Insufficient equity data for meaningful charts (need at least 2 data points)")
            return []
        
        # Generate all charts
        charts = [
            ("Equity Curve", self.create_equity_curve_chart),
            ("Drawdown Analysis", self.create_drawdown_chart),
            ("Returns Analysis", self.create_returns_analysis_chart),
            ("Trade Analysis", self.create_trade_analysis_chart),
            ("Risk Dashboard", self.create_risk_metrics_dashboard),
            ("Performance Dashboard", self.create_performance_summary_dashboard)
        ]
        
        for chart_name, chart_func in charts:
            try:
                print(f"  📈 Creating {chart_name}...")
                filename = chart_func(save_and_open)
                if filename and os.path.exists(filename):
                    generated_files.append(filename)
                    print(f"  ✅ {chart_name} created successfully")
                else:
                    print(f"  ⚠️ {chart_name} was not created (may be due to insufficient data)")
            except Exception as e:
                print(f"  ❌ Error creating {chart_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Clean up matplotlib resources
        try:
            plt.close('all')
        except:
            pass
        
        if generated_files:
            print(f"\n✅ Generated {len(generated_files)} charts in '{self.charts_dir}' directory")
            print("📁 Charts can be opened directly from file explorer")
            print("\n📊 Generated charts:")
            for filename in generated_files:
                print(f"  • {os.path.basename(filename)}")
        else:
            print("\n⚠️ No charts were generated successfully")
            print("💡 This may be due to insufficient data or missing dependencies")
        
        return generated_files


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
    print("🚀 Alpaca Trading Analytics & Chart Generator")
    print(f"API Key: {ALPACA_API_KEY}")
    
    # Check for chart generation capability
    if CHARTS_AVAILABLE:
        print("📊 Advanced analytics charts enabled")
    else:
        print("⚠️  Basic mode - install chart dependencies for advanced analytics:")
        print("   pip install matplotlib seaborn pandas numpy scipy")
    
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
    orders = get_orders(status="all", limit=MAX_ORDERS_TO_FETCH)
    if orders is not None:
        print(f"📋 Found {len(orders)} total orders")
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
    
    # Get portfolio history - try extended periods for better analysis
    print("\nFetching portfolio performance...")
    # Try different periods to get meaningful data
    portfolio = None
    for period in DEFAULT_ANALYSIS_PERIODS:
        portfolio = get_portfolio_history(period)
        if portfolio and portfolio.get('equity') and len(portfolio.get('equity', [])) > 1:
            print(f"📊 Using {period} portfolio data ({len(portfolio.get('equity', []))} data points)")
            break
    
    if portfolio:
        display_portfolio_summary(portfolio)
    else:
        print("❌ Failed to fetch portfolio history")
    
    # Get extended data for comprehensive analysis
    print("\nFetching extended data for comprehensive analysis...")
    extended_portfolio = get_extended_portfolio_history()
    
    # Try to get maximum order history
    print("🔍 Fetching maximum available order history...")
    closed_orders = None
    
    # Strategy 1: Try to get all orders first (maximum history)
    try:
        print("  📋 Attempting to fetch all available orders...")
        all_orders = get_closed_orders()
        if all_orders and len(all_orders) > 0:
            print(f"✅ Found {len(all_orders)} total orders")
            closed_orders = all_orders
    except Exception as e:
        print(f"  ❌ Failed to fetch all orders: {e}")
    
    # Strategy 2: Try different time periods if needed
    if not closed_orders or len(closed_orders) < 10:
        print("🔍 Trying date-based order fetching for more history...")
        for months_back in [12, 6, 3, 1]:  # Try 1 year first, then shorter periods
            start_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
            print(f"  📅 Trying {months_back} month(s) back ({start_date})...")
            date_orders = get_closed_orders_from_date(start_date)
            if date_orders and len(date_orders) > (len(closed_orders) if closed_orders else 0):
                print(f"✅ Found {len(date_orders)} orders from {start_date}")
                closed_orders = date_orders
                break
    
    if extended_portfolio and account:
        try:
            print("\n🔍 Performing comprehensive performance analysis...")
            
            # Create performance calculator
            calculator = PerformanceCalculator(
                extended_portfolio, 
                closed_orders if closed_orders else [], 
                account
            )
            
            # Display comprehensive metrics
            display_comprehensive_metrics(calculator)
            
            # Generate charts if available
            if CHARTS_AVAILABLE:
                print("\n🎨 Generating analytics charts...")
                chart_generator = ChartGenerator(calculator)
                generated_files = chart_generator.generate_all_charts(save_and_open=True)
                
                if generated_files:
                    print(f"\n✅ Successfully generated {len(generated_files)} charts:")
                    for filename in generated_files:
                        print(f"  📊 {filename}")
                else:
                    print("⚠️ No charts were generated")
            else:
                print("\n📊 Chart generation skipped - dependencies not available")
                
        except Exception as e:
            print(f"❌ Error during comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ Insufficient data for comprehensive analysis")
        if not extended_portfolio:
            print("  • Could not fetch extended portfolio history")
        if not account:
            print("  • Could not fetch account information")

    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    
    if CHARTS_AVAILABLE:
        print("📈 Analytics Summary:")
        print("  • Comprehensive performance metrics calculated")
        print("  • Professional charts generated and saved")
        print("  • Charts automatically opened in default viewer")
        print("  • Files saved in 'charts' directory for future reference")
        print("\n💡 Tip: Charts are high-resolution PNG files suitable for presentations")
    else:
        print("📊 Install chart dependencies for visual analytics:")
        print("  pip install matplotlib seaborn pandas numpy scipy")
    
    print("\n🎯 Next Steps:")
    print("  • Review performance metrics and charts")
    print("  • Analyze risk-adjusted returns")
    print("  • Compare against benchmarks")
    print("  • Consider portfolio adjustments based on insights")


def test_chart_generation():
    """Test chart generation with sample data."""
    if not CHARTS_AVAILABLE:
        print("❌ Chart dependencies not available")
        print("📦 Install with: pip install matplotlib seaborn pandas numpy scipy")
        return False
    
    print("🧪 Testing chart generation with sample data...")
    
    try:
        # Create sample data
        import random
        random.seed(42)  # For reproducible results
        
        dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
        equity_values = []
        current_value = 10000
        
        for _ in range(100):
            change = random.uniform(-0.02, 0.02)  # ±2% daily change
            current_value *= (1 + change)
            equity_values.append(current_value)
        
        print(f"📊 Generated sample data: {len(equity_values)} data points")
        print(f"📈 Value range: ${equity_values[0]:,.2f} → ${equity_values[-1]:,.2f}")
        
        # Create sample portfolio history
        sample_portfolio = {
            'equity': equity_values,
            'timestamp': [int(d.timestamp()) for d in dates]
        }
        
        # Create sample account
        sample_account = {
            'equity': str(equity_values[-1]),
            'cash': '5000',
            'portfolio_value': str(equity_values[-1])
        }
        
        # Create sample orders
        sample_orders = []
        
        print("🔧 Creating performance calculator...")
        calculator = PerformanceCalculator(sample_portfolio, sample_orders, sample_account)
        
        print("🎨 Creating chart generator...")
        chart_generator = ChartGenerator(calculator)
        
        # Test all chart generation
        print("📈 Testing all chart generation...")
        print(f"🔍 Calculator has {len(calculator.equity_values)} equity values")
        print(f"🔍 Calculator has {len(calculator.dates)} dates")
        
        generated_files = chart_generator.generate_all_charts(save_and_open=True)
        
        if generated_files:
            print(f"✅ Chart generation test passed: {len(generated_files)} charts created")
            for filename in generated_files:
                print(f"  📊 {os.path.basename(filename)}")
            return True
        else:
            print("❌ Chart generation test failed - no charts created")
            return False
            
    except Exception as e:
        print(f"❌ Chart generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # Check if user wants to test chart generation
        if len(sys.argv) > 1 and sys.argv[1] == "--test-charts":
            test_chart_generation()
        else:
            main()
    except KeyboardInterrupt:
        print("\n\n👋 Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
