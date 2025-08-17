#!/usr/bin/env python3
"""
Alpaca Trading API Analytics Script

A comprehensive trading analytics tool that fetches data from Alpaca API
and generates professional performance reports with charts.

Author: Trading Analytics Team
Version: 2.0.0
License: MIT
"""

import requests
import json
import math
import statistics
import logging
import asyncio
import aiohttp
import concurrent.futures
import multiprocessing as mp
from functools import lru_cache, partial
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os
import subprocess
import platform
from contextlib import contextmanager
import time
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Performance optimization for AMD Ryzen 7 7840U
class PerformanceConfig:
    """Performance configuration optimized for AMD Ryzen 7 7840U."""
    
    def __init__(self):
        # Detect system capabilities
        self.cpu_count = mp.cpu_count()  # 16 threads on 7840U
        self.physical_cores = psutil.cpu_count(logical=False)  # 8 cores on 7840U
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Optimized settings for 7840U (8C/16T, efficient mobile CPU)
        self.max_workers = min(8, self.physical_cores)  # Use physical cores for CPU-bound tasks
        self.io_workers = min(16, self.cpu_count)  # Use all threads for I/O-bound tasks
        self.chart_workers = min(4, self.physical_cores // 2)  # Conservative for chart generation
        self.memory_limit_mb = int(self.memory_gb * 1024 * 0.7)  # Use 70% of available RAM
        self.enable_vectorization = True
        self.use_async_io = True
        self.chunk_size = 1000  # Optimal for memory usage
        
        logger.info(f"🚀 Performance Config for AMD Ryzen 7 7840U:")
        logger.info(f"   CPU: {self.physical_cores}C/{self.cpu_count}T, RAM: {self.memory_gb:.1f}GB")
        logger.info(f"   Workers: {self.max_workers} CPU, {self.io_workers} I/O, {self.chart_workers} Charts")

# Constants
class TradingEnvironment(Enum):
    """Trading environment types."""
    PAPER = "paper"
    LIVE = "live"

@dataclass
class APIConfig:
    """API configuration dataclass optimized for performance."""
    paper_url: str = "https://paper-api.alpaca.markets"
    live_url: str = "https://api.alpaca.markets"
    timeout: int = 15  # Reduced for faster failover
    max_retries: int = 2  # Reduced for faster execution
    rate_limit_delay: float = 0.1  # Reduced delay
    connection_pool_size: int = 10  # Connection pooling
    
# Initialize performance configuration
perf_config = PerformanceConfig()

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
    logger.info("📊 Chart dependencies loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️  Chart dependencies not available: {e}")
    logger.info("📦 Install with: pip install matplotlib seaborn pandas numpy scipy")
    CHARTS_AVAILABLE = False

class OptimizedAlpacaAPIManager:
    """Optimized API manager for AMD Ryzen 7 7840U with async support and caching."""
    
    def __init__(self):
        """Initialize optimized API manager."""
        self.config = APIConfig()
        self._load_credentials()
        self.base_url: Optional[str] = None
        self.environment: Optional[TradingEnvironment] = None
        self._session: Optional[requests.Session] = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self.cache_ttl = 60  # Cache for 1 minute
        
    def _load_credentials(self) -> None:
        """Load API credentials securely."""
        try:
            from config import (
                ALPACA_API_KEY, ALPACA_SECRET_KEY, 
                DEFAULT_ANALYSIS_PERIODS, MAX_ORDERS_TO_FETCH, 
                ANALYSIS_START_DATE
            )
            self.api_key = ALPACA_API_KEY
            self.secret_key = ALPACA_SECRET_KEY
            self.default_analysis_periods = DEFAULT_ANALYSIS_PERIODS
            self.max_orders_to_fetch = MAX_ORDERS_TO_FETCH
            self.analysis_start_date = ANALYSIS_START_DATE
            logger.info("🔑 Using configuration from config.py")
        except ImportError:
            self.api_key = os.getenv('ALPACA_API_KEY', '')
            self.secret_key = os.getenv('ALPACA_SECRET_KEY', '')
            self.default_analysis_periods = ["1M", "3M", "1Y", "1W", "1D"]
            self.max_orders_to_fetch = 100
            self.analysis_start_date = None
            
            if not self.api_key or not self.secret_key:
                logger.error("⚠️  No API keys found. Please:")
                logger.error("   1. Edit config.py with your Alpaca API keys, OR")
                logger.error("   2. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
                raise ValueError("Missing API credentials")
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
        
    @property
    def session(self) -> requests.Session:
        """Get or create optimized session with connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            # Configure session for optimal performance on 7840U
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.config.connection_pool_size,
                pool_maxsize=self.config.connection_pool_size,
                max_retries=self.config.max_retries
            )
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
            self._session.headers.update(self.headers)
        return self._session
    
    def _get_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for request."""
        key = endpoint
        if params:
            key += "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return key
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[key] < self.cache_ttl
    
    def _get_cached_or_fetch(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Get data from cache or fetch from API."""
        cache_key = self._get_cache_key(endpoint, params)
        
        if self._is_cache_valid(cache_key):
            logger.debug(f"📋 Cache hit for {endpoint}")
            return self._cache[cache_key]
        
        try:
            response = self.session.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=self.config.timeout
            )
            data = validate_api_response(response, f"GET {endpoint}")
            
            # Cache the result
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = time.time()
            logger.debug(f"💾 Cached result for {endpoint}")
            
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {endpoint}: {e}")
            return None
        
    def find_working_endpoint(self) -> Optional[str]:
        """Find which endpoint works with optimized connection testing."""
        endpoints = [
            (self.config.paper_url, TradingEnvironment.PAPER),
            (self.config.live_url, TradingEnvironment.LIVE)
        ]
        
        # Use ThreadPoolExecutor for concurrent testing
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._test_endpoint, endpoint, env): (endpoint, env)
                for endpoint, env in endpoints
            }
            
            for future in concurrent.futures.as_completed(futures, timeout=10):
                endpoint, env = futures[future]
                try:
                    if future.result():
                        logger.info(f"✅ Successfully connected to: {endpoint}")
                        self.base_url = endpoint
                        self.environment = env
                        return endpoint
                except Exception as e:
                    logger.error(f"❌ Connection failed for {endpoint}: {e}")
        
        return None
    
    def _test_endpoint(self, endpoint: str, env: TradingEnvironment) -> bool:
        """Test a single endpoint."""
        try:
            logger.info(f"🔍 Testing {env.value} endpoint: {endpoint}")
            response = requests.get(
                f"{endpoint}/v2/account", 
                headers=self.headers, 
                timeout=self.config.timeout
            )
            if response.status_code == 200:
                return True
            elif response.status_code == 401:
                logger.warning(f"❌ Unauthorized for: {endpoint}")
            else:
                logger.warning(f"❌ Error {response.status_code} for: {endpoint}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Connection failed for {endpoint}: {e}")
        return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information with caching."""
        return self._get_cached_or_fetch("/v2/account")
    
    def get_orders(self, status: str = "all", limit: int = 50) -> Optional[List[Dict]]:
        """Get orders with caching."""
        params = {
            "status": status,
            "limit": min(limit, 500),  # Respect API limits
            "direction": "desc"
        }
        return self._get_cached_or_fetch("/v2/orders", params)
    
    def get_positions(self) -> Optional[List[Dict]]:
        """Get current positions with caching."""
        return self._get_cached_or_fetch("/v2/positions")
    
    def get_portfolio_history(self, period: str = "1D") -> Optional[Dict]:
        """Get portfolio history with caching."""
        params = {"period": period}
        return self._get_cached_or_fetch("/v2/account/portfolio/history", params)
    
    def cleanup(self):
        """Clean up resources."""
        if self._session:
            self._session.close()
        self._cache.clear()
        self._cache_timestamps.clear()
        gc.collect()  # Force garbage collection

# Initialize optimized API manager
try:
    api_manager = OptimizedAlpacaAPIManager()
except ValueError as e:
    logger.error(f"Failed to initialize API manager: {e}")
    sys.exit(1)


@contextmanager
def managed_matplotlib_figure(*args, **kwargs):
    """Context manager for matplotlib figures to ensure proper cleanup."""
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)

class APIException(Exception):
    """Custom exception for API-related errors."""
    pass

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_api_response(response: requests.Response, operation: str) -> Dict[str, Any]:
    """Validate API response and return JSON data."""
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error during {operation}: {e}")
        raise APIException(f"HTTP error during {operation}: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during {operation}: {e}")
        raise APIException(f"Request error during {operation}: {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error during {operation}: {e}")
        raise APIException(f"Invalid JSON response during {operation}")

def find_working_endpoint() -> Optional[str]:
    """Legacy function - use api_manager.find_working_endpoint() instead."""
    logger.warning("Using deprecated find_working_endpoint(). Use api_manager instead.")
    return api_manager.find_working_endpoint()


def get_account_info() -> Optional[Dict]:
    """Legacy function - use api_manager.get_account_info() instead."""
    logger.warning("Using deprecated get_account_info(). Use api_manager instead.")
    return api_manager.get_account_info()


def get_orders(status: str = "all", limit: int = 50) -> Optional[List[Dict]]:
    """Legacy function - use api_manager.get_orders() instead."""
    logger.warning("Using deprecated get_orders(). Use api_manager instead.")
    return api_manager.get_orders(status, limit)


def get_positions() -> Optional[List[Dict]]:
    """Legacy function - use api_manager.get_positions() instead."""
    logger.warning("Using deprecated get_positions(). Use api_manager instead.")
    return api_manager.get_positions()


def get_portfolio_history(period: str = "1D") -> Optional[Dict]:
    """Legacy function - use api_manager.get_portfolio_history() instead."""
    logger.warning("Using deprecated get_portfolio_history(). Use api_manager instead.")
    return api_manager.get_portfolio_history(period)


def get_portfolio_history_from_date(start_date: str = "2024-04-11") -> Optional[Dict]:
    """Legacy function - use api_manager methods instead."""
    logger.warning("Using deprecated get_portfolio_history_from_date(). Use api_manager instead.")
    # For now, return None as this specific functionality would need to be added to api_manager
    return None


def get_extended_portfolio_history() -> Optional[Dict]:
    """Legacy function - use api_manager methods instead."""
    logger.warning("Using deprecated get_extended_portfolio_history(). Use api_manager instead.")
    
    # Try different periods to get the best available data
    for period in api_manager.default_analysis_periods:
        try:
            data = api_manager.get_portfolio_history(period)
            if data and data.get('equity') and len(data.get('equity', [])) > 1:
                logger.info(f"✅ Got {len(data.get('equity', []))} data points from {period} period")
                return data
        except Exception as e:
            logger.debug(f"Period {period} failed: {e}")
            continue
    
    logger.warning("⚠️ Could not fetch extended history")
    return None


def get_closed_orders_from_date(start_date: str = "2024-04-11") -> Optional[List[Dict]]:
    """Legacy function - use api_manager.get_orders() instead."""
    logger.warning("Using deprecated get_closed_orders_from_date(). Use api_manager instead.")
    
    try:
        # Get closed orders and filter by date
        orders = api_manager.get_orders("closed", 500)
        if not orders:
            return None
            
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        filtered_orders = []
        
        for order in orders:
            order_date_str = order.get('filled_at') or order.get('updated_at') or order.get('created_at')
            if order_date_str:
                try:
                    order_date = datetime.fromisoformat(order_date_str.replace('Z', '+00:00'))
                    if order_date.replace(tzinfo=None) >= start_dt:
                        filtered_orders.append(order)
                except:
                    filtered_orders.append(order)
        
        logger.info(f"📊 Filtered {len(filtered_orders)} orders from {start_date} onwards")
        return filtered_orders
        
    except Exception as e:
        logger.error(f"Error filtering orders: {e}")
        return None


def get_closed_orders() -> Optional[List[Dict]]:
    """Legacy function - use api_manager.get_orders() instead."""
    logger.warning("Using deprecated get_closed_orders(). Use api_manager instead.")
    orders = api_manager.get_orders("closed", 500)
    if orders:
        logger.info(f"📋 Fetched {len(orders)} closed orders")
    return orders


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


class OptimizedPerformanceCalculator:
    """Optimized performance calculator using vectorized operations for AMD Ryzen 7 7840U."""
    
    def __init__(self, portfolio_history: Dict, orders: List[Dict], account: Dict):
        self.portfolio_history = portfolio_history
        self.orders = orders
        self.account = account
        self.equity_values = portfolio_history.get('equity', [])
        self.timestamps = portfolio_history.get('timestamp', [])
        
        # Use vectorized operations when available
        if CHARTS_AVAILABLE and perf_config.enable_vectorization:
            self._setup_vectorized_data()
        else:
            self._setup_standard_data()
    
    def _setup_vectorized_data(self):
        """Setup data using NumPy for vectorized operations."""
        # Convert to numpy arrays for faster calculations
        self.equity_array = np.array(self.equity_values, dtype=np.float64)
        self.timestamp_array = np.array(self.timestamps, dtype=np.int64)
        
        # Convert timestamps to datetime objects vectorized
        try:
            # Vectorized datetime conversion
            self.dates = [datetime.fromtimestamp(ts) for ts in self.timestamps if ts]
            self.dates_array = np.array(self.dates) if self.dates else np.array([])
        except Exception as e:
            logger.warning(f"Vectorized datetime conversion failed: {e}")
            self._setup_standard_data()
    
    def _setup_standard_data(self):
        """Setup data using standard Python operations."""
        # Convert timestamps to datetime objects
        self.dates = []
        for ts in self.timestamps:
            try:
                self.dates.append(datetime.fromtimestamp(ts))
            except:
                continue
        
        # Create numpy arrays if available
        if CHARTS_AVAILABLE:
            self.equity_array = np.array(self.equity_values, dtype=np.float64) if self.equity_values else np.array([])
        else:
            self.equity_array = None
    
    def calculate_returns(self) -> List[float]:
        """Calculate daily returns from equity curve using optimized vectorized operations."""
        if len(self.equity_values) < 2:
            return []
        
        # Use vectorized numpy operations if available
        if CHARTS_AVAILABLE and self.equity_array is not None and len(self.equity_array) > 1:
            try:
                # Vectorized calculation - much faster on multi-core CPUs
                equity_shifted = self.equity_array[:-1]  # Previous values
                equity_current = self.equity_array[1:]   # Current values
                
                # Avoid division by zero
                mask = equity_shifted > 0
                returns = np.zeros_like(equity_current)
                returns[mask] = (equity_current[mask] - equity_shifted[mask]) / equity_shifted[mask]
                
                return returns.tolist()
            except Exception as e:
                logger.warning(f"Vectorized returns calculation failed, using standard method: {e}")
        
        # Fallback to standard calculation
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
    
    def calculate_performance_rating(self) -> Dict:
        """Calculate comprehensive performance rating out of 10."""
        rating_components = {}
        total_score = 0
        max_possible_score = 0
        
        # 1. Return Performance (0-2 points)
        total_return = self.total_return()
        if total_return >= 20:
            return_score = 2.0
        elif total_return >= 10:
            return_score = 1.5
        elif total_return >= 5:
            return_score = 1.0
        elif total_return >= 0:
            return_score = 0.5
        else:
            return_score = 0.0
        
        rating_components['return_performance'] = {
            'score': return_score,
            'max_score': 2.0,
            'value': total_return,
            'description': f'Total Return: {total_return:+.1f}%'
        }
        total_score += return_score
        max_possible_score += 2.0
        
        # 2. Risk-Adjusted Performance - Sharpe Ratio (0-2 points)
        sharpe = self.calculate_sharpe_ratio()
        if sharpe >= 2.0:
            sharpe_score = 2.0
        elif sharpe >= 1.5:
            sharpe_score = 1.5
        elif sharpe >= 1.0:
            sharpe_score = 1.0
        elif sharpe >= 0.5:
            sharpe_score = 0.5
        else:
            sharpe_score = 0.0
        
        rating_components['sharpe_ratio'] = {
            'score': sharpe_score,
            'max_score': 2.0,
            'value': sharpe,
            'description': f'Sharpe Ratio: {sharpe:.2f}'
        }
        total_score += sharpe_score
        max_possible_score += 2.0
        
        # 3. Risk Management - Maximum Drawdown (0-1.5 points)
        max_dd, _, _ = self.calculate_max_drawdown()
        if max_dd <= 5:
            dd_score = 1.5
        elif max_dd <= 10:
            dd_score = 1.0
        elif max_dd <= 15:
            dd_score = 0.5
        else:
            dd_score = 0.0
        
        rating_components['risk_management'] = {
            'score': dd_score,
            'max_score': 1.5,
            'value': max_dd,
            'description': f'Max Drawdown: {max_dd:.1f}%'
        }
        total_score += dd_score
        max_possible_score += 1.5
        
        # 4. Trading Efficiency - Win Rate (0-1.5 points)
        trade_analysis = self.analyze_trades()
        win_rate = trade_analysis.get('win_rate', 0)
        
        if trade_analysis['total_trades'] > 0:
            if win_rate >= 70:
                win_score = 1.5
            elif win_rate >= 60:
                win_score = 1.0
            elif win_rate >= 50:
                win_score = 0.5
            else:
                win_score = 0.0
        else:
            win_score = 0.0  # No trades to evaluate
        
        rating_components['trading_efficiency'] = {
            'score': win_score,
            'max_score': 1.5,
            'value': win_rate,
            'description': f'Win Rate: {win_rate:.1f}%'
        }
        total_score += win_score
        max_possible_score += 1.5
        
        # 5. Profit Factor (0-1.5 points)
        profit_factor = trade_analysis.get('profit_factor', 0)
        
        if trade_analysis['total_trades'] > 0:
            if profit_factor >= 2.0:
                pf_score = 1.5
            elif profit_factor >= 1.5:
                pf_score = 1.0
            elif profit_factor >= 1.0:
                pf_score = 0.5
            else:
                pf_score = 0.0
        else:
            pf_score = 0.0
        
        rating_components['profit_factor'] = {
            'score': pf_score,
            'max_score': 1.5,
            'value': profit_factor,
            'description': f'Profit Factor: {profit_factor:.2f}'
        }
        total_score += pf_score
        max_possible_score += 1.5
        
        # 6. Consistency - Sortino Ratio (0-1.5 points)
        sortino = self.calculate_sortino_ratio()
        if sortino >= 2.0:
            sortino_score = 1.5
        elif sortino >= 1.5:
            sortino_score = 1.0
        elif sortino >= 1.0:
            sortino_score = 0.5
        else:
            sortino_score = 0.0
        
        rating_components['consistency'] = {
            'score': sortino_score,
            'max_score': 1.5,
            'value': sortino,
            'description': f'Sortino Ratio: {sortino:.2f}'
        }
        total_score += sortino_score
        max_possible_score += 1.5
        
        # Calculate final rating out of 10
        final_rating = (total_score / max_possible_score) * 10 if max_possible_score > 0 else 0
        
        # Determine rating category and emoji
        if final_rating >= 9.0:
            rating_category = "EXCEPTIONAL"
            rating_emoji = "🏆"
            rating_color = "gold"
        elif final_rating >= 8.0:
            rating_category = "EXCELLENT"
            rating_emoji = "🌟"
            rating_color = "green"
        elif final_rating >= 7.0:
            rating_category = "VERY GOOD"
            rating_emoji = "✅"
            rating_color = "lightgreen"
        elif final_rating >= 6.0:
            rating_category = "GOOD"
            rating_emoji = "👍"
            rating_color = "yellow"
        elif final_rating >= 5.0:
            rating_category = "AVERAGE"
            rating_emoji = "⚖️"
            rating_color = "orange"
        elif final_rating >= 4.0:
            rating_category = "BELOW AVERAGE"
            rating_emoji = "⚠️"
            rating_color = "orange"
        elif final_rating >= 3.0:
            rating_category = "POOR"
            rating_emoji = "❌"
            rating_color = "red"
        else:
            rating_category = "VERY POOR"
            rating_emoji = "💀"
            rating_color = "darkred"
        
        return {
            'final_rating': final_rating,
            'rating_category': rating_category,
            'rating_emoji': rating_emoji,
            'rating_color': rating_color,
            'total_score': total_score,
            'max_possible_score': max_possible_score,
            'components': rating_components,
            'summary': f"{rating_emoji} {final_rating:.1f}/10 - {rating_category}"
        }

# Compatibility alias for backward compatibility
PerformanceCalculator = OptimizedPerformanceCalculator


class ChartGenerator:
    """Generate comprehensive trading analytics charts."""
    
    def __init__(self, calculator: OptimizedPerformanceCalculator):
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
        
        # Get performance rating
        rating_data = self.calculator.calculate_performance_rating()
        
        metrics_text = f"""
KEY METRICS

{rating_data['rating_emoji']} RATING: {rating_data['final_rating']:.1f}/10
{rating_data['rating_category']}

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
        """Generate all available charts using optimized parallel processing."""
        if not CHARTS_AVAILABLE:
            logger.error("❌ Chart generation not available. Please install required dependencies:")
            logger.error("   pip install matplotlib seaborn pandas numpy scipy")
            return []
        
        logger.info("🎨 Generating comprehensive analytics charts with parallel processing...")
        
        # Validate data before generating charts
        if not self.calculator.equity_values:
            logger.error("❌ No equity data available for chart generation")
            return []
        
        if len(self.calculator.equity_values) < 2:
            logger.error("❌ Insufficient equity data for meaningful charts (need at least 2 data points)")
            return []
        
        # Define chart generation tasks with priorities
        # High priority charts (generate first)
        priority_charts = [
            ("Comprehensive Dashboard", self.create_comprehensive_merged_dashboard),
            ("Performance Dashboard", self.create_performance_summary_dashboard),
        ]
        
        # Standard charts (can be generated in parallel)
        standard_charts = [
            ("Equity Curve", self.create_equity_curve_chart),
            ("Drawdown Analysis", self.create_drawdown_chart),
            ("Returns Analysis", self.create_returns_analysis_chart),
            ("Trade Analysis", self.create_trade_analysis_chart),
            ("Risk Dashboard", self.create_risk_metrics_dashboard),
        ]
        
        generated_files = []
        
        # Generate priority charts first (sequential for memory management)
        for chart_name, chart_func in priority_charts:
            try:
                logger.info(f"🎯 Creating priority chart: {chart_name}...")
                start_time = time.time()
                filename = chart_func(save_and_open)
                if filename and os.path.exists(filename):
                    generated_files.append(filename)
                    elapsed = time.time() - start_time
                    logger.info(f"✅ {chart_name} created in {elapsed:.2f}s")
                else:
                    logger.warning(f"⚠️ {chart_name} was not created")
                
                # Memory cleanup after each priority chart
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Error creating {chart_name}: {e}")
        
        # Generate standard charts in parallel
        if standard_charts:
            logger.info(f"🚀 Creating {len(standard_charts)} charts in parallel using {perf_config.chart_workers} workers...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=perf_config.chart_workers) as executor:
                # Submit all chart generation tasks
                future_to_chart = {
                    executor.submit(self._generate_single_chart, chart_name, chart_func, save_and_open): chart_name
                    for chart_name, chart_func in standard_charts
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_chart, timeout=300):
                    chart_name = future_to_chart[future]
                    try:
                        result = future.result()
                        if result:
                            generated_files.append(result)
                            logger.info(f"✅ {chart_name} completed successfully")
                        else:
                            logger.warning(f"⚠️ {chart_name} was not created")
                    except Exception as e:
                        logger.error(f"❌ Error creating {chart_name}: {e}")
        
        # Final cleanup
        self._cleanup_matplotlib_resources()
        gc.collect()
        
        if generated_files:
            logger.info(f"✅ Generated {len(generated_files)} charts using AMD Ryzen 7 7840U optimization")
            logger.info("📁 Charts can be opened directly from file explorer")
            logger.info("📊 Generated charts:")
            for filename in generated_files:
                logger.info(f"  • {os.path.basename(filename)}")
        else:
            logger.warning("⚠️ No charts were generated successfully")
            
        return generated_files
    
    def _generate_single_chart(self, chart_name: str, chart_func, save_and_open: bool) -> Optional[str]:
        """Generate a single chart in a thread-safe manner."""
        try:
            start_time = time.time()
            filename = chart_func(save_and_open)
            elapsed = time.time() - start_time
            
            if filename and os.path.exists(filename):
                logger.debug(f"📈 {chart_name} generated in {elapsed:.2f}s")
                return filename
            else:
                logger.warning(f"⚠️ {chart_name} generation failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error in {chart_name}: {e}")
            return None
        finally:
            # Thread-local cleanup
            try:
                plt.close('all')
            except:
                pass
    
    def _cleanup_matplotlib_resources(self):
        """Clean up all matplotlib resources."""
        try:
            plt.close('all')
            # Clear matplotlib cache
            if hasattr(plt, 'rcdefaults'):
                plt.rcdefaults()
            gc.collect()
        except Exception as e:
            logger.debug(f"Matplotlib cleanup warning: {e}")
    
    def create_comprehensive_merged_dashboard(self, save_and_open: bool = True) -> Optional[str]:
        """Create a comprehensive merged dashboard with all charts and statistics."""
        if not CHARTS_AVAILABLE:
            print("❌ Cannot create merged dashboard - dependencies missing")
            return None
        
        if not self.calculator.equity_values:
            print("❌ Cannot create merged dashboard - no data available")
            return None
        
        try:
            # Create a large figure for the comprehensive dashboard
            fig = plt.figure(figsize=(24, 16))
            
            # Create a complex grid layout
            gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.3, 
                                 height_ratios=[1, 1, 1, 0.8], 
                                 width_ratios=[1, 1, 1, 1, 1, 1])
            
            # Prepare data
            dates = self.calculator.dates
            equity = self.calculator.equity_values
            
            if not dates or len(dates) != len(equity):
                dates = list(range(len(equity)))
            
            # Calculate all metrics
            total_return = self.calculator.total_return()
            cagr = self.calculator.annualized_return()
            sharpe = self.calculator.calculate_sharpe_ratio()
            sortino = self.calculator.calculate_sortino_ratio()
            calmar = self.calculator.calculate_calmar_ratio()
            max_dd, recovery_days, drawdown_days = self.calculator.calculate_max_drawdown()
            volatility = self.calculator.calculate_volatility()
            var_5 = self.calculator.calculate_var(0.05)
            trade_analysis = self.calculator.analyze_trades()
            returns = self.calculator.calculate_returns()
            
            # 1. MAIN EQUITY CURVE (Top row, spans 4 columns)
            ax_equity = fig.add_subplot(gs[0, :4])
            ax_equity.plot(dates, equity, linewidth=3, color='#2E86AB', label='Portfolio Value')
            ax_equity.fill_between(dates, equity, alpha=0.3, color='#2E86AB')
            
            # Add peak and trough annotations
            if len(equity) > 0:
                peak_idx = np.argmax(equity)
                trough_idx = np.argmin(equity)
                
                ax_equity.scatter(dates[peak_idx], equity[peak_idx], color='green', s=100, zorder=5)
                ax_equity.annotate(f'Peak: ${equity[peak_idx]:,.0f}', 
                                  xy=(dates[peak_idx], equity[peak_idx]),
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                                  fontsize=10)
                
                ax_equity.scatter(dates[trough_idx], equity[trough_idx], color='red', s=100, zorder=5)
                ax_equity.annotate(f'Trough: ${equity[trough_idx]:,.0f}', 
                                  xy=(dates[trough_idx], equity[trough_idx]),
                                  xytext=(10, -20), textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.7),
                                  fontsize=10)
            
            ax_equity.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
            ax_equity.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax_equity.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax_equity.grid(True, alpha=0.3)
            ax_equity.legend()
            
            # 2. KEY METRICS PANEL (Top right, spans 2 columns)
            ax_metrics = fig.add_subplot(gs[0, 4:])
            ax_metrics.axis('off')
            
            current_value = equity[-1] if equity else 0
            
            # Get performance rating
            rating_data = self.calculator.calculate_performance_rating()
            
            metrics_text = f"""
KEY PERFORMANCE METRICS

Portfolio Value: ${current_value:,.0f}
Total Return: {total_return:+.2f}%
Annualized Return (CAGR): {cagr:+.2f}%

{rating_data['rating_emoji']} PERFORMANCE RATING
{rating_data['final_rating']:.1f}/10 - {rating_data['rating_category']}

RISK-ADJUSTED METRICS
Sharpe Ratio: {sharpe:.2f}
Sortino Ratio: {sortino:.2f}
Calmar Ratio: {calmar:.2f}

RISK METRICS
Max Drawdown: {max_dd:.2f}%
Volatility: {volatility:.2f}%
Value at Risk (5%): ${var_5:,.0f}
            """
            
            ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # 3. DRAWDOWN CHART (Second row, left 2 columns)
            ax_drawdown = fig.add_subplot(gs[1, :2])
            
            peak = equity[0] if equity else 0
            drawdowns = []
            for value in equity:
                if value > peak:
                    peak = value
                drawdown = (value - peak) / peak * 100 if peak > 0 else 0
                drawdowns.append(drawdown)
            
            ax_drawdown.fill_between(dates, drawdowns, 0, color='red', alpha=0.3, label='Drawdown')
            ax_drawdown.plot(dates, drawdowns, color='darkred', linewidth=2)
            
            if drawdowns:
                max_dd_idx = np.argmin(drawdowns)
                max_dd_value = drawdowns[max_dd_idx]
                ax_drawdown.scatter(dates[max_dd_idx], max_dd_value, color='red', s=100, zorder=5)
                ax_drawdown.annotate(f'Max DD: {max_dd_value:.1f}%', 
                                   xy=(dates[max_dd_idx], max_dd_value),
                                   xytext=(10, 10), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.8),
                                   fontsize=9)
            
            ax_drawdown.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
            ax_drawdown.set_ylabel('Drawdown (%)', fontsize=10)
            ax_drawdown.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax_drawdown.grid(True, alpha=0.3)
            
            # 4. RETURNS DISTRIBUTION (Second row, middle 2 columns)
            ax_returns = fig.add_subplot(gs[1, 2:4])
            
            if returns:
                ax_returns.hist(returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax_returns.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, 
                                  label=f'Mean: {np.mean(returns):.4f}')
                ax_returns.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, 
                                  label=f'Median: {np.median(returns):.4f}')
                ax_returns.legend(fontsize=9)
            
            ax_returns.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
            ax_returns.set_xlabel('Daily Return', fontsize=10)
            ax_returns.set_ylabel('Frequency', fontsize=10)
            ax_returns.grid(True, alpha=0.3)
            
            # 5. TRADE STATISTICS PANEL (Second row, right 2 columns)
            ax_trade_stats = fig.add_subplot(gs[1, 4:])
            ax_trade_stats.axis('off')
            
            if trade_analysis['total_trades'] > 0:
                trade_stats_text = f"""
TRADE ANALYSIS

Total Trades: {trade_analysis['total_trades']}
Winning Trades: {trade_analysis['winning_trades']}
Losing Trades: {trade_analysis['losing_trades']}
Win Rate: {trade_analysis['win_rate']:.1f}%

TRADE EFFICIENCY
Profit Factor: {trade_analysis['profit_factor']:.2f}
Average Win: ${trade_analysis['avg_win']:.2f}
Average Loss: ${trade_analysis['avg_loss']:.2f}
Expectancy: ${trade_analysis['expectancy']:.2f}

Win/Loss Ratio: {trade_analysis['avg_win']/trade_analysis['avg_loss'] if trade_analysis['avg_loss'] > 0 else 0:.2f}
                """
            else:
                trade_stats_text = """
TRADE ANALYSIS

No completed trades found
for detailed analysis.

Consider making some trades
to see comprehensive
trade statistics.
                """
            
            ax_trade_stats.text(0.05, 0.95, trade_stats_text, transform=ax_trade_stats.transAxes,
                               fontsize=10, verticalalignment='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            
            # 6. TRADE P&L SCATTER (Third row, left 2 columns)
            ax_trade_pnl = fig.add_subplot(gs[2, :2])
            
            if trade_analysis['total_trades'] > 0 and 'trade_pnls' in trade_analysis:
                trade_pnls = trade_analysis['trade_pnls']
                trade_numbers = list(range(1, len(trade_pnls) + 1))
                colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
                
                ax_trade_pnl.scatter(trade_numbers, trade_pnls, c=colors, alpha=0.7, s=30)
                ax_trade_pnl.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax_trade_pnl.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
                ax_trade_pnl.set_xlabel('Trade Number', fontsize=10)
                ax_trade_pnl.set_ylabel('P&L ($)', fontsize=10)
                ax_trade_pnl.grid(True, alpha=0.3)
            else:
                ax_trade_pnl.text(0.5, 0.5, 'No Trade Data\nAvailable', ha='center', va='center',
                                 transform=ax_trade_pnl.transAxes, fontsize=12)
                ax_trade_pnl.set_title('Individual Trade P&L', fontsize=14, fontweight='bold')
            
            # 7. CUMULATIVE P&L (Third row, middle 2 columns)
            ax_cum_pnl = fig.add_subplot(gs[2, 2:4])
            
            if trade_analysis['total_trades'] > 0 and 'trade_pnls' in trade_analysis:
                trade_pnls = trade_analysis['trade_pnls']
                cumulative_pnl = np.cumsum(trade_pnls)
                trade_numbers = list(range(1, len(trade_pnls) + 1))
                
                ax_cum_pnl.plot(trade_numbers, cumulative_pnl, linewidth=2, color='blue')
                ax_cum_pnl.fill_between(trade_numbers, cumulative_pnl, alpha=0.3, color='blue')
                ax_cum_pnl.set_title('Cumulative Trade P&L', fontsize=14, fontweight='bold')
                ax_cum_pnl.set_xlabel('Trade Number', fontsize=10)
                ax_cum_pnl.set_ylabel('Cumulative P&L ($)', fontsize=10)
                ax_cum_pnl.grid(True, alpha=0.3)
            else:
                ax_cum_pnl.text(0.5, 0.5, 'No Trade Data\nAvailable', ha='center', va='center',
                               transform=ax_cum_pnl.transAxes, fontsize=12)
                ax_cum_pnl.set_title('Cumulative Trade P&L', fontsize=14, fontweight='bold')
            
            # 8. WIN/LOSS PIE CHART (Third row, right 2 columns)
            ax_pie = fig.add_subplot(gs[2, 4:])
            
            if trade_analysis['total_trades'] > 0:
                win_count = trade_analysis['winning_trades']
                loss_count = trade_analysis['losing_trades']
                
                if win_count > 0 or loss_count > 0:
                    labels = ['Winning Trades', 'Losing Trades']
                    sizes = [win_count, loss_count]
                    colors = ['lightgreen', 'lightcoral']
                    explode = (0.05, 0.05)
                    
                    ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                              shadow=True, startangle=90, textprops={'fontsize': 10})
                    ax_pie.set_title(f'Win/Loss Distribution\n(Win Rate: {trade_analysis["win_rate"]:.1f}%)', 
                                    fontsize=14, fontweight='bold')
                else:
                    ax_pie.text(0.5, 0.5, 'No Trade Data', ha='center', va='center', fontsize=12)
                    ax_pie.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
            else:
                ax_pie.text(0.5, 0.5, 'No Trade Data', ha='center', va='center', fontsize=12)
                ax_pie.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
            
            # 9. RISK GAUGES (Bottom row, spans all columns)
            ax_risk_gauges = fig.add_subplot(gs[3, :])
            ax_risk_gauges.axis('off')
            
            # Create mini gauge visualizations using bar charts
            gauge_metrics = [
                ('Sharpe Ratio', sharpe, [0, 1, 2, 3], ['red', 'yellow', 'lightgreen', 'green']),
                ('Sortino Ratio', sortino, [0, 1, 2, 3], ['red', 'yellow', 'lightgreen', 'green']),
                ('Max Drawdown (%)', max_dd, [0, 5, 10, 20], ['green', 'lightgreen', 'yellow', 'red']),
                ('Volatility (%)', volatility, [0, 10, 20, 40], ['green', 'lightgreen', 'yellow', 'red'])
            ]
            
            gauge_positions = [0.1, 0.3, 0.5, 0.7]
            
            for i, (name, value, thresholds, colors) in enumerate(gauge_metrics):
                x_pos = gauge_positions[i]
                
                # Determine color based on value and thresholds
                color_idx = 0
                if name in ['Max Drawdown (%)', 'Volatility (%)']:  # Reverse logic for risk metrics
                    for j, threshold in enumerate(thresholds[1:], 1):
                        if value >= threshold:
                            color_idx = j
                else:  # Normal logic for performance metrics
                    color_idx = len(colors) - 1
                    for j, threshold in enumerate(thresholds[1:], 1):
                        if value < threshold:
                            color_idx = j - 1
                            break
                
                color = colors[min(color_idx, len(colors) - 1)]
                
                # Create a simple bar gauge
                bar_height = min(value / max(thresholds) if max(thresholds) > 0 else 0, 1.0)
                
                # Draw the gauge background
                ax_risk_gauges.barh(i, 1.0, left=x_pos, height=0.15, color='lightgray', alpha=0.3)
                # Draw the gauge value
                ax_risk_gauges.barh(i, bar_height, left=x_pos, height=0.15, color=color, alpha=0.8)
                
                # Add text labels
                ax_risk_gauges.text(x_pos + 0.05, i, f'{name}\n{value:.2f}', 
                                   fontsize=10, fontweight='bold', va='center')
            
            ax_risk_gauges.set_xlim(0, 1)
            ax_risk_gauges.set_ylim(-0.5, len(gauge_metrics) - 0.5)
            ax_risk_gauges.set_title('Risk Metrics Dashboard', fontsize=16, fontweight='bold', pad=20)
            
            # Format x-axis for dates if we have actual dates
            if isinstance(dates[0], datetime):
                for ax in [ax_equity, ax_drawdown]:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Main title with performance rating
            rating_data = self.calculator.calculate_performance_rating()
            main_title = f'Comprehensive Trading Analytics Dashboard - {rating_data["rating_emoji"]} {rating_data["final_rating"]:.1f}/10'
            plt.suptitle(main_title, fontsize=20, fontweight='bold', y=0.98)
            
            # Add footer with timestamp
            footer_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Alpaca Trading Analytics"
            fig.text(0.5, 0.01, footer_text, ha='center', fontsize=10, alpha=0.7)
            
            plt.tight_layout()
            
            if save_and_open:
                filename = f"{self.charts_dir}/comprehensive_dashboard.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"🎯 Comprehensive dashboard saved: {filename}")
                self.open_file_in_explorer(filename)
                return filename
            
            return None
            
        except Exception as e:
            print(f"❌ Error creating comprehensive dashboard: {e}")
            plt.close('all')
            return None


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


def display_comprehensive_metrics(calculator: OptimizedPerformanceCalculator) -> None:
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
    
    # Comprehensive Performance Rating
    print("\n🏆 PERFORMANCE RATING")
    print("=" * 80)
    
    rating_data = calculator.calculate_performance_rating()
    
    # Display main rating
    print(f"\n{rating_data['rating_emoji']} OVERALL RATING: {rating_data['final_rating']:.1f}/10 - {rating_data['rating_category']}")
    print(f"📊 Total Score: {rating_data['total_score']:.1f}/{rating_data['max_possible_score']:.1f} points")
    
    # Display component breakdown
    print(f"\n📋 RATING BREAKDOWN:")
    print("-" * 50)
    
    for component_name, component_data in rating_data['components'].items():
        score = component_data['score']
        max_score = component_data['max_score']
        description = component_data['description']
        percentage = (score / max_score * 100) if max_score > 0 else 0
        
        # Create visual bar
        bar_length = 20
        filled_length = int(bar_length * percentage / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        print(f"{description:<25} │{bar}│ {score:.1f}/{max_score:.1f} ({percentage:.0f}%)")
    
    # Performance interpretation
    print(f"\n💡 PERFORMANCE INTERPRETATION:")
    print("-" * 50)
    
    if rating_data['final_rating'] >= 9.0:
        print("🏆 EXCEPTIONAL: Outstanding performance across all metrics!")
        print("   Your trading strategy demonstrates exceptional skill and risk management.")
    elif rating_data['final_rating'] >= 8.0:
        print("🌟 EXCELLENT: Strong performance with excellent risk-adjusted returns!")
        print("   You're in the top tier of traders with consistent profitability.")
    elif rating_data['final_rating'] >= 7.0:
        print("✅ VERY GOOD: Solid performance with good risk management!")
        print("   Your strategy shows strong potential with room for optimization.")
    elif rating_data['final_rating'] >= 6.0:
        print("👍 GOOD: Decent performance with acceptable risk levels!")
        print("   You're on the right track - focus on improving weak areas.")
    elif rating_data['final_rating'] >= 5.0:
        print("⚖️ AVERAGE: Performance is in line with market averages!")
        print("   Consider refining your strategy and risk management approach.")
    elif rating_data['final_rating'] >= 4.0:
        print("⚠️ BELOW AVERAGE: Performance needs improvement!")
        print("   Review your strategy and consider reducing risk exposure.")
    elif rating_data['final_rating'] >= 3.0:
        print("❌ POOR: Significant improvements needed!")
        print("   Consider paper trading to refine your approach before risking capital.")
    else:
        print("💀 VERY POOR: Major strategy overhaul required!")
        print("   Stop trading with real money and focus on education and practice.")
    
    print("\n" + "=" * 80)


def optimized_main():
    """Optimized main function for AMD Ryzen 7 7840U."""
    start_time = time.time()
    
    logger.info("🚀 Alpaca Trading Analytics & Chart Generator (AMD Ryzen 7 7840U Optimized)")
    logger.info(f"🔧 Performance: {perf_config.physical_cores}C/{perf_config.cpu_count}T, {perf_config.memory_gb:.1f}GB RAM")
    
    # Check for chart generation capability
    if CHARTS_AVAILABLE:
        logger.info("📊 Advanced analytics charts enabled with parallel processing")
    else:
        logger.warning("⚠️  Basic mode - install chart dependencies for advanced analytics:")
        logger.warning("   pip install matplotlib seaborn pandas numpy scipy aiohttp psutil")
    
    try:
        # Initialize API connection
        logger.info("🔍 Establishing optimized API connection...")
        if not api_manager.find_working_endpoint():
            logger.error("❌ Failed to establish API connection")
            logger.error("🔧 Troubleshooting tips:")
            logger.error("1. Verify your API credentials are correct")
            logger.error("2. Check if your account is activated")
            logger.error("3. Ensure you have the correct permissions")
            logger.error("4. Try generating new API keys from your Alpaca dashboard")
            return
        
        # Parallel data fetching using ThreadPoolExecutor
        logger.info("⚡ Fetching data in parallel...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=perf_config.io_workers) as executor:
            # Submit all data fetching tasks concurrently
            futures = {
                'account': executor.submit(api_manager.get_account_info),
                'orders': executor.submit(api_manager.get_orders, "all", api_manager.max_orders_to_fetch),
                'positions': executor.submit(api_manager.get_positions),
            }
            
            # Add portfolio history requests for multiple periods
            portfolio_futures = {}
            for period in api_manager.default_analysis_periods:
                portfolio_futures[period] = executor.submit(api_manager.get_portfolio_history, period)
            
            # Collect results as they complete
            account = None
            orders = None
            positions = None
            portfolio = None
            
            try:
                # Get basic data with timeout
                for future_name, future in futures.items():
                    try:
                        result = future.result(timeout=30)
                        if future_name == 'account':
                            account = result
                        elif future_name == 'orders':
                            orders = result
                        elif future_name == 'positions':
                            positions = result
                            
                        if result:
                            logger.info(f"✅ {future_name.title()} data fetched successfully")
                        else:
                            logger.warning(f"⚠️ {future_name.title()} data not available")
                            
                    except concurrent.futures.TimeoutError:
                        logger.error(f"⏰ Timeout fetching {future_name}")
                    except Exception as e:
                        logger.error(f"❌ Error fetching {future_name}: {e}")
                
                # Get portfolio data (try periods in order of preference)
                for period in api_manager.default_analysis_periods:
                    try:
                        result = portfolio_futures[period].result(timeout=20)
                        if result and result.get('equity') and len(result.get('equity', [])) > 1:
                            portfolio = result
                            logger.info(f"📊 Using {period} portfolio data ({len(result.get('equity', []))} data points)")
                            break
                    except Exception as e:
                        logger.debug(f"Period {period} failed: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error in parallel data fetching: {e}")
        
        # Validate required data
        if not account:
            logger.error("❌ Failed to fetch account information - cannot proceed")
            return
            
        # Display results efficiently
        display_account_info(account)
        
        if orders:
            logger.info(f"📋 Found {len(orders)} total orders")
            display_recent_orders(orders)
        
        if positions:
            display_positions(positions)
            
        if portfolio:
            display_portfolio_summary(portfolio)
        
        # Comprehensive analysis with optimization
        if portfolio and account:
            logger.info("🔍 Performing optimized comprehensive analysis...")
            
            # Use the best available portfolio data
            extended_portfolio = portfolio
            closed_orders = orders if orders else []
            
            # Create performance calculator with memory optimization
            calculator = OptimizedPerformanceCalculator(extended_portfolio, closed_orders, account)
            
            # Display metrics
            display_comprehensive_metrics(calculator)
            
            # Generate charts with parallel processing
            if CHARTS_AVAILABLE:
                logger.info("🎨 Generating analytics charts with parallel optimization...")
                chart_generator = ChartGenerator(calculator)
                
                chart_start = time.time()
                generated_files = chart_generator.generate_all_charts(save_and_open=True)
                chart_elapsed = time.time() - chart_start
                
                if generated_files:
                    logger.info(f"✅ Generated {len(generated_files)} charts in {chart_elapsed:.2f}s")
                    for filename in generated_files:
                        logger.info(f"  📊 {os.path.basename(filename)}")
                else:
                    logger.warning("⚠️ No charts were generated")
            else:
                logger.info("📊 Chart generation skipped - dependencies not available")
        else:
            logger.warning("⚠️ Insufficient data for comprehensive analysis")
            
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup resources
        api_manager.cleanup()
        gc.collect()
    
    # Performance summary
    total_elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"⚡ Total execution time: {total_elapsed:.2f}s (optimized for AMD Ryzen 7 7840U)")
    logger.info(f"💾 Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
    
    if CHARTS_AVAILABLE:
        logger.info("📈 Analytics Summary:")
        logger.info("  • Comprehensive performance metrics calculated")
        logger.info("  • Professional charts generated with parallel processing")
        logger.info("  • Charts automatically opened in default viewer")
        logger.info("  • Files saved in 'charts' directory for future reference")
        logger.info("💡 Tip: Charts are high-resolution PNG files suitable for presentations")
    else:
        logger.info("📊 Install dependencies for full optimization:")
        logger.info("  pip install matplotlib seaborn pandas numpy scipy aiohttp psutil")
    
    logger.info("🎯 Next Steps:")
    logger.info("  • Review performance metrics and charts")
    logger.info("  • Analyze risk-adjusted returns")
    logger.info("  • Compare against benchmarks")
    logger.info("  • Consider portfolio adjustments based on insights")

# Alias for backward compatibility
main = optimized_main


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


def generate_comprehensive_dashboard_only():
    """Generate only the comprehensive merged dashboard."""
    print("🎯 Generating Comprehensive Dashboard Only...")
    
    # Get account information
    account = get_account_info()
    if not account:
        print("❌ Failed to fetch account information")
        return
    
    # Get extended portfolio data
    print("🔍 Fetching portfolio data...")
    extended_portfolio = get_extended_portfolio_history()
    
    if not extended_portfolio:
        print("❌ Could not fetch portfolio data")
        return
    
    # Get order data
    print("🔍 Fetching order history...")
    closed_orders = get_closed_orders()
    
    try:
        # Create performance calculator
        calculator = PerformanceCalculator(
            extended_portfolio, 
            closed_orders if closed_orders else [], 
            account
        )
        
        # Generate comprehensive dashboard
        if CHARTS_AVAILABLE:
            print("🎨 Creating comprehensive merged dashboard...")
            chart_generator = ChartGenerator(calculator)
            filename = chart_generator.create_comprehensive_merged_dashboard(save_and_open=True)
            
            if filename:
                print(f"✅ Comprehensive dashboard created: {filename}")
                print("🎯 This single chart contains all your analytics in one view!")
            else:
                print("❌ Failed to create comprehensive dashboard")
        else:
            print("❌ Chart dependencies not available")
            print("📦 Install with: pip install matplotlib seaborn pandas numpy scipy")
            
    except Exception as e:
        print(f"❌ Error creating comprehensive dashboard: {e}")


if __name__ == "__main__":
    try:
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--test-charts":
                test_chart_generation()
            elif sys.argv[1] == "--comprehensive" or sys.argv[1] == "--merged":
                generate_comprehensive_dashboard_only()
            elif sys.argv[1] == "--help":
                print("🚀 Alpaca Trading Analytics & Chart Generator")
                print("\nUsage:")
                print("  python alpaca.py                    # Full analysis with all charts")
                print("  python alpaca.py --test-charts      # Test chart generation")
                print("  python alpaca.py --comprehensive    # Generate only merged dashboard")
                print("  python alpaca.py --merged           # Generate only merged dashboard")
                print("  python alpaca.py --help             # Show this help")
                print("\nFeatures:")
                print("  • Comprehensive performance analytics")
                print("  • Professional chart generation (7 charts)")
                print("  • Risk-adjusted metrics")
                print("  • Trade analysis and statistics")
                print("  • High-resolution PNG output")
                print("  • Comprehensive merged dashboard")
            else:
                print(f"❌ Unknown option: {sys.argv[1]}")
                print("Use --help to see available options")
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
