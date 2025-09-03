#!/usr/bin/env python3
"""
Alpaca Trading Analytics - Refactored Professional Version

A clean, modular, and secure implementation of Alpaca trading analytics
with proper error handling, type safety, and enterprise-grade architecture.

Author: Senior Development Team
Version: 3.0.0
License: MIT
"""

import sys
import logging
from typing import Optional
from decimal import Decimal

# Import our modular components
from models import AccountInfo, Position, Order, PortfolioHistory
from services import SecureAlpacaClient, Environment, AlpacaAPIError, FinancialAnalytics, PerformanceMetrics, ChartGenerator
from config import config, SecureConfig, LogLevel

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.value),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.log_filename) if config.log_to_file else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlpacaAnalytics:
    """Main analytics application with clean architecture."""
    
    def __init__(self, environment: Environment = None):
        """Initialize analytics application."""
        self.environment = environment or (Environment.PAPER if config.use_paper_trading else Environment.LIVE)
        self.client = None
        self.analytics = FinancialAnalytics()
        self.chart_generator = ChartGenerator() if config.enable_charts else None
        
        trading_type = "PAPER TRADING" if self.environment == Environment.PAPER else "LIVE TRADING"
        logger.info(f">> Alpaca Analytics v3.0.0 - {trading_type} Environment")
    
    def __enter__(self):
        """Context manager entry."""
        self.client = SecureAlpacaClient(self.environment)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.client:
            self.client.close()
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            return self.client.test_connection()
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_account_summary(self) -> Optional[AccountInfo]:
        """Get account information with error handling."""
        try:
            account = self.client.get_account()
            logger.info(f"[OK] Account loaded: {account.account_number} ({account.status.value})")
            return account
        except AlpacaAPIError as e:
            logger.error(f"Failed to get account information: {e}")
            return None
    
    def get_positions_summary(self) -> Optional[list[Position]]:
        """Get current positions with error handling."""
        try:
            positions = self.client.get_positions()
            logger.info(f"[DATA] Loaded {len(positions)} positions")
            return positions
        except AlpacaAPIError as e:
            logger.error(f"Failed to get positions: {e}")
            return None
    
    def get_orders_summary(self, limit: int = 2000) -> Optional[list[Order]]:
        """Get order history with error handling."""
        try:
            orders = self.client.get_orders(limit=limit)
            logger.info(f"[LIST] Loaded {len(orders)} orders")
            
            # Debug: Check order statuses
            if orders:
                status_counts = {}
                filled_count = 0
                for order in orders:
                    status = order.status.value
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if order.filled_qty > 0:
                        filled_count += 1
                
                logger.info(f"Order status breakdown: {status_counts}")
                logger.info(f"Orders with filled_qty > 0: {filled_count}")
            
            return orders
        except AlpacaAPIError as e:
            logger.error(f"Failed to get orders: {e}")
            return None
    
    def get_portfolio_history_summary(self, period: str = "1Y") -> Optional[PortfolioHistory]:
        """Get portfolio history with fallback periods."""
        for attempt_period in [period, "6M", "3M", "1M", "1W", "1D"]:
            try:
                history = self.client.get_portfolio_history(attempt_period)
                if history:
                    logger.info(f"[UP] Portfolio history loaded: {attempt_period} ({len(history.equity)} data points)")
                    return history
            except AlpacaAPIError as e:
                logger.warning(f"Failed to get portfolio history for {attempt_period}: {e}")
        
        logger.warning("No portfolio history available")
        return None
    
    def calculate_performance_metrics(self, portfolio_history: PortfolioHistory, 
                                    orders: list[Order], positions: list[Position] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        return self.analytics.calculate_comprehensive_metrics(portfolio_history, orders, positions)
    
    def display_account_info(self, account: AccountInfo):
        """Display account information in a clean format."""
        print("\n" + "="*60)
        print("[BANK] ACCOUNT INFORMATION")
        print("="*60)
        
        print(f"Account ID: {account.account_number}")
        print(f"Status: {account.status.value}")
        print(f"Trading Active: {'[OK] Yes' if account.is_active else '[X] No'}")
        print(f"Pattern Day Trader: {'Yes' if account.pattern_day_trader else 'No'}")
        
        print("\n[MONEY] BALANCE INFORMATION:")
        print(f"  Equity: ${account.equity:,.2f}")
        print(f"  Cash: ${account.cash:,.2f}")
        print(f"  Buying Power: ${account.buying_power:,.2f}")
        print(f"  Portfolio Value: ${account.portfolio_value:,.2f}")
        
        print(f"  Daily P&L: ${account.daily_pnl:,.2f} ({account.daily_pnl_percent:+.2f}%)")
        print(f"  Day Trade Count: {account.day_trade_count}")
        
        # Professional balance validation
        if account.equity >= 90000:
            print(f"\n[OK] SUCCESS: Professional trading balance detected (${account.equity:,.2f})")
            print("[MONEY] Account is properly configured for professional trading!")
        elif account.equity >= 50000:
            print(f"\n[DATA] Account balance: ${account.equity:,.2f}")
            print("[INFO] This appears to be after some trading activity")
        else:
            print(f"\n[WARN]  Account balance: ${account.equity:,.2f}")
            print("[TIP] Consider increasing balance for professional trading")
    
    def display_positions(self, positions: list[Position]):
        """Display current positions."""
        if not positions:
            print("\n[DATA] No current positions")
            return
        
        print("\n" + "="*60)
        print("[DATA] CURRENT POSITIONS")
        print("="*60)
        
        total_value = Decimal('0')
        total_pnl = Decimal('0')
        
        for pos in positions:
            emoji = "[UP]" if pos.is_profitable else "[DOWN]"
            print(f"\n{emoji} {pos.symbol}")
            print(f"   Quantity: {pos.qty} shares")
            print(f"   Market Value: ${pos.market_value:,.2f}")
            print(f"   Cost Basis: ${pos.cost_basis:,.2f}")
            print(f"   Unrealized P&L: ${pos.unrealized_pl:,.2f} ({pos.unrealized_plpc:+.2f}%)")
            print(f"   Avg Entry Price: ${pos.avg_entry_price:.2f}")
            
            total_value += pos.market_value
            total_pnl += pos.unrealized_pl
        
        print(f"\n[PORTFOLIO] Total Position Value: ${total_value:,.2f}")
        print(f"[MONEY] Total Unrealized P&L: ${total_pnl:,.2f}")
    
    def display_performance_metrics(self, metrics: PerformanceMetrics, orders: list[Order], account: AccountInfo):
        """Display comprehensive performance metrics."""
        print("\n" + "="*80)
        print("[DATA] COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*80)
        
        print("\n[METRICS] PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Total Return: {metrics.total_return:+.2f}%")
        print(f"Annualized Return (CAGR): {metrics.annualized_return:+.2f}%")
        print(f"Win Rate: {metrics.win_rate:.1f}%")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Average Win: ${metrics.avg_win:.2f}")
        print(f"Average Loss: ${metrics.avg_loss:.2f}")
        print(f"Expectancy: ${metrics.expectancy:.2f}")
        
        print("\n[TRADE] DETAILED TRADING STATISTICS")
        print("-" * 40)
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Winning Trades: {metrics.winning_trades}")
        print(f"Losing Trades: {metrics.losing_trades}")
        print(f"Average Trade Size: ${metrics.avg_trade_size:,.2f}")
        print(f"Average Winning Trade Size: ${metrics.avg_winning_trade_size:,.2f}")
        print(f"Average Losing Trade Size: ${metrics.avg_losing_trade_size:,.2f}")
        print(f"Expected Win Per Trade: ${metrics.expected_win_per_trade:.2f}")
        print(f"Cost Per Loss Trade: ${metrics.cost_per_loss:.2f}")
        print(f"Total P&L: ${metrics.total_pnl:+,.2f}")
        
        print("\n[BALANCE] RISK-ADJUSTED METRICS")
        print("-" * 40)
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
        
        print("\n[WARN] RISK METRICS")
        print("-" * 40)
        print(f"Maximum Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"Volatility (Annualized): {metrics.volatility:.2f}%")
        print(f"Value at Risk (5%): ${metrics.var_5:.2f}")
        
        print("\n[TRADE] TRADE EFFICIENCY METRICS")
        print("-" * 40)
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Winning Trades: {metrics.winning_trades}")
        print(f"Losing Trades: {metrics.losing_trades}")
        
        # Comprehensive strategy score
        strategy_score = self.analytics.score_strategy(metrics)
        print(f"\n[SCORE] OVERALL STRATEGY RATING")
        print("=" * 50)
        print(f"SCORE: {strategy_score['score']:.1f}/10.0 | GRADE: {strategy_score['grade']} | {strategy_score['assessment']}")
        print(f"\nSCORING BREAKDOWN:")
        for reason in strategy_score['reasoning']:
            print(f"  • {reason}")
        
    
    def _display_performance_rating(self, metrics: PerformanceMetrics):
        """Display performance rating."""
        score = 0.0
        max_score = 10.0
        
        # Total return (0-2 points)
        if metrics.total_return >= 20:
            score += 2.0
        elif metrics.total_return >= 10:
            score += 1.5
        elif metrics.total_return >= 5:
            score += 1.0
        elif metrics.total_return >= 0:
            score += 0.5
        
        # Sharpe ratio (0-2 points)
        if metrics.sharpe_ratio >= 2.0:
            score += 2.0
        elif metrics.sharpe_ratio >= 1.5:
            score += 1.5
        elif metrics.sharpe_ratio >= 1.0:
            score += 1.0
        elif metrics.sharpe_ratio >= 0.5:
            score += 0.5
        
        # Max drawdown (0-1.5 points)
        if metrics.max_drawdown <= 5:
            score += 1.5
        elif metrics.max_drawdown <= 10:
            score += 1.0
        elif metrics.max_drawdown <= 20:
            score += 0.5
        
        # Win rate (0-1.5 points)
        if metrics.win_rate >= 60:
            score += 1.5
        elif metrics.win_rate >= 50:
            score += 1.0
        elif metrics.win_rate >= 40:
            score += 0.5
        
        # Profit factor (0-1.5 points)
        if metrics.profit_factor >= 2.0:
            score += 1.5
        elif metrics.profit_factor >= 1.5:
            score += 1.0
        elif metrics.profit_factor >= 1.0:
            score += 0.5
        
        # Sortino ratio (0-1.5 points)
        if metrics.sortino_ratio >= 2.0:
            score += 1.5
        elif metrics.sortino_ratio >= 1.5:
            score += 1.0
        elif metrics.sortino_ratio >= 1.0:
            score += 0.5
        
        print(f"\n[RATING] PERFORMANCE RATING")
        print("="*80)
        print(f"[BALANCE] OVERALL RATING: {score:.1f}/{max_score} - {self._get_rating_label(score, max_score)}")
        print(f"[DATA] Total Score: {score:.1f}/{max_score} points")
        
        # Rating breakdown
        print(f"\n[LIST] RATING BREAKDOWN:")
        print("-" * 50)
        bar_length = 20
        
        categories = [
            ("Total Return", metrics.total_return, "%", 2.0),
            ("Sharpe Ratio", metrics.sharpe_ratio, "", 2.0),
            ("Max Drawdown", 20 - metrics.max_drawdown, "% (inverted)", 1.5),
            ("Win Rate", metrics.win_rate, "%", 1.5),
            ("Profit Factor", metrics.profit_factor, "", 1.5),
            ("Sortino Ratio", metrics.sortino_ratio, "", 1.5)
        ]
        
        for name, value, unit, max_points in categories:
            # Calculate points for this category
            if "Total Return" in name:
                if value >= 20: points = 2.0
                elif value >= 10: points = 1.5
                elif value >= 5: points = 1.0
                elif value >= 0: points = 0.5
                else: points = 0.0
            elif "Sharpe" in name:
                if value >= 2.0: points = 2.0
                elif value >= 1.5: points = 1.5
                elif value >= 1.0: points = 1.0
                elif value >= 0.5: points = 0.5
                else: points = 0.0
            elif "Drawdown" in name:
                actual_dd = 20 - value
                if actual_dd <= 5: points = 1.5
                elif actual_dd <= 10: points = 1.0
                elif actual_dd <= 20: points = 0.5
                else: points = 0.0
            elif "Win Rate" in name:
                if value >= 60: points = 1.5
                elif value >= 50: points = 1.0
                elif value >= 40: points = 0.5
                else: points = 0.0
            elif "Profit Factor" in name:
                if value >= 2.0: points = 1.5
                elif value >= 1.5: points = 1.0
                elif value >= 1.0: points = 0.5
                else: points = 0.0
            elif "Sortino" in name:
                if value >= 2.0: points = 1.5
                elif value >= 1.5: points = 1.0
                elif value >= 1.0: points = 0.5
                else: points = 0.0
            else:
                points = 0.0
            
            # Create progress bar
            filled_length = int(bar_length * points / max_points)
            bar = "#" * filled_length + "-" * (bar_length - filled_length)
            percentage = (points / max_points) * 100
            
            display_value = value if "Drawdown" not in name else 20 - value
            print(f"{name}: {display_value:.1f}{unit:<12} |{bar}| {points:.1f}/{max_points} ({percentage:.0f}%)")
    
    def _get_rating_label(self, score: float, max_score: float) -> str:
        """Get rating label based on score."""
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return "EXCEPTIONAL"
        elif percentage >= 80:
            return "EXCELLENT"
        elif percentage >= 70:
            return "GOOD"
        elif percentage >= 60:
            return "ABOVE AVERAGE"
        elif percentage >= 50:
            return "AVERAGE"
        elif percentage >= 40:
            return "BELOW AVERAGE"
        elif percentage >= 30:
            return "POOR"
        else:
            return "VERY POOR"
    
    def run_analysis(self):
        """Run complete analytics analysis."""
        try:
            # Test connection
            if not self.test_connection():
                logger.error("[X] Failed to connect to Alpaca API")
                return False
            
            logger.info("[OK] Connected to Alpaca API successfully")
            
            # Get account information
            account = self.get_account_summary()
            if not account:
                logger.error("[X] Failed to get account information")
                return False
            
            # Display account info
            self.display_account_info(account)
            
            # Get positions
            positions = self.get_positions_summary()
            if positions:
                self.display_positions(positions)
            
            # Get orders
            orders = self.get_orders_summary()
            if not orders:
                logger.warning("[WARN] No order history available")
                orders = []
            
            # Get portfolio history
            portfolio_history = self.get_portfolio_history_summary()
            if not portfolio_history:
                logger.warning("[WARN] No portfolio history available - this is normal for new accounts")
                print(f"\n[LIST] Account Summary Complete!")
                print(f"[MONEY] Current Balance: ${account.equity:,.2f}")
                print(f"[CASH] Available Cash: ${account.cash:,.2f}")
                print(f"🛒 Buying Power: ${account.buying_power:,.2f}")
                return True
            
            # Calculate and display performance metrics
            metrics = self.calculate_performance_metrics(portfolio_history, orders, positions)
            self.display_performance_metrics(metrics, orders, account)
            
            # Generate charts if enabled
            if self.chart_generator:
                logger.info("[CHARTS] Generating performance charts...")
                charts = self.chart_generator.generate_all_charts(
                    portfolio_history, metrics, account, positions or []
                )
                
                if charts:
                    print(f"\n{'='*60}")
                    print("[CHARTS] GENERATED PERFORMANCE CHARTS")
                    print('='*60)
                    
                    for chart_type, filename in charts.items():
                        print(f"[OK] {chart_type.replace('_', ' ').title()}: {filename}")
                    
                    print(f"\n[INFO] {len(charts)} charts saved to charts/ directory")
                    print("[INFO] Open the charts to view detailed performance analysis")
                else:
                    logger.warning("[WARN] Chart generation failed")
            
            logger.info("[OK] Analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"[X] Analysis failed: {e}")
            return False

def main():
    """Main entry point."""
    print(">> Alpaca Trading Analytics v3.0.0 - Professional Edition")
    print("=" * 60)
    
    # Validate configuration
    if not SecureConfig.validate_config():
        print("[X] Configuration Error: API keys not found")
        print("[TIP] Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("   Or ensure config.py file exists with valid keys")
        return 1
    
    # Run analytics
    with AlpacaAnalytics() as analytics:
        success = analytics.run_analysis()
        
        if success:
            print(f"\n{'='*60}")
            print("[OK] Alpaca Analytics Complete!")
            print(f"{'='*60}")
            return 0
        else:
            print(f"\n{'='*60}")
            print("[X] Analysis failed - check logs for details")
            print(f"{'='*60}")
            return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n[X] Unexpected error: {e}")
        sys.exit(1)
