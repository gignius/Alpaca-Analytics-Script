"""
Professional chart generation for Alpaca trading analytics.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

from models import AccountInfo, Position, Order, PortfolioHistory

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Professional chart generation for trading analytics."""
    
    def __init__(self, chart_dir: str = "charts", dpi: int = 300, format: str = "png"):
        self.chart_dir = Path(chart_dir)
        self.chart_dir.mkdir(exist_ok=True)
        self.dpi = dpi
        self.format = format
        
        # Set professional styling
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Configure matplotlib
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        
    def generate_equity_curve(self, portfolio_history: PortfolioHistory, 
                            account: AccountInfo) -> str:
        """Generate equity curve chart."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            
            # Convert to pandas for easier plotting
            df = pd.DataFrame({
                'timestamp': portfolio_history.timestamp,
                'equity': [float(e) for e in portfolio_history.equity],
                'pnl': [float(p) for p in portfolio_history.profit_loss],
            })
            
            # Main equity curve
            ax1.plot(df['timestamp'], df['equity'], linewidth=2, color='#2E86C1', label='Portfolio Value')
            ax1.fill_between(df['timestamp'], df['equity'], alpha=0.3, color='#2E86C1')
            
            # Add current value line
            ax1.axhline(y=float(account.equity), color='red', linestyle='--', 
                       label=f'Current: ${float(account.equity):,.2f}')
            
            ax1.set_title(f'Portfolio Equity Curve - Paper Trading Account {account.account_number}', 
                         fontsize=16, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Daily P&L subplot
            colors = ['green' if p >= 0 else 'red' for p in df['pnl']]
            ax2.bar(df['timestamp'], df['pnl'], color=colors, alpha=0.7, width=1)
            ax2.set_title('Daily P&L', fontsize=14)
            ax2.set_ylabel('P&L ($)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Format dates on both subplots
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            filename = self.chart_dir / f"equity_curve.{self.format}"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Equity curve chart saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error generating equity curve: {e}")
            plt.close()
            return ""
    
    def generate_performance_dashboard(self, metrics, account: AccountInfo) -> str:
        """Generate comprehensive performance metrics dashboard."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Performance Dashboard - Paper Trading Account {account.account_number}', 
                        fontsize=18, fontweight='bold')
            
            # 1. Key Metrics Gauge Chart
            metrics_data = {
                'Total Return': float(metrics.total_return),
                'Annual Return': float(metrics.annualized_return),
                'Sharpe Ratio': float(metrics.sharpe_ratio),
                'Max Drawdown': -float(metrics.max_drawdown),
            }
            
            colors = ['green' if v >= 0 else 'red' for v in metrics_data.values()]
            bars = ax1.bar(metrics_data.keys(), metrics_data.values(), color=colors, alpha=0.7)
            ax1.set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Value (%)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, (key, value) in zip(bars, metrics_data.items()):
                height = bar.get_height()
                ax1.annotate(f'{value:.2f}%' if 'Ratio' not in key else f'{value:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points", ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. Risk Metrics
            risk_data = {
                'Volatility': float(metrics.volatility),
                'Max Drawdown': float(metrics.max_drawdown),
                'Sortino Ratio': float(metrics.sortino_ratio),
                'Calmar Ratio': float(metrics.calmar_ratio),
            }
            
            # Create a radar-like bar chart
            bars2 = ax2.bar(risk_data.keys(), risk_data.values(), 
                           color=['orange', 'red', 'green', 'blue'], alpha=0.7)
            ax2.set_title('Risk Analysis', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Value')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Add value labels
            for bar, value in zip(bars2, risk_data.values()):
                height = bar.get_height()
                ax2.annotate(f'{value:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points", ha='center', va='bottom' if height >= 0 else 'top')
            
            # 3. Trade Performance
            trade_data = {
                'Total Trades': metrics.total_trades,
                'Winning Trades': metrics.winning_trades,
                'Losing Trades': metrics.losing_trades,
                'Win Rate (%)': float(metrics.win_rate),
            }
            
            # Create pie chart for win/loss ratio if trades exist
            if metrics.total_trades > 0:
                win_loss_data = [metrics.winning_trades, metrics.losing_trades]
                labels = [f'Winners ({metrics.winning_trades})', f'Losers ({metrics.losing_trades})']
                colors = ['green', 'red']
                
                # Only show non-zero segments
                filtered_data = [(data, label, color) for data, label, color in zip(win_loss_data, labels, colors) if data > 0]
                if filtered_data:
                    data_vals, data_labels, data_colors = zip(*filtered_data)
                    ax3.pie(data_vals, labels=data_labels, colors=data_colors, autopct='%1.1f%%', startangle=90)
                    ax3.set_title(f'Win/Loss Distribution\n(Win Rate: {float(metrics.win_rate):.1f}%)', 
                                 fontsize=14, fontweight='bold')
                else:
                    ax3.text(0.5, 0.5, 'No Trade Data', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                    ax3.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No Trades to Display', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Win/Loss Distribution', fontsize=14, fontweight='bold')
            
            # 4. Account Summary
            account_data = {
                'Portfolio Value': float(account.portfolio_value),
                'Cash': float(account.cash),
                'Buying Power': float(account.buying_power),
                'Daily P&L': float(account.daily_pnl),
            }
            
            # Create horizontal bar chart
            y_pos = np.arange(len(account_data))
            values = list(account_data.values())
            colors = ['green' if v >= 0 else 'red' for v in values]
            
            bars4 = ax4.barh(y_pos, values, color=colors, alpha=0.7)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(account_data.keys())
            ax4.set_title('Account Summary', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Amount ($)')
            ax4.grid(True, alpha=0.3)
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add value labels
            for bar, value in zip(bars4, values):
                width = bar.get_width()
                ax4.annotate(f'${value:,.0f}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(5 if width >= 0 else -5, 0),
                           textcoords="offset points", ha='left' if width >= 0 else 'right', va='center')
            
            plt.tight_layout()
            
            # Save chart
            filename = self.chart_dir / f"performance_dashboard.{self.format}"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance dashboard saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error generating performance dashboard: {e}")
            plt.close()
            return ""
    
    def generate_comprehensive_dashboard(self, portfolio_history: PortfolioHistory, 
                                       metrics, account: AccountInfo, positions: List[Position], orders: List[Order] = None) -> str:
        """Generate a comprehensive all-in-one dashboard."""
        try:
            # Calculate strategy score for display
            from .analytics import FinancialAnalytics
            analytics = FinancialAnalytics()
            strategy_score = analytics.score_strategy(metrics)
            
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle(f'Comprehensive Trading Dashboard - Paper Trading Account {account.account_number}\n'
                        f'STRATEGY SCORE: {strategy_score["score"]:.1f}/10 ({strategy_score["grade"]}) - {strategy_score["assessment"]}', 
                        fontsize=20, fontweight='bold')
            
            # Create grid layout
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Main equity curve (top row, spans 2 columns)
            ax1 = fig.add_subplot(gs[0, :2])
            df = pd.DataFrame({
                'timestamp': portfolio_history.timestamp,
                'equity': [float(e) for e in portfolio_history.equity],
            })
            ax1.plot(df['timestamp'], df['equity'], linewidth=3, color='#2E86C1', label='Portfolio Value')
            ax1.fill_between(df['timestamp'], df['equity'], alpha=0.3, color='#2E86C1')
            ax1.axhline(y=float(account.equity), color='red', linestyle='--', alpha=0.7,
                       label=f'Current: ${float(account.equity):,.0f}')
            ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # 2. Performance metrics (top right)
            ax2 = fig.add_subplot(gs[0, 2:])
            perf_metrics = {
                'Total Return': f"{float(metrics.total_return):.2f}%",
                'Annual Return': f"{float(metrics.annualized_return):.2f}%",
                'Sharpe Ratio': f"{float(metrics.sharpe_ratio):.2f}",
                'Max Drawdown': f"{float(metrics.max_drawdown):.2f}%",
                'Win Rate': f"{float(metrics.win_rate):.1f}%",
                'Total Trades': f"{metrics.total_trades}",
            }
            
            # Create text table
            ax2.axis('off')
            table_data = [[k, v] for k, v in perf_metrics.items()]
            table = ax2.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                             cellLoc='left', loc='center', colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            # Color code the values
            for i in range(1, len(table_data) + 1):
                cell = table[(i, 1)]
                if 'Return' in table_data[i-1][0] and float(table_data[i-1][1].replace('%', '')) > 0:
                    cell.set_facecolor('#90EE90')
                elif 'Drawdown' in table_data[i-1][0]:
                    cell.set_facecolor('#FFB6C1')
                elif 'Sharpe' in table_data[i-1][0] and float(table_data[i-1][1]) > 1:
                    cell.set_facecolor('#90EE90')
            
            ax2.set_title('Key Performance Metrics', fontsize=16, fontweight='bold')
            
            # 3. Top positions (middle left)
            ax3 = fig.add_subplot(gs[1, :2])
            if positions:
                # Get top 10 positions by market value
                top_positions = sorted(positions, key=lambda p: float(p.market_value), reverse=True)[:10]
                symbols = [p.symbol for p in top_positions]
                values = [float(p.market_value) for p in top_positions]
                pnls = [float(p.unrealized_pl) for p in top_positions]
                
                colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
                bars = ax3.barh(symbols, values, color=colors, alpha=0.7)
                ax3.set_title('Top 10 Positions by Value', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Market Value ($)')
                ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # Add P&L labels
                for bar, pnl in zip(bars, pnls):
                    width = bar.get_width()
                    ax3.annotate(f'${pnl:,.0f}',
                               xy=(width, bar.get_y() + bar.get_height() / 2),
                               xytext=(5, 0), textcoords="offset points",
                               ha='left', va='center', fontsize=10)
            else:
                ax3.text(0.5, 0.5, 'No Position Data', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Top Positions', fontsize=14, fontweight='bold')
            
            # 4. Risk metrics gauge (middle right)
            ax4 = fig.add_subplot(gs[1, 2:])
            risk_metrics = {
                'Volatility': float(metrics.volatility),
                'Sortino': float(metrics.sortino_ratio),
                'Calmar': float(metrics.calmar_ratio),
            }
            
            # Create gauge-style visualization
            angles = np.linspace(0, np.pi, len(risk_metrics))
            values = list(risk_metrics.values())
            
            ax4.bar(range(len(risk_metrics)), values, color=['orange', 'green', 'blue'], alpha=0.7)
            ax4.set_xticks(range(len(risk_metrics)))
            ax4.set_xticklabels(risk_metrics.keys())
            ax4.set_title('Risk Metrics', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Value')
            ax4.grid(True, alpha=0.3)
            
            # 5. Account summary (bottom row)
            ax5 = fig.add_subplot(gs[2, :])
            
            # Create summary statistics with strategy score breakdown
            summary_text = f"""
PAPER TRADING ACCOUNT SUMMARY
Account: {account.account_number} | Status: {account.status.value}

STRATEGY PERFORMANCE SCORE: {strategy_score["score"]:.1f}/10.0 - {strategy_score["assessment"]}
SCORING BREAKDOWN:
{chr(10).join(f"• {reason}" for reason in strategy_score["reasoning"][:4])}

PORTFOLIO METRICS:
• Total Equity: ${float(account.equity):,.2f}
• Portfolio Value: ${float(account.portfolio_value):,.2f} 
• Cash Position: ${float(account.cash):,.2f}
• Buying Power: ${float(account.buying_power):,.2f}
• Daily P&L: ${float(account.daily_pnl):,.2f} ({float(account.daily_pnl_percent):.2f}%)

PERFORMANCE SUMMARY:
• Total Return: {float(metrics.total_return):.2f}% | Annualized: {float(metrics.annualized_return):.2f}%
• Risk-Adjusted Returns: Sharpe {float(metrics.sharpe_ratio):.2f} | Sortino {float(metrics.sortino_ratio):.2f}
• Risk Control: Max DD {float(metrics.max_drawdown):.2f}% | Volatility {float(metrics.volatility):.2f}%
• Trading Activity: {metrics.total_trades} trades | {float(metrics.win_rate):.1f}% win rate
• Expected Value: ${float(metrics.expectancy):.2f} per trade | Profit Factor: {float(metrics.profit_factor):.2f}

POSITIONS: {len(positions)} active positions | Total Value: ${sum(float(p.market_value) for p in positions):,.2f}
            """
            
            ax5.text(0.05, 0.95, summary_text.strip(), transform=ax5.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            ax5.axis('off')
            
            # Save comprehensive dashboard
            filename = self.chart_dir / f"comprehensive_dashboard.{self.format}"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Comprehensive dashboard saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive dashboard: {e}")
            plt.close()
            return ""
    
    def generate_drawdown_chart(self, portfolio_history: PortfolioHistory, account: AccountInfo) -> str:
        """Generate drawdown analysis chart."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Convert to pandas for easier plotting
            df = pd.DataFrame({
                'timestamp': portfolio_history.timestamp,
                'equity': [float(e) for e in portfolio_history.equity],
            })
            
            # Calculate running maximum (peak) and drawdown
            df['peak'] = df['equity'].cummax()
            df['drawdown'] = ((df['equity'] - df['peak']) / df['peak'] * 100)
            
            # Plot underwater chart
            ax.fill_between(df['timestamp'], df['drawdown'], 0, alpha=0.7, color='red', label='Drawdown')
            ax.plot(df['timestamp'], df['drawdown'], color='darkred', linewidth=1)
            
            ax.set_title(f'Portfolio Drawdown Analysis - Paper Trading Account {account.account_number}', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Add statistics text box
            max_dd = df['drawdown'].min()
            avg_dd = df['drawdown'][df['drawdown'] < 0].mean() if any(df['drawdown'] < 0) else 0
            
            stats_text = f"Max Drawdown: {max_dd:.2f}%\nAvg Drawdown: {avg_dd:.2f}%"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            
            # Save chart
            filename = self.chart_dir / f"drawdown_chart.{self.format}"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Drawdown chart saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error generating drawdown chart: {e}")
            plt.close()
            return ""

    def generate_returns_analysis(self, portfolio_history: PortfolioHistory, account: AccountInfo) -> str:
        """Generate returns analysis chart."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Returns Analysis - Paper Trading Account {account.account_number}', 
                        fontsize=16, fontweight='bold')
            
            # Calculate returns starting from first trade
            equity_values = [float(e) for e in portfolio_history.equity]
            if len(equity_values) < 2:
                ax1.text(0.5, 0.5, 'Insufficient data for returns analysis', 
                        ha='center', va='center', transform=ax1.transAxes)
                return ""
                
            # Calculate daily returns
            returns = []
            for i in range(1, len(equity_values)):
                if equity_values[i-1] > 0:
                    daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1] * 100
                    returns.append(daily_return)
            
            if not returns:
                ax1.text(0.5, 0.5, 'No valid returns data', ha='center', va='center', transform=ax1.transAxes)
                return ""
            
            # 1. Daily returns time series
            dates = portfolio_history.timestamp[1:len(returns)+1]
            colors = ['green' if r >= 0 else 'red' for r in returns]
            ax1.bar(dates, returns, color=colors, alpha=0.7, width=1)
            ax1.set_title('Daily Returns (%)', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Return (%)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 2. Returns distribution histogram
            ax2.hist(returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title('Returns Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Daily Return (%)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=np.mean(returns), color='red', linestyle='--', alpha=0.8, label=f'Mean: {np.mean(returns):.3f}%')
            ax2.legend()
            
            # 3. Cumulative returns
            cumulative_returns = np.cumprod([1 + r/100 for r in returns]) - 1
            ax3.plot(dates, [r*100 for r in cumulative_returns], linewidth=2, color='blue')
            ax3.fill_between(dates, [r*100 for r in cumulative_returns], alpha=0.3, color='blue')
            ax3.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Cumulative Return (%)')
            ax3.grid(True, alpha=0.3)
            
            # 4. Rolling volatility (30-day window)
            if len(returns) >= 30:
                rolling_vol = pd.Series(returns).rolling(window=30).std() * np.sqrt(252)  # Annualized
                vol_dates = dates[29:]  # Skip first 29 days
                ax4.plot(vol_dates, rolling_vol[29:], linewidth=2, color='orange')
                ax4.fill_between(vol_dates, rolling_vol[29:], alpha=0.3, color='orange')
                ax4.set_title('30-Day Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Volatility (%)')
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for rolling volatility', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('30-Day Rolling Volatility', fontsize=14, fontweight='bold')
            
            ax4.grid(True, alpha=0.3)
            
            # Format dates on all subplots
            for ax in [ax1, ax3, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            filename = self.chart_dir / f"returns_analysis.{self.format}"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Returns analysis chart saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error generating returns analysis: {e}")
            plt.close()
            return ""

    def generate_trade_analysis(self, orders: List[Order], account: AccountInfo) -> str:
        """Generate trade analysis charts."""
        try:
            # Filter filled orders
            filled_orders = [order for order in orders if order.status.value == 'filled' and order.filled_qty > 0]
            
            if not filled_orders:
                return ""
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Trade Analysis - Paper Trading Account {account.account_number}', 
                        fontsize=16, fontweight='bold')
            
            # Group trades by symbol and calculate P&L
            trade_pnl = {}
            for order in filled_orders:
                symbol = order.symbol
                if symbol not in trade_pnl:
                    trade_pnl[symbol] = []
                
                # Approximate P&L calculation
                filled_value = float(order.filled_qty) * float(order.filled_avg_price or order.limit_price or 0)
                if order.side.value == 'buy':
                    trade_pnl[symbol].append(-filled_value)  # Cost
                else:  # sell
                    trade_pnl[symbol].append(filled_value)   # Revenue
            
            # Calculate net P&L per symbol
            symbol_pnl = {symbol: sum(pnls) for symbol, pnls in trade_pnl.items()}
            winners = {k: v for k, v in symbol_pnl.items() if v > 0}
            losers = {k: v for k, v in symbol_pnl.items() if v < 0}
            
            # 1. P&L by symbol (top 10 winners and losers)
            if winners or losers:
                top_winners = sorted(winners.items(), key=lambda x: x[1], reverse=True)[:10]
                top_losers = sorted(losers.items(), key=lambda x: x[1])[:10]
                
                symbols = [x[0] for x in top_winners + top_losers]
                pnls = [x[1] for x in top_winners + top_losers]
                colors = ['green' if p > 0 else 'red' for p in pnls]
                
                ax1.barh(symbols, pnls, color=colors, alpha=0.7)
                ax1.set_title('Top Winners & Losers by Symbol', fontsize=14, fontweight='bold')
                ax1.set_xlabel('P&L ($)')
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            else:
                ax1.text(0.5, 0.5, 'No P&L data available', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('P&L by Symbol', fontsize=14, fontweight='bold')
            
            # 2. Trade frequency by symbol (top 15)
            symbol_counts = {}
            for order in filled_orders:
                symbol_counts[order.symbol] = symbol_counts.get(order.symbol, 0) + 1
            
            top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            symbols = [x[0] for x in top_symbols]
            counts = [x[1] for x in top_symbols]
            
            ax2.bar(symbols, counts, alpha=0.7, color='skyblue')
            ax2.set_title('Trade Frequency by Symbol (Top 15)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Number of Trades')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # 3. Trade size distribution
            trade_sizes = [float(order.filled_qty) * float(order.filled_avg_price or order.limit_price or 0) 
                          for order in filled_orders if order.filled_avg_price or order.limit_price]
            
            if trade_sizes:
                ax3.hist(trade_sizes, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
                ax3.set_title('Trade Size Distribution', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Trade Value ($)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
                ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # Add statistics
                avg_size = np.mean(trade_sizes)
                ax3.axvline(x=avg_size, color='red', linestyle='--', alpha=0.8, 
                          label=f'Avg: ${avg_size:,.0f}')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No trade size data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Trade Size Distribution', fontsize=14, fontweight='bold')
            
            # 4. Trading activity over time
            trade_dates = [order.created_at.date() for order in filled_orders]
            date_counts = {}
            for date in trade_dates:
                date_counts[date] = date_counts.get(date, 0) + 1
            
            if date_counts:
                dates = sorted(date_counts.keys())
                counts = [date_counts[date] for date in dates]
                
                ax4.plot(dates, counts, marker='o', linewidth=2, markersize=4, color='purple')
                ax4.fill_between(dates, counts, alpha=0.3, color='purple')
                ax4.set_title('Trading Activity Over Time', fontsize=14, fontweight='bold')
                ax4.set_ylabel('Number of Trades')
                ax4.grid(True, alpha=0.3)
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No trading activity data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Trading Activity Over Time', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            filename = self.chart_dir / f"trade_analysis.{self.format}"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Trade analysis chart saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error generating trade analysis: {e}")
            plt.close()
            return ""

    def generate_risk_dashboard(self, metrics, account: AccountInfo) -> str:
        """Generate risk metrics dashboard with gauges."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Risk Dashboard - Paper Trading Account {account.account_number}', 
                        fontsize=16, fontweight='bold')
            
            # 1. Sharpe Ratio Gauge (0-4+ range: <1=Poor, 1-2=Fair, 2-3=Good, 3+=Excellent)  
            sharpe = float(metrics.sharpe_ratio)
            clamped_sharpe = min(sharpe, 4)
            sharpe_thresholds = [0.25, 0.5, 0.75, 1.0]  # 1/4=Poor, 2/4=Fair, 3/4=Good, 4/4=Excellent
            self._create_gauge(ax1, clamped_sharpe, 0, 4, 'Sharpe Ratio', ['Poor', 'Fair', 'Good', 'Excellent'], thresholds=sharpe_thresholds)
            
            # 2. Sortino Ratio Gauge (0-8+ range: <2=Poor, 2-4=Fair, 4-6=Good, 6+=Excellent)
            sortino = float(metrics.sortino_ratio)  
            clamped_sortino = min(sortino, 8)
            sortino_thresholds = [0.25, 0.5, 0.75, 1.0]  # 2/8=Poor, 4/8=Fair, 6/8=Good, 8/8=Excellent
            self._create_gauge(ax2, clamped_sortino, 0, 8, 'Sortino Ratio', ['Poor', 'Fair', 'Good', 'Excellent'], thresholds=sortino_thresholds)
            
            # 3. Max Drawdown (inverted - lower is better)
            max_dd = float(metrics.max_drawdown)
            self._create_gauge(ax3, 20 - max_dd, 0, 20, 'Drawdown Control', 
                             ['Poor', 'Fair', 'Good', 'Excellent'], 
                             f'Max DD: {max_dd:.2f}%')
            
            # 4. Volatility Assessment
            volatility = float(metrics.volatility)
            vol_score = max(0, 50 - volatility)  # Lower volatility = higher score
            self._create_gauge(ax4, vol_score, 0, 50, 'Volatility Control', 
                             ['High Vol', 'Moderate', 'Low Vol', 'Very Low'],
                             f'Volatility: {volatility:.1f}%')
            
            plt.tight_layout()
            
            # Save chart
            filename = self.chart_dir / f"risk_dashboard.{self.format}"
            plt.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Risk dashboard saved: {filename}")
            return str(filename)
            
        except Exception as e:
            logger.error(f"Error generating risk dashboard: {e}")
            plt.close()
            return ""

    def _create_gauge(self, ax, value, min_val, max_val, title, labels, subtitle=None, thresholds=None):
        """Create a gauge chart with proper threshold-based segments."""
        # Normalize value to 0-1 range
        norm_value = (value - min_val) / (max_val - min_val)
        norm_value = max(0, min(1, norm_value))  # Clamp to 0-1
        
        # Default thresholds if not provided (equal segments)
        if thresholds is None:
            thresholds = [0.25, 0.5, 0.75, 1.0]
        
        # Create gauge using a pie chart with threshold-based segments
        # Color order: low (red) → medium (yellow) → improving (light green) → high (green)
        colors = ['red', 'yellow', 'lightgreen', 'green']
        
        # Calculate segment sizes based on thresholds
        sizes = []
        prev_threshold = 0
        for threshold in thresholds:
            sizes.append(threshold - prev_threshold)
            prev_threshold = threshold
        
        # Create wedges
        wedges, texts = ax.pie(sizes, colors=colors, startangle=90, counterclock=False, 
                              wedgeprops=dict(width=0.3))
        
        # Add indicator needle
        angle = 90 - (norm_value * 180)  # Convert to angle (90 to -90 degrees)
        angle_rad = np.radians(angle)
        needle_length = 0.7
        
        ax.annotate('', xy=(needle_length * np.cos(angle_rad), needle_length * np.sin(angle_rad)),
                   xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        
        # Add center circle
        circle = plt.Circle((0, 0), 0.1, color='black')
        ax.add_patch(circle)
        
        # Add labels at segment centers
        cumulative = 0
        for i, (label, size) in enumerate(zip(labels, sizes)):
            # Calculate angle for center of segment
            segment_center = cumulative + size / 2
            angle = 90 - (segment_center * 180)
            angle_rad = np.radians(angle)
            
            # Position label
            label_radius = 0.85
            x = label_radius * np.cos(angle_rad)
            y = label_radius * np.sin(angle_rad)
            
            ax.text(x, y, label, ha='center', va='center', fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.7))
            
            cumulative += size
        
        # Add title and value
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.text(0, -1.3, f'{value:.2f}', ha='center', va='center', fontsize=16, fontweight='bold')
        
        if subtitle:
            ax.text(0, -1.5, subtitle, ha='center', va='center', fontsize=10, style='italic')
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.6, 1.2)
        ax.set_aspect('equal')

    def generate_all_charts(self, portfolio_history: PortfolioHistory, metrics, 
                          account: AccountInfo, positions: List[Position], orders: List[Order] = None) -> Dict[str, str]:
        """Generate all charts and return filenames."""
        charts = {}
        
        try:
            logger.info("Starting comprehensive chart generation...")
            
            # Generate all individual charts
            charts['equity_curve'] = self.generate_equity_curve(portfolio_history, account)
            charts['drawdown_chart'] = self.generate_drawdown_chart(portfolio_history, account)
            charts['returns_analysis'] = self.generate_returns_analysis(portfolio_history, account)
            charts['risk_dashboard'] = self.generate_risk_dashboard(metrics, account)
            charts['performance_dashboard'] = self.generate_performance_dashboard(metrics, account)
            charts['comprehensive_dashboard'] = self.generate_comprehensive_dashboard(
                portfolio_history, metrics, account, positions, orders)
            
            # Generate trade analysis if orders are provided
            if orders:
                charts['trade_analysis'] = self.generate_trade_analysis(orders, account)
            
            # Filter out empty results
            charts = {k: v for k, v in charts.items() if v}
            
            logger.info(f"Generated {len(charts)} charts successfully")
            return charts
            
        except Exception as e:
            logger.error(f"Error in comprehensive chart generation: {e}")
            return {}
