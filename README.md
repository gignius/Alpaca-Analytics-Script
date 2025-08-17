# 📈 Alpaca Trading Analytics & Chart Generator

A comprehensive Python script for analyzing your Alpaca trading account performance with detailed metrics, professional visualizations, and advanced portfolio analytics. Generate publication-ready charts and comprehensive performance reports.

## 🚀 Features

### 📊 **Account Analytics**
- **Real-time account information** - Current balance, buying power, equity
- **Portfolio performance metrics** - Returns, volatility, Sharpe ratio
- **Position tracking** - Current holdings with P&L analysis
- **Order history** - Recent trades with detailed execution data

### 📈 **Advanced Performance Metrics**
- **Total & Annualized Returns** - CAGR calculation with time-weighted performance
- **Risk-Adjusted Metrics** - Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk Analysis** - Maximum drawdown, volatility, Value at Risk (VaR)
- **Trade Analytics** - Win rate, profit factor, expectancy, win/loss ratios
- **Statistical Analysis** - Returns distribution, rolling metrics, Q-Q plots

### 🎨 **Professional Chart Generation**
- **Equity Curve Chart** - Portfolio performance over time with annotations
- **Drawdown Analysis** - Underwater chart showing risk periods
- **Returns Analysis** - Distribution histograms and rolling metrics
- **Trade Analysis** - P&L scatter plots and cumulative performance
- **Risk Dashboard** - Gauge charts for key risk metrics
- **Performance Dashboard** - Comprehensive multi-panel overview
- **🎯 Comprehensive Merged Dashboard** - All analytics in one large graphic with text statistics

### 🔄 **Dual Environment Support**
- **Paper Trading** - Safe testing environment
- **Live Trading** - Real account monitoring
- **Automatic endpoint detection** - Seamlessly switches between environments

### 📅 **Historical Analysis**
- **Extended portfolio history** - Up to 6 months of data analysis
- **Trade pattern analysis** - Performance trends over time
- **Drawdown calculation** - Risk assessment with recovery analysis
- **Equity curve diagnostics** - Trend strength and smoothness metrics

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Valid Alpaca trading account
- API keys from Alpaca dashboard

### Quick Setup (Recommended)
```bash
python setup.py
```
This will automatically install all dependencies and help configure your API keys.

### Manual Installation
```bash
pip install -r requirements.txt
```

### Dependencies
**Core Requirements:**
- `requests` - API communication
- `matplotlib` - Chart generation
- `seaborn` - Statistical visualization
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scipy` - Statistical functions

**For Chart Generation:**
```bash
pip install matplotlib seaborn pandas numpy scipy
```

### Setup
1. **Clone this repository**
   ```bash
   git clone https://github.com/gignius/Alpaca-Analytics-Script.git
   cd Alpaca-Analytics-Script
   ```

2. **Quick Setup (Recommended)**
   ```bash
   python setup.py
   ```
   This will:
   - Install all required dependencies
   - Create a secure config file
   - Guide you through API key setup
   - Test your connection

3. **Manual Configuration**
   
   Create a `config.py` file with your credentials:
   ```python
   ALPACA_API_KEY = "your_api_key_here"
   ALPACA_SECRET_KEY = "your_secret_key_here"
   DEFAULT_ANALYSIS_PERIODS = ["1M", "3M", "1Y", "1W", "1D"]
   MAX_ORDERS_TO_FETCH = 100
   ANALYSIS_START_DATE = None  # Or "2024-04-11" for specific date
   ```

4. **Run the script**
   ```bash
   python alpaca.py                    # Full analysis with all charts
   python alpaca.py --comprehensive    # Generate only merged dashboard
   python alpaca.py --test-charts      # Test chart generation
   python alpaca.py --help             # Show all options
   ```

## 🔑 Getting Alpaca API Keys

1. **Sign up** at [Alpaca Trading](https://alpaca.markets/)
2. **Navigate to** your dashboard
3. **Go to** "API Keys" section
4. **Generate** new API keys
5. **Choose** Paper Trading for testing or Live Trading for real money

⚠️ **Security Note**: Never commit your actual API keys to version control. Consider using environment variables:

```python
import os
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
```

## 📊 Sample Output

### Account Overview
```
🏦 ACCOUNT INFORMATION
============================================================
Account ID: 13402eef-03a8-4cab-9cf6-c52755feac5d
Account Status: ACTIVE
Trading Blocked: False
Pattern Day Trader: False

💰 BALANCE INFORMATION:
  Equity: $103,445.81
  Cash: $-100,952.17
  Buying Power: $168,505.68
  Portfolio Value: $103,445.81
  Day Trade Count: 0
  Daily P&L: $0.00 (+0.00%)
```

### Comprehensive Performance Analysis
```
📊 COMPREHENSIVE PERFORMANCE ANALYSIS
================================================================================

🎯 PERFORMANCE METRICS
----------------------------------------
Total Return: +15.73%
Annualized Return (CAGR): +8.42%
Win Rate: 67.3%
Profit Factor: 1.45
Average Win: $24.02
Average Loss: $16.54
Expectancy: $5.12

⚖️ RISK-ADJUSTED METRICS
----------------------------------------
Sharpe Ratio: 1.87
Sortino Ratio: 2.34
Calmar Ratio: 0.68

⚠️ RISK METRICS
----------------------------------------
Maximum Drawdown: 12.35%
Drawdown Duration: 15 days
Recovery Time: 8 days
Volatility (Annualized): 18.76%
Value at Risk (5%): $2,456.78

⚡ TRADE EFFICIENCY METRICS
----------------------------------------
Total Trades: 119
Winning Trades: 80
Losing Trades: 39
Annual Turnover: 239.9 trades/year

📋 PERFORMANCE SUMMARY
----------------------------------------
Performance Rating: Excellent (5/5)
🎉 Excellent performance! Strong risk-adjusted returns.
```

### Generated Charts
```
🎨 Generating comprehensive analytics charts...
  📈 Creating Equity Curve...
  ✅ Equity Curve created successfully
  📈 Creating Drawdown Analysis...
  ✅ Drawdown Analysis created successfully
  📈 Creating Returns Analysis...
  ✅ Returns Analysis created successfully
  📈 Creating Trade Analysis...
  ✅ Trade Analysis created successfully
  📈 Creating Risk Dashboard...
  ✅ Risk Dashboard created successfully
  📈 Creating Performance Dashboard...
  ✅ Performance Dashboard created successfully
  📈 Creating Comprehensive Dashboard...
  ✅ Comprehensive Dashboard created successfully

✅ Generated 7 charts in 'charts' directory
📁 Charts automatically opened in default viewer

📊 Generated charts:
  • equity_curve.png
  • drawdown_chart.png
  • returns_analysis.png
  • trade_analysis.png
  • risk_dashboard.png
  • performance_dashboard.png
  • comprehensive_dashboard.png  ⭐ ALL-IN-ONE VIEW
```

### Comprehensive Dashboard Only
```bash
python alpaca.py --comprehensive
```
```
🎯 Generating Comprehensive Dashboard Only...
🎨 Creating comprehensive merged dashboard...
✅ Comprehensive dashboard created: charts/comprehensive_dashboard.png
🎯 This single chart contains all your analytics in one view!
```

### Current Positions
```
📊 CURRENT POSITIONS
============================================================

📈 AAPL
   Quantity: 150 shares
   Market Value: $27,784.50
   Cost Basis: $25,432.10
   Unrealized P&L: $2,352.40 (+9.25%)
   Avg Entry Price: $169.55

📈 TSLA
   Quantity: 75 shares
   Market Value: $18,650.25
   Cost Basis: $19,798.07
   Unrealized P&L: $-1,147.82 (-5.80%)
   Avg Entry Price: $263.97

💼 Total Position Value: $204,397.98
```

## 🎯 Use Cases

### 📈 **Day Traders**
- Monitor intraday performance and positions
- Track real-time P&L across multiple positions
- Analyze trading patterns with professional charts
- Generate daily performance reports

### 📊 **Swing Traders**
- Review weekly/monthly performance trends
- Calculate risk-adjusted returns with visual analysis
- Monitor portfolio diversification through charts
- Track drawdown periods and recovery

### 🎓 **Learning Traders**
- Practice with paper trading analytics
- Understand performance metrics through visualizations
- Build trading discipline with comprehensive data
- Learn from statistical analysis and Q-Q plots

### 🏢 **Portfolio Managers**
- Monitor multiple strategies with dashboard views
- Generate professional performance reports
- Risk management with advanced metrics
- Present results with publication-ready charts

### 📊 **Quantitative Analysts**
- Statistical analysis of returns distribution
- Rolling metrics and trend analysis
- Risk assessment with VaR calculations
- Performance attribution analysis

## 🔧 Configuration Options

### Time Periods
Configure analysis periods in `config.py`:
```python
DEFAULT_ANALYSIS_PERIODS = ["1M", "3M", "1Y", "1W", "1D"]
```
- **1D** - Intraday analysis
- **1W** - Weekly performance  
- **1M** - Monthly trends
- **3M** - Quarterly analysis
- **6M** - Semi-annual review
- **1Y** - Annual performance
- **Custom** - Specify start date: `ANALYSIS_START_DATE = "2024-04-11"`

### Analysis Depth
- **Basic Mode** - Essential metrics and recent activity
- **Comprehensive Mode** - Full statistical analysis with risk metrics
- **Chart Generation** - Professional visualizations with all metrics
- **Historical Mode** - Extended backtesting from specific date

### Chart Options
- **High-Resolution Output** - 300 DPI PNG files
- **Automatic Opening** - Charts open in default image viewer
- **Professional Styling** - Seaborn themes with custom colors
- **Multi-Panel Dashboards** - Comprehensive overview charts
- **Risk Color Coding** - Visual indicators for risk levels

### Output Formats
- **Console Display** - Real-time terminal output with emojis
- **Professional Charts** - High-resolution PNG files
- **Structured Data** - JSON-compatible performance metrics
- **Chart Directory** - Organized file management in `charts/` folder

## 🛡️ Security Best Practices

### API Key Management
- ✅ Use environment variables for production
- ✅ Regenerate keys regularly
- ✅ Use paper trading for testing
- ❌ Never commit keys to version control

### Paper vs Live Trading
- **Paper Trading**: Risk-free testing environment
- **Live Trading**: Real money - use with caution
- **Automatic Detection**: Script tries both endpoints

## 🐛 Troubleshooting

### Common Issues

#### "Unauthorized" Error
```
❌ Unauthorized for: https://api.alpaca.markets
```
**Solutions:**
- Verify API keys are correct in `config.py`
- Check if account is activated
- Ensure proper permissions are set
- Try regenerating API keys from Alpaca dashboard

#### Chart Generation Issues
```
❌ Chart dependencies not available
```
**Solutions:**
- Install chart dependencies: `pip install matplotlib seaborn pandas numpy scipy`
- Run the setup script: `python setup.py`
- Test chart generation: `python alpaca.py --test-charts`

#### No Trading History
```
❌ Insufficient data for comprehensive performance analysis
```
**Solutions:**
- Make some trades in paper environment
- Use a longer time period for analysis
- Check if account has historical data
- Try basic analysis with available data

#### Connection Errors
```
❌ Connection failed: Connection timeout
```
**Solutions:**
- Check internet connection
- Verify Alpaca API status
- Try again after a few minutes
- Check firewall settings

#### Chart Display Issues
```
⚠️ Could not open file automatically
```
**Solutions:**
- Charts are still saved in `charts/` directory
- Manually open the PNG files
- Check default image viewer settings
- Files are high-resolution and suitable for presentations

## 📚 Understanding the Metrics

### **Sharpe Ratio**
- Measures risk-adjusted returns
- Higher is better (>1.0 is good, >2.0 is excellent)
- Compares excess return to volatility

### **Maximum Drawdown**
- Largest peak-to-trough decline
- Risk measurement (lower is better)
- Important for risk management

### **CAGR (Compound Annual Growth Rate)**
- Smoothed annual return rate
- Accounts for compounding effects
- Better than simple average for long-term analysis

### **Sortino Ratio**
- Similar to Sharpe ratio but only penalizes downside volatility
- Higher values indicate better risk-adjusted performance
- More relevant for asymmetric return distributions

### **Calmar Ratio**
- CAGR divided by maximum drawdown
- Measures return per unit of worst-case risk
- Higher values indicate better risk management

### **Value at Risk (VaR)**
- Maximum expected loss at 5% confidence level
- Risk management metric for position sizing
- Helps understand potential downside exposure

### **Win Rate**
- Percentage of profitable trades
- Not the only important metric
- Should be combined with profit factor and expectancy

### **Profit Factor**
- Gross profit divided by gross loss
- Values > 1.0 indicate profitable trading
- Higher values show better trade selection

### **Expectancy**
- Average profit/loss per trade
- Positive values indicate profitable system
- Key metric for long-term success

## 📊 Chart Gallery

### 📈 Equity Curve Chart
- **Main Portfolio Line**: Shows portfolio value progression over time
- **Peak/Trough Annotations**: Highlights best and worst performance points
- **Daily Returns Subplot**: Bar chart showing daily percentage changes
- **Professional Formatting**: Date axes, currency formatting, grid overlays

### 📉 Drawdown Analysis (Underwater Chart)
- **Drawdown Visualization**: Shows portfolio decline from peaks
- **Maximum Drawdown Annotation**: Highlights worst drawdown period
- **Risk Assessment**: Visual representation of portfolio risk periods
- **Recovery Analysis**: Shows time to recover from drawdowns

### 📊 Returns Analysis
- **Distribution Histogram**: Shows frequency of daily returns
- **Q-Q Plot**: Tests if returns follow normal distribution
- **Rolling Sharpe Ratio**: 30-day rolling risk-adjusted performance
- **Rolling Volatility**: 30-day rolling volatility analysis

### 💰 Trade Analysis
- **P&L Scatter Plot**: Individual trade performance visualization
- **Cumulative P&L**: Running total of trade profits/losses
- **Win/Loss Pie Chart**: Visual breakdown of successful vs unsuccessful trades
- **Performance Metrics**: Bar chart of key trading statistics

### ⚠️ Risk Dashboard
- **Gauge Charts**: Visual representation of key risk metrics
- **Sharpe Ratio Gauge**: Risk-adjusted return measurement
- **Sortino Ratio Gauge**: Downside risk-adjusted performance
- **Drawdown & Volatility**: Risk level indicators with color coding

### 🎯 Performance Dashboard
- **Multi-Panel Overview**: Comprehensive performance summary
- **Key Metrics Panel**: Essential statistics in one view
- **Equity Curve**: Main portfolio performance chart
- **Risk Analysis**: Drawdown and volatility subplots
- **Trade Summary**: Win/loss analysis and key ratios

### ⭐ Comprehensive Merged Dashboard
- **All-in-One View**: Every chart and statistic in one large graphic
- **Text Statistics**: All key metrics displayed as text within the image
- **Professional Layout**: 24x16 inch high-resolution dashboard
- **Complete Analysis**: Equity curve, drawdowns, returns, trades, and risk metrics
- **Presentation Ready**: Perfect for reports, presentations, and sharing
- **Single File**: Everything you need in one comprehensive PNG file

## 🎨 Chart Features

### Professional Quality
- **High Resolution**: 300 DPI PNG output suitable for presentations
- **Publication Ready**: Professional styling with clean layouts
- **Color Coded**: Risk levels and performance indicators
- **Annotated**: Key points highlighted with explanatory text

### Automatic Management
- **Auto-Generated**: Charts created automatically with each analysis
- **Auto-Opened**: Charts open in default image viewer
- **Organized Storage**: Saved in dedicated `charts/` directory
- **Cross-Platform**: Works on Windows, macOS, and Linux

### Customizable Styling
- **Seaborn Themes**: Professional statistical visualization styling
- **Grid Overlays**: Enhanced readability with subtle grid lines
- **Consistent Branding**: Uniform color schemes across all charts
- **Responsive Layout**: Adapts to different screen sizes and resolutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This tool is for educational and informational purposes only.**

- **Not Financial Advice**: This software does not provide investment advice
- **Use at Your Own Risk**: Trading involves substantial risk of loss
- **Paper Trading Recommended**: Test thoroughly before using with real money
- **No Guarantees**: Past performance does not guarantee future results

Always consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for providing the trading API
- Python community for excellent libraries
- Trading community for feedback and suggestions

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in minutes
- **[Chart Features](CHART_FEATURES.md)** - Detailed chart documentation
- **[Changelog](CHANGELOG.md)** - Version history and new features
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project

## 📞 Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: Check the documentation files for detailed guides
- **Community**: Join discussions in the issues section
- **Quick Help**: Run `python alpaca.py --test-charts` to test your setup

## 🔄 Version Information

**Current Version**: 2.0.0
- ✅ Professional chart generation
- ✅ Advanced analytics engine
- ✅ Risk-adjusted metrics
- ✅ Comprehensive performance analysis
- ✅ High-resolution output
- ✅ Cross-platform support

---

**Happy Trading! 📈🚀**

*Remember: The best trader is an educated trader. Use this tool to learn, analyze, and improve your trading performance with professional-grade analytics and visualizations.*
