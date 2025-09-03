# 📊 Alpaca Trading Analytics - Chart Features

## Overview
Enhanced Alpaca trading script with comprehensive analytics and professional chart generation capabilities.

## 🚀 New Features

### Advanced Analytics
- **Performance Metrics**: Total return, CAGR, Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk Analysis**: Maximum drawdown, volatility, Value at Risk (VaR)
- **Trade Analysis**: Win rate, profit factor, average win/loss, expectancy
- **Statistical Analysis**: Returns distribution, rolling metrics, Q-Q plots

### Professional Charts
1. **Equity Curve Chart** - Portfolio value over time with peak/trough annotations
2. **Drawdown Analysis** - Underwater chart showing portfolio drawdowns
3. **Returns Analysis** - Distribution histograms and rolling metrics
4. **Trade Analysis** - P&L scatter plots and cumulative performance
5. **Risk Dashboard** - Gauge charts for key risk metrics
6. **Performance Dashboard** - Comprehensive overview with all key metrics
7. **⭐ Comprehensive Merged Dashboard** - All charts and statistics in one large graphic

### Chart Features
- High-resolution PNG output (300 DPI)
- Automatic opening in default image viewer
- Professional styling with seaborn themes
- Interactive annotations and legends
- Grid layouts for multi-panel dashboards
- Color-coded risk indicators

## 📦 Installation

### Quick Setup
```bash
python install_deps.py
```

### Manual Installation
```bash
pip install -r requirements.txt
```

### Dependencies
- `matplotlib` - Chart generation
- `seaborn` - Statistical visualization
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scipy` - Statistical functions
- `requests` - API communication

## 🎯 Usage

### Basic Usage
```bash
python alpaca.py                    # Generate all 7 charts
python alpaca.py --comprehensive    # Generate only merged dashboard
python alpaca.py --test-charts      # Test chart generation
```

### What Happens
1. Fetches trading data from Alpaca API
2. Calculates comprehensive performance metrics
3. Generates 7 professional charts (including merged dashboard)
4. Automatically opens charts in default viewer
5. Saves charts in `charts/` directory

### Generated Files
- `equity_curve.png` - Portfolio performance over time
- `drawdown_chart.png` - Risk analysis (underwater chart)
- `returns_analysis.png` - Statistical distribution analysis
- `trade_analysis.png` - Individual trade performance
- `risk_dashboard.png` - Risk metrics overview
- `performance_dashboard.png` - Comprehensive summary
- `comprehensive_dashboard.png` - ⭐ **ALL-IN-ONE MERGED VIEW**

## 🖼️ Chart Details

### Equity Curve Chart
- Main portfolio value line chart
- Peak and trough annotations
- Daily returns bar chart subplot
- Professional date formatting

### Drawdown Chart
- Underwater visualization of portfolio drawdowns
- Maximum drawdown annotation
- Risk period identification
- Recovery analysis

### Returns Analysis
- Daily returns histogram
- Q-Q plot for normality testing
- Rolling Sharpe ratio
- Rolling volatility

### Trade Analysis
- Individual trade P&L scatter plot
- Cumulative P&L progression
- Win/loss pie chart
- Key trading metrics bar chart

### Risk Dashboard
- Gauge charts for Sharpe ratio
- Sortino ratio visualization
- Maximum drawdown gauge
- Volatility assessment

### Performance Dashboard
- Multi-panel comprehensive overview
- Key metrics summary panel
- Drawdown analysis subplot
- Returns distribution
- Trade analysis summary
- Risk metrics panel

### ⭐ Comprehensive Merged Dashboard
- **All-in-One Analytics**: Every chart and statistic in one large 24x16 inch graphic
- **Text Statistics Embedded**: All key metrics displayed as text within the image
- **9 Analysis Sections**:
  1. **Main Equity Curve** - Portfolio performance with peak/trough annotations
  2. **Key Metrics Panel** - All performance statistics in text format
  3. **Drawdown Chart** - Risk analysis with maximum drawdown highlighted
  4. **Returns Distribution** - Histogram with mean and median lines
  5. **Trade Statistics Panel** - Complete trade analysis in text format
  6. **Trade P&L Scatter** - Individual trade performance visualization
  7. **Cumulative P&L** - Running total of trade profits/losses
  8. **Win/Loss Pie Chart** - Visual breakdown of successful trades
  9. **Risk Gauges** - Visual indicators for Sharpe, Sortino, drawdown, and volatility
- **Professional Layout**: Grid-based design with proper spacing and alignment
- **High Resolution**: 300 DPI output perfect for presentations and reports
- **Complete Information**: No need to view multiple files - everything in one image
- **Presentation Ready**: Ideal for sharing with stakeholders, reports, or social media

## 🎨 Customization

### Chart Styling
Charts use professional seaborn styling with:
- High-contrast color schemes
- Grid overlays for readability
- Professional fonts and sizing
- Color-coded risk indicators

### File Management
- Charts saved in dedicated `charts/` directory
- High-resolution PNG format
- Timestamped for historical tracking
- Automatic directory creation

## 🔧 Platform Support

### Windows
- Uses `os.startfile()` to open charts
- Compatible with Windows 10/11
- Works with default image viewer

### macOS
- Uses `open` command
- Compatible with Preview and other viewers
- Full retina display support

### Linux
- Uses `xdg-open` command
- Compatible with most image viewers
- Works with GNOME, KDE, XFCE

## 📈 Performance Metrics Explained

### Return Metrics
- **Total Return**: Overall portfolio performance percentage
- **CAGR**: Compound Annual Growth Rate (annualized return)
- **Win Rate**: Percentage of profitable trades

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Return per unit of total risk
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: CAGR divided by maximum drawdown

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **VaR**: Value at Risk (5% confidence level)

### Trade Metrics
- **Profit Factor**: Gross profit / gross loss
- **Expectancy**: Average trade profit/loss
- **Win/Loss Ratio**: Average win / average loss

## 💡 Tips

### Best Practices
1. Review charts regularly for pattern recognition
2. Compare metrics against benchmarks
3. Use charts for presentations and reporting
4. Archive charts for historical comparison

### Troubleshooting
- Ensure API keys are correctly configured
- Check internet connection for data fetching
- Verify sufficient trading history exists
- Install all dependencies for full functionality

## 🚀 Future Enhancements

Potential future features:
- PDF report generation
- Interactive HTML charts
- Benchmark comparison charts
- Monte Carlo simulation
- Portfolio optimization charts
- Correlation analysis
- Sector allocation charts
- Custom date range selection

## 📞 Support

For issues or questions:
1. Check that all dependencies are installed
2. Verify API credentials are correct
3. Ensure sufficient trading data exists
4. Review error messages in console output
