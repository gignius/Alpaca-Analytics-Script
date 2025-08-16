# 📈 Alpaca Trading Analytics Tool

A comprehensive Python script for analyzing your Alpaca trading account performance with detailed metrics, visualizations, and portfolio analytics.

## 🚀 Features

### 📊 **Account Analytics**
- **Real-time account information** - Current balance, buying power, equity
- **Portfolio performance metrics** - Returns, volatility, Sharpe ratio
- **Position tracking** - Current holdings with P&L analysis
- **Order history** - Recent trades with detailed execution data

### 📈 **Advanced Performance Metrics**
- **Total & Annualized Returns** - CAGR calculation with time-weighted performance
- **Risk Metrics** - Volatility, maximum drawdown, Sharpe ratio
- **Win/Loss Analysis** - Success rates, average gains/losses
- **Trading Statistics** - Frequency analysis and pattern recognition

### 🔄 **Dual Environment Support**
- **Paper Trading** - Safe testing environment
- **Live Trading** - Real account monitoring
- **Automatic endpoint detection** - Seamlessly switches between environments

### 📅 **Historical Analysis**
- **Extended portfolio history** - Customizable date ranges
- **Trade pattern analysis** - Performance trends over time
- **Drawdown calculation** - Risk assessment metrics
- **Equity curve visualization** - Portfolio growth tracking

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Valid Alpaca trading account
- API keys from Alpaca dashboard

### Dependencies
```bash
pip install requests
```

### Setup
1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/alpaca-trading-analytics.git
   cd alpaca-trading-analytics
   ```

2. **Configure API credentials**
   
   Edit the script and replace with your credentials:
   ```python
   ALPACA_API_KEY = "your_api_key_here"
   ALPACA_SECRET_KEY = "your_secret_key_here"
   ```

3. **Run the script**
   ```bash
   python alpaca.py
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
🏦 Account Information:
  📊 Account Value: $125,487.32
  💰 Cash Balance: $12,450.67
  📈 Buying Power: $24,901.34
  ⚡ Day Trading Buying Power: $49,802.68
  📊 Portfolio Value: $113,036.65
```

### Performance Metrics
```
📈 Performance Analysis:
  📊 Total Return: +15.73%
  📅 Annualized Return (CAGR): +8.42%
  📉 Maximum Drawdown: -12.35%
  ⚡ Volatility (Annual): 18.76%
  📊 Sharpe Ratio: 0.87
  🎯 Win Rate: 67.3%
```

### Recent Trades
```
📋 Recent Orders (Last 20):
  🟢 2024-01-15 09:30:15 | BUY 100 AAPL @ $185.23 | FILLED
  🔴 2024-01-14 15:45:32 | SELL 50 TSLA @ $248.67 | FILLED
  🟢 2024-01-14 10:15:45 | BUY 200 SPY @ $478.91 | FILLED
```

### Current Positions
```
💼 Current Positions:
  📊 AAPL: 150 shares | $27,784.50 | +12.3% (+$3,045.23)
  📊 TSLA: 75 shares  | $18,650.25 | -5.8% (-$1,147.82)
  📊 SPY:  200 shares | $95,782.00 | +8.1% (+$7,234.56)
```

## 🎯 Use Cases

### 📈 **Day Traders**
- Monitor intraday performance and positions
- Track real-time P&L across multiple positions
- Analyze trading patterns and success rates

### 📊 **Swing Traders**
- Review weekly/monthly performance trends
- Calculate risk-adjusted returns
- Monitor portfolio diversification

### 🎓 **Learning Traders**
- Practice with paper trading analytics
- Understand performance metrics
- Build trading discipline with data

### 🏢 **Portfolio Managers**
- Monitor multiple strategies
- Generate performance reports
- Risk management and drawdown analysis

## 🔧 Configuration Options

### Time Periods
The script supports various analysis periods:
- **1D** - Intraday analysis
- **1W** - Weekly performance
- **1M** - Monthly trends
- **Custom** - Specify start date for historical analysis

### Analysis Depth
- **Basic Mode** - Essential metrics and recent activity
- **Comprehensive Mode** - Full statistical analysis with risk metrics
- **Historical Mode** - Extended backtesting from specific date

### Output Formats
- **Console Display** - Real-time terminal output with emojis
- **JSON Export** - Structured data for further analysis
- **CSV Export** - Spreadsheet-compatible format

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
- Verify API keys are correct
- Check if account is activated
- Ensure proper permissions are set
- Try regenerating API keys

#### No Trading History
```
❌ Insufficient data for comprehensive performance analysis
```
**Solutions:**
- Make some trades in paper environment
- Use a longer time period for analysis
- Check if account has historical data

#### Connection Errors
```
❌ Connection failed: Connection timeout
```
**Solutions:**
- Check internet connection
- Verify Alpaca API status
- Try again after a few minutes

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

### **Win Rate**
- Percentage of profitable trades
- Not the only important metric
- Should be combined with risk/reward ratio

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

## 📞 Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join discussions in the issues section

---

**Happy Trading! 📈🚀**

*Remember: The best trader is an educated trader. Use this tool to learn, analyze, and improve your trading performance.*
