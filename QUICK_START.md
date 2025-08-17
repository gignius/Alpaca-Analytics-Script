# 🚀 Quick Start Guide

Get up and running with Alpaca Trading Analytics in minutes!

## ⚡ One-Command Setup

```bash
python setup.py
```

This will:
- ✅ Install all dependencies automatically
- ✅ Guide you through API key configuration
- ✅ Create secure config file
- ✅ Test your connection

## 🎯 Basic Usage

### Run Full Analysis
```bash
python alpaca.py
```
**What you get:**
- Complete account overview
- Performance metrics and statistics
- 7 professional charts automatically generated
- Risk analysis and trade statistics
- ⭐ Comprehensive merged dashboard with everything in one view

### Generate Only Merged Dashboard
```bash
python alpaca.py --comprehensive
```
**What you get:**
- Single large graphic with all analytics
- All statistics embedded as text in the image
- Perfect for sharing and presentations
- Complete analysis in one file

### Test Chart Generation
```bash
python alpaca.py --test-charts
```
**What it does:**
- Creates sample data
- Tests all chart generation functions
- Verifies dependencies are working
- Saves test charts to `charts/` folder

## 📊 What Charts Are Generated?

1. **📈 Equity Curve** - Portfolio performance over time
2. **📉 Drawdown Analysis** - Risk periods and recovery
3. **📊 Returns Analysis** - Statistical distribution
4. **💰 Trade Analysis** - Individual trade performance
5. **⚠️ Risk Dashboard** - Key risk metrics
6. **🎯 Performance Dashboard** - Comprehensive overview
7. **⭐ Comprehensive Dashboard** - ALL analytics in one large graphic with text stats

## 🔑 Getting API Keys

1. **Sign up** at [Alpaca Markets](https://alpaca.markets/)
2. **Go to** your dashboard
3. **Navigate to** "API Keys" section
4. **Generate** new keys
5. **Choose** Paper Trading for testing

## 📁 File Structure After Setup

```
your-project/
├── alpaca.py              # Main script
├── config.py              # Your API keys (auto-generated)
├── charts/                # Generated charts folder
│   ├── equity_curve.png
│   ├── drawdown_chart.png
│   ├── returns_analysis.png
│   ├── trade_analysis.png
│   ├── risk_dashboard.png
│   ├── performance_dashboard.png
│   └── comprehensive_dashboard.png  ⭐ ALL-IN-ONE
├── requirements.txt       # Dependencies
└── setup.py              # Setup script
```

## 🎨 Sample Output

### Console Output
```
🚀 Alpaca Trading Analytics & Chart Generator
📊 Advanced analytics charts enabled
✅ Successfully connected to: https://paper-api.alpaca.markets

🏦 ACCOUNT INFORMATION
============================================================
Account Status: ACTIVE
Equity: $103,445.81
Portfolio Value: $103,445.81

📊 COMPREHENSIVE PERFORMANCE ANALYSIS
================================================================================
🎯 PERFORMANCE METRICS
Total Return: +15.73%
Annualized Return (CAGR): +8.42%
Win Rate: 67.3%
Sharpe Ratio: 1.87

🎨 Generating comprehensive analytics charts...
✅ Generated 7 charts in 'charts' directory
📁 Charts automatically opened in default viewer
⭐ Comprehensive dashboard contains everything in one view!
```

### Generated Charts
All charts are:
- **High Resolution** (300 DPI)
- **Professional Quality**
- **Automatically Opened**
- **Saved in `charts/` folder**

## 🛠️ Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### API Connection Issues
- ✅ Check API keys in `config.py`
- ✅ Verify account is activated
- ✅ Try paper trading first

### Chart Issues
```bash
python alpaca.py --test-charts
```

## 💡 Pro Tips

### For Best Results
1. **Use Paper Trading** first to test
2. **Make some trades** to get meaningful data
3. **Run regularly** to track performance
4. **Save charts** for presentations

### Performance Analysis
- **Sharpe Ratio > 1.0** = Good risk-adjusted returns
- **Max Drawdown < 10%** = Good risk management
- **Win Rate > 50%** = More wins than losses
- **Profit Factor > 1.5** = Strong trading system

### Chart Usage
- **Equity Curve**: Track overall performance
- **Drawdown**: Monitor risk periods
- **Returns**: Understand volatility patterns
- **Trade Analysis**: Improve trade selection
- **Risk Dashboard**: Quick risk assessment
- **Performance Dashboard**: Complete overview

## 🎯 Next Steps

1. **Review your charts** - Look for patterns and trends
2. **Analyze metrics** - Compare against benchmarks
3. **Identify improvements** - Focus on weak areas
4. **Track progress** - Run analysis regularly
5. **Share results** - Use charts in presentations

## 📞 Need Help?

- **Check README.md** for detailed documentation
- **Review CHART_FEATURES.md** for chart explanations
- **Open GitHub issue** for bugs or questions
- **Test with sample data** using `--test-charts`

---

**Happy Trading! 📈🚀**

*Remember: This tool is for analysis and education. Always trade responsibly!*