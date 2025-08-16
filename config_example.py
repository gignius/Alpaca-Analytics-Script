"""
Example configuration file for Alpaca Trading Analytics
Copy this file to config.py and add your actual API keys
"""

# Alpaca API Configuration
# Get these from your Alpaca dashboard: https://app.alpaca.markets/
ALPACA_API_KEY = "YOUR_API_KEY_HERE"
ALPACA_SECRET_KEY = "YOUR_SECRET_KEY_HERE"

# Optional: Environment-specific settings
USE_PAPER_TRADING = True  # Set to False for live trading
DEFAULT_ANALYSIS_PERIOD = "1M"  # Default time period for analysis
MAX_ORDERS_TO_DISPLAY = 50  # Maximum orders to show in recent orders

# API Rate limiting settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Output formatting
CURRENCY_SYMBOL = "$"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S UTC"
PRECISION_DECIMAL_PLACES = 2
