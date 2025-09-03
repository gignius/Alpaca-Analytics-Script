#!/usr/bin/env python3
"""
Simple connection test for Alpaca Trading Analytics
Tests API connectivity without performing full analysis
"""

import sys
from pathlib import Path

# Try to import configuration
try:
    if Path("config.py").exists():
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
    else:
        # Fall back to alpaca.py constants
        from alpaca import ALPACA_API_KEY, ALPACA_SECRET_KEY
except ImportError:
    print("❌ Configuration not found. Please run setup.py first.")
    sys.exit(1)

import requests

# API Configuration
ENDPOINTS = [
    "https://paper-api.alpaca.markets",  # Paper trading
    "https://api.alpaca.markets"         # Live trading
]

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

def test_endpoint(endpoint_url: str) -> bool:
    """Test a single endpoint for connectivity."""
    try:
        print(f"🔍 Testing: {endpoint_url}")
        response = requests.get(f"{endpoint_url}/v2/account", headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            account_data = response.json()
            account_status = account_data.get('status', 'unknown')
            account_value = float(account_data.get('portfolio_value', 0))
            
            print(f"✅ Connected successfully!")
            print(f"   📊 Account Status: {account_status}")
            print(f"   💰 Portfolio Value: ${account_value:,.2f}")
            
            # Determine if this is paper or live
            trading_type = "Paper" if "paper" in endpoint_url else "Live"
            print(f"   🎯 Trading Type: {trading_type}")
            
            return True
            
        elif response.status_code == 401:
            print(f"❌ Unauthorized - Check your API credentials")
            return False
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏱️  Timeout - API may be slow or unavailable")
        return False
    except requests.exceptions.ConnectionError:
        print(f"🌐 Connection Error - Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Test connection to Alpaca API."""
    print("🧪 Alpaca API Connection Test")
    print("=" * 40)
    
    # Check API key configuration
    if ALPACA_API_KEY in ["YOUR_API_KEY_HERE", "", None]:
        print("❌ API key not configured. Please run setup.py first.")
        return False
    
    if ALPACA_SECRET_KEY in ["YOUR_SECRET_KEY_HERE", "", None]:
        print("❌ Secret key not configured. Please run setup.py first.")
        return False
    
    print("🔑 API keys configured ✅")
    print("\n🔗 Testing API endpoints...")
    
    # Test both endpoints
    paper_success = test_endpoint(ENDPOINTS[0])
    print()
    live_success = test_endpoint(ENDPOINTS[1])
    
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"   📄 Paper Trading: {'✅ Connected' if paper_success else '❌ Failed'}")
    print(f"   💰 Live Trading:  {'✅ Connected' if live_success else '❌ Failed'}")
    
    if paper_success or live_success:
        print("\n🎉 Connection test successful!")
        print("   You can now run the full analytics: python alpaca.py")
        return True
    else:
        print("\n❌ Connection test failed!")
        print("   Please check your API credentials and try again.")
        print("   🔧 Troubleshooting:")
        print("   1. Verify API keys in your Alpaca dashboard")
        print("   2. Check if your account is activated")
        print("   3. Ensure proper API permissions are set")
        print("   4. Try regenerating your API keys")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
