#!/usr/bin/env python3
"""
Simple script to check Alpaca account balance
"""

import requests
import json
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, USE_PAPER_TRADING

def check_balance():
    """Check current account balance."""
    
    # Set up API endpoint
    if USE_PAPER_TRADING:
        base_url = "https://paper-api.alpaca.markets"
        print("🧪 Using PAPER TRADING account")
    else:
        base_url = "https://api.alpaca.markets"
        print("💰 Using LIVE TRADING account")
    
    # Set up headers
    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
    }
    
    try:
        # Get account info
        response = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        
        if response.status_code == 200:
            account = response.json()
            
            print("\n" + "="*50)
            print("💰 ACCOUNT BALANCE")
            print("="*50)
            
            equity = float(account.get('equity', 0))
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))
            portfolio_value = float(account.get('portfolio_value', 0))
            
            print(f"💵 Equity: ${equity:,.2f}")
            print(f"💵 Cash: ${cash:,.2f}")
            print(f"💵 Buying Power: ${buying_power:,.2f}")
            print(f"💵 Portfolio Value: ${portfolio_value:,.2f}")
            
            # Check if this looks like the expected $100k
            if equity < 50000:  # Less than $50k
                print("\n⚠️  WARNING: Account balance is less than $50,000")
                print("📝 Expected: $100,000 for professional trading setup")
                print("🔧 This might be a new paper trading account")
                
                print("\n💡 SOLUTION OPTIONS:")
                print("1. 🎮 Reset your paper trading account to $100k in Alpaca dashboard")
                print("2. 📊 Update the crypto bot to use current balance as starting point")
                print("3. 💰 Add virtual funds in Alpaca paper trading settings")
                
            elif equity >= 90000:  # Close to $100k
                print("\n✅ Account balance looks correct for professional trading!")
                
            else:
                print(f"\n📊 Current balance: ${equity:,.2f}")
                print("💭 This might be after some trading activity")
            
            # Show account status
            status = account.get('status', 'unknown')
            print(f"\n📋 Account Status: {status}")
            
            return account
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🔍 Checking Alpaca account balance...")
    account = check_balance()
    
    if account:
        print("\n✅ Balance check completed!")
    else:
        print("\n❌ Failed to check balance")
