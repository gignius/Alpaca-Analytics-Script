#!/usr/bin/env python3
"""
Dependency Installation Script for Alpaca Trading Analytics
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        print("🚀 Installing Alpaca Trading Analytics dependencies...")
        print("=" * 60)
        
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("❌ requirements.txt not found!")
            return False
        
        # Install packages
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All dependencies installed successfully!")
            print("\n📊 Available features:")
            print("  • Comprehensive trading analytics")
            print("  • Professional chart generation") 
            print("  • Risk metrics calculation")
            print("  • Performance dashboards")
            print("  • Automatic chart opening in viewer/explorer")
            
            print("\n🎯 Next steps:")
            print("  1. Update your API keys in alpaca.py")
            print("  2. Run: python alpaca.py")
            print("  3. View generated charts in the 'charts' directory")
            
            return True
        else:
            print("❌ Installation failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def main():
    """Main installation function"""
    print("Alpaca Trading Analytics - Dependency Installer")
    print("=" * 60)
    
    success = install_requirements()
    
    if success:
        print("\n🎉 Setup complete! You're ready to generate trading analytics.")
    else:
        print("\n💡 Try manual installation:")
        print("pip install matplotlib seaborn pandas numpy scipy requests")

if __name__ == "__main__":
    main()
