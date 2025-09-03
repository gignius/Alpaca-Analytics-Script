"""
Secure configuration management for Alpaca trading analytics.
"""

import os
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass
class TradingConfig:
    """Trading configuration settings."""
    
    # Environment settings
    use_paper_trading: bool = True
    
    # Analysis settings
    default_analysis_periods: list = None
    max_orders_to_fetch: int = 500
    analysis_start_date: Optional[str] = None
    
    # Performance settings
    max_workers: int = 4
    chart_workers: int = 2
    memory_limit_mb: int = 2048
    
    # Chart settings
    chart_dpi: int = 300
    chart_format: str = "png"
    enable_charts: bool = True
    
    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_filename: str = "alpaca_analytics.log"
    
    def __post_init__(self):
        if self.default_analysis_periods is None:
            self.default_analysis_periods = ["1Y", "6M", "3M", "1M", "1W", "1D"]

class SecureConfig:
    """Secure configuration manager using environment variables."""
    
    @staticmethod
    def get_api_key() -> str:
        """Get API key from environment or config file."""
        api_key = os.getenv('ALPACA_API_KEY')
        
        if not api_key:
            # Fallback to config file (less secure)
            try:
                import config
                api_key = getattr(config, 'ALPACA_API_KEY', None)
            except ImportError:
                raise ValueError("ALPACA_API_KEY not found in environment variables or config file")
        
        return api_key
    
    @staticmethod
    def get_secret_key() -> str:
        """Get secret key from environment or config file."""
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not secret_key:
            # Fallback to config file (less secure)
            try:
                import config
                secret_key = getattr(config, 'ALPACA_SECRET_KEY', None)
            except ImportError:
                raise ValueError("ALPACA_SECRET_KEY not found in environment variables or config file")
        
        return secret_key
    
    @staticmethod
    def get_use_paper_trading() -> bool:
        """Get paper trading setting."""
        paper_trading = os.getenv('ALPACA_USE_PAPER_TRADING', 'true').lower()
        return paper_trading in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_log_level() -> LogLevel:
        """Get logging level."""
        level = os.getenv('ALPACA_LOG_LEVEL', 'INFO').upper()
        try:
            return LogLevel(level)
        except ValueError:
            return LogLevel.INFO
    
    @staticmethod
    def validate_config() -> bool:
        """Validate that required configuration is present."""
        try:
            SecureConfig.get_api_key()
            SecureConfig.get_secret_key()
            return True
        except ValueError:
            return False

# Global configuration instance
config = TradingConfig()

# Validation on import
if not SecureConfig.validate_config():
    import warnings
    warnings.warn(
        "API keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.",
        UserWarning
    )
