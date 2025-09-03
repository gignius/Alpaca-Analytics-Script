"""
Secure and robust Alpaca API client with proper error handling.
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import os
from contextlib import contextmanager
from datetime import datetime

from models import AccountInfo, Position, Order, PortfolioHistory

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Trading environment."""
    PAPER = "paper"
    LIVE = "live"

@dataclass
class APIConfig:
    """API configuration with secure defaults."""
    paper_url: str = "https://paper-api.alpaca.markets"
    live_url: str = "https://api.alpaca.markets"
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    retry_status_codes: List[int] = None
    
    def __post_init__(self):
        if self.retry_status_codes is None:
            self.retry_status_codes = [429, 500, 502, 503, 504]

class AlpacaAPIError(Exception):
    """Custom exception for Alpaca API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class SecureAlpacaClient:
    """Secure Alpaca API client with proper error handling and validation."""
    
    def __init__(self, environment: Environment = Environment.PAPER):
        self.environment = environment
        self.config = APIConfig()
        self._session = None
        self._setup_session()
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Setup authentication from environment variables."""
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if self.api_key and self.secret_key:
            environment_type = "PAPER TRADING" if self.environment == Environment.PAPER else "LIVE TRADING"
            logger.warning(f"Using {environment_type} API keys from environment variables")
            logger.debug(f"API Key from env: {self.api_key[:8]}..." if self.api_key else "None")
        
        if not self.api_key or not self.secret_key:
            # Fallback to config file (less secure)
            try:
                import api_keys
                logger.debug(f"API keys imported successfully")
                api_key = getattr(api_keys, 'ALPACA_API_KEY', None)
                secret_key = getattr(api_keys, 'ALPACA_SECRET_KEY', None)
                logger.debug(f"Config API key: {api_key[:8] if api_key else 'None'}...")
                logger.debug(f"Config secret key: {secret_key[:8] if secret_key else 'None'}...")
                self.api_key = api_key
                self.secret_key = secret_key
                environment_type = "PAPER TRADING" if self.environment == Environment.PAPER else "LIVE TRADING"
                logger.warning(f"Using {environment_type} API keys from config file - consider using environment variables")
                logger.debug(f"Final API Key: {self.api_key[:8]}..." if self.api_key else "None")
            except ImportError:
                raise AlpacaAPIError("No API keys found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
    
    def _setup_session(self):
        """Setup HTTP session with retry strategy."""
        self._session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            status_forcelist=self.config.retry_status_codes,
            backoff_factor=self.config.backoff_factor,
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
    
    @property
    def base_url(self) -> str:
        """Get base URL for current environment."""
        return self.config.paper_url if self.environment == Environment.PAPER else self.config.live_url
    
    @contextmanager
    def _handle_api_errors(self, endpoint: str):
        """Context manager for consistent error handling."""
        try:
            yield
        except requests.exceptions.Timeout:
            raise AlpacaAPIError(f"Timeout when calling {endpoint}")
        except requests.exceptions.ConnectionError:
            raise AlpacaAPIError(f"Connection error when calling {endpoint}")
        except requests.exceptions.HTTPError as e:
            raise AlpacaAPIError(f"HTTP error {e.response.status_code} when calling {endpoint}", 
                               e.response.status_code, e.response.json() if e.response.content else None)
        except Exception as e:
            raise AlpacaAPIError(f"Unexpected error when calling {endpoint}: {str(e)}")
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     data: Optional[Dict] = None) -> Dict:
        """Make authenticated API request with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        with self._handle_api_errors(endpoint):
            response = self._session.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=self.config.timeout
            )
            
            if response.status_code == 401:
                raise AlpacaAPIError("Unauthorized - check your API keys", 401)
            elif response.status_code == 403:
                environment_type = "PAPER TRADING" if self.environment == Environment.PAPER else "LIVE TRADING"
                raise AlpacaAPIError(f"Forbidden - insufficient permissions (using {environment_type} keys)", 403)
            elif response.status_code == 429:
                raise AlpacaAPIError("Rate limit exceeded - please wait", 429)
            
            response.raise_for_status()
            
            try:
                return response.json()
            except ValueError:
                logger.warning(f"Non-JSON response from {endpoint}")
                return {}
    
    def get_account(self) -> AccountInfo:
        """Get account information with validation."""
        # Paper trading may have different account endpoint behavior
        if self.environment == Environment.PAPER:
            logger.info("Accessing paper trading account information")
        
        data = self._make_request('GET', '/v2/account')
        return AccountInfo.from_api_response(data)
    
    def get_positions(self) -> List[Position]:
        """Get current positions with validation."""
        # Paper trading accounts may start with no positions
        if self.environment == Environment.PAPER:
            logger.info("Fetching paper trading positions")
        
        data = self._make_request('GET', '/v2/positions')
        return [Position.from_api_response(pos) for pos in data]
    
    def get_orders(self, status: str = "all", limit: int = 500, 
                   after: Optional[datetime] = None, until: Optional[datetime] = None) -> List[Order]:
        """Get orders with validation."""
        params = {
            "status": status,
            "limit": min(limit, 500),  # Respect API limits
            "direction": "desc"
        }
        
        if after:
            params["after"] = after.isoformat()
        if until:
            params["until"] = until.isoformat()
        
        data = self._make_request('GET', '/v2/orders', params=params)
        return [Order.from_api_response(order) for order in data]
    
    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D", 
                             extended_hours: bool = False) -> Optional[PortfolioHistory]:
        """Get portfolio history with validation."""
        params = {
            "period": period,
            "timeframe": timeframe,
            "extended_hours": str(extended_hours).lower()
        }
        
        try:
            if self.environment == Environment.PAPER:
                logger.info(f"Fetching paper trading portfolio history for period: {period}")
            
            data = self._make_request('GET', '/v2/account/portfolio/history', params=params)
            return PortfolioHistory.from_api_response(data)
        except AlpacaAPIError as e:
            if e.status_code == 422:
                if self.environment == Environment.PAPER:
                    logger.warning(f"Paper trading portfolio history not available for period {period} - may need trading activity first")
                else:
                    logger.warning(f"Portfolio history not available for period {period}")
                return None
            raise
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            if self.environment == Environment.PAPER:
                logger.info("Testing paper trading API connection...")
            else:
                logger.info("Testing live trading API connection...")
            
            self.get_account()
            
            if self.environment == Environment.PAPER:
                logger.info("Paper trading API connection successful")
            
            return True
        except AlpacaAPIError as e:
            if self.environment == Environment.PAPER:
                logger.error(f"Paper trading connection test failed: {e}")
                logger.info("Note: Paper trading accounts may require specific setup in Alpaca dashboard")
            else:
                logger.error(f"Live trading connection test failed: {e}")
            return False
    
    def close(self):
        """Close the session."""
        if self._session:
            self._session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
