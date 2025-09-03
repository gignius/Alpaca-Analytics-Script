"""
Account data models with validation and type safety.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from decimal import Decimal
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AccountStatus(Enum):
    """Account status enumeration."""
    ACTIVE = "ACTIVE"
    ACCOUNT_UPDATED = "ACCOUNT_UPDATED"
    APPROVAL_PENDING = "APPROVAL_PENDING"
    SUBMITTED = "SUBMITTED"
    INACTIVE = "INACTIVE"
    REJECTED = "REJECTED"

class OrderStatus(Enum):
    """Order status enumeration."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"
    HELD = "held"

@dataclass
class AccountInfo:
    """Account information with validation."""
    
    id: str
    account_number: str
    status: AccountStatus
    currency: str
    buying_power: Decimal
    regt_buying_power: Decimal
    daytrading_buying_power: Decimal
    cash: Decimal
    portfolio_value: Decimal
    equity: Decimal
    last_equity: Decimal
    multiplier: str
    day_trade_count: int
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime
    trade_suspended_by_user: bool
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'AccountInfo':
        """Create AccountInfo from API response with validation."""
        try:
            return cls(
                id=str(data['id']),
                account_number=str(data['account_number']),
                status=AccountStatus(data['status']),
                currency=str(data['currency']),
                buying_power=Decimal(str(data['buying_power'])),
                regt_buying_power=Decimal(str(data['regt_buying_power'])),
                daytrading_buying_power=Decimal(str(data['daytrading_buying_power'])),
                cash=Decimal(str(data['cash'])),
                portfolio_value=Decimal(str(data['portfolio_value'])),
                equity=Decimal(str(data['equity'])),
                last_equity=Decimal(str(data['last_equity'])),
                multiplier=str(data['multiplier']),
                day_trade_count=int(data['daytrade_count']),
                pattern_day_trader=bool(data['pattern_day_trader']),
                trading_blocked=bool(data.get('trading_blocked', False)),
                transfers_blocked=bool(data.get('transfers_blocked', False)),
                account_blocked=bool(data.get('account_blocked', False)),
                created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
                trade_suspended_by_user=bool(data['trade_suspended_by_user'])
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse account data: {e}")
            raise ValueError(f"Invalid account data: {e}")
    
    @property
    def is_active(self) -> bool:
        """Check if account is active and ready for trading."""
        return (
            self.status == AccountStatus.ACTIVE and
            not self.trading_blocked and
            not self.account_blocked
        )
    
    @property
    def daily_pnl(self) -> Decimal:
        """Calculate daily P&L."""
        return self.equity - self.last_equity
    
    @property
    def daily_pnl_percent(self) -> Decimal:
        """Calculate daily P&L percentage."""
        if self.last_equity == 0:
            return Decimal('0')
        return (self.daily_pnl / self.last_equity) * 100

@dataclass
class Position:
    """Position data model."""
    
    asset_id: str
    symbol: str
    exchange: str
    asset_class: str
    qty: Decimal
    avg_entry_price: Decimal
    side: str
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pl: Decimal
    unrealized_plpc: Decimal
    current_price: Optional[Decimal] = None
    lastday_price: Optional[Decimal] = None
    change_today: Optional[Decimal] = None
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'Position':
        """Create Position from API response."""
        try:
            return cls(
                asset_id=str(data['asset_id']),
                symbol=str(data['symbol']),
                exchange=str(data['exchange']),
                asset_class=str(data['asset_class']),
                qty=Decimal(str(data['qty'])),
                avg_entry_price=Decimal(str(data['avg_entry_price'])),
                side=str(data['side']),
                market_value=Decimal(str(data['market_value'])),
                cost_basis=Decimal(str(data['cost_basis'])),
                unrealized_pl=Decimal(str(data['unrealized_pl'])),
                unrealized_plpc=Decimal(str(data['unrealized_plpc'])),
                current_price=Decimal(str(data['current_price'])) if data.get('current_price') else None,
                lastday_price=Decimal(str(data['lastday_price'])) if data.get('lastday_price') else None,
                change_today=Decimal(str(data['change_today'])) if data.get('change_today') else None
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse position data: {e}")
            raise ValueError(f"Invalid position data: {e}")
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pl > 0

@dataclass
class Order:
    """Order data model."""
    
    id: str
    client_order_id: str
    created_at: datetime
    updated_at: datetime
    submitted_at: datetime
    filled_at: Optional[datetime]
    expired_at: Optional[datetime]
    canceled_at: Optional[datetime]
    failed_at: Optional[datetime]
    replaced_at: Optional[datetime]
    asset_id: str
    symbol: str
    asset_class: str
    qty: Decimal
    filled_qty: Decimal
    type: str
    side: str
    time_in_force: str
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: OrderStatus
    extended_hours: bool
    legs: Optional[List[Dict]] = None
    trail_percent: Optional[Decimal] = None
    trail_price: Optional[Decimal] = None
    hwm: Optional[Decimal] = None
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'Order':
        """Create Order from API response."""
        try:
            return cls(
                id=str(data['id']),
                client_order_id=str(data['client_order_id']),
                created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
                submitted_at=datetime.fromisoformat(data['submitted_at'].replace('Z', '+00:00')),
                filled_at=datetime.fromisoformat(data['filled_at'].replace('Z', '+00:00')) if data.get('filled_at') else None,
                expired_at=datetime.fromisoformat(data['expired_at'].replace('Z', '+00:00')) if data.get('expired_at') else None,
                canceled_at=datetime.fromisoformat(data['canceled_at'].replace('Z', '+00:00')) if data.get('canceled_at') else None,
                failed_at=datetime.fromisoformat(data['failed_at'].replace('Z', '+00:00')) if data.get('failed_at') else None,
                replaced_at=datetime.fromisoformat(data['replaced_at'].replace('Z', '+00:00')) if data.get('replaced_at') else None,
                asset_id=str(data['asset_id']),
                symbol=str(data['symbol']),
                asset_class=str(data['asset_class']),
                qty=Decimal(str(data['qty'])),
                filled_qty=Decimal(str(data['filled_qty'])),
                type=str(data['type']),
                side=str(data['side']),
                time_in_force=str(data['time_in_force']),
                limit_price=Decimal(str(data['limit_price'])) if data.get('limit_price') else None,
                stop_price=Decimal(str(data['stop_price'])) if data.get('stop_price') else None,
                status=OrderStatus(data['status']),
                extended_hours=bool(data.get('extended_hours', False)),
                legs=data.get('legs'),
                trail_percent=Decimal(str(data['trail_percent'])) if data.get('trail_percent') else None,
                trail_price=Decimal(str(data['trail_price'])) if data.get('trail_price') else None,
                hwm=Decimal(str(data['hwm'])) if data.get('hwm') else None
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse order data: {e}")
            raise ValueError(f"Invalid order data: {e}")
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage."""
        if self.qty == 0:
            return Decimal('0')
        return (self.filled_qty / self.qty) * 100

@dataclass
class PortfolioHistory:
    """Portfolio history data model."""
    
    timestamp: List[datetime]
    equity: List[Decimal]
    profit_loss: List[Decimal]
    profit_loss_pct: List[Decimal]
    base_value: Decimal
    timeframe: str
    
    @classmethod
    def from_api_response(cls, data: Dict) -> 'PortfolioHistory':
        """Create PortfolioHistory from API response."""
        try:
            timestamps = [datetime.fromtimestamp(ts) for ts in data['timestamp']]
            equity = [Decimal(str(val)) for val in data['equity']]
            profit_loss = [Decimal(str(val)) for val in data['profit_loss']]
            profit_loss_pct = [Decimal(str(val)) for val in data['profit_loss_pct']]
            
            return cls(
                timestamp=timestamps,
                equity=equity,
                profit_loss=profit_loss,
                profit_loss_pct=profit_loss_pct,
                base_value=Decimal(str(data['base_value'])),
                timeframe=str(data['timeframe'])
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse portfolio history: {e}")
            raise ValueError(f"Invalid portfolio history data: {e}")
    
    @property
    def total_return(self) -> Decimal:
        """Calculate total return percentage."""
        if not self.equity or self.base_value == 0:
            return Decimal('0')
        
        initial_value = self.equity[0] if self.equity[0] > 0 else self.base_value
        final_value = self.equity[-1]
        
        return ((final_value - initial_value) / initial_value) * 100
    
    @property
    def max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown."""
        if not self.equity:
            return Decimal('0')
        
        peak = self.equity[0]
        max_dd = Decimal('0')
        
        for value in self.equity:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
