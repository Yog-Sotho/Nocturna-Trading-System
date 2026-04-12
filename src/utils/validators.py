"""
NOCTURNA Trading System - Input Validators
Production-grade input validation using Pydantic schemas.
"""

import os
import sys
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError, ConfigError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ValidationException(Exception):
    """Custom exception for validation errors."""

    def __init__(self, errors: List[Dict]):
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")


# =============================================================================
# ENUMS FOR VALIDATION
# =============================================================================

class OrderSide(str, Enum):
    """Order side enum."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class RiskLevel(str, Enum):
    """Risk level enum."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    AGGRESSIVE = "AGGRESSIVE"


class TradingMode(str, Enum):
    """Trading mode enum."""
    PAPER = "PAPER"
    LIVE = "LIVE"


# =============================================================================
# CONFIGURATION VALIDATORS
# =============================================================================

class ConfigurationSchema(BaseModel):
    """
    Validates and types the system configuration.
    """
    symbols: Optional[List[str]] = Field(
        default_factory=list,
        description="List of trading symbols",
        max_length=50
    )

    update_interval: Optional[int] = Field(
        default=60,
        ge=1,
        le=3600,
        description="Update interval in seconds"
    )

    max_position_size: Optional[float] = Field(
        default=0.2,
        ge=0.001,
        le=1.0,
        description="Maximum position size as fraction of portfolio"
    )

    risk_level: Optional[RiskLevel] = Field(
        default=RiskLevel.LOW,
        description="Risk level setting"
    )

    trading_mode: Optional[TradingMode] = Field(
        default=TradingMode.PAPER,
        description="Trading mode (PAPER or LIVE)"
    )

    grid_spacing: Optional[float] = Field(
        default=0.005,
        ge=0.0001,
        le=0.1,
        description="Grid spacing for EVE mode"
    )

    atr_mult_sl: Optional[float] = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="ATR multiplier for stop loss"
    )

    atr_mult_tp: Optional[float] = Field(
        default=4.0,
        ge=0.5,
        le=20.0,
        description="ATR multiplier for take profit"
    )

    volatility_threshold: Optional[float] = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Volatility threshold for mode switching"
    )

    trend_strength_threshold: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Trend strength threshold"
    )

    reversal_confirmation_bars: Optional[int] = Field(
        default=3,
        ge=1,
        le=20,
        description="Confirmation bars for reversal signals"
    )

    breakout_volume_mult: Optional[float] = Field(
        default=1.5,
        ge=1.0,
        le=5.0,
        description="Volume multiplier for breakout confirmation"
    )

    max_daily_loss: Optional[float] = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Maximum daily loss as fraction of capital"
    )

    max_drawdown: Optional[float] = Field(
        default=0.15,
        ge=0.001,
        le=0.5,
        description="Maximum drawdown as fraction of capital"
    )

    emergency_close_positions: Optional[bool] = Field(
        default=False,
        description="Whether to close positions on emergency stop"
    )

    @field_validator('symbols', mode='before')
    @classmethod
    def validate_symbols(cls, v):
        """Validate stock symbols."""
        if not isinstance(v, list):
            return []
        validated = []
        for symbol in v:
            if isinstance(symbol, str):
                symbol = symbol.upper().strip()
                # Stock symbol pattern: 1-5 uppercase letters
                if re.match(r'^[A-Z]{1,5}$', symbol):
                    validated.append(symbol)
        return validated

    @model_validator(mode='after')
    def validate_atr_multipliers(self):
        """Ensure ATR multipliers are logically consistent."""
        if self.atr_mult_tp and self.atr_mult_sl:
            if self.atr_mult_tp < self.atr_mult_sl:
                raise ValueError('Take profit ATR multiplier must be >= stop loss ATR multiplier')
        return self


# =============================================================================
# TRADING SIGNAL VALIDATORS
# =============================================================================

class TradingSignalSchema(BaseModel):
    """
    Validates and types trading signals/orders.
    """
    symbol: str = Field(
        description="Stock symbol (1-5 uppercase letters)"
    )

    side: OrderSide = Field(
        description="Order side"
    )

    order_type: OrderType = Field(
        default=OrderType.MARKET,
        description="Order type"
    )

    quantity: float = Field(
        gt=0,
        le=1000000,
        description="Order quantity"
    )

    price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Limit price (required for limit orders)"
    )

    stop_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Stop price (required for stop orders)"
    )

    stop_loss: Optional[float] = Field(
        default=None,
        gt=0,
        description="Stop loss price"
    )

    take_profit: Optional[float] = Field(
        default=None,
        gt=0,
        description="Take profit price"
    )

    time_in_force: Optional[str] = Field(
        default='day',
        description="Time in force (day, gtc, opg, cls, etc.)"
    )

    trail_trigger: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Trailing stop trigger as profit fraction"
    )

    trail_offset: Optional[float] = Field(
        default=None,
        ge=0,
        le=0.1,
        description="Trailing stop offset fraction"
    )

    @field_validator('symbol', mode='before')
    @classmethod
    def validate_symbol(cls, v):
        """Validate stock symbol format."""
        if not isinstance(v, str):
            raise ValueError('Symbol must be a string')
        symbol = v.upper().strip()
        if not re.match(r'^[A-Z]{1,5}$', symbol):
            raise ValueError('Symbol must be 1-5 uppercase letters')
        return symbol

    @field_validator('quantity', mode='before')
    @classmethod
    def validate_quantity(cls, v):
        """Validate quantity is positive and reasonable."""
        try:
            qty = float(v)
        except (ValueError, TypeError):
            raise ValueError('Quantity must be a number')
        if qty <= 0:
            raise ValueError('Quantity must be positive')
        if qty > 1000000:
            raise ValueError('Quantity exceeds maximum allowed (1,000,000)')
        return qty

    @model_validator(mode='after')
    def validate_order_requirements(self):
        """Ensure orders have required fields."""
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError('Limit price required for limit orders')
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError('Stop price required for stop orders')
        if self.order_type == OrderType.STOP_LIMIT:
            if self.price is None:
                raise ValueError('Limit price required for stop limit orders')
            if self.stop_price is None:
                raise ValueError('Stop price required for stop limit orders')
        return self


# =============================================================================
# EMERGENCY STOP VALIDATOR
# =============================================================================

class EmergencyStopSchema(BaseModel):
    """Validates emergency stop requests."""
    reason: Optional[str] = Field(
        default="Emergency stop triggered via API",
        max_length=500,
        description="Reason for emergency stop"
    )

    close_all_positions: bool = Field(
        default=True,
        description="Whether to close all positions"
    )

    notify: bool = Field(
        default=True,
        description="Whether to send notifications"
    )

    @field_validator('reason', mode='before')
    @classmethod
    def validate_reason(cls, v):
        """Sanitize reason string."""
        if not isinstance(v, str):
            return "Emergency stop triggered via API"
        # Remove potentially dangerous characters
        v = re.sub(r'[<>\"\'\\]', '', v)
        return v.strip()[:500]


# =============================================================================
# USER AUTHENTICATION VALIDATORS
# =============================================================================

class LoginSchema(BaseModel):
    """Validates login requests."""
    username: str = Field(
        min_length=3,
        max_length=50,
        description="Username or email"
    )

    password: str = Field(
        min_length=8,
        max_length=128,
        description="Password"
    )

    remember_me: bool = Field(
        default=False,
        description="Extend token expiration"
    )

    @field_validator('username', mode='before')
    @classmethod
    def sanitize_username(cls, v):
        """Sanitize username input."""
        if not isinstance(v, str):
            raise ValueError('Username must be a string')
        return v.strip().lower()


class RegisterSchema(BaseModel):
    """Validates registration requests."""
    username: str = Field(
        min_length=3,
        max_length=50,
        description="Desired username"
    )

    email: str = Field(
        description="Email address"
    )

    password: str = Field(
        min_length=12,
        max_length=128,
        description="Password"
    )

    password_confirm: str = Field(
        description="Password confirmation"
    )

    @field_validator('username', mode='before')
    @classmethod
    def sanitize_username(cls, v):
        """Sanitize username."""
        if not isinstance(v, str):
            raise ValueError('Username must be a string')
        v = v.strip()
        # Only allow alphanumeric and underscore
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()

    @field_validator('email', mode='before')
    @classmethod
    def validate_email(cls, v):
        """Validate email format."""
        if not isinstance(v, str):
            raise ValueError('Email must be a string')
        v = v.strip().lower()
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v

    @field_validator('password', mode='before')
    @classmethod
    def validate_password_strength(cls, v):
        """Validate password strength."""
        if not isinstance(v, str):
            raise ValueError('Password must be a string')
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v

    @model_validator(mode='after')
    def validate_passwords_match(self):
        """Ensure passwords match."""
        if self.password != self.password_confirm:
            raise ValueError('Passwords do not match')
        return self


# =============================================================================
# VALIDATION HELPER FUNCTIONS
# =============================================================================

def validate_config_input(data: Dict) -> tuple[bool, Optional[ConfigurationSchema], List[Dict]]:
    """
    Validate configuration input.

    Args:
        data: Configuration dictionary

    Returns:
        Tuple of (is_valid, validated_config, errors)
    """
    try:
        config = ConfigurationSchema(**data)
        return True, config, []
    except ValidationError as e:
        errors = [{
            'field': '.'.join(str(p) for p in e['loc']),
            'message': e['msg'],
            'type': e['type']
        } for e in e.errors()]
        return False, None, errors


def validate_trading_signal(data: Dict) -> tuple[bool, Optional[TradingSignalSchema], List[Dict]]:
    """
    Validate trading signal input.

    Args:
        data: Trading signal dictionary

    Returns:
        Tuple of (is_valid, validated_signal, errors)
    """
    try:
        signal = TradingSignalSchema(**data)
        return True, signal, []
    except ValidationError as e:
        errors = [{
            'field': '.'.join(str(p) for p in err['loc']),
            'message': err['msg'],
            'type': err['type']
        } for err in e.errors()]
        return False, None, errors


def validate_emergency_stop(data: Dict) -> tuple[bool, Optional[EmergencyStopSchema], List[Dict]]:
    """
    Validate emergency stop request.

    Args:
        data: Emergency stop data dictionary

    Returns:
        Tuple of (is_valid, validated_request, errors)
    """
    try:
        request = EmergencyStopSchema(**data)
        return True, request, []
    except ValidationError as e:
        errors = [{
            'field': '.'.join(str(p) for p in err['loc']),
            'message': err['msg'],
            'type': err['type']
        } for err in e.errors()]
        return False, None, errors


def validate_login(data: Dict) -> tuple[bool, Optional[LoginSchema], List[Dict]]:
    """
    Validate login request.

    Args:
        data: Login data dictionary

    Returns:
        Tuple of (is_valid, validated_request, errors)
    """
    try:
        request = LoginSchema(**data)
        return True, request, []
    except ValidationError as e:
        errors = [{
            'field': '.'.join(str(p) for p in err['loc']),
            'message': err['msg'],
            'type': err['type']
        } for err in e.errors()]
        return False, None, errors


def validate_registration(data: Dict) -> tuple[bool, Optional[RegisterSchema], List[Dict]]:
    """
    Validate registration request.

    Args:
        data: Registration data dictionary

    Returns:
        Tuple of (is_valid, validated_request, errors)
    """
    try:
        request = RegisterSchema(**data)
        return True, request, []
    except ValidationError as e:
        errors = [{
            'field': '.'.join(str(p) for p in err['loc']),
            'message': err['msg'],
            'type': err['type']
        } for err in e.errors()]
        return False, None, errors
