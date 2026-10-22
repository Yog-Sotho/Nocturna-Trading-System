"""
NOCTURNA Trading System - Input validators.

Pydantic schemas for request bodies. Each ``validate_*`` helper returns
(is_valid, model or None, errors) so routes can answer 400 with field-level
messages.
"""

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# =============================================================================
# MANUAL ORDER VALIDATOR
# =============================================================================

class ManualOrderSchema(BaseModel):
    """An operator's order. It goes through the same risk gate as strategy orders."""
    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.]{1,10}(/[A-Z]{2,5})?$")
    side: Literal["buy", "sell"]
    qty: Decimal = Field(gt=0, le=Decimal("1e9"), max_digits=28, decimal_places=9)
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "market"
    # None: GTC for crypto, DAY for equities
    time_in_force: Literal["day", "gtc", "ioc", "opg", "cls"] | None = None
    limit_price: Decimal | None = Field(default=None, gt=0, max_digits=28, decimal_places=9)
    stop_price: Decimal | None = Field(default=None, gt=0, max_digits=28, decimal_places=9)

    @field_validator('symbol', mode='before')
    @classmethod
    def normalise_symbol(cls, v):
        if not isinstance(v, str):
            raise ValueError('symbol must be a string')
        return v.strip().upper()

    @model_validator(mode='after')
    def prices_match_type(self):
        needs_limit = self.order_type in ("limit", "stop_limit")
        needs_stop = self.order_type in ("stop", "stop_limit")
        if needs_limit != (self.limit_price is not None):
            raise ValueError(f"{self.order_type} orders {'need' if needs_limit else 'take no'} limit_price")
        if needs_stop != (self.stop_price is not None):
            raise ValueError(f"{self.order_type} orders {'need' if needs_stop else 'take no'} stop_price")
        return self


# =============================================================================
# EMERGENCY STOP VALIDATOR
# =============================================================================

class EmergencyStopSchema(BaseModel):
    """Validates emergency stop requests."""
    reason: str | None = Field(
        default="Emergency stop triggered via API",
        max_length=500,
        description="Reason for emergency stop"
    )

    close_all_positions: bool = Field(
        default=True,
        description="Whether to close all positions"
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
# VALIDATION HELPERS
# =============================================================================

def _validate(schema: type[BaseModel], data: dict) -> tuple[bool, BaseModel | None, list[dict]]:
    try:
        return True, schema(**data), []
    except ValidationError as e:
        errors = [{
            'field': '.'.join(str(p) for p in err['loc']),
            'message': err['msg'],
            'type': err['type']
        } for err in e.errors()]
        return False, None, errors


def validate_manual_order(data: dict) -> tuple[bool, ManualOrderSchema | None, list[dict]]:
    return _validate(ManualOrderSchema, data)


def validate_emergency_stop(data: dict) -> tuple[bool, EmergencyStopSchema | None, list[dict]]:
    return _validate(EmergencyStopSchema, data)


def validate_login(data: dict) -> tuple[bool, LoginSchema | None, list[dict]]:
    return _validate(LoginSchema, data)


def validate_registration(data: dict) -> tuple[bool, RegisterSchema | None, list[dict]]:
    return _validate(RegisterSchema, data)
