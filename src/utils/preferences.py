"""
NOCTURNA Trading System - Each user's dashboard preferences.

How the dashboard looks to one person: the name it greets them by, light or
dark, the accent colour, and which overview cards it shows in which order.
They are kept in the user's ``settings`` (``users.settings["dashboard"]``)
and change nothing about trading.
"""

from __future__ import annotations

from typing import Any

KEY = "dashboard"
MAX_NAME = 40
THEMES = ("auto", "dark", "light")
ACCENTS = ("violet", "indigo", "magenta", "teal", "amber")
#: The overview's cards, in their default order
CARDS = ("account", "equity", "mix", "positions", "risk", "orders", "fills")
#: Night by default; "auto" follows the device
DEFAULTS: dict[str, Any] = {"display_name": "", "theme": "dark", "accent": "violet", "cards": list(CARDS),
                            "hidden": []}


class PreferencesError(ValueError):
    """The preferences cannot be saved; the message says why."""


def of(user: Any) -> dict[str, Any]:
    """The user's preferences, defaults filled in; anything stored that is no longer valid is dropped."""
    stored = (user.settings or {}).get(KEY)
    try:
        return parse(stored, DEFAULTS) if isinstance(stored, dict) else dict(DEFAULTS)
    except PreferencesError:
        return dict(DEFAULTS)


def _cards(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(c, str) for c in value):
        raise PreferencesError(f"{what} must be a list of card names")
    unknown = sorted(set(value) - set(CARDS))
    if unknown:
        raise PreferencesError(f"unknown cards: {', '.join(unknown)}; known: {', '.join(CARDS)}")
    if len(set(value)) != len(value):
        raise PreferencesError(f"{what} names a card twice")
    return list(value)


def parse(data: Any, current: dict[str, Any]) -> dict[str, Any]:
    """
    ``current`` with the fields ``data`` changes.

    Raises:
        PreferencesError: For an unknown field or a value out of range.
    """
    if not isinstance(data, dict):
        raise PreferencesError("the body must be a JSON object")
    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        raise PreferencesError(f"unknown fields: {', '.join(unknown)}")
    data = {k: v for k, v in data.items() if v is not None}  # null: unchanged
    out = {**DEFAULTS, **current}
    if "display_name" in data:
        name = data["display_name"]
        if not isinstance(name, str) or len(name.strip()) > MAX_NAME or any(not c.isprintable() for c in name):
            raise PreferencesError(f"display_name must be printable text of at most {MAX_NAME} characters")
        out["display_name"] = name.strip()
    for field, choices in (("theme", THEMES), ("accent", ACCENTS)):
        if field in data:
            if data[field] not in choices:
                raise PreferencesError(f"{field} must be one of: {', '.join(choices)}")
            out[field] = data[field]
    if "cards" in data:
        order = _cards(data["cards"], "cards")
        out["cards"] = order + [c for c in CARDS if c not in order]  # a card added later goes last
    if "hidden" in data:
        out["hidden"] = _cards(data["hidden"], "hidden")
    return out


def save(user: Any, preferences: dict[str, Any]) -> None:
    """Store them on the user (the caller commits); a new dict, so the JSON column sees the change."""
    user.settings = {**(user.settings or {}), KEY: preferences}
