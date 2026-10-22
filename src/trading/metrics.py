"""
NOCTURNA Trading System - Prometheus metrics.

``render`` turns the engine's views into the Prometheus text exposition format
(version 0.0.4), served at GET /metrics. Every metric is a gauge: the ledger's
totals start again from the broker account when the engine restarts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
STATES = ("stopped", "starting", "running", "failed", "unreachable")

Sample = tuple[Mapping[str, str], float]


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _number(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


class _Exposition:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def gauge(self, name: str, help_text: str, samples: Iterable[Sample]) -> None:
        samples = list(samples)
        if not samples:
            return
        self.lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
        for labels, value in samples:
            text = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
            self.lines.append(f"{name}{{{text}}} {_number(value)}" if text else f"{name} {_number(value)}")

    def value(self, name: str, help_text: str, value: Any) -> None:
        if value is not None:
            self.gauge(name, help_text, [({}, float(value))])


def render(views: Mapping[str, Any]) -> str:
    status = views["status"]
    out = _Exposition()
    out.gauge("nocturna_engine_state", "1 for the engine's current state",
              [({"state": s}, float(status["state"] == s)) for s in STATES])
    out.value("nocturna_engine_configured", "1 when Alpaca credentials are set", status["configured"])
    out.value("nocturna_engine_halted", "1 while trading is halted", status.get("halted"))
    out.value("nocturna_equity_usd", "Account equity", status.get("equity"))
    out.value("nocturna_cash_usd", "Cash", status.get("cash"))
    out.value("nocturna_realized_pnl_usd", "Realized profit and loss since the engine started",
              status.get("realized_pnl"))
    out.value("nocturna_fees_usd", "Fees paid since the engine started", status.get("fees"))
    out.value("nocturna_open_orders", "Working orders", status.get("open_orders"))
    out.value("nocturna_positions", "Open positions", status.get("positions"))
    if status.get("last_poll"):
        out.value("nocturna_last_poll_timestamp_seconds", "When market data was last polled",
                  datetime.fromisoformat(status["last_poll"]).timestamp())
    risk = status.get("risk") or {}
    out.value("nocturna_cooling_down", "1 after the daily loss limit, until the next trading day",
              risk.get("cooling_down"))
    out.value("nocturna_drawdown_ratio", "Drawdown from the high-water mark", risk.get("drawdown"))
    out.value("nocturna_orders_today", "Orders sent today", risk.get("orders_today"))
    out.value("nocturna_volatility_scale_ratio", "Exposure scale from volatility targeting",
              risk.get("volatility_scale"))
    out.value("nocturna_drawdown_scale_ratio", "Exposure scale from the drawdown ladder", risk.get("drawdown_scale"))
    report = status.get("last_reconciliation")
    if report is not None:
        out.value("nocturna_reconciliation_in_sync", "1 when the last reconciliation found nothing to correct",
                  report["in_sync"])
    positions = (views.get("positions") or {}).get("positions", [])
    out.gauge("nocturna_position_market_value_usd", "Market value of each position",
              [({"symbol": p["symbol"]}, float(p["market_value"])) for p in positions])
    out.gauge("nocturna_position_unrealized_pnl_usd", "Unrealized profit and loss of each position",
              [({"symbol": p["symbol"]}, float(p["unrealized_pnl"])) for p in positions])
    targets = views.get("targets") or {}
    out.gauge("nocturna_target_weight_ratio", "The engine's target weight for each symbol",
              [({"symbol": s}, float(w)) for s, w in targets.items() if w is not None])
    return "\n".join(out.lines) + "\n"
