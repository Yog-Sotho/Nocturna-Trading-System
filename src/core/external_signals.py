# FILE LOCATION: src/core/external_signals.py
"""
NOCTURNA Trading System - External Signal Aggregator (F16)
Integrates third-party signal sources for supplementary confirmation.

Supported sources:
  1. Alternative.me Crypto Fear & Greed Index (free, no key)
  2. Alpaca News API (uses existing broker credentials)
  3. x70.ai Trading Signals (requires separate API key — optional)

Architecture: Each source provides a normalized score [-1.0, +1.0] with confidence.
The aggregator blends all available sources into a composite external signal.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class ExternalSignal:
    """Normalized external signal."""
    source: str
    score: float          # -1.0 (extreme bearish) to +1.0 (extreme bullish)
    confidence: float     # 0.0 to 1.0
    raw_value: float      # Original value from source
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)


class ExternalSignalAggregator:
    """
    Aggregates external signal sources into a unified signal for strategy filtering.
    Thread-safe: designed to be polled periodically from the trading engine.
    """

    # Cache TTL per source (seconds)
    CACHE_TTL = {
        'fear_greed': 900,      # 15 minutes (index updates every 15-60 min)
        'alpaca_news': 300,     # 5 minutes
        'x70_signals': 600,     # 10 minutes
    }

    # Source weights for composite score
    SOURCE_WEIGHTS = {
        'fear_greed': 0.3,     # Contrarian — strong macro filter
        'alpaca_news': 0.3,    # News sentiment — real-time
        'x70_signals': 0.4,    # AI signals — highest signal quality
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        self._cache: dict[str, ExternalSignal] = {}
        self._lock = threading.Lock()

        # Source availability
        self._sources_enabled = {
            'fear_greed': self.config.get('enable_fear_greed', True),
            'alpaca_news': self.config.get('enable_alpaca_news', True),
            'x70_signals': self.config.get('enable_x70', False),  # Requires API key
        }

        self.logger.info(
            f"External Signal Aggregator initialized. Sources: "
            f"{[k for k, v in self._sources_enabled.items() if v]}"
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_composite_signal(self, symbol: str = 'BTC') -> dict:
        """
        Get blended external signal from all available sources.

        Returns:
            {
                'score': float (-1 to 1),
                'confidence': float (0 to 1),
                'sources': {source_name: signal_dict},
                'timestamp': str
            }
        """
        signals = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for source, enabled in self._sources_enabled.items():
            if not enabled:
                continue

            sig = self._get_cached_or_fetch(source, symbol)
            if sig is None:
                continue

            signals[source] = {
                'score': sig.score,
                'confidence': sig.confidence,
                'raw_value': sig.raw_value,
                'age_seconds': (datetime.now(UTC) - sig.timestamp).total_seconds(),
            }

            weight = self.SOURCE_WEIGHTS.get(source, 0.25) * sig.confidence
            weighted_sum += sig.score * weight
            total_weight += weight

        composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        composite_conf = min(total_weight / sum(self.SOURCE_WEIGHTS.values()), 1.0)

        return {
            'score': composite_score,
            'confidence': composite_conf,
            'sources': signals,
            'source_count': len(signals),
            'timestamp': datetime.now(UTC).isoformat(),
        }

    def get_fear_greed_signal(self) -> ExternalSignal | None:
        """Get Fear & Greed Index signal directly."""
        return self._get_cached_or_fetch('fear_greed')

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def _get_cached_or_fetch(self, source: str, symbol: str = 'BTC') -> ExternalSignal | None:
        """Return cached signal if fresh, otherwise fetch."""
        cache_key = f"{source}_{symbol}"
        ttl = self.CACHE_TTL.get(source, 600)

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                age = (datetime.now(UTC) - cached.timestamp).total_seconds()
                if age < ttl:
                    return cached

        # Fetch fresh data
        sig = None
        try:
            if source == 'fear_greed':
                sig = self._fetch_fear_greed()
            elif source == 'alpaca_news':
                sig = self._fetch_alpaca_news_sentiment(symbol)
            elif source == 'x70_signals':
                sig = self._fetch_x70_signal(symbol)
        except Exception as e:
            self.logger.warning(f"External signal fetch failed ({source}): {e}")

        if sig:
            with self._lock:
                self._cache[cache_key] = sig

        return sig

    # =========================================================================
    # SOURCE: Alternative.me Fear & Greed Index
    # =========================================================================

    def _fetch_fear_greed(self) -> ExternalSignal | None:
        """
        Fetch Crypto Fear & Greed Index from Alternative.me.
        Free API, no key required. Returns 0-100 index.
        0 = Extreme Fear (contrarian bullish), 100 = Extreme Greed (contrarian bearish).
        """
        try:
            import json
            import urllib.request

            url = 'https://api.alternative.me/fng/?limit=1'
            req = urllib.request.Request(url, headers={'User-Agent': 'NOCTURNA/2.0'})

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            if 'data' not in data or not data['data']:
                return None

            fng_value = int(data['data'][0]['value'])  # 0-100
            classification = data['data'][0].get('value_classification', '')

            # Convert to contrarian signal: Fear = bullish opportunity, Greed = bearish warning
            # 0-25: Extreme Fear → score +0.6 to +1.0 (strong buy zone)
            # 25-45: Fear → score +0.1 to +0.5
            # 45-55: Neutral → score -0.1 to +0.1
            # 55-75: Greed → score -0.1 to -0.5
            # 75-100: Extreme Greed → score -0.6 to -1.0 (strong sell zone)
            contrarian_score = (50 - fng_value) / 50.0  # Linear: 0→+1.0, 100→-1.0

            # Confidence is higher at extremes
            distance_from_center = abs(fng_value - 50) / 50.0
            confidence = 0.4 + (distance_from_center * 0.6)  # 0.4 at center, 1.0 at extremes

            return ExternalSignal(
                source='fear_greed',
                score=contrarian_score,
                confidence=confidence,
                raw_value=fng_value,
                metadata={'classification': classification},
            )

        except Exception as e:
            self.logger.debug(f"Fear & Greed fetch error: {e}")
            return None

    # =========================================================================
    # SOURCE: Alpaca News Sentiment
    # =========================================================================

    def _fetch_alpaca_news_sentiment(self, symbol: str) -> ExternalSignal | None:
        """
        Fetch recent news for a symbol via Alpaca and run basic sentiment scoring.
        Uses VADER if available, otherwise simple keyword count.
        """
        try:
            import os

            alpaca_key = os.environ.get('ALPACA_API_KEY')
            alpaca_secret = os.environ.get('ALPACA_SECRET_KEY')
            if not alpaca_key or not alpaca_secret:
                return None

            import json
            import urllib.request

            # Alpaca news endpoint
            news_url = f"https://data.alpaca.markets/v1beta1/news?symbols={symbol}&limit=10&sort=desc"

            req = urllib.request.Request(news_url, headers={
                'APCA-API-KEY-ID': alpaca_key,
                'APCA-API-SECRET-KEY': alpaca_secret,
            })

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            news_items = data.get('news', [])
            if not news_items:
                return None

            # Score headlines
            scores = []
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                vader = SentimentIntensityAnalyzer()
                for item in news_items:
                    headline = item.get('headline', '')
                    if headline:
                        scores.append(vader.polarity_scores(headline)['compound'])
            except ImportError:
                # Fallback: simple keyword scoring
                bullish = {'surge', 'rally', 'bullish', 'gain', 'rise', 'beat', 'upgrade', 'strong'}
                bearish = {'crash', 'plunge', 'bearish', 'decline', 'loss', 'miss', 'downgrade', 'weak'}
                for item in news_items:
                    words = set(item.get('headline', '').lower().split())
                    bull = len(words & bullish)
                    bear = len(words & bearish)
                    if bull + bear > 0:
                        scores.append((bull - bear) / (bull + bear))

            if not scores:
                return None

            import numpy as np
            avg_score = float(np.mean(scores))
            confidence = min(len(scores) / 10.0, 1.0)  # More news = higher confidence

            return ExternalSignal(
                source='alpaca_news',
                score=avg_score,
                confidence=confidence,
                raw_value=avg_score,
                metadata={'article_count': len(scores)},
            )

        except Exception as e:
            self.logger.debug(f"Alpaca news sentiment error: {e}")
            return None

    # =========================================================================
    # SOURCE: x70.ai Trading Signals (optional)
    # =========================================================================

    def _fetch_x70_signal(self, symbol: str) -> ExternalSignal | None:
        """
        Fetch AI trading signals from x70.ai.
        Requires API key set in config or env var X70_API_KEY.
        Returns directional signal with confidence.
        """
        try:
            import json
            import os
            import urllib.request

            api_key = self.config.get('x70_api_key') or os.environ.get('X70_API_KEY')
            if not api_key:
                return None

            url = f"https://signals.x70.ai/api/skill/signals?status=active&coin={symbol}"
            req = urllib.request.Request(url, headers={
                'Authorization': f'Bearer {api_key}',
                'User-Agent': 'NOCTURNA/2.0',
            })

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            signals = data.get('data', [])
            if not signals:
                return None

            # Take highest-confidence signal for this symbol
            best = max(signals, key=lambda s: s.get('confidence', 0))
            direction = best.get('direction', '').lower()
            conf = best.get('confidence', 50) / 100.0

            # Convert to normalized score
            if direction in ('bull', 'bullish', 'long'):
                score = conf
            elif direction in ('bear', 'bearish', 'short'):
                score = -conf
            else:
                score = 0.0

            return ExternalSignal(
                source='x70_signals',
                score=score,
                confidence=conf,
                raw_value=best.get('confidence', 0),
                metadata={
                    'direction': direction,
                    'entry': best.get('entry_price'),
                    'sl': best.get('stop_loss'),
                    'tp': best.get('take_profit'),
                },
            )

        except Exception as e:
            self.logger.debug(f"x70.ai signal fetch error: {e}")
            return None

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> dict:
        """Get aggregator status."""
        with self._lock:
            cache_status = {}
            for key, sig in self._cache.items():
                age = (datetime.now(UTC) - sig.timestamp).total_seconds()
                cache_status[key] = {
                    'score': sig.score,
                    'confidence': sig.confidence,
                    'age_seconds': age,
                    'fresh': age < self.CACHE_TTL.get(sig.source, 600),
                }

        return {
            'sources_enabled': self._sources_enabled,
            'cache': cache_status,
            'timestamp': datetime.now(UTC).isoformat(),
        }
