"""
Sentiment Analysis Engine for NOCTURNA v2.0
Analyzes market sentiment from news, social media, and other textual data.
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

import numpy as np


class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

@dataclass
class SentimentData:
    """Represents sentiment data."""
    source: str
    symbol: str
    text: str
    score: float
    confidence: float
    timestamp: datetime
    metadata: dict | None = None

class SentimentAnalyzer:
    """
    Advanced sentiment analysis system for trading.
    Combines multiple sources and analysis techniques.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Primary scorer: VADER (if available)
        self.vader_analyzer = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("VADER sentiment analyzer loaded as primary scorer")
        except ImportError:
            self.logger.warning("vaderSentiment not installed — using word-list fallback only")

        # Supplemental word-list dictionaries
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.financial_terms = self._load_financial_terms()

        # Source weights for aggregation
        self.source_weights = {
            'news': 1.0,
            'twitter': 0.7,
            'reddit': 0.6,
            'analyst_reports': 1.2,
            'earnings_calls': 1.1
        }

        # Cache and history
        self.sentiment_cache = {}
        self.sentiment_history = []

        self.logger.info("Sentiment Analyzer initialized")

    def _load_positive_words(self) -> set:
        """Carica dizionario di parole positive."""
        positive_words = {
            'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'increase',
            'strong', 'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'rally',
            'surge', 'boom', 'breakthrough', 'success', 'excellent', 'outstanding',
            'robust', 'solid', 'impressive', 'momentum', 'optimistic', 'confident',
            'expansion', 'recovery', 'uptrend', 'breakout', 'acceleration'
        }
        return positive_words

    def _load_negative_words(self) -> set:
        """Load negative sentiment words (financial context)."""
        negative_words = {
            'bearish', 'negative', 'decline', 'loss', 'fall', 'decrease',
            'weak', 'miss', 'underperform', 'downgrade', 'sell', 'crash',
            'plunge', 'collapse', 'concern', 'worry', 'risk', 'threat',
            'disappointing', 'poor', 'terrible', 'awful', 'pessimistic',
            'recession', 'crisis', 'volatility', 'uncertainty', 'correction',
            'breakdown'
        }
        return negative_words

    def _load_financial_terms(self) -> dict[str, float]:
        """Carica termini finanziari con pesi."""
        financial_terms = {
            'earnings': 1.5,
            'revenue': 1.3,
            'profit': 1.4,
            'margin': 1.2,
            'guidance': 1.6,
            'forecast': 1.3,
            'outlook': 1.4,
            'target': 1.2,
            'estimate': 1.1,
            'consensus': 1.3,
            'analyst': 1.2,
            'rating': 1.4,
            'recommendation': 1.5
        }
        return financial_terms

    def analyze_text(self, text: str, symbol: str = None) -> dict:
        """
        Analyze text sentiment using VADER (primary) + financial word-list (supplemental).

        Args:
            text: Text to analyze
            symbol: Stock symbol (optional)

        Returns:
            Sentiment analysis results
        """
        try:
            # Preprocessing
            cleaned_text = self._preprocess_text(text)
            words = cleaned_text.split()

            if not words:
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'classification': SentimentScore.NEUTRAL.name,
                    'word_count': 0
                }

            # VADER score (primary — if available)
            vader_score = 0.0
            vader_available = False
            if self.vader_analyzer:
                vader_result = self.vader_analyzer.polarity_scores(text)
                vader_score = vader_result['compound']  # Already -1 to +1
                vader_available = True

            # Word-list score (supplemental)
            positive_score = 0
            negative_score = 0
            total_weight = 0

            for word in words:
                word_lower = word.lower()
                weight = 1.0

                # Apply financial term weight
                if word_lower in self.financial_terms:
                    weight = self.financial_terms[word_lower]

                if word_lower in self.positive_words:
                    positive_score += weight
                    total_weight += weight
                elif word_lower in self.negative_words:
                    negative_score += weight
                    total_weight += weight

            if total_weight > 0:
                wordlist_score = (positive_score - negative_score) / total_weight
            else:
                wordlist_score = 0.0

            # Apply contextual modifiers to word-list score
            wordlist_score = self._apply_contextual_modifiers(cleaned_text, wordlist_score)
            wordlist_score = np.tanh(wordlist_score)

            # Blend: 70% VADER + 30% word-list (if VADER available)
            if vader_available:
                final_score = vader_score * 0.7 + wordlist_score * 0.3
            else:
                final_score = wordlist_score

            # Confidence based on word coverage
            confidence = min(total_weight / len(words), 1.0)
            if vader_available:
                confidence = min(confidence + 0.3, 1.0)  # Boost confidence with VADER

            # Classify
            classification = self._classify_sentiment(final_score)

            result = {
                'score': final_score,
                'confidence': confidence,
                'classification': classification.name,
                'word_count': len(words),
                'positive_words': positive_score,
                'negative_words': negative_score,
                'financial_relevance': total_weight / len(words) if words else 0.0,
                'scorer': 'vader+wordlist' if vader_available else 'wordlist'
            }

            return result

        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'classification': SentimentScore.NEUTRAL.name,
                'error': str(e)
            }

    def _preprocess_text(self, text: str) -> str:
        """Preprocessing del testo."""
        # Converti in lowercase
        text = text.lower()

        # Rimuovi URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Rimuovi menzioni e hashtag (ma mantieni il contenuto)
        text = re.sub(r'[@#]', '', text)

        # Rimuovi punteggiatura eccessiva
        text = re.sub(r'[^\w\s]', ' ', text)

        # Rimuovi spazi multipli
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _apply_contextual_modifiers(self, text: str, base_score: float) -> float:
        """Apply contextual modifiers to sentiment score. Each modifier type applied once."""
        modified_score = base_score

        # Negation — count occurrences, apply once (partial inversion)
        negation_patterns = [
            r'not\s+\w+', r'no\s+\w+', r'never\s+\w+',
            r'nothing\s+\w+', r'none\s+\w+'
        ]
        has_negation = any(re.search(p, text) for p in negation_patterns)
        if has_negation:
            modified_score *= -0.5  # Apply once regardless of how many negations

        # Intensifiers — detect presence, apply single max boost
        intensifiers = ['very', 'extremely', 'highly', 'significantly', 'substantially']
        has_intensifier = any(word in text for word in intensifiers)
        if has_intensifier:
            modified_score *= 1.3  # Single application

        # Diminishers — detect presence, apply single reduction
        diminishers = ['slightly', 'somewhat', 'barely', 'hardly', 'scarcely']
        has_diminisher = any(word in text for word in diminishers)
        if has_diminisher:
            modified_score *= 0.7  # Single application

        # Uncertainty — detect presence, apply single dampening
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could']
        has_uncertainty = any(word in text for word in uncertainty_words)
        if has_uncertainty:
            modified_score *= 0.8  # Single application

        return modified_score

    def _classify_sentiment(self, score: float) -> SentimentScore:
        """Classify the sentiment score."""
        if score >= 0.6:
            return SentimentScore.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentScore.POSITIVE
        elif score <= -0.6:
            return SentimentScore.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentScore.NEGATIVE
        else:
            return SentimentScore.NEUTRAL

    def analyze_news_batch(self, news_articles: list[dict]) -> dict:
        """
        Analyze a batch of news articles.

        Args:
            news_articles: Lista di articoli con 'title', 'content', 'symbol', 'timestamp'

        Returns:
            Analisi aggregata del sentiment
        """
        try:
            if not news_articles:
                return {'error': 'No articles provided'}

            symbol_sentiments = {}

            for article in news_articles:
                symbol = article.get('symbol', 'UNKNOWN')
                title = article.get('title', '')
                content = article.get('content', '')
                timestamp = article.get('timestamp', datetime.now(UTC))

                # Combine title and content
                full_text = f"{title} {content}"

                # Analyze sentiment
                sentiment_result = self.analyze_text(full_text, symbol)

                # Aggrega per simbolo
                if symbol not in symbol_sentiments:
                    symbol_sentiments[symbol] = {
                        'scores': [],
                        'confidences': [],
                        'articles': 0,
                        'latest_timestamp': timestamp
                    }

                symbol_sentiments[symbol]['scores'].append(sentiment_result['score'])
                symbol_sentiments[symbol]['confidences'].append(sentiment_result['confidence'])
                symbol_sentiments[symbol]['articles'] += 1

                if timestamp > symbol_sentiments[symbol]['latest_timestamp']:
                    symbol_sentiments[symbol]['latest_timestamp'] = timestamp

            # Calculate aggregated sentiment by symbol
            aggregated_results = {}

            for symbol, data in symbol_sentiments.items():
                scores = np.array(data['scores'])
                confidences = np.array(data['confidences'])

                # Weighted average
                if len(confidences) > 0 and np.sum(confidences) > 0:
                    weighted_score = np.average(scores, weights=confidences)
                    avg_confidence = np.mean(confidences)
                else:
                    weighted_score = np.mean(scores) if len(scores) > 0 else 0.0
                    avg_confidence = 0.0

                aggregated_results[symbol] = {
                    'sentiment_score': weighted_score,
                    'confidence': avg_confidence,
                    'classification': self._classify_sentiment(weighted_score).name,
                    'article_count': data['articles'],
                    'score_std': np.std(scores) if len(scores) > 1 else 0.0,
                    'latest_timestamp': data['latest_timestamp'].isoformat()
                }

            return {
                'success': True,
                'symbols': aggregated_results,
                'total_articles': len(news_articles),
                'analysis_timestamp': datetime.now(UTC).isoformat()
            }

        except Exception as e:
            self.logger.error(f"News batch analysis error: {e}")
            return {'error': str(e)}

    def get_market_sentiment_signal(self, symbol: str,
                                   lookback_hours: int = 24) -> dict:
        """
        Generate a sentiment signal for the market.

        Args:
            symbol: Simbolo del titolo
            lookback_hours: Ore di lookback per l'analisi

        Returns:
            Segnale di sentiment
        """
        try:
            # Filtra sentiment history per simbolo e timeframe
            cutoff_time = datetime.now(UTC) - timedelta(hours=lookback_hours)

            relevant_sentiments = [
                s for s in self.sentiment_history
                if s.symbol == symbol and s.timestamp >= cutoff_time
            ]

            if not relevant_sentiments:
                return {
                    'signal': 'NEUTRAL',
                    'strength': 0.0,
                    'confidence': 0.0,
                    'data_points': 0
                }

            # Calculate aggregated sentiment with temporal decay
            current_time = datetime.now(UTC)
            weighted_scores = []
            weights = []

            for sentiment in relevant_sentiments:
                # Calculate weight based on age and confidence
                age_hours = (current_time - sentiment.timestamp).total_seconds() / 3600
                time_weight = np.exp(-age_hours / 12)  # Decay con half-life di 12 ore

                total_weight = time_weight * sentiment.confidence

                # Applica peso della fonte
                source_weight = self.source_weights.get(sentiment.source, 1.0)
                total_weight *= source_weight

                weighted_scores.append(sentiment.score * total_weight)
                weights.append(total_weight)

            # Calculate final sentiment
            if sum(weights) > 0:
                final_score = sum(weighted_scores) / sum(weights)
                avg_confidence = np.mean([s.confidence for s in relevant_sentiments])
            else:
                final_score = 0.0
                avg_confidence = 0.0

            # Generate signal
            signal_strength = abs(final_score)

            if final_score > 0.3:
                signal = 'BULLISH'
            elif final_score < -0.3:
                signal = 'BEARISH'
            else:
                signal = 'NEUTRAL'

            return {
                'signal': signal,
                'strength': signal_strength,
                'confidence': avg_confidence,
                'sentiment_score': final_score,
                'data_points': len(relevant_sentiments),
                'timeframe_hours': lookback_hours,
                'classification': self._classify_sentiment(final_score).name
            }

        except Exception as e:
            self.logger.error(f"Sentiment signal error: {e}")
            return {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

    def add_sentiment_data(self, source: str, symbol: str, text: str,
                          metadata: dict = None):
        """
        Aggiunge nuovi dati di sentiment al sistema.

        Args:
            source: Fonte dei dati
            symbol: Simbolo del titolo
            text: Testo da analizzare
            metadata: Metadati aggiuntivi
        """
        try:
            # Analyze sentiment
            sentiment_result = self.analyze_text(text, symbol)

            # Crea oggetto sentiment
            sentiment_data = SentimentData(
                source=source,
                symbol=symbol,
                text=text,
                score=sentiment_result['score'],
                confidence=sentiment_result['confidence'],
                timestamp=datetime.now(UTC),
                metadata=metadata or {}
            )

            # Aggiungi alla history
            self.sentiment_history.append(sentiment_data)

            # Mantieni solo ultimi N giorni
            max_age = timedelta(days=7)
            cutoff_time = datetime.now(UTC) - max_age

            self.sentiment_history = [
                s for s in self.sentiment_history
                if s.timestamp >= cutoff_time
            ]

            self.logger.debug(f"Aggiunto sentiment per {symbol}: {sentiment_result['score']:.3f}")

        except Exception as e:
            self.logger.error(f"Error adding sentiment: {e}")

    def get_sentiment_trends(self, symbol: str, days: int = 7) -> dict:
        """
        Analyze sentiment trends for a symbol.

        Args:
            symbol: Simbolo del titolo
            days: Giorni di storia da analizzare

        Returns:
            Analisi dei trend
        """
        try:
            # Filtra dati per simbolo e timeframe
            cutoff_time = datetime.now(UTC) - timedelta(days=days)

            symbol_sentiments = [
                s for s in self.sentiment_history
                if s.symbol == symbol and s.timestamp >= cutoff_time
            ]

            if len(symbol_sentiments) < 2:
                return {
                    'trend': 'INSUFFICIENT_DATA',
                    'data_points': len(symbol_sentiments)
                }

            # Ordina per timestamp
            symbol_sentiments.sort(key=lambda x: x.timestamp)

            # Calculate trend
            timestamps = [s.timestamp for s in symbol_sentiments]
            scores = [s.score for s in symbol_sentiments]

            # Regressione lineare semplice
            x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
            y = np.array(scores)

            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]

                # Classify trend
                if slope > 0.001:
                    trend = 'IMPROVING'
                elif slope < -0.001:
                    trend = 'DETERIORATING'
                else:
                    trend = 'STABLE'
            else:
                slope = 0.0
                trend = 'STABLE'

            # Calculate sentiment volatility
            sentiment_volatility = np.std(scores) if len(scores) > 1 else 0.0

            # Analyze distribution by source
            source_breakdown = {}
            for sentiment in symbol_sentiments:
                source = sentiment.source
                if source not in source_breakdown:
                    source_breakdown[source] = {
                        'count': 0,
                        'avg_score': 0.0,
                        'scores': []
                    }

                source_breakdown[source]['count'] += 1
                source_breakdown[source]['scores'].append(sentiment.score)

            # Calcola medie per fonte
            for _source, data in source_breakdown.items():
                data['avg_score'] = np.mean(data['scores'])
                del data['scores']  # Rimuovi per ridurre dimensione output

            return {
                'trend': trend,
                'slope': slope,
                'current_sentiment': scores[-1] if scores else 0.0,
                'avg_sentiment': np.mean(scores) if scores else 0.0,
                'sentiment_volatility': sentiment_volatility,
                'data_points': len(symbol_sentiments),
                'timeframe_days': days,
                'source_breakdown': source_breakdown,
                'first_timestamp': timestamps[0].isoformat() if timestamps else None,
                'last_timestamp': timestamps[-1].isoformat() if timestamps else None
            }

        except Exception as e:
            self.logger.error(f"Errore nell'analisi trend: {e}")
            return {'error': str(e)}

    def get_sentiment_summary(self) -> dict:
        """Genera un summary completo del sentiment."""
        try:
            if not self.sentiment_history:
                return {'status': 'No sentiment data available'}

            # Raggruppa per simbolo
            symbol_groups = {}
            for sentiment in self.sentiment_history:
                symbol = sentiment.symbol
                if symbol not in symbol_groups:
                    symbol_groups[symbol] = []
                symbol_groups[symbol].append(sentiment)

            # Analizza ogni simbolo
            symbol_summaries = {}
            for symbol, sentiments in symbol_groups.items():
                recent_sentiments = [
                    s for s in sentiments
                    if s.timestamp >= datetime.now(UTC) - timedelta(hours=24)
                ]

                if recent_sentiments:
                    avg_score = np.mean([s.score for s in recent_sentiments])
                    avg_confidence = np.mean([s.confidence for s in recent_sentiments])

                    symbol_summaries[symbol] = {
                        'sentiment_score': avg_score,
                        'confidence': avg_confidence,
                        'classification': self._classify_sentiment(avg_score).name,
                        'data_points_24h': len(recent_sentiments),
                        'total_data_points': len(sentiments)
                    }

            # Statistiche globali
            all_recent = [
                s for s in self.sentiment_history
                if s.timestamp >= datetime.now(UTC) - timedelta(hours=24)
            ]

            global_stats = {
                'total_symbols': len(symbol_summaries),
                'total_data_points_24h': len(all_recent),
                'total_data_points': len(self.sentiment_history),
                'avg_market_sentiment': np.mean([s.score for s in all_recent]) if all_recent else 0.0,
                'sentiment_distribution': {
                    'very_positive': 0,
                    'positive': 0,
                    'neutral': 0,
                    'negative': 0,
                    'very_negative': 0
                }
            }

            # Calcola distribuzione
            for sentiment in all_recent:
                classification = self._classify_sentiment(sentiment.score)
                if classification == SentimentScore.VERY_POSITIVE:
                    global_stats['sentiment_distribution']['very_positive'] += 1
                elif classification == SentimentScore.POSITIVE:
                    global_stats['sentiment_distribution']['positive'] += 1
                elif classification == SentimentScore.NEUTRAL:
                    global_stats['sentiment_distribution']['neutral'] += 1
                elif classification == SentimentScore.NEGATIVE:
                    global_stats['sentiment_distribution']['negative'] += 1
                elif classification == SentimentScore.VERY_NEGATIVE:
                    global_stats['sentiment_distribution']['very_negative'] += 1

            return {
                'status': 'active',
                'timestamp': datetime.now(UTC).isoformat(),
                'global_stats': global_stats,
                'symbol_summaries': symbol_summaries
            }

        except Exception as e:
            self.logger.error(f"Errore nel summary sentiment: {e}")
            return {'error': str(e)}

