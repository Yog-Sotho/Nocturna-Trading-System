"""
Sentiment Analysis Engine per NOCTURNA v2.0
Analizza il sentiment del mercato da news, social media e altri dati testuali.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re
import json
from dataclasses import dataclass
from enum import Enum

class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

@dataclass
class SentimentData:
    """Rappresenta dati di sentiment."""
    source: str
    symbol: str
    text: str
    score: float
    confidence: float
    timestamp: datetime
    metadata: Dict = None

class SentimentAnalyzer:
    """
    Sistema avanzato di sentiment analysis per trading.
    Combina multiple fonti e tecniche di analisi.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dizionari di sentiment
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.financial_terms = self._load_financial_terms()
        
        # Pesi per diverse fonti
        self.source_weights = {
            'news': 1.0,
            'twitter': 0.7,
            'reddit': 0.6,
            'analyst_reports': 1.2,
            'earnings_calls': 1.1
        }
        
        # Cache per sentiment
        self.sentiment_cache = {}
        self.sentiment_history = []
        
        self.logger.info("Sentiment Analyzer inizializzato")
    
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
        """Carica dizionario di parole negative."""
        negative_words = {
            'bearish', 'negative', 'decline', 'loss', 'fall', 'decrease',
            'weak', 'miss', 'underperform', 'downgrade', 'sell', 'crash',
            'plunge', 'collapse', 'concern', 'worry', 'risk', 'threat',
            'disappointing', 'poor', 'terrible', 'awful', 'pessimistic',
            'recession', 'crisis', 'volatility', 'uncertainty', 'correction',
            'breakdown', 'resistance', 'support', 'oversold', 'overbought'
        }
        return negative_words
    
    def _load_financial_terms(self) -> Dict[str, float]:
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
    
    def analyze_text(self, text: str, symbol: str = None) -> Dict:
        """
        Analizza il sentiment di un testo.
        
        Args:
            text: Testo da analizzare
            symbol: Simbolo del titolo (opzionale)
            
        Returns:
            Risultati dell'analisi sentiment
        """
        try:
            # Preprocessing del testo
            cleaned_text = self._preprocess_text(text)
            words = cleaned_text.split()
            
            if not words:
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'classification': SentimentScore.NEUTRAL.name,
                    'word_count': 0
                }
            
            # Calcola score base
            positive_score = 0
            negative_score = 0
            total_weight = 0
            
            for word in words:
                word_lower = word.lower()
                weight = 1.0
                
                # Applica peso per termini finanziari
                if word_lower in self.financial_terms:
                    weight = self.financial_terms[word_lower]
                
                if word_lower in self.positive_words:
                    positive_score += weight
                    total_weight += weight
                elif word_lower in self.negative_words:
                    negative_score += weight
                    total_weight += weight
            
            # Calcola score normalizzato
            if total_weight > 0:
                raw_score = (positive_score - negative_score) / total_weight
            else:
                raw_score = 0.0
            
            # Applica modificatori contestuali
            raw_score = self._apply_contextual_modifiers(cleaned_text, raw_score)
            
            # Normalizza score tra -1 e 1
            normalized_score = np.tanh(raw_score)
            
            # Calcola confidence
            confidence = min(total_weight / len(words), 1.0)
            
            # Classifica sentiment
            classification = self._classify_sentiment(normalized_score)
            
            result = {
                'score': normalized_score,
                'confidence': confidence,
                'classification': classification.name,
                'word_count': len(words),
                'positive_words': positive_score,
                'negative_words': negative_score,
                'financial_relevance': total_weight / len(words) if words else 0.0
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi sentiment: {e}")
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
        """Applica modificatori contestuali al score."""
        modified_score = base_score
        
        # Negazioni
        negation_patterns = [
            r'not\s+\w+',
            r'no\s+\w+',
            r'never\s+\w+',
            r'nothing\s+\w+',
            r'none\s+\w+'
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, text):
                modified_score *= -0.5  # Inverte parzialmente il sentiment
        
        # Intensificatori
        intensifiers = ['very', 'extremely', 'highly', 'significantly', 'substantially']
        for intensifier in intensifiers:
            if intensifier in text:
                modified_score *= 1.3
        
        # Diminutivi
        diminishers = ['slightly', 'somewhat', 'barely', 'hardly', 'scarcely']
        for diminisher in diminishers:
            if diminisher in text:
                modified_score *= 0.7
        
        # Incertezza
        uncertainty_words = ['maybe', 'perhaps', 'possibly', 'might', 'could']
        for word in uncertainty_words:
            if word in text:
                modified_score *= 0.8
        
        return modified_score
    
    def _classify_sentiment(self, score: float) -> SentimentScore:
        """Classifica il sentiment score."""
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
    
    def analyze_news_batch(self, news_articles: List[Dict]) -> Dict:
        """
        Analizza un batch di articoli di news.
        
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
                timestamp = article.get('timestamp', datetime.now())
                
                # Combina titolo e contenuto
                full_text = f"{title} {content}"
                
                # Analizza sentiment
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
            
            # Calcola sentiment aggregato per simbolo
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
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi batch news: {e}")
            return {'error': str(e)}
    
    def get_market_sentiment_signal(self, symbol: str, 
                                   lookback_hours: int = 24) -> Dict:
        """
        Genera un segnale di sentiment per il mercato.
        
        Args:
            symbol: Simbolo del titolo
            lookback_hours: Ore di lookback per l'analisi
            
        Returns:
            Segnale di sentiment
        """
        try:
            # Filtra sentiment history per simbolo e timeframe
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            
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
            
            # Calcola sentiment aggregato con decay temporale
            current_time = datetime.now()
            weighted_scores = []
            weights = []
            
            for sentiment in relevant_sentiments:
                # Calcola peso basato su età e confidence
                age_hours = (current_time - sentiment.timestamp).total_seconds() / 3600
                time_weight = np.exp(-age_hours / 12)  # Decay con half-life di 12 ore
                
                total_weight = time_weight * sentiment.confidence
                
                # Applica peso della fonte
                source_weight = self.source_weights.get(sentiment.source, 1.0)
                total_weight *= source_weight
                
                weighted_scores.append(sentiment.score * total_weight)
                weights.append(total_weight)
            
            # Calcola sentiment finale
            if sum(weights) > 0:
                final_score = sum(weighted_scores) / sum(weights)
                avg_confidence = np.mean([s.confidence for s in relevant_sentiments])
            else:
                final_score = 0.0
                avg_confidence = 0.0
            
            # Genera segnale
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
            self.logger.error(f"Errore nel segnale sentiment: {e}")
            return {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def add_sentiment_data(self, source: str, symbol: str, text: str,
                          metadata: Dict = None):
        """
        Aggiunge nuovi dati di sentiment al sistema.
        
        Args:
            source: Fonte dei dati
            symbol: Simbolo del titolo
            text: Testo da analizzare
            metadata: Metadati aggiuntivi
        """
        try:
            # Analizza sentiment
            sentiment_result = self.analyze_text(text, symbol)
            
            # Crea oggetto sentiment
            sentiment_data = SentimentData(
                source=source,
                symbol=symbol,
                text=text,
                score=sentiment_result['score'],
                confidence=sentiment_result['confidence'],
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            # Aggiungi alla history
            self.sentiment_history.append(sentiment_data)
            
            # Mantieni solo ultimi N giorni
            max_age = timedelta(days=7)
            cutoff_time = datetime.now() - max_age
            
            self.sentiment_history = [
                s for s in self.sentiment_history
                if s.timestamp >= cutoff_time
            ]
            
            self.logger.debug(f"Aggiunto sentiment per {symbol}: {sentiment_result['score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiunta sentiment: {e}")
    
    def get_sentiment_trends(self, symbol: str, days: int = 7) -> Dict:
        """
        Analizza i trend di sentiment per un simbolo.
        
        Args:
            symbol: Simbolo del titolo
            days: Giorni di storia da analizzare
            
        Returns:
            Analisi dei trend
        """
        try:
            # Filtra dati per simbolo e timeframe
            cutoff_time = datetime.now() - timedelta(days=days)
            
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
            
            # Calcola trend
            timestamps = [s.timestamp for s in symbol_sentiments]
            scores = [s.score for s in symbol_sentiments]
            
            # Regressione lineare semplice
            x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
            y = np.array(scores)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                # Classifica trend
                if slope > 0.001:
                    trend = 'IMPROVING'
                elif slope < -0.001:
                    trend = 'DETERIORATING'
                else:
                    trend = 'STABLE'
            else:
                slope = 0.0
                trend = 'STABLE'
            
            # Calcola volatilità del sentiment
            sentiment_volatility = np.std(scores) if len(scores) > 1 else 0.0
            
            # Analizza distribuzione per fonte
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
            for source, data in source_breakdown.items():
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
    
    def get_sentiment_summary(self) -> Dict:
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
                    if s.timestamp >= datetime.now() - timedelta(hours=24)
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
                if s.timestamp >= datetime.now() - timedelta(hours=24)
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
                'timestamp': datetime.now().isoformat(),
                'global_stats': global_stats,
                'symbol_summaries': symbol_summaries
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel summary sentiment: {e}")
            return {'error': str(e)}

