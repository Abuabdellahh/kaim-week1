"""
Financial News Sentiment Analysis Module
Provides advanced sentiment analysis capabilities for financial news text.
"""

import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
from textblob import TextBlob
import re

nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)

class FinancialSentimentAnalyzer:
    """Advanced sentiment analysis for financial news text."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.sia = SentimentIntensityAnalyzer()
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by removing special characters and normalizing.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        try:
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Convert to lowercase
            text = text.lower()
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using multiple methods.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Sentiment scores from different methods
        """
        try:
            # VADER sentiment
            vader_scores = self.sia.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_scores = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
            
            # Custom financial sentiment score
            financial_terms = ['profit', 'loss', 'growth', 'decline', 'bear', 'bull']
            financial_score = sum(term in text.lower() for term in financial_terms)
            
            return {
                'vader': vader_scores['compound'],
                'textblob': textblob_scores['polarity'],
                'financial_terms': financial_score,
                'subjectivity': textblob_scores['subjectivity']
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'vader': 0.0, 'textblob': 0.0, 'financial_terms': 0, 'subjectivity': 0.0}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            
        Returns:
            List[Dict[str, float]]: List of sentiment scores for each text
        """
        try:
            results = []
            for text in texts:
                processed_text = self.preprocess_text(text)
                sentiment = self.analyze_sentiment(processed_text)
                results.append(sentiment)
            return results
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return [{'vader': 0.0, 'textblob': 0.0, 'financial_terms': 0, 'subjectivity': 0.0}] * len(texts)
    
    def get_sentiment_trend(self, texts: List[str], timestamps: List[datetime]) -> pd.DataFrame:
        """
        Analyze sentiment trend over time.
        
        Args:
            texts (List[str]): List of texts
            timestamps (List[datetime]): Corresponding timestamps
            
        Returns:
            pd.DataFrame: DataFrame with sentiment scores over time
        """
        try:
            if len(texts) != len(timestamps):
                raise ValueError("Texts and timestamps must have the same length")
            
            results = self.analyze_batch(texts)
            df = pd.DataFrame(results)
            df['timestamp'] = timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate rolling averages
            df['vader_rolling'] = df['vader'].rolling(window=5).mean()
            df['textblob_rolling'] = df['textblob'].rolling(window=5).mean()
            
            return df
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            raise
    
    def get_sentiment_summary(self, texts: List[str]) -> Dict[str, float]:
        """
        Get summary statistics of sentiment scores.
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            dict: Summary statistics of sentiment scores
        """
        try:
            results = self.analyze_batch(texts)
            df = pd.DataFrame(results)
            return {
                'vader_mean': df['vader'].mean(),
                'vader_std': df['vader'].std(),
                'textblob_mean': df['textblob'].mean(),
                'textblob_std': df['textblob'].std(),
                'financial_terms_mean': df['financial_terms'].mean(),
                'subjectivity_mean': df['subjectivity'].mean()
            }
        except Exception as e:
            logger.error(f"Error in summary: {str(e)}")
            return {
                'vader_mean': 0.0, 'vader_std': 0.0,
                'textblob_mean': 0.0, 'textblob_std': 0.0,
                'financial_terms_mean': 0.0, 'subjectivity_mean': 0.0
            }
