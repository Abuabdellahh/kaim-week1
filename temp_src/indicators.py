"""
Financial Technical Indicators Module
Provides implementation of various technical indicators for financial analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from datetime import datetime
import talib

class TechnicalIndicators:
    """Class for calculating technical indicators."""
    
    def __init__(self):
        """Initialize the technical indicators class."""
        pass
    
    def calculate_moving_averages(self, prices: pd.Series, 
                                 periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calculate multiple moving averages.
        
        Args:
            prices (pd.Series): Series of closing prices
            periods (List[int]): List of periods for moving averages
            
        Returns:
            Dict[str, pd.Series]: Dictionary of moving averages
        """
        try:
            mas = {}
            for period in periods:
                ma = prices.rolling(window=period).mean()
                mas[f'ma_{period}'] = ma
            return mas
        except Exception as e:
            logger.error(f"Error calculating moving averages: {str(e)}")
            return {}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices (pd.Series): Series of closing prices
            period (int): RSI period
            
        Returns:
            pd.Series: RSI values
        """
        try:
            return talib.RSI(prices, timeperiod=period)
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series()
    
    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            prices (pd.Series): Series of closing prices
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing MACD components
        """
        try:
            macd, signal, hist = talib.MACD(prices)
            return {
                'macd': macd,
                'signal': signal,
                'histogram': hist
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {}
    
    def calculate_bollinger_bands(self, prices: pd.Series, 
                                 period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices (pd.Series): Series of closing prices
            period (int): Period for moving average
            std_dev (float): Number of standard deviations
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing Bollinger Bands components
        """
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return {
                'sma': sma,
                'upper_band': upper_band,
                'lower_band': lower_band
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {}
    
    def calculate_volume_indicators(self, volume: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators.
        
        Args:
            volume (pd.Series): Series of volume data
            
        Returns:
            Dict[str, pd.Series]: Dictionary of volume indicators
        """
        try:
            indicators = {}
            
            # Simple Moving Average of Volume
            indicators['volume_ma_20'] = volume.rolling(window=20).mean()
            
            # Volume Rate of Change
            indicators['volume_roc'] = volume.pct_change()
            
            # On Balance Volume
            indicators['obv'] = volume.cumsum()
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return {}
    
    def calculate_volatility(self, prices: pd.Series, 
                            period: int = 20) -> pd.Series:
        """
        Calculate historical volatility.
        
        Args:
            prices (pd.Series): Series of closing prices
            period (int): Period for volatility calculation
            
        Returns:
            pd.Series: Volatility values
        """
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
            return volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return pd.Series()
    
    def get_indicators_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of all indicators.
        
        Args:
            df (pd.DataFrame): DataFrame containing price and volume data
            
        Returns:
            Dict[str, Any]: Summary statistics of indicators
        """
        try:
            indicators = {}
            
            # Calculate all indicators
            mas = self.calculate_moving_averages(df['close'])
            rsi = self.calculate_rsi(df['close'])
            macd = self.calculate_macd(df['close'])
            bollinger = self.calculate_bollinger_bands(df['close'])
            volume = self.calculate_volume_indicators(df['volume'])
            volatility = self.calculate_volatility(df['close'])
            
            # Get summary statistics
            indicators['moving_averages'] = {k: v.describe().to_dict() for k, v in mas.items()}
            indicators['rsi'] = rsi.describe().to_dict()
            indicators['macd'] = {k: v.describe().to_dict() for k, v in macd.items()}
            indicators['bollinger_bands'] = {k: v.describe().to_dict() for k, v in bollinger.items()}
            indicators['volume'] = {k: v.describe().to_dict() for k, v in volume.items()}
            indicators['volatility'] = volatility.describe().to_dict()
            
            return indicators
        except Exception as e:
            logger.error(f"Error getting indicators summary: {str(e)}")
            return {}
