"""
Financial News EDA Module
Provides comprehensive exploratory data analysis for financial news datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import warnings
from loguru import logger

warnings.filterwarnings('ignore')

class FinancialNewsEDA:
    """Class for performing EDA on financial news datasets."""
    
    def __init__(self, data_path: str, encoding: str = 'utf-8'):
        """
        Initialize the EDA class.
        
        Args:
            data_path (str): Path to the CSV file containing financial news data
            encoding (str): File encoding, defaults to 'utf-8'
        """
        self.data_path = data_path
        self.encoding = encoding
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the data."""
        try:
            df = pd.read_csv(self.data_path, encoding=self.encoding)
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def analyze_publisher_stats(self) -> Dict[str, Any]:
        """
        Analyze publisher statistics.
        
        Returns:
            dict: Statistics about publishers and domains
        """
        try:
            stats = {
                'unique_publishers': len(self.data['publisher'].unique()),
                'unique_domains': len(self.data['domain'].unique()),
                'publisher_distribution': self.data['publisher'].value_counts().to_dict(),
                'domain_distribution': self.data['domain'].value_counts().to_dict()
            }
            return stats
        except Exception as e:
            logger.error(f"Error in publisher analysis: {str(e)}")
            raise
    
    def analyze_temporal_patterns(self, bin_size: str = 'D') -> Dict[str, Any]:
        """
        Analyze temporal patterns in the data.
        
        Args:
            bin_size (str): Time bin size ('D' for daily, 'H' for hourly, etc.)
        
        Returns:
            dict: Temporal analysis results
        """
        try:
            # Daily counts
            daily_counts = self.data.groupby(self.data['timestamp'].dt.floor(bin_size))\
                                  .size()\
                                  .reset_index(name='count')
            
            # Hourly distribution
            hourly_distribution = self.data.groupby(self.data['timestamp'].dt.hour)\
                                        .size()\
                                        .reset_index(name='count')
            
            return {
                'daily_counts': daily_counts,
                'hourly_distribution': hourly_distribution,
                'time_range': {
                    'start': self.data['timestamp'].min(),
                    'end': self.data['timestamp'].max()
                }
            }
        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            raise
    
    def analyze_stock_coverage(self) -> Dict[str, Any]:
        """
        Analyze stock coverage in the news data.
        
        Returns:
            dict: Statistics about stock mentions
        """
        try:
            if 'stock' not in self.data.columns:
                raise ValueError("Stock column not found in dataset")
            
            stats = {
                'unique_stocks': len(self.data['stock'].unique()),
                'stock_distribution': self.data['stock'].value_counts().to_dict(),
                'stock_coverage': self.data.groupby('stock').size().reset_index(name='count')
            }
            return stats
        except Exception as e:
            logger.error(f"Error in stock analysis: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of the dataset.
        
        Returns:
            dict: Summary statistics
        """
        try:
            summary = {
                'total_records': len(self.data),
                'date_range': {
                    'start': self.data['timestamp'].min(),
                    'end': self.data['timestamp'].max(),
                    'duration_days': (self.data['timestamp'].max() - self.data['timestamp'].min()).days
                },
                'missing_values': self.data.isnull().sum().to_dict(),
                'data_types': self.data.dtypes.to_dict()
            }
            return summary
        except Exception as e:
            logger.error(f"Error in data summary: {str(e)}")
            raise
    
    def plot_temporal_patterns(self) -> None:
        """Generate plots for temporal patterns."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn')
        
        # Daily counts
        daily = self.analyze_temporal_patterns()['daily_counts']
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='timestamp', y='count', data=daily)
        plt.title('News Articles per Day')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Hourly distribution
        hourly = self.analyze_temporal_patterns()['hourly_distribution']
        plt.figure(figsize=(12, 6))
        sns.barplot(x='timestamp', y='count', data=hourly)
        plt.title('Hourly Distribution of News Articles')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.show()
