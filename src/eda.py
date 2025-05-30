import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FinancialNewsEDA:
    """
    Comprehensive EDA class for financial news dataset analysis.
    Provides modular functions for various exploratory data analysis tasks.
    """
    
    def __init__(self, filepath):
        """Initialize with data loading and basic validation."""
        self.df = self.load_data(filepath)
        self.validate_data()
    
    def load_data(self, filepath):
        """Load financial news dataset from CSV with error handling."""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    
    def validate_data(self):
        """Validate required columns exist in the dataset."""
        if self.df is None:
            return False
        
        required_columns = ['headline', 'publisher', 'date', 'stock']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"Warning: Missing required columns: {missing_columns}")
            return False
        
        print("Data validation passed")
        return True
    
    def analyze_headline_lengths(self):
        """Calculate comprehensive statistics for headline lengths."""
        if self.df is None:
            return None
        
        self.df['headline_word_count'] = self.df['headline'].apply(lambda x: len(str(x).split()))
        self.df['headline_char_count'] = self.df['headline'].apply(lambda x: len(str(x)))
        
        stats = {
            'word_count_stats': self.df['headline_word_count'].describe(),
            'char_count_stats': self.df['headline_char_count'].describe()
        }
        
        return stats
    
    def analyze_publishers(self):
        """Comprehensive publisher analysis including domain extraction."""
        if self.df is None:
            return None
        
        publisher_counts = self.df['publisher'].value_counts()
        self.df['publisher_domain'] = self.df['publisher'].apply(self._extract_domain)
        domain_counts = self.df['publisher_domain'].value_counts()
        
        return {
            'publisher_counts': publisher_counts,
            'domain_counts': domain_counts,
            'unique_publishers': len(publisher_counts),
            'unique_domains': len(domain_counts)
        }
    
    def _extract_domain(self, publisher):
        """Helper function to extract domain from email-like publisher names."""
        if '@' in str(publisher):
            return str(publisher).split('@')[-1]
        return str(publisher)
    
    def analyze_temporal_patterns(self):
        """Comprehensive time series analysis of publication patterns."""
        if self.df is None:
            return None
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['pub_date'] = self.df['date'].dt.date
        self.df['pub_hour'] = self.df['date'].dt.hour
        self.df['pub_weekday'] = self.df['date'].dt.day_name()
        
        temporal_analysis = {
            'daily_counts': self.df.groupby('pub_date').size(),
            'hourly_distribution': self.df['pub_hour'].value_counts().sort_index(),
            'weekday_distribution': self.df['pub_weekday'].value_counts(),
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max(),
                'total_days': (self.df['date'].max() - self.df['date'].min()).days
            }
        }
        
        return temporal_analysis
    
    def analyze_stock_coverage(self):
        """Analyze stock symbol coverage and distribution."""
        if self.df is None:
            return None
        
        stock_analysis = {
            'stock_counts': self.df['stock'].value_counts(),
            'unique_stocks': len(self.df['stock'].unique()),
            'articles_per_stock_stats': self.df['stock'].value_counts().describe()
        }
        
        return stock_analysis
    
    def generate_visualizations(self, save_path='notebooks/'):
        """Generate and save key visualizations."""
        if self.df is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Headline length distribution
        axes[0, 0].hist(self.df['headline_word_count'], bins=30, alpha=0.7)
        axes[0, 0].set_title('Distribution of Headline Word Counts')
        axes[0, 0].set_xlabel('Word Count')
        axes[0, 0].set_ylabel('Frequency')
        
        # Top publishers
        top_publishers = self.df['publisher'].value_counts().head(10)
        axes[0, 1].barh(range(len(top_publishers)), top_publishers.values)
        axes[0, 1].set_yticks(range(len(top_publishers)))
        axes[0, 1].set_yticklabels(top_publishers.index, fontsize=8)
        axes[0, 1].set_title('Top 10 Publishers by Article Count')
        
        # Publication frequency over time
        daily_counts = self.df.groupby('pub_date').size()
        axes[1, 0].plot(daily_counts.index, daily_counts.values)
        axes[1, 0].set_title('Daily Publication Frequency')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Number of Articles')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Top stocks by coverage
        top_stocks = self.df['stock'].value_counts().head(10)
        axes[1, 1].bar(range(len(top_stocks)), top_stocks.values)
        axes[1, 1].set_xticks(range(len(top_stocks)))
        axes[1, 1].set_xticklabels(top_stocks.index, rotation=45)
        axes[1, 1].set_title('Top 10 Stocks by News Coverage')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}eda_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive EDA summary report."""
        if self.df is None:
            return "No data available for analysis"
        
        headline_stats = self.analyze_headline_lengths()
        publisher_stats = self.analyze_publishers()
        temporal_stats = self.analyze_temporal_patterns()
        stock_stats = self.analyze_stock_coverage()
        
        report = f"""
FINANCIAL NEWS DATASET - EDA SUMMARY REPORT
==========================================

Dataset Overview:
- Total Articles: {len(self.df):,}
- Date Range: {temporal_stats['date_range']['start'].strftime('%Y-%m-%d')} to {temporal_stats['date_range']['end'].strftime('%Y-%m-%d')}
- Total Days Covered: {temporal_stats['date_range']['total_days']}
- Unique Publishers: {publisher_stats['unique_publishers']}
- Unique Stocks Covered: {stock_stats['unique_stocks']}

Headline Analysis:
- Average Word Count: {headline_stats['word_count_stats']['mean']:.1f}
- Median Word Count: {headline_stats['word_count_stats']['50%']:.1f}
- Max Word Count: {headline_stats['word_count_stats']['max']:.0f}

Publication Patterns:
- Average Articles per Day: {len(self.df) / temporal_stats['date_range']['total_days']:.1f}
- Most Active Publisher: {publisher_stats['publisher_counts'].index[0]} ({publisher_stats['publisher_counts'].iloc[0]} articles)
- Most Covered Stock: {stock_stats['stock_counts'].index[0]} ({stock_stats['stock_counts'].iloc[0]} articles)
"""
        return report

def main():
    data_path = "data/financial_news.csv"
    eda = FinancialNewsEDA(data_path)
    if eda.df is None:
        return

    print("=== SUMMARY REPORT ===")
    print(eda.generate_summary_report())

if __name__ == "__main__":
    main()
