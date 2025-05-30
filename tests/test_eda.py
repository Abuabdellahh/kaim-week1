import unittest
import pandas as pd
from src.eda import FinancialNewsEDA

class TestFinancialNewsEDA(unittest.TestCase):
    
    def setUp(self):
        # Create a small DataFrame and write to CSV
        self.test_df = pd.DataFrame({
            'headline': ['Up 5%', 'Earnings beat', 'New product launch'],
            'publisher': ['a@ex.com', 'b@ex.com', 'a@ex.com'],
            'date': ['2025-01-01 09:00:00', '2025-01-02 10:30:00', '2025-01-03 14:15:00'],
            'stock': ['AAPL', 'GOOGL', 'AAPL']
        })
        self.test_file = 'tests/tmp_test.csv'
        self.test_df.to_csv(self.test_file, index=False)
    
    def test_extract_domain(self):
        eda = FinancialNewsEDA.__new__(FinancialNewsEDA)
        self.assertEqual(eda._extract_domain('user@domain.com'), 'domain.com')
        self.assertEqual(eda._extract_domain('Reuters'), 'Reuters')
    
    def test_load_and_validate(self):
        eda = FinancialNewsEDA(self.test_file)
        self.assertIsNotNone(eda.df)
        self.assertTrue(eda.validate_data())
    
    def test_analyze_headline_lengths(self):
        eda = FinancialNewsEDA(self.test_file)
        stats = eda.analyze_headline_lengths()
        self.assertIn('word_count_stats', stats)
        self.assertIn('char_count_stats', stats)
    
    def test_analyze_publishers(self):
        eda = FinancialNewsEDA(self.test_file)
        pub_stats = eda.analyze_publishers()
        self.assertEqual(pub_stats['unique_publishers'], 2)
        self.assertEqual(pub_stats['unique_domains'], 2)
    
    def test_temporal_patterns(self):
        eda = FinancialNewsEDA(self.test_file)
        temp = eda.analyze_temporal_patterns()
        self.assertIn('daily_counts', temp)
        self.assertIn('hourly_distribution', temp)
    
    def test_stock_coverage(self):
        eda = FinancialNewsEDA(self.test_file)
        stock = eda.analyze_stock_coverage()
        self.assertEqual(stock['unique_stocks'], 2)

if __name__ == '__main__':
    unittest.main()
