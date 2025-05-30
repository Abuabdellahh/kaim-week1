import pandas as pd
import talib

def load_stock_data(filepath):
    """Load stock data from CSV."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def apply_indicators(df):
    """Apply technical indicators using TA-Lib."""
    df['SMA'] = talib.SMA(df['Close'], timeperiod=10)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

def main():
    stock_path = "data/stock_prices.csv"
    df = load_stock_data(stock_path)
    df = apply_indicators(df)
    print(df[['date', 'Close', 'SMA', 'RSI', 'MACD', 'MACD_signal']].tail())

if __name__ == "__main__":
    main()
