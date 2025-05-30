import pandas as pd
from textblob import TextBlob

def calculate_sentiment(df):
    """Add polarity and subjectivity scores to the DataFrame using TextBlob."""
    df['headline'] = df['headline'].fillna("")
    df['polarity'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

def main():
    data_path = "data/financial_news.csv"
    df = pd.read_csv(data_path)
    df = calculate_sentiment(df)
    print(df[['headline', 'polarity', 'subjectivity']].head())

if __name__ == "__main__":
    main()
