from transformers import pipeline

# Initialize the FinBERT sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

def get_sentiment(text):
    sentiment = sentiment_analyzer(text)
    return sentiment

# Example usage: Fetch news articles or tweets and perform sentiment analysis
news_article = "The company's stock surged after reporting record earnings."
sentiment_result = get_sentiment(news_article)
print(f"Sentiment: {sentiment_result}")

