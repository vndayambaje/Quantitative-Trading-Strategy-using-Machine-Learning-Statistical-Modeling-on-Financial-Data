import yfinance as yf
import pandas as pd

# Function to fetch stock data using Ticker's history method
def fetch_stock_data(symbol, start_date="2018-01-01"):
    # Fetch stock data from yFinance
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date)
    
    # Handle missing 'Dividends' and 'Stock Splits' columns
    if 'Dividends' in data.columns:
        data['Dividends'].fillna(0, inplace=True)
    
    if 'Stock Splits' in data.columns:
        data['Stock Splits'].fillna(0, inplace=True)

    return data

# Example: Fetch data for Apple and S&P500
# You can expand this to fetch data for multiple stocks by looping over a list of symbols.
apple_data = fetch_stock_data('AAPL')
sp500_data = fetch_stock_data('SPY')  # S&P 500 Index

