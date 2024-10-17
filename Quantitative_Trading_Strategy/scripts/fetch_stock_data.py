import yfinance as yf
import pandas as pd
import talib as ta

# Fetch stock data from yFinance
def fetch_stock_data(symbols, start_date="2018-01-01"):
    stock_data = {}
    
    for symbol in symbols:
        data = yf.download(symbol, start=start_date)
        data['SMA_20'] = ta.SMA(data['Close'], timeperiod=20)
        data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
        stock_data[symbol] = data

    return stock_data

# Example usage
symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
stock_data = fetch_stock_data(symbols)
