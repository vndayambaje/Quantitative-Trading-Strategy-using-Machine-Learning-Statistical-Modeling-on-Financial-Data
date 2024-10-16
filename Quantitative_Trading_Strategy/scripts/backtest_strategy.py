import pandas as pd

# Backtest trading strategy
def backtest_strategize(stock_data, signals, initial_balance=10000):
    balance = initial_balance
    shares_held = 0
    balance_history = []

    # Simulate trades based on buy/sell signals
    for index, row in stock_data.iterrows():
        signal = signals.loc[index] if index in signals.index else 0

        # Buy condition
        if signal == 1 and balance >= row['Close']:
            shares_to_buy = balance // row['Close']
            balance -= shares_to_buy * row['Close']
            shares_held += shares_to_buy
        
        # Sell condition
        elif signal == -1 and shares_held > 0:
            balance += shares_held * row['Close']
            shares_held = 0
        
        # Track balance history
        balance_history.append(balance + shares_held * row['Close'])

    stock_data['Balance'] = balance_history
    return stock_data


# Function to compare models and backtest
def compare_models(stock_data, rf_model, xgb_model, lstm_model, X):
    rf_signals = pd.Series(rf_model.predict(X), index=X.index)
    xgb_signals = pd.Series(xgb_model.predict(X), index=X.index)

    # Reshape for LSTM model
    X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))
    lstm_predictions = lstm_model.predict(X_reshaped).flatten()  # Flatten to match index size
    lstm_signals = pd.Series((lstm_predictions > 0.5).astype(int), index=X.index)

    # Backtest each model
    rf_backtest = backtest_strategize(stock_data.copy(), rf_signals)
    xgb_backtest = backtest_strategize(stock_data.copy(), xgb_signals)
    lstm_backtest = backtest_strategize(stock_data.copy(), lstm_signals)

    return rf_backtest, xgb_backtest, lstm_backtest




