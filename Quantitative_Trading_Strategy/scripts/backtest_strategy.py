def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0
    
    # Buy signal: when RSI is below 30 (oversold) and price crosses above SMA
    signals['Signal'] = ((data['RSI'] < 30) & (data['Close'] > data['SMA_20'])).astype(int)
    
    # Sell signal: when RSI is above 70 (overbought) and price crosses below SMA
    signals['Exit'] = ((data['RSI'] > 70) & (data['Close'] < data['SMA_20'])).astype(int)
    
    return signals

def backtest_strategy(data, signals, initial_balance=10000):
    # Ensure the data aligns with the signals length
    data = data.iloc[-len(signals):].copy()  # Slice the data to match the signal's length

    balance = initial_balance
    position = 0  # 0 = no position, 1 = long position
    balance_history = []

    for i in range(len(signals)):
        if signals[i] == 1 and position == 0:  # Buy signal
            position = balance / data['Close'][i]  # Buy shares
            balance = 0
        elif signals[i] == -1 and position > 0:  # Sell signal
            balance = position * data['Close'][i]  # Sell shares
            position = 0
        balance_history.append(balance if balance > 0 else position * data['Close'][i])

    data['Balance'] = balance_history
    return data


