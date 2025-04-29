import pandas as pd
import numpy as np

def calculate_ma(df, window=20):
    """Calculate Moving Average
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices
        window (int): Window size for moving average
    
    Returns:
        pd.Series: Moving average values
    """
    return df['Close'].rolling(window=window).mean()

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices
        period (int): RSI calculation period
    
    Returns:
        pd.Series: RSI values
    """
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
    
    Returns:
        tuple: (MACD line, Signal line, MACD histogram)
    """
    # Calculate EMAs
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate MACD histogram
    macd_hist = macd_line - signal_line
    
    return macd_line, signal_line, macd_hist 