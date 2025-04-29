import yfinance as yf
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timedelta

class StockDataError(Exception):
    """Custom exception for stock data fetching errors"""
    pass

def get_stock_data(ticker: str, period: str = "1mo", interval: str = "1d") -> Tuple[pd.DataFrame, dict]:
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock symbol (e.g., 'AAPL')
        period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        Tuple[pd.DataFrame, dict]: DataFrame with stock data and info dictionary
        
    Raises:
        StockDataError: If there's an error fetching the data
    """
    if not ticker:
        raise StockDataError("Stock symbol cannot be empty")
    
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get stock info
        info = stock.info
        if not info:
            raise StockDataError(f"Could not find stock with symbol '{ticker}'")
        
        # Download stock data with the specified period and interval
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            raise StockDataError(f"No data available for stock '{ticker}'")
        
        # Sort index to ensure chronological order
        df = df.sort_index()
        
        return df, info
        
    except Exception as e:
        if "Invalid ticker" in str(e):
            raise StockDataError(f"Invalid stock symbol: {ticker}")
        raise StockDataError(f"Error fetching stock data: {str(e)}")

def get_company_info(ticker: str) -> dict:
    """
    Get detailed company information from Yahoo Finance
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL')
    
    Returns:
        dict: Dictionary containing company information
    """
    try:
        # Create Ticker object
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price data for today's changes
        try:
            hist = stock.history(period="2d")
            if not hist.empty and len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                day_change = current_price - prev_close
                day_change_pct = (day_change / prev_close) * 100
            else:
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                prev_close = info.get('previousClose', 0)
                day_change = current_price - prev_close
                day_change_pct = (day_change / prev_close) * 100 if prev_close else 0
        except Exception as e:
            print(f"Error getting price data: {e}")
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            day_change = 0
            day_change_pct = 0
        
        # Extract and format relevant information
        company_info = {
            'symbol': ticker,
            'name': info.get('longName', 'N/A'),
            'exchange': info.get('exchange', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'website': info.get('website', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A'),
            'employees': info.get('fullTimeEmployees', 'N/A'),
            'country': info.get('country', 'N/A'),
            'city': info.get('city', 'N/A'),
            'address': info.get('address1', 'N/A'),
            'phone': info.get('phone', 'N/A'),
            'currency': info.get('currency', 'USD'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            'avg_volume': info.get('averageVolume', 'N/A'),
            'current_price': current_price,
            'day_change': day_change,
            'day_change_pct': day_change_pct
        }
        
        # Format market cap
        if company_info['market_cap'] > 0:
            if company_info['market_cap'] >= 1e12:
                market_cap_str = f"${company_info['market_cap']/1e12:.2f}T"
            elif company_info['market_cap'] >= 1e9:
                market_cap_str = f"${company_info['market_cap']/1e9:.2f}B"
            else:
                market_cap_str = f"${company_info['market_cap']/1e6:.2f}M"
            company_info['market_cap_str'] = market_cap_str
        else:
            company_info['market_cap_str'] = 'N/A'
        
        return company_info
        
    except Exception as e:
        raise StockDataError(f"Error fetching company info for {ticker}: {str(e)}") 