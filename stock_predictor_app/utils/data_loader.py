import yfinance as yf

def load_stock_data(ticker, start="2015-01-01", end=None):
    df = yf.download(ticker, start=start, end=end)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']] 