from sklearn.metrics import mean_squared_error

def create_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    return df

def calculate_mse(true_values, predicted_values):
    return mean_squared_error(true_values, predicted_values) 