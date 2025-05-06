from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

def create_features(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df = df.dropna()
    return df

def calculate_mse(true_values, predicted_values):
    return mean_squared_error(true_values, predicted_values)

def calculate_mape(true_values, predicted_values):
    """Calculate Mean Absolute Percentage Error"""
    try:
        # Handle zero values in true_values to prevent division by zero
        if np.any(np.array(true_values) == 0):
            # Replace zeros with a small value (1% of the mean of non-zero values)
            non_zero = np.array(true_values)[np.array(true_values) != 0]
            epsilon = 0.01 * np.mean(non_zero) if len(non_zero) > 0 else 1e-10
            true_values_adj = np.array([max(v, epsilon) for v in true_values])
            return mean_absolute_percentage_error(true_values_adj, predicted_values) * 100
        else:
            return mean_absolute_percentage_error(true_values, predicted_values) * 100
    except Exception as e:
        print(f"Error calculating MAPE: {str(e)}")
        return 0.0 