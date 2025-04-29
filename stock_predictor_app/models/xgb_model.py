import xgboost as xgb
from stock_predictor_app.utils.feature_engineering import calculate_mse

def predict_xgb(df, days=30):
    df = df.copy()
    df['Target'] = df['Close'].shift(-days)
    df = df.dropna()
    X = df.drop(['Target'], axis=1).values
    y = df['Target'].values

    model = xgb.XGBRegressor()
    model.fit(X, y)

    future_input = df.drop(['Target'], axis=1).iloc[-1:].values
    prediction = model.predict(future_input)[0:days]

    true_values = df['Target'].values[-days:]
    mse = calculate_mse(true_values, prediction[:len(true_values)])
    return prediction, mse 