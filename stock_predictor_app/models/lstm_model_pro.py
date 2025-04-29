import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from stock_predictor_app.utils.feature_engineering import calculate_mse

def predict_lstm(df, days=30, mse_threshold=1000):
    df = df[['Close']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(60, len(scaled) - days):
        X.append(scaled[i-60:i])
        y.append(scaled[i:i+days, 0])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(days)
    ])
    model.compile(optimizer='adam', loss='mse')

    best_mse = float('inf')
    for _ in range(10):
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
        pred_val = model.predict(X_val)
        val_mse = calculate_mse(y_val[:, 0], pred_val[:, 0])
        if val_mse < best_mse:
            best_mse = val_mse
        if val_mse <= mse_threshold:
            break

    last_60 = scaled[-60:]
    input_seq = last_60.reshape(1, 60, 1)
    pred_scaled = model.predict(input_seq)
    prediction = scaler.inverse_transform(np.concatenate((last_60, pred_scaled.T), axis=0))[-days:, 0]

    true_values = df['Close'].values[-days:]
    future_mse = calculate_mse(true_values, prediction)
    return prediction, future_mse, best_mse 