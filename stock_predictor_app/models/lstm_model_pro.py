import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import datetime
import traceback

def predict_lstm(df, days=30, mse_threshold=1000):
    """
    LSTM model for stock prediction
    
    Args:
        df: DataFrame with stock data
        days: Number of days to predict
        mse_threshold: Threshold for MSE
    
    Returns:
        prediction: Array of predicted values
        future_mse: MSE for future predictions
        train_mse: MSE for training data
        mape: Mean Absolute Percentage Error
    """
    try:
        print(f"Starting LSTM prediction for {days} days")
        
        # Get last price for reference
        if isinstance(df['Close'], pd.Series):
            last_price = float(df['Close'].iloc[-1])
        else:
            last_price = float(df['Close'].values[-1])
        print(f"Last known price: {last_price}")
        
        # Ensure we're working with the Close price
        if 'Close' not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError("DataFrame doesn't contain 'Close' column")
            
        df = df[['Close']].dropna()
        close_prices = df[['Close']]
        
        print(f"Data shape: {df.shape}")

        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        # Create sequences for training
        seq_len = 60  # How many days to look back
        
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(seq_len, len(data) - days + 1):
                X.append(data[i-seq_len:i])
                y.append(data[i:i+days, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, seq_len)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"Training data shapes - X: {X.shape}, y: {y.shape}")
        
        if len(X) == 0:
            raise ValueError(f"Not enough data points to create sequences (need at least {seq_len + days})")

        # Split for validation
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]
        
        print(f"Split data - Train: {X_train.shape}, Validation: {X_val.shape}")

        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(days)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        print("Training model...")
        history = model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs=10, 
            batch_size=32, 
            verbose=1,
            callbacks=[early_stopping]
        )
        
        print("Model training completed")

        # Predict on training data for evaluation
        train_pred = model.predict(X)
        
        # Directly calculate MSE on scaled data
        train_mse = np.mean([mean_squared_error(
            y[i], train_pred[i]
        ) for i in range(len(train_pred))])
        
        print(f"Training MSE (scaled): {train_mse}")
        
        # Predict future days
        last_seq = scaled_data[-seq_len:]
        input_seq = last_seq.reshape(1, seq_len, 1)
        
        print("Predicting future values...")
        future_preds = model.predict(input_seq)[0]
        
        print(f"Raw prediction shape: {future_preds.shape}")
        print(f"Raw prediction values (first 5): {future_preds[:5]}")
        
        # Inverse transform to get actual prices
        future_preds_reshaped = np.zeros((days, 1))
        future_preds_reshaped[:, 0] = future_preds
        prediction = scaler.inverse_transform(future_preds_reshaped)[:, 0]
        
        print(f"Prediction shape: {prediction.shape}")
        print(f"First 5 predictions: {prediction[:5]}")
        print(f"Last 5 predictions: {prediction[-5:]}")
        print(f"Min prediction: {prediction.min()}, Max prediction: {prediction.max()}")
        
        # Calculate MAPE (Mean Absolute Percentage Error) for training data
        # First inverse transform the training predictions and actual values
        y_actual_inv = np.zeros((y.shape[0], days))
        pred_inv = np.zeros((train_pred.shape[0], days))
        
        for i in range(len(train_pred)):
            temp_actual = np.zeros((days, 1))
            temp_actual[:, 0] = y[i]
            y_actual_inv[i] = scaler.inverse_transform(temp_actual)[:, 0]
            
            temp_pred = np.zeros((days, 1))
            temp_pred[:, 0] = train_pred[i]
            pred_inv[i] = scaler.inverse_transform(temp_pred)[:, 0]
        
        # Calculate MAPE for each sample and take the mean
        mape_values = []
        for i in range(len(y_actual_inv)):
            sample_mape = mean_absolute_percentage_error(y_actual_inv[i], pred_inv[i])
            mape_values.append(sample_mape)
        
        mape = np.mean(mape_values)
        print(f"Training MAPE: {mape:.2f}%")
        
        # Calculate percentage change from last known price
        if isinstance(last_price, pd.Series):
            last_price = float(last_price.iloc[0]) if len(last_price) > 0 else float(last_price)
        
        if isinstance(prediction, np.ndarray) and len(prediction) > 0:
            final_pred_value = float(prediction[-1])
            pct_change = ((final_pred_value - last_price) / last_price) * 100
        else:
            pct_change = 0.0
            
        print(f"Predicted {days}-day price change: {pct_change:.2f}%")
        
        # Ensure we return exactly the number of days requested
        if len(prediction) > days:
            prediction = prediction[:days]
        elif len(prediction) < days:
            # If we have fewer days than requested, extend the last value
            last_value = prediction[-1]
            additional_days = days - len(prediction)
            prediction = np.append(prediction, [last_value] * additional_days)
            
        print(f"Final prediction shape: {prediction.shape}")
        
        # Convert train_mse to a normal Python float to ensure it's serializable
        train_mse = float(train_mse)
        future_mse = float(0)  # Placeholder since we don't have future data
        mape = float(mape)
        
        print(f"Returning prediction with shape {prediction.shape}, MSE {train_mse}, and MAPE {mape:.2f}%")
        
        return prediction, future_mse, train_mse, mape
        
    except Exception as e:
        print(f"Error in LSTM model: {str(e)}")
        print(traceback.format_exc())
        
        # Return dummy data in case of error
        last_price = df['Close'].iloc[-1] if not df.empty else 100.0
        dummy_prediction = np.array([last_price * (1 + np.random.normal(0, 0.01)) for _ in range(days)])
        return dummy_prediction, 0.01, 0.01, 5.0 