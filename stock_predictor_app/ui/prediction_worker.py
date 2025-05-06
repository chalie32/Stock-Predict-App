from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import traceback
import json

class PredictionWorker(QObject):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, symbol, model, days, last_price):
        super().__init__()
        self.symbol = symbol
        self.model = model
        self.days = days
        self.last_price = last_price
        
    def _ensure_flat_prediction(self, prediction):
        """
        Ensure prediction is a flat 1D array by flattening if needed
        """
        # If prediction is already a flat array, return it
        if not isinstance(prediction, np.ndarray):
            prediction = np.array(prediction)
            
        # Check if prediction contains nested arrays and flatten if needed
        if len(prediction.shape) > 1:
            print(f"Flattening nested prediction array with shape {prediction.shape}")
            prediction = prediction.flatten()
            
        # Handle case where each item might be a list or array
        elif isinstance(prediction[0], (list, np.ndarray)):
            print("Flattening nested prediction values")
            prediction = np.array([float(p[0]) if isinstance(p, (list, np.ndarray)) else float(p) for p in prediction])
            
        return prediction

    def run(self):
        try:
            # Emit progress updates
            self.progress.emit("Initializing model...")
            print(f"Starting prediction for {self.symbol} using {self.model} for {self.days} days")
            
            # Download historical data
            self.progress.emit("Downloading historical data...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)  # About 5 years of data
            
            df = yf.download(self.symbol, start=start_date, end=end_date)
            
            if df.empty:
                self.error.emit(f"Could not download data for {self.symbol}")
                return
                
            self.progress.emit("Processing data...")
            print(f"Downloaded data: {len(df)} rows for {self.symbol}")
            
            # Ensure we have data
            if len(df) < 60 + self.days:  # Need at least seq_len + prediction days
                self.error.emit(f"Not enough historical data for {self.symbol}")
                return
            
            self.progress.emit("Training model...")
            
            # Run the model prediction
            if self.model == "LSTM":
                try:
                    from stock_predictor_app.models.lstm_model_pro import predict_lstm
                    prediction, future_mse, train_mse, mape = predict_lstm(df, days=self.days)
                    
                    # Make sure prediction is the right length
                    if len(prediction) != self.days:
                        print(f"WARNING: Prediction length mismatch - got {len(prediction)}, expected {self.days}")
                        self.error.emit(f"Model prediction length ({len(prediction)}) doesn't match requested days ({self.days})")
                        return
                        
                    self.progress.emit("Generating predictions...")
                    
                    # Ensure prediction is flat
                    prediction = self._ensure_flat_prediction(prediction)
                    
                    # Get the last known price
                    if isinstance(df['Close'], pd.Series):
                        last_price = float(df['Close'].iloc[-1])
                    else:
                        last_price = float(df['Close'].values[-1])
                    
                    # Calculate predicted change percentage
                    final_pred = float(prediction[-1])
                    pct_change = ((final_pred - last_price) / last_price) * 100
                    
                    # Prepare the result dictionary with all required fields
                    result = {
                        'prediction': prediction.tolist(),
                        'mse': float(train_mse),
                        'mape': float(mape),
                        'status': 'Completed',
                        'data_points': len(df),
                        'change_pct': float(pct_change)
                    }
                    
                    print(f"Sending prediction result with {len(prediction)} days, MSE {train_mse}, and MAPE {mape:.2f}%")
                    print(f"Full result object: {json.dumps({k: str(v) if isinstance(v, list) else v for k, v in result.items()})}")
                except Exception as model_error:
                    error_details = f"LSTM model error: {str(model_error)}\n{traceback.format_exc()}"
                    print(error_details)
                    self.error.emit(error_details)
                    return
            elif self.model == "RF":
                try:
                    from stock_predictor_app.models.rf_model import predict_rf
                    prediction, future_mse, train_mse, mape = predict_rf(df, days=self.days)
                    
                    # Make sure prediction is the right length
                    if len(prediction) != self.days:
                        print(f"WARNING: Prediction length mismatch - got {len(prediction)}, expected {self.days}")
                        self.error.emit(f"Model prediction length ({len(prediction)}) doesn't match requested days ({self.days})")
                        return
                    
                    self.progress.emit("Generating predictions...")
                    
                    # Ensure prediction is flat
                    prediction = self._ensure_flat_prediction(prediction)
                    
                    # Get the last known price
                    if isinstance(df['Close'], pd.Series):
                        last_price = float(df['Close'].iloc[-1])
                    else:
                        last_price = float(df['Close'].values[-1])
                    
                    # Calculate predicted change percentage
                    final_pred = float(prediction[-1])
                    pct_change = ((final_pred - last_price) / last_price) * 100
                    
                    # Prepare the result dictionary
                    result = {
                        'prediction': prediction.tolist(),
                        'mse': float(train_mse),
                        'mape': float(mape),
                        'status': 'Completed',
                        'data_points': len(df),
                        'change_pct': float(pct_change)
                    }
                    
                    print(f"Sending RF prediction with {len(prediction)} days, MSE {train_mse}, and MAPE {mape:.2f}%")
                except Exception as model_error:
                    error_details = f"RF model error: {str(model_error)}\n{traceback.format_exc()}"
                    print(error_details)
                    self.error.emit(error_details)
                    return
            elif self.model == "XGB":
                try:
                    from stock_predictor_app.models.xgb_model import predict_xgb
                    prediction, future_mse, train_mse, mape = predict_xgb(df, days=self.days)
                    
                    # Make sure prediction is the right length
                    if len(prediction) != self.days:
                        print(f"WARNING: Prediction length mismatch - got {len(prediction)}, expected {self.days}")
                        self.error.emit(f"Model prediction length ({len(prediction)}) doesn't match requested days ({self.days})")
                        return
                    
                    self.progress.emit("Generating predictions...")
                    
                    # Ensure prediction is flat
                    prediction = self._ensure_flat_prediction(prediction)
                    
                    # Get the last known price
                    if isinstance(df['Close'], pd.Series):
                        last_price = float(df['Close'].iloc[-1])
                    else:
                        last_price = float(df['Close'].values[-1])
                    
                    # Calculate predicted change percentage
                    final_pred = float(prediction[-1])
                    pct_change = ((final_pred - last_price) / last_price) * 100
                    
                    # Prepare the result dictionary
                    result = {
                        'prediction': prediction.tolist(),
                        'mse': float(train_mse),
                        'mape': float(mape),
                        'status': 'Completed',
                        'data_points': len(df),
                        'change_pct': float(pct_change)
                    }
                    
                    print(f"Sending XGB prediction with {len(prediction)} days, MSE {train_mse}, and MAPE {mape:.2f}%")
                except Exception as model_error:
                    error_details = f"XGB model error: {str(model_error)}\n{traceback.format_exc()}"
                    print(error_details)
                    self.error.emit(error_details)
                    return
            elif self.model == "ARIMA":
                try:
                    from stock_predictor_app.models.arima_model import predict_arima
                    prediction, future_mse, train_mse, mape = predict_arima(df, days=self.days)
                    
                    # Make sure prediction is the right length
                    if len(prediction) != self.days:
                        print(f"WARNING: Prediction length mismatch - got {len(prediction)}, expected {self.days}")
                        self.error.emit(f"Model prediction length ({len(prediction)}) doesn't match requested days ({self.days})")
                        return
                    
                    self.progress.emit("Generating predictions...")
                    
                    # Ensure prediction is flat
                    prediction = self._ensure_flat_prediction(prediction)
                    
                    # Get the last known price
                    if isinstance(df['Close'], pd.Series):
                        last_price = float(df['Close'].iloc[-1])
                    else:
                        last_price = float(df['Close'].values[-1])
                    
                    # Calculate predicted change percentage
                    final_pred = float(prediction[-1])
                    pct_change = ((final_pred - last_price) / last_price) * 100
                    
                    # Prepare the result dictionary
                    result = {
                        'prediction': prediction.tolist(),
                        'mse': float(train_mse),
                        'mape': float(mape),
                        'status': 'Completed',
                        'data_points': len(df),
                        'change_pct': float(pct_change)
                    }
                    
                    print(f"Sending ARIMA prediction with {len(prediction)} days, MSE {train_mse}, and MAPE {mape:.2f}%")
                except Exception as model_error:
                    error_details = f"ARIMA model error: {str(model_error)}\n{traceback.format_exc()}"
                    print(error_details)
                    self.error.emit(error_details)
                    return
            else:
                # Fallback to dummy data if model not recognized
                self.progress.emit("Model not implemented, using simulated data...")
                prediction = np.array([self.last_price * (1 + np.random.normal(0, 0.02)) 
                                      for _ in range(self.days)])
                train_mse = np.random.uniform(0.001, 0.01)
                mape = np.random.uniform(1.0, 5.0)
                
                print("WARNING: Using dummy prediction data")
                
                result = {
                    'prediction': prediction.tolist(),
                    'mse': float(train_mse),
                    'mape': float(mape),
                    'status': 'Completed',
                    'data_points': 1000,  # Dummy value
                    'change_pct': float(np.random.normal(0, 5))  # Dummy value
                }
            
            self.progress.emit("Finalizing results...")
            
            # Debug check the result structure
            print(f"Result keys: {result.keys()}")
            print(f"Prediction type: {type(result['prediction'])}, Length: {len(result['prediction'])}")
            print(f"MSE type: {type(result['mse'])}")
            
            self.finished.emit(result)
            
        except Exception as e:
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            print(f"Error in prediction worker: {error_details}")
            self.error.emit(str(e)) 