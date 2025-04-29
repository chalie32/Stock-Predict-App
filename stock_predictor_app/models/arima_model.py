"""
Advanced ARIMA model implementation for stock price prediction.
Includes both standard ARIMA and auto-tuning ARIMA implementations.
"""

from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from stock_predictor_app.utils.feature_engineering import calculate_mse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARIMAPredictor:
    """
    Advanced ARIMA model for stock price prediction with both standard and auto-tuning capabilities.
    """
    
    def __init__(self, use_auto: bool = True):
        """
        Initialize the ARIMA predictor.
        
        Args:
            use_auto (bool): Whether to use auto-tuning ARIMA. Defaults to True.
        """
        self.use_auto = use_auto
        self.model = None
        self.order = None
        self.training_summary = {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and validate data for ARIMA modeling.
        
        Args:
            df (pd.DataFrame): Input dataframe with 'Close' prices
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Training and test data arrays
        
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Ensure we have Close prices
            if 'Close' not in df.columns:
                raise ValueError("DataFrame must contain 'Close' column")
            
            # Clean and convert data
            close_prices = df['Close'].dropna().astype(float)
            
            if len(close_prices) < 30:
                raise ValueError("Insufficient data: need at least 30 data points")
            
            # Split into train/test (80/20)
            train_size = int(len(close_prices) * 0.8)
            train = close_prices[:train_size]
            test = close_prices[train_size:]
            
            return train, test
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train_auto_arima(self, train_data: np.ndarray) -> None:
        """
        Train auto-ARIMA model with optimal parameter selection.
        
        Args:
            train_data (np.ndarray): Training data array
        """
        try:
            logger.info("Training auto-ARIMA model...")
            
            self.model = auto_arima(
                train_data,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                m=1,  # Non-seasonal model
                start_P=0, seasonal=False,
                d=1, D=0,  # Let the model determine d
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            self.order = self.model.order
            self.training_summary = {
                'aic': self.model.aic(),
                'order': self.order,
                'model_params': self.model.get_params()
            }
            
            logger.info(f"Auto-ARIMA training complete. Selected order: {self.order}")
            
        except Exception as e:
            logger.error(f"Auto-ARIMA training failed: {str(e)}")
            raise
    
    def train_standard_arima(self, train_data: np.ndarray, order: Tuple[int, int, int] = (5, 1, 0)) -> None:
        """
        Train standard ARIMA model with specified parameters.
        
        Args:
            train_data (np.ndarray): Training data array
            order (Tuple[int, int, int]): ARIMA order (p,d,q). Defaults to (5,1,0)
        """
        try:
            logger.info(f"Training standard ARIMA model with order {order}...")
            
            self.model = ARIMA(train_data, order=order)
            self.model = self.model.fit()
            self.order = order
            
            self.training_summary = {
                'aic': self.model.aic,
                'order': self.order,
                'model_params': self.model.params
            }
            
            logger.info("Standard ARIMA training complete")
            
        except Exception as e:
            logger.error(f"Standard ARIMA training failed: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame, days: int = 30) -> Tuple[np.ndarray, float, Dict]:
        """
        Generate predictions using either auto or standard ARIMA.
        
        Args:
            df (pd.DataFrame): Input dataframe with historical data
            days (int): Number of days to forecast
            
        Returns:
            Tuple[np.ndarray, float, Dict]: Predictions, MSE score, and model info
        """
        try:
            # Prepare data
            train_data, test_data = self.prepare_data(df)
            
            # Train model
            if self.use_auto:
                self.train_auto_arima(train_data)
            else:
                self.train_standard_arima(train_data)
            
            # Generate predictions
            predictions = self.model.predict(n_periods=days) if self.use_auto else self.model.forecast(steps=days)
            
            # Calculate MSE if we have test data
            mse = calculate_mse(test_data[-len(predictions):], predictions[:len(test_data)])
            
            # Prepare return information
            model_info = {
                'model_type': 'Auto-ARIMA' if self.use_auto else 'Standard ARIMA',
                'order': self.order,
                'training_summary': self.training_summary,
                'prediction_length': len(predictions)
            }
            
            return predictions, mse, model_info
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return np.array([]), np.nan, {}
    
    def get_confidence_intervals(self, days: int = 30, alpha: float = 0.05) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            days (int): Number of days to forecast
            alpha (float): Significance level for confidence intervals
            
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: Lower and upper confidence bounds
        """
        try:
            if self.model is None:
                raise ValueError("Model must be trained before calculating confidence intervals")
            
            if self.use_auto:
                # For auto_arima
                _, conf_int = self.model.predict(n_periods=days, return_conf_int=True, alpha=alpha)
                lower_bound, upper_bound = conf_int[:, 0], conf_int[:, 1]
            else:
                # For standard ARIMA
                forecast = self.model.get_forecast(steps=days)
                conf_int = forecast.conf_int(alpha=alpha)
                lower_bound = conf_int.iloc[:, 0].values
                upper_bound = conf_int.iloc[:, 1].values
            
            return lower_bound, upper_bound
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence intervals: {str(e)}")
            return None
    
    def get_model_diagnostics(self) -> Dict:
        """
        Get model diagnostic information.
        
        Returns:
            Dict: Dictionary containing model diagnostics
        """
        if self.model is None:
            return {}
        
        try:
            diagnostics = {
                'model_type': 'Auto-ARIMA' if self.use_auto else 'Standard ARIMA',
                'order': self.order,
                'aic': self.training_summary.get('aic'),
                'training_summary': self.training_summary
            }
            
            if not self.use_auto:
                diagnostics.update({
                    'resid_std': float(self.model.resid.std()),
                    'nobs': self.model.nobs
                })
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Failed to get model diagnostics: {str(e)}")
            return {} 