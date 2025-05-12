# Stock Predict App

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
    <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" alt="Platform">
</div>

## 📊 Overview

Stock Predict App is a sophisticated desktop application for stock market analysis and prediction. Built with PyQt6, it combines advanced machine learning models (Random Forest, LSTM, XGBoost, and ARIMA) with an elegant, modern user interface to provide powerful stock price forecasting tools for investors and traders.

## ✨ Key Features

- **📈 Real-time Market Data**: Fetch and display up-to-date stock price information using Yahoo Finance API
- **📊 Advanced Interactive Charts**: Candlestick, line, and OHLC charts with customizable technical indicators (MA, RSI, MACD)
- **🤖 Multiple Prediction Models**:
  - Random Forest with market behavior pattern analysis
  - Long Short-Term Memory (LSTM) neural networks
  - XGBoost with optimized hyperparameters
  - ARIMA time series forecasting
- **📰 Financial News Integration**: Latest stock-related news via NewsAPI
- **💾 Historical Prediction Tracking**: Save and compare prediction records over time
- **🎨 Modern, Customizable UI**: Dark mode interface with elegant animations and responsive design

## 🖼️ Screenshots



## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Internet connection for fetching stock data

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Stock-Predict-App.git
   cd Stock-Predict-App
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your NewsAPI key:
   - Get a free API key from [NewsAPI](https://newsapi.org)
   - Open `stock_predictor_app/config.py` and replace the existing NewsAPI key with yours

## 🎮 Usage

Run the application with:
```bash
python -m stock_predictor_app.main
```

### Quick Start Guide

1. **Search for a stock**: Enter a stock symbol (e.g., AAPL, MSFT) in the search box
2. **Analyze historical data**: View candlestick charts with customizable technical indicators
3. **Generate predictions**: Select a prediction model and forecast horizon
4. **Save and compare predictions**: Track prediction accuracy over time

## ⚙️ Configuration

The application uses a `settings.json` file located in the `stock_predictor_app` directory for customization:

```json
{
    "Chart Settings": {
        "chart_type": "Candlestick",
        "default_ma": {
            "MA20": true,
            "MA50": true,
            "MA200": true
        },
        "show_volume": true
    },
    "Analysis Settings": {
        "default_timeframe": "1 Year",
        "default_model": "LSTM",
        "prediction_days": 30
    },
    "Alert Settings": {
        "price_alerts": true,
        "price_change_threshold": 5
    }
}
```

## 🔧 Technology Stack

- **UI Framework**: PyQt6
- **Data Visualization**: Matplotlib, mplfinance
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, TensorFlow/Keras, XGBoost, StatsModels
- **API Integration**: Yahoo Finance, NewsAPI

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data retrieval
- [mplfinance](https://github.com/matplotlib/mplfinance) for financial charting capabilities
- [NewsAPI](https://newsapi.org) for financial news integration 