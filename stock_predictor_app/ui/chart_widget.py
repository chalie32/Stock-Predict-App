from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QCheckBox, QFrame
import matplotlib
matplotlib.use('QtAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import numpy as np
from datetime import datetime, time
from stock_predictor_app.data.indicators import calculate_ma, calculate_rsi, calculate_macd
from stock_predictor_app.data.settings import Settings
from matplotlib.ticker import FuncFormatter

def is_market_day(date):
    """Check if the given date is a potential trading day (weekday)"""
    # Convert to datetime if it's not already
    if not isinstance(date, datetime):
        date = pd.to_datetime(date)
    
    # Check if it's a weekday (Monday = 0, Sunday = 6)
    return date.weekday() < 5

class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = Settings()
        
        # Create main layout with proper spacing
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)  # Reduced spacing
        
        # Create a container for controls with modern styling
        controls_frame = QFrame()
        controls_frame.setObjectName("container")
        controls_frame.setMaximumHeight(60)  # Limit height of controls
        
        # Use QHBoxLayout for controls with better spacing
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setSpacing(10)  # Reduced spacing
        controls_layout.setContentsMargins(12, 8, 12, 8)  # Reduced margins
        
        # Create indicator checkboxes with icons
        self.ma20_cb = QCheckBox("MA20")
        self.ma50_cb = QCheckBox("MA50")
        self.ma200_cb = QCheckBox("MA200")
        self.rsi_cb = QCheckBox("RSI")
        self.macd_cb = QCheckBox("MACD")
        
        # Load settings
        chart_settings = self.settings.get_setting("Chart Settings", "default_ma")
        if chart_settings:
            self.ma20_cb.setChecked(chart_settings.get("MA20", True))
            self.ma50_cb.setChecked(chart_settings.get("MA50", True))
            self.ma200_cb.setChecked(chart_settings.get("MA200", False))
        
        # Connect signals
        self.ma20_cb.stateChanged.connect(lambda state: self.update_indicator_setting("MA20", state))
        self.ma50_cb.stateChanged.connect(lambda state: self.update_indicator_setting("MA50", state))
        self.ma200_cb.stateChanged.connect(lambda state: self.update_indicator_setting("MA200", state))
        self.rsi_cb.stateChanged.connect(lambda state: self.update_indicator_setting("RSI", state))
        self.macd_cb.stateChanged.connect(lambda state: self.update_indicator_setting("MACD", state))
        
        # Create a container for MA indicators with more compact styling
        ma_container = QFrame()
        ma_container.setStyleSheet("""
            QFrame {
                background-color: #21262d;
                border-radius: 8px;
                padding: 2px;
            }
            QCheckBox {
                color: #c9d1d9;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
                min-width: 60px;
                max-width: 80px;
            }
            QCheckBox:hover {
                background-color: #30363d;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #30363d;
                border-radius: 4px;
                background: #0d1117;
            }
            QCheckBox::indicator:checked {
                border-color: #1f6feb;
            }
            QCheckBox::indicator:hover {
                border-color: #388bfd;
            }
        """)
        ma_layout = QHBoxLayout(ma_container)
        ma_layout.setSpacing(2)
        ma_layout.setContentsMargins(4, 2, 4, 2)
        
        # Add badges to MA checkboxes with colors - make more compact
        self.ma20_cb.setText("MA20")
        self.ma20_cb.setStyleSheet("""
            QCheckBox {
                color: #c9d1d9;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QCheckBox:hover {
                background-color: #30363d;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #30363d;
                border-radius: 4px;
                background: #0d1117;
            }
            QCheckBox::indicator:checked {
                background: #fb923c;
                border-color: #fb923c;
            }
            QCheckBox:checked {
                color: #fb923c;
                font-weight: bold;
            }
        """)
        
        self.ma50_cb.setText("MA50")
        self.ma50_cb.setStyleSheet("""
            QCheckBox {
                color: #c9d1d9;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QCheckBox:hover {
                background-color: #30363d;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #30363d;
                border-radius: 4px;
                background: #0d1117;
            }
            QCheckBox::indicator:checked {
                background: #3b82f6;
                border-color: #3b82f6;
            }
            QCheckBox:checked {
                color: #3b82f6;
                font-weight: bold;
            }
        """)
        
        self.ma200_cb.setText("MA200")
        self.ma200_cb.setStyleSheet("""
            QCheckBox {
                color: #c9d1d9;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QCheckBox:hover {
                background-color: #30363d;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #30363d;
                border-radius: 4px;
                background: #0d1117;
            }
            QCheckBox::indicator:checked {
                background: #f43f5e;
                border-color: #f43f5e;
            }
            QCheckBox:checked {
                color: #f43f5e;
                font-weight: bold;
            }
        """)
        
        ma_layout.addWidget(self.ma20_cb)
        ma_layout.addWidget(self.ma50_cb)
        ma_layout.addWidget(self.ma200_cb)
        
        # Create a container for other indicators
        other_container = QFrame()
        other_container.setStyleSheet("""
            QFrame {
                background-color: #21262d;
                border-radius: 8px;
                padding: 2px;
            }
        """)
        other_layout = QHBoxLayout(other_container)
        other_layout.setSpacing(2)
        other_layout.setContentsMargins(4, 2, 4, 2)
        
        # Add icons to other indicators
        self.rsi_cb.setText("RSI")
        self.rsi_cb.setStyleSheet("""
            QCheckBox {
                color: #c9d1d9;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QCheckBox:hover {
                background-color: #30363d;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #30363d;
                border-radius: 4px;
                background: #0d1117;
            }
            QCheckBox::indicator:checked {
                background: #9b59b6;
                border-color: #9b59b6;
            }
            QCheckBox:checked {
                color: #9b59b6;
                font-weight: bold;
            }
        """)
        
        self.macd_cb.setText("MACD")
        self.macd_cb.setStyleSheet("""
            QCheckBox {
                color: #c9d1d9;
                padding: 4px 8px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 500;
            }
            QCheckBox:hover {
                background-color: #30363d;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #30363d;
                border-radius: 4px;
                background: #0d1117;
            }
            QCheckBox::indicator:checked {
                background: #22c55e;
                border-color: #22c55e;
            }
            QCheckBox:checked {
                color: #22c55e;
                font-weight: bold;
            }
        """)
        
        other_layout.addWidget(self.rsi_cb)
        other_layout.addWidget(self.macd_cb)
        
        # Add containers to controls layout
        controls_layout.addWidget(ma_container)
        controls_layout.addWidget(other_container)
        controls_layout.addStretch()
        
        main_layout.addWidget(controls_frame)
        
        # Create chart container with modern styling
        chart_container = QFrame()
        chart_container.setObjectName("container")
        chart_container.setStyleSheet("""
            QFrame#container {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 12px;
            }
        """)
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        chart_layout.setSpacing(0)
        
        # Setup matplotlib figure with dynamic size
        self.figure = Figure(facecolor='#161b22')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        chart_layout.addWidget(self.canvas)
        
        main_layout.addWidget(chart_container, stretch=1)
        
        # Store the current data
        self.current_data = None
        self.current_symbol = None
        
        # Initialize subplot references
        self.price_ax = None
        self.volume_ax = None
        self.rsi_ax = None
        self.macd_ax = None
        
        # Get initial settings
        self.chart_type = self.settings.get_setting("Chart Settings", "chart_type")
        self.show_volume = self.settings.get_setting("Chart Settings", "show_volume")
        
        # Setup initial plot
        self.setup_plot()
    
    def setup_plot(self):
        """Setup the plot with potential for multiple subplots"""
        self.figure.clear()
        
        # Create GridSpec with better proportions for indicator visibility
        if self.rsi_cb.isChecked() and self.macd_cb.isChecked():
            # Both RSI and MACD are shown - give more space to indicators
            gs = self.figure.add_gridspec(4, 1, height_ratios=[4, 1, 1.5, 1.5], hspace=0.2)
        elif self.rsi_cb.isChecked():
            # Only RSI is shown
            gs = self.figure.add_gridspec(3, 1, height_ratios=[5, 1, 2], hspace=0.2)
        elif self.macd_cb.isChecked():
            # Only MACD is shown
            gs = self.figure.add_gridspec(3, 1, height_ratios=[5, 1, 2], hspace=0.2)
        else:
            # No indicators shown, just price and volume
            gs = self.figure.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.2)
        
        # Create and style subplots
        self.price_ax = self.figure.add_subplot(gs[0])
        self.volume_ax = self.figure.add_subplot(gs[1], sharex=self.price_ax)
        
        if self.rsi_cb.isChecked():
            self.rsi_ax = self.figure.add_subplot(gs[2], sharex=self.price_ax)
            self.macd_ax = None if not self.macd_cb.isChecked() else self.figure.add_subplot(gs[3], sharex=self.price_ax)
        elif self.macd_cb.isChecked():
            self.macd_ax = self.figure.add_subplot(gs[2], sharex=self.price_ax)
            self.rsi_ax = None
        else:
            self.rsi_ax = None
            self.macd_ax = None
        
        # Style all subplots with better visibility
        for ax in [self.price_ax, self.volume_ax, self.rsi_ax, self.macd_ax]:
            if ax is not None:
                ax.grid(True, linestyle='--', alpha=0.1, color='#8b949e')
                ax.set_facecolor('#0d1117')
                ax.tick_params(colors='#8b949e', labelsize=9)  # Smaller font size
                # Increase y-axis width for better visibility of numbers
                ax.yaxis.set_tick_params(pad=5)
                for spine in ax.spines.values():
                    spine.set_color('#30363d')
                    spine.set_linewidth(1)
        
        # Explicitly set margins for better spacing
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        self.canvas.draw()
    
    def filter_trading_days(self, df):
        """Filter out non-trading days"""
        if df is None or df.empty:
            return df
            
        # Convert index to datetime if it's not already
        df.index = pd.to_datetime(df.index)
        
        # Keep only days with trading activity (Volume > 0)
        df = df[df['Volume'] > 0]
        
        # Sort chronologically
        df = df.sort_index()
        
        return df

    def update_chart(self, df, symbol):
        """Update the chart with new data"""
        if df is None or df.empty:
            print("Cannot update chart: DataFrame is empty or None")
            self.display_error_message("No data available")
            return
        
        try:
            # Ensure we have a copy of the data
            df = df.copy()
            
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    print(f"Error converting index to datetime: {e}")
                    self.display_error_message("Invalid date format in data")
                    return
            
            # Filter out non-trading days
            df = self.filter_trading_days(df)
            
            if df.empty:
                print(f"No trading data available for {symbol}")
                self.display_error_message(f"No trading data available for {symbol}")
                return
            
            # Convert price columns to numeric, replacing errors with NaN
            price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Store current data
            self.current_data = df
            self.current_symbol = symbol
            
            # Clear the figure and setup plot
            self.figure.clear()
            self.setup_plot()
            
            # Get chart type from instance or settings
            chart_type = self.chart_type or self.settings.get_setting("Chart Settings", "chart_type") or "Line"
            print(f"Rendering chart type: {chart_type}")
            
            # Plot main chart
            try:
                if chart_type == "Candlestick":
                    # Check required columns for candlestick
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    if all(col in df.columns for col in required_cols):
                        self.plot_candlesticks(df)
                    else:
                        print("Missing required columns for candlestick chart")
                        self.plot_line(df)  # Fallback to line chart
                elif chart_type == "OHLC":
                    # Check required columns for OHLC
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    if all(col in df.columns for col in required_cols):
                        self.plot_ohlc(df)
                    else:
                        print("Missing required columns for OHLC chart")
                        self.plot_line(df)  # Fallback to line chart
                else:  # Default to line chart
                    self.plot_line(df)
                    
                # Plot additional components if we have valid data
                if not df.empty and 'Close' in df.columns:
                    self.plot_moving_averages(df)
                    if self.rsi_cb.isChecked():
                        self.plot_rsi(df)
                    if self.macd_cb.isChecked():
                        self.plot_macd(df)
                    if self.show_volume and 'Volume' in df.columns:
                        self.plot_volume(df)
                    
                # Update layout and labels
                self.update_chart_labels(symbol, chart_type)
                self.canvas.draw()
                
            except Exception as e:
                print(f"Error plotting chart: {e}")
                import traceback
                traceback.print_exc()
                self.display_error_message(f"Error plotting chart: {str(e)}")
                
        except Exception as e:
            print(f"Error updating chart: {e}")
            import traceback
            traceback.print_exc()
            self.display_error_message(f"Error updating chart: {str(e)}")

    def display_error_message(self, message):
        """Display an error message on the chart"""
        self.setup_plot()
        self.price_ax.text(0.5, 0.5, message,
                          horizontalalignment='center',
                          verticalalignment='center',
                          color='#f87171',
                          transform=self.price_ax.transAxes)
        self.canvas.draw()

    def update_chart_labels(self, symbol, chart_type):
        """Update chart labels and formatting"""
        # Set labels
        self.price_ax.set_xlabel("")
        self.price_ax.set_ylabel("Price ($)", color='#8b949e', labelpad=5)
        
        if self.volume_ax:
            self.volume_ax.set_xlabel("Date", color='#8b949e', labelpad=5)
            
        # Format x-axis dates
        n_points = len(self.current_data)
        step = max(n_points // 8, 1)  # Show about 8 dates
        x = range(n_points)
        
        for ax in [self.price_ax, self.volume_ax, self.rsi_ax, self.macd_ax]:
            if ax is not None:
                ax.set_xticks(x[::step])
                ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in self.current_data.index[::step]], 
                                 rotation=45, ha='right')
        
        # Format dates
        self.figure.autofmt_xdate()
        
        # Add title
        self.price_ax.set_title(f"{symbol} - {chart_type} Chart", 
                             fontsize=12, color='#e6edf3',
                             fontweight='bold', pad=10)
        
        # Update layout
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    def plot_candlesticks(self, df):
        """Plot candlestick chart"""
        try:
            if df is None or df.empty:
                print("Cannot plot candlesticks: Empty dataframe")
                return

            # Check if we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_columns):
                print(f"Cannot plot candlesticks: Missing required columns. Available columns: {df.columns.tolist()}")
                raise ValueError("Missing required OHLC columns")
                
            # Define colors
            up_color = '#4ade80'  # Green for up days
            down_color = '#f87171'  # Red for down days
            
            # Ensure data is numeric and handle any NaN values
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values to prevent crashes
            df = df.dropna(subset=required_columns)
            
            if df.empty:
                raise ValueError("No valid data points after cleaning")
            
            # Create figure-level data structures for better performance
            x = np.arange(len(df))
            prices = df[['High', 'Low']].values.flatten()
            price_range = np.ptp(prices)  # Peak to peak (max - min)
            
            # Set price range with padding
            y_min = np.min(prices) - price_range * 0.05
            y_max = np.max(prices) + price_range * 0.05
            self.price_ax.set_ylim(y_min, y_max)
            
            # Plot candlesticks more efficiently
            for i, (idx, row) in enumerate(df.iterrows()):
                try:
                    open_price = float(row['Open'])
                    close_price = float(row['Close'])
                    high_price = float(row['High'])
                    low_price = float(row['Low'])
                    
                    # Skip if any price is invalid
                    if any(np.isnan([open_price, close_price, high_price, low_price])):
                        continue
                    
                    # Determine color based on price movement
                    color = up_color if close_price >= open_price else down_color
                    
                    # Plot candlestick body
                    body_bottom = min(open_price, close_price)
                    body_height = abs(close_price - open_price)
                    
                    # Only create rectangle if there's a valid height
                    if body_height > 0:
                        rect = Rectangle(
                            (i - 0.4, body_bottom),
                            0.8,  # Width of the candlestick
                            body_height,
                            facecolor=color,
                            edgecolor=color,
                            alpha=0.8,
                            antialiased=True
                        )
                        self.price_ax.add_patch(rect)
                    
                    # Plot wicks
                    self.price_ax.vlines(
                        i, low_price, high_price,
                        color=color,
                        linewidth=1,
                        alpha=0.8
                    )
                    
                except (ValueError, TypeError) as e:
                    print(f"Error plotting candlestick at index {i}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error in plot_candlesticks: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to trigger fallback to line chart

    def plot_line(self, df):
        """Plot line chart"""
        try:
            if 'Close' not in df.columns:
                raise ValueError("Missing 'Close' column for line chart")
                
            # Remove any NaN values to prevent plotting errors
            df = df.dropna(subset=['Close'])
            
            if df.empty:
                raise ValueError("No valid data points after cleaning")
            
            # Plot the line
            self.price_ax.plot(range(len(df)), df['Close'], 
                             color='#3b82f6', linewidth=2, 
                             label='Price')
            
            # Set price range with some padding
            prices = df['Close'].values
            price_range = np.ptp(prices)  # Peak to peak (max - min)
            y_min = np.min(prices) - price_range * 0.05
            y_max = np.max(prices) + price_range * 0.05
            self.price_ax.set_ylim(y_min, y_max)
            
        except Exception as e:
            print(f"Error plotting line chart: {e}")
            raise

    def plot_ohlc(self, df):
        """Plot OHLC chart"""
        try:
            up_color = '#4ade80'  # Green for up days
            down_color = '#f87171'  # Red for down days
            
            for i, (idx, row) in enumerate(df.iterrows()):
                open_price = row['Open']
                close_price = row['Close']
                high_price = row['High']
                low_price = row['Low']
                
                # Determine color based on price movement
                color = up_color if close_price >= open_price else down_color
                
                # Plot vertical line (high to low)
                self.price_ax.plot(
                    [i, i],
                    [low_price, high_price],
                    color=color,
                    linewidth=1.5
                )
                
                # Plot open tick
                self.price_ax.plot(
                    [i - 0.2, i],
                    [open_price, open_price],
                    color=color,
                    linewidth=1.5
                )
                
                # Plot close tick
                self.price_ax.plot(
                    [i, i + 0.2],
                    [close_price, close_price],
                    color=color,
                    linewidth=1.5
                )
            
            # Set price range with some padding
            prices = df[['High', 'Low']].values.flatten()
            price_range = max(prices) - min(prices)
            self.price_ax.set_ylim(
                min(prices) - price_range * 0.05,
                max(prices) + price_range * 0.05
            )
            
        except Exception as e:
            print(f"Error plotting OHLC chart: {e}")
            raise
    
    def plot_moving_averages(self, df):
        """Plot selected moving averages"""
        x = range(len(df))
        
        ma_settings = [
            (self.ma20_cb, 20, '#fb923c'),  # Orange
            (self.ma50_cb, 50, '#3b82f6'),  # Blue
            (self.ma200_cb, 200, '#f43f5e')  # Red
        ]
        
        for checkbox, period, color in ma_settings:
            if checkbox.isChecked():
                ma = calculate_ma(df, period)
                self.price_ax.plot(
                    x,
                    ma,
                    color=color,
                    linewidth=1.2,  # Slightly thicker lines
                    label=f'MA{period}',
                    alpha=0.9  # More visible
                )
        
        # Add a legend with improved visibility if any MA is enabled
        if any(checkbox.isChecked() for checkbox, _, _ in ma_settings):
            legend = self.price_ax.legend(
                loc='upper left',
                frameon=True,
                facecolor='#161b2299',  # Subtle background
                edgecolor='#30363d',
                fontsize=8,
                ncol=3,  # Display in one row
                framealpha=0.8,
                labelspacing=0.2,
                handlelength=1.0,
                handletextpad=0.5
            )
            # Improve legend text visibility
            for text in legend.get_texts():
                text.set_color('#e6edf3')
    
    def plot_rsi(self, df):
        """Plot RSI indicator"""
        if self.rsi_cb.isChecked() and self.rsi_ax is not None:
            try:
                x = range(len(df))
                rsi = calculate_rsi(df)
                
                # Plot RSI line with thicker line
                self.rsi_ax.plot(x, rsi, color='#9b59b6', linewidth=1.8, zorder=2)
                
                # Add overbought/oversold areas with improved visibility
                self.rsi_ax.axhspan(70, 100, color='#f4394f15', zorder=1)
                self.rsi_ax.axhspan(0, 30, color='#4ade8015', zorder=1)
                
                # Add reference lines with improved visibility
                self.rsi_ax.axhline(y=70, color='#f4394f60', linestyle='--', linewidth=1, zorder=1)
                self.rsi_ax.axhline(y=30, color='#4ade8060', linestyle='--', linewidth=1, zorder=1)
                self.rsi_ax.axhline(y=50, color='#8b949e30', linestyle='-', linewidth=1, zorder=1)
                
                # Add label above the indicator for better identification
                self.rsi_ax.text(0.02, 0.95, "RSI (14)", transform=self.rsi_ax.transAxes,
                               fontsize=9, fontweight='bold', color='#9b59b6', va='top')
                
                self.rsi_ax.set_ylim(0, 100)
                self.rsi_ax.set_ylabel('RSI', color='#8b949e', fontsize=9)
                
                # Use less frequent y-axis ticks for clarity
                self.rsi_ax.set_yticks([0, 30, 50, 70, 100])
                
                # Remove x-axis labels if not the last subplot
                if self.macd_cb.isChecked():
                    self.rsi_ax.set_xticklabels([])
                
            except Exception as e:
                print(f"Error plotting RSI: {e}")
                raise

    def plot_macd(self, df):
        """Plot MACD indicator"""
        if self.macd_cb.isChecked() and self.macd_ax is not None:
            try:
                x = range(len(df))
                macd_line, signal_line, macd_hist = calculate_macd(df)
                
                # Plot histogram with improved visibility
                pos_hist = macd_hist.copy()
                neg_hist = macd_hist.copy()
                pos_hist[pos_hist <= 0] = np.nan
                neg_hist[neg_hist > 0] = np.nan
                
                # Use wider bars for better visibility
                bar_width = 0.7
                
                # Add gradient effect to histogram bars with better visibility
                self.macd_ax.bar(x, pos_hist, color='#4ade80a0', width=bar_width, zorder=2)
                self.macd_ax.bar(x, neg_hist, color='#f87171a0', width=bar_width, zorder=2)
                
                # Plot lines with improved visibility
                self.macd_ax.plot(x, macd_line, color='#3b82f6', linewidth=1.8, 
                               label='MACD', zorder=3)
                self.macd_ax.plot(x, signal_line, color='#f43f5e', linewidth=1.8, 
                               label='Signal', zorder=3)
                
                # Add zero line with better visibility
                self.macd_ax.axhline(y=0, color='#8b949e40', linestyle='-', linewidth=1, zorder=1)
                
                # Add label above the indicator for better identification
                self.macd_ax.text(0.02, 0.95, "MACD (12,26,9)", transform=self.macd_ax.transAxes,
                               fontsize=9, fontweight='bold', color='#22c55e', va='top')
                
                # Legend with smaller size and better positioning
                self.macd_ax.legend(loc='upper right', frameon=False, fontsize=8, 
                                 ncol=2, handlelength=1.5, handletextpad=0.5)
                
                self.macd_ax.set_ylabel('MACD', color='#8b949e', fontsize=9)
                
            except Exception as e:
                print(f"Error plotting MACD: {e}")
                raise
    
    def plot_volume(self, df):
        """Plot volume bars"""
        if not self.volume_ax:
            return
        
        try:
            # Calculate colors for volume bars
            colors = ['#4ade8080' if close >= open else '#f8717180' 
                     for close, open in zip(df['Close'], df['Open'])]
            
            # Plot volume bars with better visibility
            self.volume_ax.bar(
                range(len(df)),
                df['Volume'],
                color=colors,
                width=0.7,  # Slightly narrower for clarity
                zorder=2,  # Place bars above grid
                linewidth=0
            )
            
            # Format volume numbers with K/M/B and better spacing
            self.volume_ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, p: format_volume(x))
            )
            
            # Show only a few y-axis labels to avoid crowding
            max_volume = df['Volume'].max()
            self.volume_ax.set_yticks([0, max_volume/2, max_volume])
            
            # Label the axis more clearly
            self.volume_ax.set_ylabel('Vol', color='#8b949e', fontsize=9, labelpad=3)
            
            # Remove volume x-axis labels (since they're shown in price chart)
            self.volume_ax.set_xticklabels([])
            
        except Exception as e:
            print(f"Error plotting volume: {e}")
            raise

    def update_indicator_setting(self, indicator, state):
        """Update the indicator setting and redraw chart"""
        if indicator == "RSI":
            self.rsi_cb.setChecked(state)
        elif indicator == "MACD":
            self.macd_cb.setChecked(state)
        
        # Redraw chart if we have data
        if self.current_data is not None:
            self.update_chart(self.current_data, self.current_symbol)
    
    def update_moving_averages(self, ma_settings):
        """Update the moving average settings and redraw chart"""
        # Update checkbox state based on settings
        for ma_name in ma_settings:
            if ma_name == "MA50":
                self.ma50_cb.setChecked(ma_settings[ma_name])
            elif ma_name == "MA100":
                self.ma100_cb.setChecked(ma_settings[ma_name])
            elif ma_name == "MA200":
                self.ma200_cb.setChecked(ma_settings[ma_name])
        
        # Redraw chart if we have data
        if self.current_data is not None:
            self.update_chart(self.current_data, self.current_symbol)
    
    def set_chart_type(self, chart_type):
        """Set the chart type and update the chart"""
        valid_types = ["Candlestick", "Line", "OHLC"]
        if chart_type not in valid_types:
            print(f"Invalid chart type: {chart_type}. Using Line chart.")
            chart_type = "Line"
            
        print(f"Setting chart type to: {chart_type}")
        
        # Save the setting first
        try:
            self.settings.update_setting("Chart Settings", "chart_type", chart_type)
        except Exception as e:
            print(f"Error saving chart type setting: {e}")
            # Continue anyway since we can still update the local chart
        
        # Update instance variable
        self.chart_type = chart_type
        
        # Update the chart if we have data
        if self.current_data is not None:
            try:
                # Make a copy of the data to prevent modifications
                df_copy = self.current_data.copy()
                
                # Ensure the data has the required columns for candlestick/OHLC
                if chart_type in ["Candlestick", "OHLC"]:
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    missing_cols = [col for col in required_cols if col not in df_copy.columns]
                    if missing_cols:
                        print(f"Missing required columns for {chart_type} chart: {missing_cols}")
                        print("Falling back to Line chart")
                        return self.set_chart_type("Line")
                
                self.update_chart(df_copy, self.current_symbol)
                print(f"Chart updated successfully to type: {chart_type}")
            except Exception as e:
                print(f"Error updating chart to {chart_type}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to line chart if not already trying line chart
                if chart_type != "Line":
                    print("Falling back to Line chart")
                    self.set_chart_type("Line")
    
    def toggle_volume(self, show_volume):
        """Toggle volume display and update chart"""
        self.show_volume = show_volume
        if self.current_data is not None:
            self.update_chart(self.current_data, self.current_symbol)

def format_volume(value):
    """Format volume numbers with K/M/B suffixes"""
    if value >= 1e9:
        return f'{value/1e9:.1f}B'
    elif value >= 1e6:
        return f'{value/1e6:.1f}M'
    elif value >= 1e3:
        return f'{value/1e3:.1f}K'
    return str(int(value)) 