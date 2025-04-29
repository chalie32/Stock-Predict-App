from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTabWidget, QFrame, QSizePolicy, QPushButton,
    QComboBox, QLineEdit, QGroupBox, QGridLayout, QSpacerItem,
    QMessageBox, QStackedWidget, QListWidget, QListWidgetItem,
    QProgressBar, QScrollArea, QSpinBox, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon
from stock_predictor_app.data.stock_fetcher import get_stock_data, get_company_info, StockDataError
from stock_predictor_app.ui.chart_widget import ChartWidget
from stock_predictor_app.ui.tabs import CompanyInfoTab, NewsTab
from stock_predictor_app.ui.prediction_worker import PredictionWorker
from datetime import datetime, timedelta
import pandas as pd
from stock_predictor_app.data.settings import Settings
from stock_predictor_app.utils.prediction_record import PredictionRecord
import numpy as np
import time

STYLE_SHEET = """
    /* Global Styles */
    QMainWindow {
        background-color: #0d1117;
    }
    
    /* Left Menu Styling */
    QFrame#leftMenu {
        background-color: #161b22;
        border-right: 1px solid #30363d;
        padding: 0;
        margin: 0;
        min-width: 250px;
        max-width: 250px;
    }
    
    QPushButton#menuButton {
        text-align: left;
        padding: 14px 24px;
        border: none;
        border-radius: 12px;
        margin: 6px 12px;
        color: #8b949e;
        background-color: rgba(255, 255, 255, 0.03);
        font-size: 14px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    QPushButton#menuButton:hover {
        background-color: rgba(255, 255, 255, 0.08);
        color: #ffffff;
        transform: translateX(2px);
    }
    
    QPushButton#menuButton:checked {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #1f6feb,
                                  stop:1 #2ea043);
        color: #ffffff;
        font-weight: 600;
    }
    
    QFrame#menuTitleContainer {
        background-color: transparent;
        border: none;
        margin: 0;
        padding: 24px 20px;
    }
    
    QLabel#menuTitle {
        color: #ffffff;
        font-size: 22px;
        font-weight: bold;
        background-color: transparent;
        border: none;
        margin: 0;
        padding: 0;
    }
    
    /* Main Content Styling */
    QFrame#mainContent {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 16px;
    }
    
    /* Dashboard Cards */
    QFrame#dashboardCard {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                  stop:0 #161b22,
                                  stop:1 #1a1f29);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 28px;
    }
    
    QFrame#dashboardCard:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                  stop:0 #1c2129,
                                  stop:1 #1f242e);
        border: 1px solid #388bfd;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    QLabel#cardTitle {
        color: #c9d1d9;
        font-size: 16px;
        font-weight: bold;
        letter-spacing: 0.3px;
    }
    
    QLabel#cardValue {
        color: #1f6feb;
        font-size: 28px;
        font-weight: bold;
        letter-spacing: -0.5px;
    }
    
    QLabel#cardTrend {
        color: #2ea043;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    QLabel#cardTrendDown {
        color: #f85149;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    /* Form Controls */
    QLineEdit {
        padding: 14px;
        border: 1px solid #30363d;
        border-radius: 12px;
        background: rgba(22, 27, 34, 0.7);
        color: #c9d1d9;
        font-size: 14px;
        letter-spacing: 0.3px;
    }
    
    QLineEdit:focus {
        border: 2px solid #1f6feb;
        background: rgba(31, 111, 235, 0.1);
        padding: 13px;
    }
    
    QComboBox {
        padding: 13px;
        border: 1px solid #30363d;
        border-radius: 12px;
        background: rgba(22, 27, 34, 0.7);
        color: #c9d1d9;
        min-width: 150px;
        font-size: 14px;
        letter-spacing: 0.3px;
    }
    
    QComboBox:hover {
        border: 1px solid #1f6feb;
        background: rgba(31, 111, 235, 0.1);
    }
    
    QComboBox::drop-down {
        border: none;
        width: 24px;
    }
    
    QComboBox::down-arrow {
        image: url(resources/down-arrow-light.png);
        width: 12px;
        height: 12px;
    }
    
    QComboBox QAbstractItemView {
        border: 1px solid #30363d;
        border-radius: 12px;
        background: #161b22;
        selection-background-color: #1f6feb;
        selection-color: #ffffff;
        padding: 4px;
    }
    
    QComboBox QAbstractItemView::item {
        padding: 12px;
        min-height: 24px;
        color: #c9d1d9;
        border-radius: 8px;
    }
    
    QComboBox QAbstractItemView::item:hover {
        background-color: rgba(31, 111, 235, 0.1);
        color: #ffffff;
    }
    
    QSpinBox {
        padding: 12px;
        border: 1px solid #30363d;
        border-radius: 12px;
        background: rgba(22, 27, 34, 0.7);
        color: #c9d1d9;
        min-width: 100px;
    }
    
    QSpinBox::up-button, QSpinBox::down-button {
        width: 24px;
        background: rgba(33, 38, 45, 0.8);
        border-radius: 6px;
        margin: 4px;
    }
    
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background: #30363d;
    }
    
    /* Buttons */
    QPushButton {
        padding: 14px 28px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 14px;
        color: #ffffff;
        border: none;
        letter-spacing: 0.3px;
    }
    
    QPushButton:disabled {
        background-color: #30363d;
        color: #8b949e;
        opacity: 0.7;
    }
    
    QPushButton#searchButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #1f6feb,
                                  stop:1 #2ea043);
        min-width: 120px;
    }
    
    QPushButton#searchButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #388bfd,
                                  stop:1 #3fb950);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    
    QPushButton#predictButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #238636,
                                  stop:1 #2ea043);
        min-width: 120px;
    }
    
    QPushButton#predictButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #2ea043,
                                  stop:1 #3fb950);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    
    QPushButton#deleteButton {
        background: rgba(218, 54, 51, 0.8);
        min-width: 36px;
        max-width: 36px;
        min-height: 36px;
        max-height: 36px;
        padding: 8px;
        border-radius: 18px;
    }
    
    QPushButton#deleteButton:hover {
        background-color: #f85149;
        transform: scale(1.1);
        transition: all 0.2s ease;
    }
    
    QPushButton#heroButton {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #1f6feb,
                                  stop:1 #2ea043);
        color: white;
        border: none;
        padding: 16px 32px;
        font-size: 16px;
        letter-spacing: 0.5px;
    }
    
    QPushButton#heroButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #388bfd,
                                  stop:1 #3fb950);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    QPushButton#secondaryButton {
        background-color: rgba(33, 38, 45, 0.8);
        color: #c9d1d9;
        border: 1px solid #30363d;
        font-weight: 600;
        padding: 10px 16px;
    }
    
    QPushButton#secondaryButton:hover {
        background-color: rgba(48, 54, 61, 0.8);
        border-color: #8b949e;
    }
    
    /* Tabs */
    QTabWidget::pane {
        border: none;
        background: transparent;
    }
    
    QTabWidget::tab-bar {
        alignment: left;
    }
    
    QTabBar::tab {
        padding: 12px 24px;
        margin-right: 4px;
        color: #8b949e;
        border-bottom: 2px solid transparent;
        background: transparent;
        font-size: 14px;
        letter-spacing: 0.3px;
    }
    
    QTabBar::tab:selected {
        color: #1f6feb;
        border-bottom: 2px solid #1f6feb;
        font-weight: 600;
    }
    
    QTabBar::tab:hover:!selected {
        color: #c9d1d9;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0 0;
    }
    
    /* Scrollbars */
    QScrollArea {
        border: none;
        background: transparent;
    }
    
    QScrollBar:vertical {
        border: none;
        background: rgba(13, 17, 23, 0.8);
        width: 8px;
        margin: 0px;
    }
    
    QScrollBar::handle:vertical {
        background: rgba(48, 54, 61, 0.8);
        border-radius: 4px;
        min-height: 30px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: #388bfd;
    }
    
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    QScrollBar:horizontal {
        border: none;
        background: rgba(13, 17, 23, 0.8);
        height: 8px;
        margin: 0px;
    }
    
    QScrollBar::handle:horizontal {
        background: rgba(48, 54, 61, 0.8);
        border-radius: 4px;
        min-width: 30px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background: #388bfd;
    }
    
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    
    /* Progress Bars */
    QProgressBar {
        border: none;
        border-radius: 6px;
        background-color: rgba(33, 38, 45, 0.8);
        text-align: center;
        color: #ffffff;
        font-weight: 600;
        height: 8px;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                  stop:0 #1f6feb,
                                  stop:1 #2ea043);
        border-radius: 6px;
    }
"""

class MenuButton(QPushButton):
    def __init__(self, text, icon_text, parent=None):
        super().__init__(parent)
        self.setText(f"{icon_text} {text}")
        self.setObjectName("menuButton")
        self.setCheckable(True)
        self.setAutoExclusive(True)

class DashboardCard(QFrame):
    def __init__(self, title, value, trend=None, parent=None):
        super().__init__(parent)
        self.setObjectName("dashboardCard")
        layout = QVBoxLayout(self)
        
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setObjectName("cardValue")
        layout.addWidget(value_label)
        
        if trend:
            trend_label = QLabel(trend)
            trend_label.setObjectName("cardTrend" if not trend.startswith("-") else "cardTrendDown")
            layout.addWidget(trend_label)

class MainWindow(QMainWindow):
    def __init__(self):
        """Initialize the main window"""
        try:
            super().__init__()
            print("Initializing MainWindow...")
            
            self.setWindowTitle("Stock Market Analysis")
            self.setMinimumSize(1800, 1000)
            self.setStyleSheet(STYLE_SHEET)
            
            print("Setting up instance variables...")
            # Store widgets we need to access later
            self.symbol_input = None
            self.timerange_combo = None
            self.chart_widget = None
            self.company_info_tab = None
            self.stacked_widget = None
            
            # Cache for stock data
            self.current_symbol = None
            self.cached_data = None
            self.cached_info = None
            
            print("Initializing prediction record manager...")
            # Initialize prediction record manager
            self.prediction_record = PredictionRecord()
            
            print("Setting up UI...")
            self.setup_ui()
            
            print("MainWindow initialization complete.")
        except Exception as e:
            print(f"Error during MainWindow initialization: {e}")
            raise  # Re-raise the exception for proper error handling
    
    def setup_ui(self):
        """Set up the main user interface"""
        try:
            print("Setting up central widget...")
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QHBoxLayout(central_widget)
            main_layout.setSpacing(20)
            main_layout.setContentsMargins(0, 0, 20, 20)
            
            print("Setting up left menu...")
            # Left Menu setup...
            self.setup_left_menu(main_layout)
            
            print("Setting up center area...")
            # Center area setup...
            self.setup_center_area(main_layout)
            
            print("Setting up right panel...")
            # Right panel setup...
            self.setup_right_panel(main_layout)
            
            print("Setting default page...")
            # Set default page
            self.home_btn.setChecked(True)
            self.stacked_widget.setCurrentIndex(0)
            
            print("UI setup complete.")
        except Exception as e:
            print(f"Error during UI setup: {e}")
            raise  # Re-raise the exception for proper error handling
    
    def setup_left_menu(self, main_layout):
        """Set up the left menu"""
        try:
            left_menu = QFrame()
            left_menu.setObjectName("leftMenu")
            left_menu_layout = QVBoxLayout(left_menu)
            left_menu_layout.setContentsMargins(0, 0, 0, 20)
            left_menu_layout.setSpacing(5)
            
            # Menu title container
            title_container = QFrame()
            title_container.setObjectName("menuTitleContainer")
            title_layout = QVBoxLayout(title_container)
            title_layout.setContentsMargins(0, 0, 0, 0)
            
            # Menu title
            menu_title = QLabel("Stock Predictor")
            menu_title.setObjectName("menuTitle")
            title_layout.addWidget(menu_title)
            
            left_menu_layout.addWidget(title_container)
            
            # Menu buttons
            self.home_btn = MenuButton("Home", "🏠")
            self.analysis_btn = MenuButton("Analysis", "📊")
            self.records_btn = MenuButton("Records", "📝")
            self.settings_btn = MenuButton("Settings", "⚙️")
            self.info_btn = MenuButton("Info", "ℹ️")
            
            # Connect button signals with direct method calls to ensure they work
            self.home_btn.clicked.connect(lambda: self.switch_page(0))
            self.analysis_btn.clicked.connect(lambda: self.switch_page(1))
            self.records_btn.clicked.connect(lambda: self.switch_page(2))
            self.settings_btn.clicked.connect(lambda: self.switch_page(3))
            self.info_btn.clicked.connect(lambda: self.switch_page(4))
            
            # Add buttons to menu
            left_menu_layout.addWidget(self.home_btn)
            left_menu_layout.addWidget(self.analysis_btn)
            left_menu_layout.addWidget(self.records_btn)
            left_menu_layout.addWidget(self.settings_btn)
            left_menu_layout.addWidget(self.info_btn)
            left_menu_layout.addStretch()
            
            main_layout.addWidget(left_menu)
            
            # Make sure Records button is visible
            self.records_btn.setVisible(True)
            print("Left menu setup complete. Records button connected.")
        except Exception as e:
            print(f"Error setting up left menu: {e}")
            raise
    
    def setup_center_area(self, main_layout):
        """Set up the center area with stacked widget"""
        try:
            center_layout = QHBoxLayout()
            
            # Stacked Widget for different pages
            self.stacked_widget = QStackedWidget()
            
            print("Creating home page...")
            self.setup_home_page()
            print(f"Home page added at index {self.stacked_widget.count()-1}")
            
            print("Creating analysis page...")
            self.setup_analysis_page()
            print(f"Analysis page added at index {self.stacked_widget.count()-1}")
            
            print("Creating records page...")
            self.setup_records_page()
            print(f"Records page added at index {self.stacked_widget.count()-1}")
            
            print("Creating settings page...")
            self.setup_settings_page()
            print(f"Settings page added at index {self.stacked_widget.count()-1}")
            
            print("Creating info page...")
            self.setup_info_page()
            print(f"Info page added at index {self.stacked_widget.count()-1}")
            
            # Verify correct page order
            print(f"Total pages in stacked widget: {self.stacked_widget.count()}")
            print(f"Index of current page: {self.stacked_widget.currentIndex()}")
            
            center_layout.addWidget(self.stacked_widget, stretch=1)
            main_layout.addLayout(center_layout)
            
            # Ensure all pages are accessible
            for i in range(self.stacked_widget.count()):
                widget = self.stacked_widget.widget(i)
                if widget:
                    print(f"Page {i} exists: {widget.__class__.__name__}")
                else:
                    print(f"Page {i} is missing!")
                
        except Exception as e:
            print(f"Error setting up center area: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def setup_right_panel(self, main_layout):
        """Set up the right panel"""
        try:
            right_panel = QFrame()
            right_panel.setObjectName("rightPanel")
            right_panel.setMinimumWidth(400)
            right_panel.setMaximumWidth(500)
            right_layout = QVBoxLayout(right_panel)
            right_layout.setContentsMargins(15, 15, 15, 15)
            right_layout.setSpacing(15)
            
            # Company Info and News Tabs
            info_tabs = QTabWidget()
            
            # Company Info tab
            self.company_info_tab = CompanyInfoTab()
            info_tabs.addTab(self.company_info_tab, "📊 Company Info")
            
            # News tab
            self.news_tab = NewsTab()
            info_tabs.addTab(self.news_tab, "📰 News")
            
            right_layout.addWidget(info_tabs)
            main_layout.addWidget(right_panel)
        except Exception as e:
            print(f"Error setting up right panel: {e}")
            raise
    
    def setup_home_page(self):
        """Set up the home page with dashboard elements"""
        try:
            page = QScrollArea()
            page.setWidgetResizable(True)
            page.setStyleSheet("QScrollArea { border: none; background: #0d1117; }")
            
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(24)
            
            # Welcome section with gradient background
            welcome_frame = QFrame()
            welcome_frame.setStyleSheet("""
            QFrame {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                             stop:0 #1a1f29,
                                             stop:1 #222a3a);
                    border: 1px solid #2f3b54;
                    border-radius: 16px;
                    padding: 24px;
            }
        """)
            welcome_layout = QVBoxLayout(welcome_frame)
            welcome_layout.setContentsMargins(30, 35, 30, 35)
            welcome_layout.setSpacing(16)
            
            # Welcome title
            welcome_title = QLabel("Welcome to Stock Predictor")
            welcome_title.setStyleSheet("""
                font-size: 28px;
                font-weight: bold;
                color: #e6edf3;
                letter-spacing: -0.5px;
            """)
            welcome_layout.addWidget(welcome_title)
            
            # Welcome subtitle
            welcome_subtitle = QLabel("Advanced stock market analysis and prediction platform")
            welcome_subtitle.setStyleSheet("""
                font-size: 16px;
                color: #8b949e;
                margin-bottom: 20px;
                letter-spacing: 0.2px;
            """)
            welcome_layout.addWidget(welcome_subtitle)
            
            # Get started button
            get_started_btn = QPushButton("Start Analysis")
            get_started_btn.setObjectName("heroButton")
            get_started_btn.clicked.connect(lambda: self.switch_page(1))
            welcome_layout.addWidget(get_started_btn, 0, Qt.AlignmentFlag.AlignLeft)
            
            layout.addWidget(welcome_frame)
            
            # Market Overview with improved styling
            market_title_layout = QHBoxLayout()
            
            market_title = QLabel("Market Overview")
            market_title.setStyleSheet("""
                font-size: 22px;
                font-weight: 700;
                color: #e6edf3;
                margin-top: 10px;
                letter-spacing: -0.3px;
            """)
            market_title_layout.addWidget(market_title)
            
            refresh_btn = QPushButton("↻ Refresh")
            refresh_btn.setStyleSheet("""
            QPushButton {
                    background-color: transparent;
                border: 1px solid #30363d;
                    border-radius: 8px;
                    color: #58a6ff;
                    font-size: 13px;
                font-weight: 600;
                    padding: 6px 12px;
            }
            QPushButton:hover {
                    background-color: rgba(56, 139, 253, 0.1);
                    border-color: #58a6ff;
            }
        """)
            refresh_btn.clicked.connect(self.refresh_market_overview)
            market_title_layout.addStretch()
            market_title_layout.addWidget(refresh_btn)
            
            layout.addLayout(market_title_layout)
            
            # Market cards in grid with improved styling
            cards_frame = QFrame()
            cards_frame.setStyleSheet("""
            QFrame {
                background-color: transparent;
                    border: none;
            }
        """)
            cards_layout = QGridLayout(cards_frame)
            cards_layout.setContentsMargins(0, 0, 0, 0)
            cards_layout.setSpacing(20)
            
            # Create market summary cards
            self.market_cards = {}
            
            indices = [
                {"symbol": "^DJI", "name": "Dow Jones", "value": "--", "change": "--"},
                {"symbol": "^GSPC", "name": "S&P 500", "value": "--", "change": "--"},
                {"symbol": "^IXIC", "name": "NASDAQ", "value": "--", "change": "--"},
                {"symbol": "^VIX", "name": "Volatility Index", "value": "--", "change": "--"}
            ]
            
            for i, index in enumerate(indices):
                card = self.create_market_card(index["name"], index["value"], index["change"])
                row, col = divmod(i, 2)
                cards_layout.addWidget(card, row, col)
                self.market_cards[index["symbol"]] = card
            
            layout.addWidget(cards_frame)
            
            # Do NOT automatically refresh market data on startup
            # This will be refreshed manually by the user
            
            page.setWidget(content)
            self.stacked_widget.addWidget(page)
            
        except Exception as e:
            print(f"Error setting up home page: {e}")
            raise
    
    def create_market_card(self, title, value, change):
        """Create a card for market overview"""
        card = QFrame()
        card.setObjectName("dashboardCard")
        
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        
        # Title
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        
        # Value
        value_label = QLabel(value)
        value_label.setObjectName("cardValue")
        layout.addWidget(value_label)
        
        # Change
        change_label = QLabel(change)
        if not change.startswith("-"):
            change_label.setObjectName("cardTrend")
            change_label.setText(f"↑ {change}")
        else:
            change_label.setObjectName("cardTrendDown")
            change_label.setText(f"↓ {change}")
        layout.addWidget(change_label)
        
        # Store references to the labels for later access
        card.value_label = value_label
        card.change_label = change_label
        
        return card
    
    def refresh_market_overview(self):
        """Fetch and update market overview data"""
        try:
            print("Refreshing market overview...")
            # Update UI with loading state
            for symbol, card in self.market_cards.items():
                card.value_label.setText("Loading...")
                card.change_label.setText("Please wait...")
                card.change_label.setStyleSheet("color: #58a6ff;")
                
            # Use a QTimer to process updates individually rather than all at once
            self.refresh_index = 0
            self.symbols = list(self.market_cards.keys())
            
            # Process next symbol after a delay
            QTimer.singleShot(100, self.process_next_market_card)
            
        except Exception as e:
            print(f"Error starting market refresh: {e}")
            import traceback
            print(traceback.format_exc())
            # Show error in UI
            self.update_all_cards_with_error()
    
    def process_next_market_card(self):
        """Process the next market card in the queue"""
        try:
            # Check if we have more symbols to process
            if self.refresh_index >= len(self.symbols):
                print("Market refresh complete")
                return
                
            # Get current symbol
            symbol = self.symbols[self.refresh_index]
            print(f"Processing market data for {symbol}")
            
            try:
                # Get data for this symbol
                data = get_stock_data(symbol, "2d")[0]
                if not data.empty and len(data) >= 2:
                    current_price = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2]  # Previous day's close
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    # Format the price with appropriate decimal places
                    if current_price < 10:
                        price_format = "${:.3f}"
                    elif current_price < 1000:
                        price_format = "${:.2f}"
                    else:
                        price_format = "${:,.2f}"
                    
                    # Update the UI with formatted values
                    self.market_cards[symbol].value_label.setText(price_format.format(current_price))
                    
                    # Format change text with appropriate decimal places
                    if abs(change) < 0.01:
                        change_format = "{:.4f}"
                    else:
                        change_format = "{:.2f}"
                    
                    change_text = f"{'↑' if change >= 0 else '↓'} {change_format.format(abs(change))} ({abs(change_pct):.2f}%)"
                    self.market_cards[symbol].change_label.setText(change_text)
                    self.market_cards[symbol].change_label.setStyleSheet(
                        f"color: {'#4ade80' if change >= 0 else '#f87171'}; font-weight: bold;"
                    )
                    
                else:
                    # Display error message if no data is available
                    self.market_cards[symbol].value_label.setText("--")
                    self.market_cards[symbol].change_label.setText("Data unavailable")
                    self.market_cards[symbol].change_label.setStyleSheet("color: #8b949e;")
            except Exception as e:
                print(f"Error refreshing {symbol}: {e}")
                self.market_cards[symbol].value_label.setText("Error")
                self.market_cards[symbol].change_label.setText("Could not fetch data")
                self.market_cards[symbol].change_label.setStyleSheet("color: #f87171;")
                
            # Increment index and process next symbol after a delay
            self.refresh_index += 1
            QTimer.singleShot(300, self.process_next_market_card)
            
        except Exception as e:
            print(f"Error in process_next_market_card: {e}")
            import traceback
            print(traceback.format_exc())
            self.update_all_cards_with_error()
    
    def update_all_cards_with_error(self):
        """Update all market cards with error message"""
        try:
            for symbol, card in self.market_cards.items():
                card.value_label.setText("Error")
                card.change_label.setText("Could not fetch data")
                card.change_label.setStyleSheet("color: #f87171;")
        except Exception as e:
            print(f"Error updating cards with error: {e}")
    
    def setup_recent_activity(self, parent_layout):
        """Set up the recent activity section"""
        try:
            # Create activity container
            activity_frame = QFrame()
            activity_frame.setObjectName("container")
            activity_layout = QVBoxLayout(activity_frame)
            activity_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create a scroll area for the activity list
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: transparent;
            }
        """)
            
            # Create a container widget for the activity items
            activity_list = QWidget()
            activity_list_layout = QVBoxLayout(activity_list)
            activity_list_layout.setContentsMargins(16, 16, 16, 16)
            activity_list_layout.setSpacing(12)
            
            # Show placeholder message instead of loading records
            no_activity = QLabel("Click 'View All Records' to see your prediction history.")
            no_activity.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_activity.setStyleSheet("color: #8b949e; padding: 20px;")
            activity_list_layout.addWidget(no_activity)
            
            # Add view records button
            view_more = QPushButton("View All Records")
            view_more.setObjectName("secondaryButton")
            view_more.clicked.connect(lambda: self.switch_page(2))
            activity_list_layout.addWidget(view_more, 0, Qt.AlignmentFlag.AlignCenter)
            
            scroll_area.setWidget(activity_list)
            activity_layout.addWidget(scroll_area)
            
            parent_layout.addWidget(activity_frame)
        except Exception as e:
            print(f"Error setting up recent activity: {e}")
            import traceback
            print(traceback.format_exc())
    
    def create_activity_item(self, record):
        """Create an activity item widget for a prediction record"""
        item = QFrame()
        item.setStyleSheet("""
            QFrame {
                border-bottom: 1px solid #30363d;
                padding: 12px 8px;
                border-radius: 8px;
            }
            QFrame:hover {
                background-color: rgba(33, 38, 45, 0.5);
            }
        """)
        
        item_layout = QHBoxLayout(item)
        item_layout.setContentsMargins(8, 4, 8, 4)
        
        # Status icon
        status_label = QLabel()
        status = record.get("status", "")
        if status == "completed" or status == "Completed":
            status_label.setText("✓")
            status_label.setStyleSheet("color: #3fb950; font-size: 18px; min-width: 30px; font-weight: bold;")
        elif status == "error" or status == "Error" or status == "Failed":
            status_label.setText("×")
            status_label.setStyleSheet("color: #f85149; font-size: 18px; min-width: 30px; font-weight: bold;")
        else:
            status_label.setText("⏳")
            status_label.setStyleSheet("color: #f0883e; font-size: 18px; min-width: 30px;")
        
        # Details
        details_layout = QVBoxLayout()
        details_layout.setSpacing(4)
        
        # Symbol and date
        symbol_date = QLabel(f"<b>{record.get('symbol', 'Unknown')}</b> · {record.get('date', record.get('timestamp', 'Unknown date'))}")
        symbol_date.setStyleSheet("color: #c9d1d9;")
        
        # Model and MSE
        model_info = QLabel(f"{record.get('model', 'Unknown')} model · MSE: {record.get('mse', 'N/A')}")
        model_info.setStyleSheet("color: #8b949e; font-size: 13px;")
        
        details_layout.addWidget(symbol_date)
        details_layout.addWidget(model_info)
        
        # View button
        view_button = QPushButton("View")
        view_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(31, 111, 235, 0.1);
                border: 1px solid rgba(31, 111, 235, 0.2);
                border-radius: 6px;
                padding: 6px 12px;
                color: #58a6ff;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(31, 111, 235, 0.2);
                border-color: #58a6ff;
            }
        """)
        view_button.clicked.connect(lambda: self.show_prediction_details(record))
        view_button.setFixedWidth(80)
        
        # Add widgets to layout
        item_layout.addWidget(status_label)
        item_layout.addLayout(details_layout, 1)
        item_layout.addWidget(view_button)
        
        return item
    
    def setup_analysis_page(self):
        """Set up the stock analysis page"""
        try:
            page = QScrollArea()
            page.setWidgetResizable(True)
            page.setStyleSheet("QScrollArea { border: none; background: #0d1117; }")
            page_widget = QWidget()
            page_layout = QVBoxLayout(page_widget)
            page_layout.setContentsMargins(20, 20, 20, 20)
            page_layout.setSpacing(20)
            
            # Page header with title
            header_layout = QHBoxLayout()
            title_label = QLabel("Stock Analysis")
            title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: 700;
            color: #e6edf3;
            letter-spacing: -0.3px;
            """)
            header_layout.addWidget(title_label)
            header_layout.addStretch()
            page_layout.addLayout(header_layout)
            
            # Create search section with compact design
            search_frame = QFrame()
            search_frame.setStyleSheet("""
            QFrame {
            background: rgba(22, 27, 34, 0.5);
            border: 1px solid #30363d;
            border-radius: 14px;
            }
            """)
            search_layout = QHBoxLayout(search_frame)
            search_layout.setContentsMargins(16, 16, 16, 16)
            search_layout.setSpacing(12)
            
            # Stock symbol input with better styling
            symbol_label = QLabel("Symbol:")
            symbol_label.setStyleSheet("color: #8b949e; font-weight: 500; min-width: 60px;")
            self.symbol_input = QLineEdit()
            self.symbol_input.setPlaceholderText("Enter stock symbol (e.g., AAPL)")
            self.symbol_input.setMinimumWidth(150)
            self.symbol_input.setMaximumWidth(180)
            
            # Time range selector with better styling
            timerange_label = QLabel("Period:")
            timerange_label.setStyleSheet("color: #8b949e; font-weight: 500; min-width: 50px;")
            self.timerange_combo = QComboBox()
            timerange_options = ["1d", "5d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
            self.timerange_combo.addItems(timerange_options)
            self.timerange_combo.setCurrentText("1mo")  # Default
            self.timerange_combo.currentIndexChanged.connect(self.update_time_range)
            self.timerange_combo.setFixedWidth(80)
            
            # Interval selector
            interval_label = QLabel("Interval:")
            interval_label.setStyleSheet("color: #8b949e; font-weight: 500; min-width: 50px;")
            self.interval_combo = QComboBox()
            self.interval_combo.addItems(["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"])
            self.interval_combo.setCurrentText("1d")  # Default
            self.interval_combo.setFixedWidth(80)
            self.interval_combo.currentIndexChanged.connect(self.update_interval)
            
            # Search button with improved styling
            search_button = QPushButton("Search")
            search_button.setObjectName("searchButton")
            search_button.clicked.connect(self.fetch_stock_data)
            search_button.setFixedWidth(100)
            
            # Add widgets to layout with better spacing
            search_layout.addWidget(symbol_label)
            search_layout.addWidget(self.symbol_input)
            search_layout.addWidget(timerange_label)
            search_layout.addWidget(self.timerange_combo)
            search_layout.addWidget(interval_label)
            search_layout.addWidget(self.interval_combo)
            search_layout.addStretch(1)
            search_layout.addWidget(search_button)
            
            page_layout.addWidget(search_frame)
            
            # Chart section with better sizing
            chart_section = QFrame()
            chart_section.setStyleSheet("""
            QFrame {
            background: rgba(22, 27, 34, 0.5);
            border: 1px solid #30363d;
            border-radius: 14px;
            }
            """)
            chart_section.setMinimumHeight(600)  # Set minimum height to ensure visibility
            chart_layout = QVBoxLayout(chart_section)
            chart_layout.setContentsMargins(12, 12, 12, 12)
            chart_layout.setSpacing(12)
            
            # Create the chart widget
            self.chart_widget = ChartWidget()
            chart_layout.addWidget(self.chart_widget)
            
            # Create prediction section with better styling
            prediction_section = QFrame()
            prediction_section.setStyleSheet("""
            QFrame {
            background: rgba(13, 17, 23, 0.3);
            border-top: 1px solid #30363d;
            border-radius: 0 0 12px 12px;
            }
            """)
            prediction_layout = QHBoxLayout(prediction_section)
            prediction_layout.setContentsMargins(16, 16, 16, 16)
            prediction_layout.setSpacing(12)
            
            # Model selector with compact styling
            model_label = QLabel("Model:")
            model_label.setStyleSheet("color: #8b949e; font-weight: 500; min-width: 40px;")
            self.model_combo = QComboBox()
            model_options = ["LSTM", "RF", "XGB", "ARIMA"]
            self.model_combo.addItems(model_options)
            self.model_combo.setFixedWidth(110)
            
            # Forecast days input with compact styling
            days_label = QLabel("Days:")
            days_label.setStyleSheet("color: #8b949e; font-weight: 500; min-width: 35px;")
            self.days_spinbox = QSpinBox()
            self.days_spinbox.setMinimum(1)
            self.days_spinbox.setMaximum(365)  # Updated to 365 days for long-term predictions
            self.days_spinbox.setValue(30)
            self.days_spinbox.setFixedWidth(60)
            
            # Predict button with improved styling
            predict_button = QPushButton("Generate Prediction")
            predict_button.setObjectName("predictButton")
            predict_button.clicked.connect(self.generate_prediction)
            predict_button.setFixedWidth(200)
            
            # Add widgets to prediction layout with better spacing
            prediction_layout.addWidget(model_label)
            prediction_layout.addWidget(self.model_combo)
            prediction_layout.addWidget(days_label)
            prediction_layout.addWidget(self.days_spinbox)
            prediction_layout.addStretch(1)
            prediction_layout.addWidget(predict_button)
            
            chart_layout.addWidget(prediction_section)
            
            # Add sections to main layout with proper proportions
            page_layout.addWidget(chart_section, 1)  # Chart takes remaining space
            
            page.setWidget(page_widget)
            self.stacked_widget.addWidget(page)
            
        except Exception as e:
            print(f"Error setting up analysis page: {e}")
            raise
    
    def setup_records_page(self):
        """Set up the prediction records page"""
        try:
            page = QScrollArea()
            page.setWidgetResizable(True)
            page.setStyleSheet("QScrollArea { border: none; background: #0d1117; }")
            
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(20, 20, 20, 20)
            layout.setSpacing(20)
            
            # Page header with title and actions
            header_layout = QHBoxLayout()
            
            # Title
            title_label = QLabel("Prediction Records")
            title_label.setStyleSheet("""
                font-size: 24px;
                font-weight: 700;
                color: #e6edf3;
                letter-spacing: -0.3px;
            """)
            header_layout.addWidget(title_label)
            
            # Refresh button
            refresh_btn = QPushButton("↻ Refresh")
            refresh_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid #30363d;
                    border-radius: 8px;
                    color: #58a6ff;
                    font-size: 13px;
                    font-weight: 600;
                    padding: 6px 12px;
                }
                QPushButton:hover {
                    background-color: rgba(56, 139, 253, 0.1);
                    border-color: #58a6ff;
                }
            """)
            refresh_btn.clicked.connect(self.refresh_records)
            header_layout.addStretch()
            header_layout.addWidget(refresh_btn)
            
            layout.addLayout(header_layout)
            
            # Records container with improved styling
            records_frame = QFrame()
            records_frame.setStyleSheet("""
                QFrame {
                    background: rgba(22, 27, 34, 0.5);
                    border: 1px solid #30363d;
                    border-radius: 14px;
                }
            """)
            records_layout = QVBoxLayout(records_frame)
            records_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create a scroll area for the records
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
            """)
            
            # Create records list widget
            self.records_list = QWidget()
            self.records_list_layout = QVBoxLayout(self.records_list)
            self.records_list_layout.setContentsMargins(16, 16, 16, 16)
            self.records_list_layout.setSpacing(16)
            
            # Placeholder message when empty with better styling
            self.no_records_label = QLabel("No prediction records found. Go to the Analysis page to generate predictions.")
            self.no_records_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.no_records_label.setStyleSheet("""
                color: #8b949e; 
                padding: 50px 30px;
                font-size: 15px;
                background: rgba(13, 17, 23, 0.4);
                border: 1px solid #30363d;
                border-radius: 12px;
            """)
            self.records_list_layout.addWidget(self.no_records_label)
            
            # Set up the records list
            scroll_area.setWidget(self.records_list)
            records_layout.addWidget(scroll_area)
            
            layout.addWidget(records_frame)
            
            page.setWidget(content)
            self.stacked_widget.addWidget(page)
            print("Records page successfully added to stacked widget")
            
            # Initialize the records list after setup
            self.refresh_records()
            
        except Exception as e:
            print(f"Error setting up records page: {e}")
            raise
    
    def refresh_records(self):
        """Refresh the prediction records display"""
        try:
            print("Refreshing records...")
            # Clear existing record widgets
            for i in reversed(range(self.records_list_layout.count())):
                widget = self.records_list_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()
            
            # Fetch all records
            records = self.prediction_record.get_records()
            print(f"Found {len(records)} records")
            
            if not records:
                # Show no records message
                self.no_records_label = QLabel("No prediction records found. Go to the Analysis page to generate predictions.")
                self.no_records_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.no_records_label.setStyleSheet("""
                    color: #8b949e; 
                    padding: 50px 30px;
                    font-size: 15px;
                    background: rgba(13, 17, 23, 0.4);
                    border: 1px solid #30363d;
                    border-radius: 12px;
                """)
                self.records_list_layout.addWidget(self.no_records_label)
            else:
                # Add records in reverse chronological order (newest first)
                for record in sorted(records, key=lambda r: r.get("timestamp", 0), reverse=True):
                    record_widget = self.create_record_widget(record)
                    self.records_list_layout.addWidget(record_widget)
                
                # Add empty space at the bottom
                self.records_list_layout.addStretch()
            
            # Force update
            self.records_list.update()
        except Exception as e:
            print(f"Error refreshing records: {e}")
            import traceback
            print(traceback.format_exc())
    
    def create_record_widget(self, record):
        """Create a widget for a prediction record"""
        # Create main record container with improved styling
        record_widget = QFrame()
        record_widget.setStyleSheet("""
            QFrame {
                background: rgba(22, 27, 34, 0.7);
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 0;
            }
            QFrame:hover {
                border-color: #58a6ff;
                background: rgba(22, 27, 34, 0.9);
            }
        """)
        record_layout = QVBoxLayout(record_widget)
        record_layout.setContentsMargins(20, 20, 20, 20)
        record_layout.setSpacing(16)
        
        # Header section with symbol, date and actions
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)
        
        # Status icon with improved styling
        status_indicator = QLabel()
        if record.get("status") == "Completed" or record.get("status") == "completed":
            status_indicator.setText("✓")
            status_indicator.setStyleSheet("""
                color: #3fb950;
                font-size: 18px;
                font-weight: bold;
                background: rgba(46, 160, 67, 0.2);
                border-radius: 12px;
                min-width: 24px;
                min-height: 24px;
                max-width: 24px;
                max-height: 24px;
                padding: 4px;
                text-align: center;
            """)
            status_indicator.setToolTip("Prediction completed successfully")
        elif record.get("status") == "Error" or record.get("status") == "error" or record.get("status") == "Failed":
            status_indicator.setText("×")
            status_indicator.setStyleSheet("""
                color: #f85149;
                font-size: 18px;
                font-weight: bold;
                background: rgba(248, 81, 73, 0.2);
                border-radius: 12px;
                min-width: 24px;
                min-height: 24px;
                max-width: 24px;
                max-height: 24px;
                padding: 4px;
                text-align: center;
            """)
            status_indicator.setToolTip("Error during prediction")
        else:
            status_indicator.setText("⏳")
            status_indicator.setStyleSheet("""
                color: #f0883e;
                font-size: 18px;
                background: rgba(240, 136, 62, 0.2);
                border-radius: 12px;
                min-width: 24px;
                min-height: 24px;
                max-width: 24px;
                max-height: 24px;
                padding: 4px;
                text-align: center;
            """)
            status_indicator.setToolTip("Prediction in progress")
        
        # Symbol and date with improved styling
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        
        symbol_label = QLabel(f"<span style='font-size: 20px; font-weight: bold; color: #e6edf3;'>{record.get('symbol', 'Unknown')}</span>")
        
        date_str = record.get('date', record.get('timestamp', 'Unknown date'))
        date_label = QLabel(f"<span style='color: #8b949e; font-size: 13px;'>{date_str}</span>")
        
        info_layout.addWidget(symbol_label)
        info_layout.addWidget(date_label)
        
        header_layout.addWidget(status_indicator)
        header_layout.addLayout(info_layout)
        header_layout.addStretch()
        
        # Actions (delete button)
        delete_btn = QPushButton()
        delete_btn.setObjectName("deleteButton")
        delete_btn.setIcon(QIcon("resources/trash.png"))  # Consider adding an icon
        delete_btn.setToolTip("Delete record")
        delete_btn.setStyleSheet("""
            QPushButton {
                background: rgba(248, 81, 73, 0.1);
                border: none;
                border-radius: 16px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
                padding: 6px;
                text-align: center;
            }
            QPushButton:hover {
                background: rgba(248, 81, 73, 0.3);
            }
        """)
        delete_btn.setText("×")
        delete_btn.clicked.connect(lambda: self.delete_record(record.get("id")))
        header_layout.addWidget(delete_btn)
        
        record_layout.addLayout(header_layout)
        
        # Add divider with improved styling
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("background-color: #30363d; max-height: 1px; margin: 0 0;")
        record_layout.addWidget(divider)
        
        # Details grid with improved styling
        details_frame = QFrame()
        details_frame.setStyleSheet("""
            QFrame {
                background: rgba(13, 17, 23, 0.3);
                border-radius: 8px;
                padding: 2px;
                }
            """)
        details_layout = QGridLayout(details_frame)
        details_layout.setColumnStretch(1, 1)
        details_layout.setColumnStretch(3, 1)
        details_layout.setContentsMargins(12, 12, 12, 12)
        details_layout.setSpacing(12)
        
        # Add details with improved styling - Fix for float values
        details = [
            ("Model", str(record.get("model", "Unknown")), "color: #58a6ff;"),
            ("Days", f"{record.get('days', record.get('forecast_days', 'N/A'))} days", "color: #f0883e;"),
            ("MSE", str(record.get("mse", "N/A")), "color: #3fb950;"),
            ("Data Points", str(record.get("data_points", "N/A")), "color: #a371f7;")
        ]
        
        for i, (label, value, color) in enumerate(details):
            row, col = divmod(i, 2)
            
            label_widget = QLabel(f"{label}:")
            label_widget.setStyleSheet("color: #8b949e; font-weight: 500; font-size: 13px;")
            
            # Ensure value is a string before passing to QLabel
            value_widget = QLabel(str(value))
            value_widget.setStyleSheet(f"{color} font-weight: 600; font-size: 14px;")
            
            details_layout.addWidget(label_widget, row, col*2)
            details_layout.addWidget(value_widget, row, col*2+1)
        
        record_layout.addWidget(details_frame)
        
        # Add view button with improved styling
        view_prediction_btn = QPushButton("View Prediction Details")
        view_prediction_btn.setStyleSheet("""
                QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                         stop:0 rgba(31, 111, 235, 0.8),
                                         stop:1 rgba(47, 128, 237, 0.8));
                    border: none;
                border-radius: 10px;
                    color: white;
                font-weight: 600;
                padding: 10px 16px;
                font-size: 13px;
                }
                QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                         stop:0 rgba(47, 128, 237, 0.9),
                                         stop:1 rgba(56, 139, 253, 0.9));
                }
            """)
        view_prediction_btn.clicked.connect(lambda: self.show_prediction_details(record))
        
        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()
        buttons_layout.addWidget(view_prediction_btn)
        
        record_layout.addLayout(buttons_layout)
        
        return record_widget
    
    def delete_record(self, record_id):
        """Delete a prediction record"""
        # Find and remove the record
        self.prediction_record.records = [r for r in self.prediction_record.records if r['id'] != record_id]
        self.prediction_record.save_records()
        self.refresh_records()
    
    def generate_prediction(self):
        """Generate prediction using selected model and parameters"""
        if not self.current_symbol or self.cached_data is None:
            QMessageBox.warning(self, "Error", "Please select a stock symbol first")
            return
        
        model = self.model_combo.currentText()
        days = self.days_spinbox.value()
        
        try:
            # Create initial prediction record
            record = {
                'id': len(self.prediction_record.records) + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.current_symbol,
                'model': model,
                'days': days,
                'prediction': None,
                'mse': None,
                'status': 'Processing'
            }
            self.prediction_record.records.append(record)
            self.prediction_record.save_records()
            
            print("Prediction record created, switching to records page...")
            # Switch to records page and refresh the view
            self.records_btn.setChecked(True)
            
            # First refresh the records
            self.refresh_records()
            
            # Then switch to the page
            self.stacked_widget.setCurrentIndex(2)
            print(f"Current index after switch: {self.stacked_widget.currentIndex()}")
            
            # Create worker thread
            self.thread = QThread()
            self.worker = PredictionWorker(
                self.current_symbol,
                model,
                days,
                self.cached_data['Close'].iloc[-1]
            )
            self.worker.moveToThread(self.thread)
            
            # Connect signals
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.prediction_completed)
            self.worker.progress.connect(lambda msg: self.update_prediction_progress(record['id'], msg))
            self.worker.error.connect(lambda err: self.prediction_error(record['id'], err))
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            # Start the thread
            self.thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start prediction: {str(e)}")
            print(f"Error in generate_prediction: {e}")
            import traceback
            print(traceback.format_exc())
    
    def update_prediction_progress(self, record_id, message):
        """Update the progress message for a prediction record"""
        for record in self.prediction_record.records:
            if record['id'] == record_id:
                record['progress_message'] = message
                break
        self.refresh_records()
    
    def prediction_completed(self, result):
        """Handle completed prediction"""
        # Update the last record (which should be the processing one)
        for record in self.prediction_record.records:
            if record['status'] == 'Processing':
                record.update(result)
                break
        
        self.prediction_record.save_records()
        self.refresh_records()
    
    def prediction_error(self, record_id, error_message):
        """Handle prediction error"""
        for record in self.prediction_record.records:
            if record['id'] == record_id:
                record['status'] = 'Failed'
                record['error'] = error_message
                break
        
        self.prediction_record.save_records()
        self.refresh_records()
        QMessageBox.critical(self, "Error", f"Prediction failed: {error_message}")
    
    def show_prediction_details(self, record):
        """Show detailed prediction results in a modern, informative dialog"""
        try:
            # Fetch historical data for the predicted stock
            try:
                data, _ = get_stock_data(record['symbol'], period="1mo")
            except StockDataError as e:
                QMessageBox.warning(self, "Warning", f"Could not fetch historical data: {str(e)}")
                data = None

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Prediction Analysis - {record['symbol']}")
            dialog.setMinimumSize(1200, 800)
            dialog.setStyleSheet("""
                QDialog {
                    background: #0d1117;
                }
                QLabel {
                    color: #c9d1d9;
                }
                QFrame {
                    background: rgba(22, 27, 34, 0.7);
                    border: 1px solid rgba(48, 54, 61, 0.5);
                    border-radius: 16px;
                }
            """)
            
            layout = QVBoxLayout(dialog)
            layout.setSpacing(16)
            layout.setContentsMargins(16, 16, 16, 16)
            
            # Header with Key Metrics
            header = QFrame()
            header.setMaximumHeight(120)  # Limit header height
            header_layout = QHBoxLayout(header)
            header_layout.setSpacing(16)
            header_layout.setContentsMargins(16, 16, 16, 16)
            
            # Symbol and Model Info
            info_layout = QVBoxLayout()
            info_layout.setSpacing(4)
            
            symbol_label = QLabel(f"📈 {record['symbol']}")
            symbol_label.setStyleSheet("font-size: 32px; font-weight: 800; color: #58a6ff;")
            info_layout.addWidget(symbol_label)
            
            model_label = QLabel(f"Model: {record['model']}")
            model_label.setStyleSheet("font-size: 16px; color: #8b949e;")
            info_layout.addWidget(model_label)
            
            header_layout.addLayout(info_layout)
            
            # Key Metrics in a horizontal layout
            metrics_layout = QHBoxLayout()
            metrics_layout.setSpacing(16)
            
            metrics = [
                ("Forecast", f"{record['days']} days", "#238636"),
                ("MSE", f"{record['mse']:.4f}", "#ff7b00"),
                ("Status", record['status'], "#4ade80" if record['status'] == "Completed" else "#f87171")
            ]
            
            for label, value, color in metrics:
                metric_frame = QFrame()
                metric_frame.setStyleSheet(f"""
                    QFrame {{
                        background: {color}10;
                        border: 1px solid {color}30;
                        border-radius: 12px;
                        padding: 12px;
                        min-width: 150px;
                    }}
                """)
                metric_layout = QVBoxLayout(metric_frame)
                metric_layout.setSpacing(4)
                
                label_widget = QLabel(label)
                label_widget.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: 600;")
                metric_layout.addWidget(label_widget)
                
                value_widget = QLabel(value)
                value_widget.setStyleSheet("color: #c9d1d9; font-size: 18px; font-weight: 700;")
                metric_layout.addWidget(value_widget)
                
                metrics_layout.addWidget(metric_frame)
            
            header_layout.addLayout(metrics_layout)
            layout.addWidget(header)
            
            # Main content area (Chart and Analysis)
            content = QFrame()
            content_layout = QHBoxLayout(content)
            content_layout.setSpacing(16)
            content_layout.setContentsMargins(0, 0, 0, 0)
            
            # Chart Frame (larger portion)
            chart_frame = QFrame()
            chart_frame.setStyleSheet("""
                QFrame {
                    background: rgba(22, 27, 34, 0.7);
                    border: 1px solid rgba(48, 54, 61, 0.5);
                    border-radius: 16px;
                    padding: 16px;
                }
            """)
            chart_layout = QVBoxLayout(chart_frame)
            chart_layout.setContentsMargins(8, 8, 8, 8)
            
            # Create matplotlib figure with proper size
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            import matplotlib.dates as mdates
            
            # Create figure with a larger size
            fig = Figure(figsize=(14, 8), dpi=100, facecolor='#161b22')
            canvas = FigureCanvas(fig)
            canvas.setMinimumHeight(500)  # Increased minimum height
            
            # Plot data
            ax = fig.add_subplot(111)
            ax.set_facecolor('#0d1117')
            ax.grid(True, linestyle='--', alpha=0.2)
            ax.tick_params(colors='#8b949e', labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#30363d')
            
            # Get the data
            if data is not None and not data.empty:
                # Plot historical data
                ax.plot(data.index[-30:], data['Close'].iloc[-30:], 
                       label='Historical', color='#58a6ff', linewidth=2)
                
                # Generate dates for prediction
                last_date = data.index[-1]
                prediction_dates = [last_date + timedelta(days=i+1) for i in range(len(record['prediction']))]
                
                # Plot prediction
                ax.plot(prediction_dates, record['prediction'], 
                       label='Prediction', color='#3fb950', linewidth=2, linestyle='--')
                
                # Add confidence intervals
                prediction_values = np.array(record['prediction'])
                std_dev = np.std(data['Close'].iloc[-30:])
                z_score = 1.96  # For 95% confidence
                
                margin = z_score * std_dev
                upper_bound = prediction_values + margin
                lower_bound = prediction_values - margin
                
                ax.fill_between(prediction_dates, lower_bound, upper_bound, 
                              color='#238636', alpha=0.1, label='95% Confidence')
                
                # Style and format
                ax.set_title(f"{record['symbol']} Stock Price Prediction", 
                            color='#c9d1d9', pad=20, fontsize=14)
                ax.set_ylabel("Price ($)", color='#8b949e', fontsize=12)
                ax.set_xlabel("Date", color='#8b949e', fontsize=12)
                
                # Format dates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
                fig.autofmt_xdate(rotation=45)
                
                # Add legend with better positioning
                ax.legend(facecolor='#161b22', edgecolor='#30363d', 
                         labelcolor='#c9d1d9', loc='upper left',
                         fontsize=10, framealpha=0.8)
                
                # Adjust layout with proper spacing
                fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
            
            chart_layout.addWidget(canvas)
            content_layout.addWidget(chart_frame, stretch=7)  # Give chart more horizontal space
            
            # Analysis Panel (smaller portion)
            analysis_frame = QFrame()
            analysis_frame.setStyleSheet("""
                QFrame {
                    background: rgba(22, 27, 34, 0.7);
                    border: 1px solid rgba(48, 54, 61, 0.5);
                    border-radius: 16px;
                    padding: 16px;
                }
            """)
            analysis_layout = QVBoxLayout(analysis_frame)
            analysis_layout.setSpacing(16)
            
            # Prediction Summary
            summary_title = QLabel("Prediction Summary")
            summary_title.setStyleSheet("font-size: 18px; font-weight: 700; color: #58a6ff;")
            analysis_layout.addWidget(summary_title)
            
            if record['prediction'] is not None and len(record['prediction']) > 0:
                last_price = data['Close'].iloc[-1] if data is not None and not data.empty else 0
                final_pred = record['prediction'][-1]
                change = ((final_pred - last_price) / last_price) * 100 if last_price != 0 else 0
                
                # Format the summary with better styling
                summary_frame = QFrame()
                summary_frame.setStyleSheet("""
                    QFrame {
                        background: rgba(35, 134, 54, 0.1);
                        border: 1px solid rgba(35, 134, 54, 0.2);
                        border-radius: 12px;
                        padding: 16px;
                    }
                """)
                summary_layout = QVBoxLayout(summary_frame)
                
                price_items = [
                    ("Last Price", f"${last_price:.2f}"),
                    ("Final Prediction", f"${final_pred:.2f}"),
                    ("Predicted Change", f"{change:+.2f}%")
                ]
                
                for label, value in price_items:
                    item_layout = QHBoxLayout()
                    label_widget = QLabel(label)
                    label_widget.setStyleSheet("color: #8b949e; font-size: 14px;")
                    value_widget = QLabel(value)
                    value_widget.setStyleSheet("color: #c9d1d9; font-size: 16px; font-weight: 600;")
                    item_layout.addWidget(label_widget)
                    item_layout.addWidget(value_widget, alignment=Qt.AlignmentFlag.AlignRight)
                    summary_layout.addLayout(item_layout)
                
                analysis_layout.addWidget(summary_frame)
            
            analysis_layout.addStretch()
            content_layout.addWidget(analysis_frame, stretch=3)  # Give analysis panel less space
            
            layout.addWidget(content, stretch=1)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet("""
                QPushButton {
                    background: #21262d;
                    border: 1px solid #30363d;
                    border-radius: 6px;
                    padding: 12px 24px;
                    color: #c9d1d9;
                    font-weight: 600;
                    min-width: 120px;
                }
                QPushButton:hover {
                    background: #30363d;
                    border-color: #58a6ff;
                }
            """)
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
            
            dialog.exec()
            
        except Exception as e:
            print(f"Error showing prediction details: {e}")
            QMessageBox.critical(self, "Error", "Failed to display prediction details. Please try again.")
    
    def setup_settings_page(self):
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setStyleSheet("QScrollArea { border: none; background: #0d1117; }")
        
        content = QFrame()
        content.setObjectName("mainContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        # Initialize settings
        self.settings = Settings()
        self.settings_widgets = {}  # Store widgets for later access
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        header_layout = QVBoxLayout(header)
        
        title = QLabel("Settings")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffffff;")
        header_layout.addWidget(title)
        
        layout.addWidget(header)
        
        # Essential Settings
        sections = [
            ("Chart Settings", [
                ("chart_type", "Chart Type", "combo", 
                 ["Candlestick", "Line", "OHLC"], 
                 "Select your preferred chart visualization style"),
                ("default_ma", "Default Moving Averages", "multi_toggle", {
                    "MA20": True,
                    "MA50": True,
                    "MA200": False
                }, "Select which moving averages to display by default"),
                ("show_volume", "Show Volume", "toggle", True, 
                 "Display trading volume data below the price chart")
            ]),
            ("Analysis Settings", [
                ("default_timeframe", "Default Timeframe", "combo", 
                 ["1 Month", "3 Months", "6 Months", "1 Year"], 
                 "Set the default time period for analysis"),
                ("default_model", "Default Model", "combo",
                 ["LSTM", "Random Forest", "XGBoost", "ARIMA"],
                 "Choose the default prediction model"),
                ("prediction_days", "Default Prediction Days", "number", 
                 30, "Set the default number of days for predictions (up to 365 days)")
            ]),
            ("Alert Settings", [
                ("price_alerts", "Price Alerts", "toggle", True,
                 "Get notified about significant price changes"),
                ("price_change_threshold", "Price Change Threshold (%)", "number",
                 5, "Minimum price change percentage to trigger an alert")
            ])
        ]
        
        for section_name, settings_list in sections:
            section_frame = QFrame()
            section_frame.setStyleSheet("""
                QFrame {
                    background-color: #161b22;
                    border: 1px solid #30363d;
                    border-radius: 12px;
                    padding: 20px;
                }
            """)
            section_layout = QVBoxLayout(section_frame)
            section_layout.setSpacing(20)
            
            # Section header
            section_header = QLabel(section_name)
            section_header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            section_header.setStyleSheet("color: #ffffff;")
            section_layout.addWidget(section_header)
            
            # Settings grid
            settings_grid = QGridLayout()
            settings_grid.setColumnStretch(1, 1)
            settings_grid.setSpacing(15)
            
            for i, (key, name, type_, options, tooltip) in enumerate(settings_list):
                # Label and tooltip
                label_layout = QVBoxLayout()
                name_label = QLabel(name)
                name_label.setStyleSheet("color: #c9d1d9; font-weight: bold;")
                label_layout.addWidget(name_label)
                
                tooltip_label = QLabel(tooltip)
                tooltip_label.setStyleSheet("color: #8b949e; font-size: 12px;")
                tooltip_label.setWordWrap(True)
                label_layout.addWidget(tooltip_label)
                
                settings_grid.addLayout(label_layout, i, 0)
                
                # Get current value from settings
                current_value = self.settings.get_setting(section_name, key)
                
                # Control
                if type_ == "combo":
                    widget = QComboBox()
                    widget.addItems(options)
                    widget.setCurrentText(str(current_value) if current_value else options[0])
                    widget.setStyleSheet("""
                        QComboBox {
                            background-color: #21262d;
                            border: 1px solid #30363d;
                            border-radius: 6px;
                            padding: 8px 12px;
                            color: #c9d1d9;
                            min-width: 150px;
                        }
                        QComboBox:hover {
                            border-color: #58a6ff;
                        }
                    """)
                    # Fix the signal connection to avoid closure issues
                    widget.currentTextChanged.connect(
                        self.create_combo_handler(section_name, key)
                    )
                
                elif type_ == "toggle":
                    widget = QPushButton()
                    widget.setCheckable(True)
                    widget.setChecked(current_value if current_value is not None else options)
                    widget.setText("Enabled" if widget.isChecked() else "Disabled")
                    widget.setStyleSheet("""
                        QPushButton {
                            background-color: #238636;
                            border: none;
                            border-radius: 6px;
                            padding: 8px 16px;
                            color: white;
                            font-weight: bold;
                            min-width: 100px;
                        }
                        QPushButton:checked {
                            background-color: #238636;
                        }
                        QPushButton:!checked {
                            background-color: #21262d;
                            color: #c9d1d9;
                        }
                    """)
                    widget.clicked.connect(
                        lambda checked, s=section_name, k=key, b=widget:
                        self.update_toggle_setting(s, k, b)
                    )
                
                elif type_ == "number":
                    widget = QSpinBox()
                    # Increase maximum range for prediction days
                    if key == "prediction_days":
                        widget.setRange(1, 365)
                    else:
                        widget.setRange(1, 1000)
                    widget.setValue(current_value if current_value is not None else options)
                    widget.setStyleSheet("""
                        QSpinBox {
                            background-color: #21262d;
                            border: 1px solid #30363d;
                            border-radius: 6px;
                            padding: 8px 12px;
                            color: #c9d1d9;
                            min-width: 100px;
                        }
                        QSpinBox:hover {
                            border-color: #58a6ff;
                        }
                    """)
                    widget.valueChanged.connect(
                        lambda value, s=section_name, k=key:
                        self.update_setting(s, k, value)
                    )
                
                elif type_ == "multi_toggle":
                    widget = QFrame()
                    toggle_layout = QHBoxLayout(widget)
                    toggle_layout.setSpacing(10)
                    
                    current_mas = current_value if current_value is not None else options
                    for ma_name, enabled in current_mas.items():
                        toggle = QPushButton(ma_name)
                        toggle.setCheckable(True)
                        toggle.setChecked(enabled)
                        toggle.setStyleSheet("""
                            QPushButton {
                                background-color: #21262d;
                                border: 1px solid #30363d;
                                border-radius: 6px;
                                padding: 8px 12px;
                                color: #c9d1d9;
                            }
                            QPushButton:checked {
                                background-color: #1f6feb;
                                border-color: #58a6ff;
                                color: white;
                            }
                        """)
                        toggle.clicked.connect(
                            lambda checked, s=section_name, k=key, ma=ma_name:
                            self.update_ma_setting(s, k, ma, checked)
                        )
                        toggle_layout.addWidget(toggle)
                
                settings_grid.addWidget(widget, i, 1)
                self.settings_widgets[(section_name, key)] = widget
            
            section_layout.addLayout(settings_grid)
            layout.addWidget(section_frame)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 10px 20px;
                color: #c9d1d9;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #8b949e;
            }
        """)
        reset_btn.clicked.connect(self.reset_settings)
        layout.addWidget(reset_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        page.setWidget(content)
        self.stacked_widget.addWidget(page)
    
    def create_combo_handler(self, section, key):
        """Create a handler function for combo box changes"""
        def handler(value):
            try:
                print(f"Combo handler called: {section} - {key} = {value}")
                # Update the setting
                self.settings.update_setting(section, key, value)
                
                # Apply the change with a small delay to ensure the setting is saved
                QTimer.singleShot(100, lambda: self.apply_setting_change(section, key, value))
            except Exception as e:
                print(f"Error in combo handler: {e}")
                import traceback
                traceback.print_exc()
        return handler
    
    def update_setting(self, section, key, value):
        """Update a setting value"""
        self.settings.update_setting(section, key, value)
        self.apply_setting_change(section, key, value)
    
    def update_toggle_setting(self, section, key, button):
        """Update a toggle setting"""
        button.setText("Enabled" if button.isChecked() else "Disabled")
        self.update_setting(section, key, button.isChecked())
    
    def update_ma_setting(self, section, key, ma_name, enabled):
        """Update moving average setting"""
        current_value = self.settings.get_setting(section, key)
        if current_value is None:
            current_value = {"MA20": True, "MA50": True, "MA200": False}
        current_value[ma_name] = enabled
        self.update_setting(section, key, current_value)
    
    def apply_setting_change(self, section, key, value):
        """Apply the setting change to the UI"""
        print(f"Applying setting change: {section} - {key} = {value}")
        
        try:
            if section == "Chart Settings":
                if key == "chart_type":
                    if hasattr(self, 'chart_widget'):
                        print(f"Setting chart type to {value}")
                        # Update the chart type with a small delay to ensure the UI is ready
                        QTimer.singleShot(100, lambda: self.safe_update_chart_type(value))
                # ... rest of the code ...
        except Exception as e:
            print(f"Error applying setting change: {e}")
            import traceback
            traceback.print_exc()

    def safe_update_chart_type(self, value):
        """Safely update the chart type with proper error handling"""
        try:
            if hasattr(self, 'chart_widget'):
                print(f"Safe update chart type to: {value}")
                self.chart_widget.set_chart_type(value)
                
                # Force a chart refresh with current data
                if (hasattr(self, 'current_symbol') and self.current_symbol and 
                    hasattr(self, 'cached_data') and self.cached_data is not None):
                    print(f"Refreshing chart with {self.current_symbol}")
                    try:
                        period_map = {
                            "1 Month": "1mo",
                            "3 Months": "3mo",
                            "6 Months": "6mo", 
                            "1 Year": "1y",
                            "5 Years": "5y",
                            "Max": "max"
                        }
                        period = period_map.get(self.timerange_combo.currentText(), "1y")
                        filtered_data = self.filter_data_by_range(self.cached_data, period)
                        if filtered_data is not None:
                            self.chart_widget.update_chart(filtered_data.copy(), self.current_symbol)
                    except Exception as e:
                        print(f"Error refreshing chart: {e}")
                        import traceback
                        traceback.print_exc()
        except Exception as e:
            print(f"Error in safe_update_chart_type: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        self.settings.reset_to_defaults()
        
        # Update all widgets with default values
        for (section, key), widget in self.settings_widgets.items():
            value = self.settings.get_setting(section, key)
            
            if isinstance(widget, QComboBox):
                widget.setCurrentText(str(value))
            elif isinstance(widget, QPushButton) and widget.isCheckable():
                if not isinstance(value, dict):  # For regular toggle buttons
                    widget.setChecked(value)
                    widget.setText("Enabled" if value else "Disabled")
            elif isinstance(widget, QSpinBox):
                widget.setValue(value)
            elif isinstance(widget, QFrame):  # For multi_toggle
                if isinstance(value, dict):  # For moving averages
                    for toggle in widget.findChildren(QPushButton):
                        ma_name = toggle.text()
                        toggle.setChecked(value.get(ma_name, False))
            
            # Apply the change
            self.apply_setting_change(section, key, value)
    
    def filter_data_by_range(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter the cached data based on the selected time range"""
        end_date = df.index[-1]
        
        if period == "1wk":
            start_date = end_date - timedelta(days=7)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "3mo":
            start_date = end_date - timedelta(days=90)
        elif period == "6mo":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "5y":
            start_date = end_date - timedelta(days=365*5)
        else:  # max
            return df
        
        return df[df.index >= start_date]
    
    def update_time_range(self):
        """Update chart when time range changes"""
        if self.cached_data is not None and self.current_symbol:
            # Update interval combo box options based on selected period
            period = self.timerange_combo.currentText()
            valid_intervals = self.get_valid_intervals(period)
            
            # Store current interval
            current_interval = self.interval_combo.currentText()
            
            # Block signals to prevent triggering update_interval during updates
            self.interval_combo.blockSignals(True)
            
            # Clear and repopulate interval combo
            self.interval_combo.clear()
            self.interval_combo.addItems(valid_intervals)
            
            # Try to restore previous interval if valid, otherwise use first valid interval
            if current_interval in valid_intervals:
                self.interval_combo.setCurrentText(current_interval)
            else:
                self.interval_combo.setCurrentIndex(0)
            
            # Unblock signals
            self.interval_combo.blockSignals(False)
            
            # Fetch data with new period/interval
            self.fetch_stock_data()
    
    def fetch_stock_data(self):
        """Fetch stock data and update the UI"""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Error", "Please enter a stock symbol")
            return
            
        try:
            # Get selected time period and interval
            period = self.timerange_combo.currentText()
            interval = self.interval_combo.currentText()
            
            print(f"Fetching data for {symbol} with period: {period}, interval: {interval}")
            
            # Show loading message
            self.chart_widget.price_ax.clear()
            self.chart_widget.price_ax.text(0.5, 0.5, "Loading data...", 
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       transform=self.chart_widget.price_ax.transAxes)
            self.chart_widget.canvas.draw()
            
            # Validate interval based on selected period
            valid_intervals = self.get_valid_intervals(period)
            if interval not in valid_intervals:
                # Default to a valid interval for this period
                interval = valid_intervals[0]
                self.interval_combo.setCurrentText(interval)
                print(f"Adjusted interval to {interval} for compatibility with period {period}")
            
            # Fetch stock data with specified period and interval
            self.cached_data, self.cached_info = get_stock_data(symbol, period, interval)
            self.current_symbol = symbol
            
            # Update chart with data
            self.chart_widget.update_chart(self.cached_data, symbol)
            
            # Fetch and update company info
            company_info = get_company_info(symbol)
            self.company_info_tab.update_info(company_info)
            
            # Update news with both ticker and company name
            self.news_tab.update_news(symbol, company_info.get('name'))
            
        except StockDataError as e:
            QMessageBox.warning(self, "Error", str(e))
            self.chart_widget.price_ax.clear()
            self.chart_widget.volume_ax.clear()
            self.chart_widget.setup_plot()
            self.chart_widget.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
            self.chart_widget.price_ax.clear()
            self.chart_widget.volume_ax.clear()
            self.chart_widget.setup_plot()
            self.chart_widget.canvas.draw()
        
    def get_valid_intervals(self, period):
        """Return valid intervals for the given period"""
        # Define valid intervals for each period based on Yahoo Finance API restrictions
        period_to_intervals = {
            # For short periods, can use minute data
            "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"],
            "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"],
            "1wk": ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"],
            
            # For medium periods, start with daily data
            "1mo": ["1d", "5d", "1wk"],
            "3mo": ["1d", "5d", "1wk", "1mo"],
            "6mo": ["1d", "5d", "1wk", "1mo"],
            
            # For long periods, use daily or longer
            "1y": ["1d", "5d", "1wk", "1mo"],
            "2y": ["1d", "5d", "1wk", "1mo"],
            "5y": ["1d", "5d", "1wk", "1mo", "3mo"],
            "10y": ["1d", "5d", "1wk", "1mo", "3mo"],
            "ytd": ["1d", "5d", "1wk", "1mo"],
            "max": ["1d", "5d", "1wk", "1mo", "3mo"]
        }
        
        return period_to_intervals.get(period, ["1d"])
    
    def update_model_description(self, model_name):
        """Update the description text based on selected model"""
        descriptions = {
            "LSTM": "Long Short-Term Memory (LSTM) neural networks are specialized deep learning models "
                   "for time series forecasting. They excel at capturing long-term patterns in stock prices "
                   "by remembering important information over extended periods.",
            
            "RF": "Random Forest (RF) is an ensemble learning method combining multiple decision trees. "
                 "It excels at capturing non-linear relationships in market data while resisting overfitting. "
                 "Strong for identifying complex patterns in volatile markets.",
            
            "XGB": "XGBoost (XGB) is a gradient boosting algorithm optimized for speed and performance. "
                  "It builds models sequentially, with each new model correcting errors from previous ones. "
                  "Excellent for capturing market trends with high precision.",
            
            "ARIMA": "Auto Regressive Integrated Moving Average (ARIMA) is a statistical model "
                    "specifically designed for time series analysis. It works well for identifying "
                    "linear relationships and handling seasonality in market movements."
        }
        self.model_description.setText(descriptions.get(model_name, ""))
    
    def setup_info_page(self):
        """Setup the info page with a stunning, modern design"""
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #0d1117;
            }
        """)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(32)
        layout.setContentsMargins(40, 40, 40, 40)

        # Hero Section with Gradient
        hero = QFrame()
        hero.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                         stop:0 #1a1f29,
                                         stop:0.5 #192233,
                                         stop:1 #1a273a);
                border: 1px solid rgba(99, 179, 237, 0.2);
                border-radius: 24px;
            }
        """)
        hero.setMinimumHeight(220)
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(40, 40, 40, 40)
        hero_layout.setSpacing(16)

        # App title with modern typography
        title = QLabel("Stock Predictor Pro")
        title.setStyleSheet("""
            color: #e6edf3;
            font-size: 36px;
            font-weight: 800;
            letter-spacing: -0.5px;
        """)
        hero_layout.addWidget(title)

        # Version & tagline
        tagline = QLabel("Advanced market analysis powered by AI")
        tagline.setStyleSheet("""
            color: #8b949e;
            font-size: 18px;
            letter-spacing: 0.2px;
            margin-bottom: 12px;
        """)
        hero_layout.addWidget(tagline)

        version_badge = QLabel("Version 2.0")
        version_badge.setStyleSheet("""
            color: #0d1117;
            background: rgba(56, 139, 253, 1);
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 14px;
            font-weight: 600;
            max-width: 100px;
        """)
        hero_layout.addWidget(version_badge, 0, Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(hero)

        # Features Section With Cards
        features_title = QLabel("Key Features")
        features_title.setStyleSheet("""
            color: #e6edf3;
            font-size: 24px;
            font-weight: 700;
            margin-top: 16px;
        """)
        layout.addWidget(features_title)

        # Features Grid
        features_grid = QGridLayout()
        features_grid.setSpacing(20)
        
        feature_cards = [
            {
                "title": "AI-Powered Predictions",
                "icon": "🤖",
                "description": "Machine learning models to predict future stock movements with low MSE",
                "color": "rgba(56, 189, 248, 0.8)"
            },
            {
                "title": "Advanced Technical Analysis",
                "icon": "📊",
                "description": "Powerful indicators including MACD, RSI, and moving averages to analyze trends",
                "color": "rgba(139, 92, 246, 0.8)"
            },
            {
                "title": "Real-Time Market Data",
                "icon": "📈",
                "description": "Live market data with instant updates for all major indices and stocks",
                "color": "rgba(34, 211, 238, 0.8)"
            },
            {
                "title": "Simple & Intuitive Interface",
                "icon": "✨",
                "description": "Clean, modern interface designed for both novice and expert traders",
                "color": "rgba(251, 146, 60, 0.8)"
            }
        ]

        for i, feature in enumerate(feature_cards):
            row, col = divmod(i, 2)
            
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background: rgba(22, 27, 34, 0.7);
                    border: 1px solid {feature["color"]};
                    border-left: 6px solid {feature["color"]};
                    border-radius: 16px;
                    padding: 24px;
                }}
                QFrame:hover {{
                    background: rgba(22, 27, 34, 0.9);
                    border: 1px solid {feature["color"]};
                    border-left: 6px solid {feature["color"]};
                    transform: translateY(-2px);
                    transition: all 0.3s ease;
                }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(16)

            # Feature header with icon
            header_layout = QHBoxLayout()
            
            icon = QLabel(feature["icon"])
            icon.setStyleSheet(f"""
                font-size: 28px;
                min-width: 40px;
                max-width: 40px;
                color: {feature["color"]};
            """)
            header_layout.addWidget(icon)
            
            title = QLabel(feature["title"])
            title.setStyleSheet(f"""
                color: #e6edf3;
                font-size: 18px;
                font-weight: 700;
            """)
            header_layout.addWidget(title)
            header_layout.addStretch()
            
            card_layout.addLayout(header_layout)
            
            # Feature description
            description = QLabel(feature["description"])
            description.setWordWrap(True)
            description.setStyleSheet("""
                color: #8b949e;
                font-size: 14px;
                line-height: 1.5;
            """)
            card_layout.addWidget(description)
            
            features_grid.addWidget(card, row, col)

        layout.addLayout(features_grid)

        # Technology Section with modern badges
        tech_title = QLabel("Built With")
        tech_title.setStyleSheet("""
            color: #e6edf3;
            font-size: 24px;
            font-weight: 700;
            margin-top: 16px;
        """)
        layout.addWidget(tech_title)

        tech_frame = QFrame()
        tech_frame.setStyleSheet("""
            QFrame {
                background: rgba(22, 27, 34, 0.7);
                border: 1px solid #30363d;
                border-radius: 16px;
                padding: 32px;
            }
        """)
        tech_layout = QHBoxLayout(tech_frame)
        tech_layout.setSpacing(20)

        technologies = [
            ("Python", "#3776AB", "Core Language"),
            ("PyQt6", "#41CD52", "UI Framework"),
            ("TensorFlow", "#FF6F00", "Neural Networks"),
            ("Scikit-learn", "#F7931E", "ML Models"),
            ("Matplotlib", "#11557C", "Data Visualization")
        ]

        for name, color, role in technologies:
            tech_badge = QFrame()
            tech_badge.setStyleSheet(f"""
                QFrame {{
                    background: {color}20;
                    border: 2px solid {color}50;
                    border-radius: 16px;
                    padding: 16px;
                }}
                QFrame:hover {{
                    background: {color}30;
                    border: 2px solid {color}70;
                }}
            """)
            tech_badge.setMinimumWidth(150)
            tech_badge.setMaximumWidth(200)
            
            badge_layout = QVBoxLayout(tech_badge)
            badge_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            tech_name = QLabel(name)
            tech_name.setStyleSheet(f"""
                color: {color};
                font-size: 18px;
                font-weight: 700;
                text-align: center;
            """)
            tech_name.setAlignment(Qt.AlignmentFlag.AlignCenter)

            tech_role = QLabel(role)
            tech_role.setStyleSheet("""
                color: #8b949e;
                font-size: 14px;
                text-align: center;
                margin-top: 4px;
            """)
            tech_role.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            badge_layout.addWidget(tech_name)
            badge_layout.addWidget(tech_role)
            tech_layout.addWidget(tech_badge)

        layout.addWidget(tech_frame)

        # Copyright & Credits
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background: transparent;
                margin-top: 20px;
            }
        """)
        footer_layout = QHBoxLayout(footer)

        copyright = QLabel("© 2025 Stock Predictor Pro. All rights reserved.")
        copyright.setStyleSheet("""
            color: #8b949e;
            font-size: 14px;
        """)
        footer_layout.addWidget(copyright)
        
        layout.addWidget(footer)
        
        # Add stretch at the end to push everything up
        layout.addStretch()

        page.setWidget(content)
        self.stacked_widget.addWidget(page)
    
    def switch_page(self, index):
        """Switch to the specified page index"""
        try:
            print(f"Switching to page {index}")
            # Store current index before switching
            current_index = self.stacked_widget.currentIndex()
            
            # Only switch if we're changing pages
            if current_index != index:
                # Update the stacked widget
                self.stacked_widget.setCurrentIndex(index)
                
                # Refresh the page content if needed
                if index == 2:  # Records page
                    print("Refreshing records page")
                    self.refresh_records()
                elif index == 0:  # Home page
                    self.refresh_market_overview()
                    
                # Update menu buttons
                menu_buttons = [self.home_btn, self.analysis_btn, self.records_btn, 
                              self.settings_btn, self.info_btn]
                              
                for i, btn in enumerate(menu_buttons):
                    if i == index:
                        btn.setChecked(True)
                    else:
                        btn.setChecked(False)
        except Exception as e:
            print(f"Error switching pages: {e}")
            import traceback
            print(traceback.format_exc())
            # If there's an error, stay on the current page
            if hasattr(self, 'stacked_widget'):
                self.stacked_widget.setCurrentIndex(current_index) 

    def update_interval(self):
        """Update the interval selection based on the selected period"""
        try:
            # Get current period and update valid intervals
            period = self.timerange_combo.currentText()
            valid_intervals = self.get_valid_intervals(period)
            
            # Store current selection if available
            current_interval = self.interval_combo.currentText() if self.interval_combo.count() > 0 else ""
            
            # Block signals to prevent multiple updates
            self.interval_combo.blockSignals(True)
            
            # Update interval options
            self.interval_combo.clear()
            for interval in valid_intervals:
                self.interval_combo.addItem(interval)
            
            # Restore previous selection if valid, otherwise select first item
            index = self.interval_combo.findText(current_interval)
            if index >= 0:
                self.interval_combo.setCurrentIndex(index)
            elif self.interval_combo.count() > 0:
                self.interval_combo.setCurrentIndex(0)
            
            self.interval_combo.blockSignals(False)
            
        except Exception as e:
            print(f"Error updating interval: {e}")
            import traceback
            print(traceback.format_exc())
            
    def prepare_initial_data(self):
        """Prepare initial data during splash screen loading
        This method is called during splash screen display to preload necessary data.
        """
        try:
            # Load settings
            self.settings = Settings()
            print("Settings loaded successfully")
            
            # Initialize any prediction models that need pre-loading
            # For example, loading model weights or parameters
            
            # Pre-load market symbols for faster first load
            # This can involve loading ticker lists, etc.
            
            # Initialize any caches for better performance
            print("Initial data preparation complete")
            
        except Exception as e:
            print(f"Error during initial data preparation: {e}")
            import traceback
            print(traceback.format_exc())