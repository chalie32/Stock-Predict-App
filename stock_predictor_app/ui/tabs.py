from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextBrowser, QGridLayout, QFrame, QListWidget, 
    QListWidgetItem, QScrollArea, QGroupBox
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QFont
from ..data.news_fetcher import get_news, NewsAPIException
import webbrowser

class CompanyInfoTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 10, 0, 10)
        
        # Create scrollable area for all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(24)
        content_layout.setContentsMargins(5, 5, 5, 10)
        
        # Add company header section
        self.header_frame = QFrame()
        self.header_frame.setObjectName("infoHeader")
        self.header_frame.setStyleSheet("""
            #infoHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                        stop:0 #1a1f29,
                                        stop:1 #222a3a);
                border: 1px solid #30363d;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 10px;
            }
        """)
        
        header_layout = QVBoxLayout(self.header_frame)
        header_layout.setContentsMargins(20, 20, 20, 20)
        
        self.company_name = QLabel("Select a Stock")
        self.company_name.setStyleSheet("""
            font-size: 26px;
            font-weight: bold;
            color: #e6edf3;
            margin-bottom: 5px;
        """)
        
        self.company_ticker = QLabel("")
        self.company_ticker.setStyleSheet("""
            font-size: 16px;
            color: #8b949e;
            margin-bottom: 10px;
        """)
        
        self.company_sector = QLabel("")
        self.company_sector.setStyleSheet("""
            font-size: 14px;
            color: #8b949e;
            margin-bottom: 5px;
        """)
        
        header_layout.addWidget(self.company_name)
        header_layout.addWidget(self.company_ticker)
        header_layout.addWidget(self.company_sector)
        
        # Add key metrics section
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)
        
        # Create frame for price metrics
        self.price_frame = self.create_metric_frame("Price", "$--", "")
        metrics_layout.addWidget(self.price_frame)
        
        # Create frame for market cap
        self.mktcap_frame = self.create_metric_frame("Market Cap", "--", "")
        metrics_layout.addWidget(self.mktcap_frame)
        
        # Create frame for PE ratio
        self.pe_frame = self.create_metric_frame("P/E Ratio", "--", "")
        metrics_layout.addWidget(self.pe_frame)
        
        header_layout.addLayout(metrics_layout)
        content_layout.addWidget(self.header_frame)
        
        # Create info section with improved layout
        info_frame = QFrame()
        info_frame.setObjectName("container")
        
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(20)
        
        # Create labels for company info in sections
        self.info_labels = {}
        
        # Basic Info Section
        basic_group = QGroupBox("Company Details")
        basic_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 15px;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        basic_layout = QGridLayout()
        basic_layout.setColumnStretch(1, 1)  # Make value column stretch
        basic_layout.setSpacing(14)
        basic_layout.setContentsMargins(15, 15, 15, 15)
        
        basic_fields = [
            ('industry', 'Industry'),
            ('country', 'Country'),
            ('exchange', 'Exchange'),
            ('website', 'Website'),
            ('employees', 'Employees')
        ]
        
        for i, (key, label) in enumerate(basic_fields):
            label_widget = QLabel(f"{label}")
            label_widget.setStyleSheet("font-weight: bold; color: #8b949e;")
            value_widget = QLabel("--")
            value_widget.setWordWrap(True)  # Enable word wrap for long text
            value_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            basic_layout.addWidget(label_widget, i, 0)
            basic_layout.addWidget(value_widget, i, 1)
            self.info_labels[key] = value_widget
        
        basic_group.setLayout(basic_layout)
        info_layout.addWidget(basic_group)
        
        # Market Data Section
        market_group = QGroupBox("Key Metrics & Statistics")
        market_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 15px;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        market_layout = QGridLayout()
        market_layout.setColumnStretch(1, 1)
        market_layout.setSpacing(14)
        market_layout.setContentsMargins(15, 15, 15, 15)
        
        market_fields = [
            ('dividend_yield', 'Dividend Yield'),
            ('beta', 'Beta'),
            ('52_week_high', '52 Week High'),
            ('52_week_low', '52 Week Low'),
            ('avg_volume', 'Avg Volume')
        ]
        
        for i, (key, label) in enumerate(market_fields):
            label_widget = QLabel(f"{label}")
            label_widget.setStyleSheet("font-weight: bold; color: #8b949e;")
            value_widget = QLabel("--")
            value_widget.setWordWrap(True)
            value_widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            market_layout.addWidget(label_widget, i, 0)
            market_layout.addWidget(value_widget, i, 1)
            self.info_labels[key] = value_widget
        
        market_group.setLayout(market_layout)
        info_layout.addWidget(market_group)
        
        content_layout.addWidget(info_frame)
        
        # Add description section with improved styling
        description_group = QGroupBox("Business Description")
        description_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 15px;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        description_layout = QVBoxLayout()
        description_layout.setContentsMargins(15, 15, 15, 15)
        
        self.description_browser = QTextBrowser()
        self.description_browser.setOpenExternalLinks(True)
        self.description_browser.setMinimumHeight(180)
        self.description_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 15px;
                color: #c9d1d9;
                font-size: 14px;
                line-height: 1.6;
            }
        """)
        description_layout.addWidget(self.description_browser)
        
        description_group.setLayout(description_layout)
        content_layout.addWidget(description_group)
        
        # Add additional space at the bottom
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
    def create_metric_frame(self, title, value, trend=None):
        """Create a metric frame for displaying important metrics"""
        frame = QFrame()
        frame.setObjectName("metricFrame")
        frame.setStyleSheet("""
            #metricFrame {
                background-color: rgba(22, 27, 34, 0.5);
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 12px;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-size: 13px;
            color: #8b949e;
        """)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #e6edf3;
        """)
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        if trend:
            trend_label = QLabel(trend)
            trend_label.setStyleSheet("""
                font-size: 13px;
                font-weight: bold;
                color: #4ade80;
            """)
            layout.addWidget(trend_label)
            frame.trend_label = trend_label
        
        # Store references for later updates
        frame.title_label = title_label
        frame.value_label = value_label
        
        return frame
    
    def update_info(self, info: dict):
        """Update the displayed company information"""
        # Update header information
        self.company_name.setText(info.get('name', 'Unknown Company'))
        self.company_ticker.setText(f"Ticker: {info.get('symbol', '--')}")
        self.company_sector.setText(f"Sector: {info.get('sector', '--')}")
        
        # Update key metrics
        if 'current_price' in info:
            self.price_frame.value_label.setText(f"${info['current_price']:.2f}")
            
            # Add trend if day change is available
            if 'day_change' in info and 'day_change_pct' in info:
                change = info['day_change']
                change_pct = info['day_change_pct']
                
                if hasattr(self.price_frame, 'trend_label'):
                    trend_text = f"{'↑' if change >= 0 else '↓'} {abs(change):.2f} ({abs(change_pct):.2f}%)"
                    self.price_frame.trend_label.setText(trend_text)
                    self.price_frame.trend_label.setStyleSheet(f"""
                        font-size: 13px;
                        font-weight: bold;
                        color: {'#4ade80' if change >= 0 else '#f87171'};
                    """)
        
        # Update market cap
        if 'market_cap' in info:
            self.mktcap_frame.value_label.setText(self.format_market_cap(info['market_cap']))
        
        # Update PE ratio
        if 'pe_ratio' in info:
            self.pe_frame.value_label.setText(f"{info['pe_ratio']:.2f}" if info['pe_ratio'] else "--")
        
        # Update basic info
        for key, label in self.info_labels.items():
            value = info.get(key, 'N/A')
            if key == 'website' and value != 'N/A':
                label.setText(f'<a href="{value}" style="color: #58a6ff; text-decoration: none;">{value}</a>')
                label.setOpenExternalLinks(True)
            elif key == 'dividend_yield' and value != 'N/A' and value:
                label.setText(f"{float(value):.2%}")
            elif key == 'avg_volume' and value != 'N/A' and value:
                label.setText(f"{int(value):,}")
            elif key == 'employees' and value != 'N/A' and value:
                label.setText(f"{int(value):,}")
            elif value not in ('N/A', None):
                label.setText(str(value))
        
        # Update description
        description = info.get('description', 'No description available.')
        self.description_browser.setPlainText(description)
    
    def format_market_cap(self, value):
        """Format market cap value for better readability"""
        try:
            if not value:
                return "--"
                
            num = float(str(value).replace(',', ''))
            if num >= 1e12:
                return f"${num/1e12:.2f}T"
            elif num >= 1e9:
                return f"${num/1e9:.2f}B"
            elif num >= 1e6:
                return f"${num/1e6:.2f}M"
            else:
                return f"${num:,.0f}"
        except (ValueError, TypeError):
            return str(value) if value else "--"

class NewsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.current_ticker = None
        self.current_company = None
        
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create header with status
        self.status_label = QLabel("Select a stock to view related news")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("sectionTitle")
        
        # Create error label
        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.setStyleSheet("color: #f85149; padding: 10px;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        
        # Create frame for news list
        news_frame = QFrame()
        news_frame.setObjectName("container")
        news_layout = QVBoxLayout(news_frame)
        news_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area for news list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Create list widget for news items
        self.news_list = QListWidget()
        self.news_list.setWordWrap(True)
        self.news_list.setSpacing(12)  # Increase spacing between items
        self.news_list.itemClicked.connect(self.open_article)
        
        # Set style for the news list
        self.news_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #30363d;
                border-radius: 8px;
            }
            QListWidget::item:hover {
                background-color: #21262d;
            }
            QListWidget::item:selected {
                background-color: #161b22;
                border: 1px solid #388bfd;
            }
        """)
        
        scroll_area.setWidget(self.news_list)
        news_layout.addWidget(scroll_area)
        
        # Add widgets to layout
        layout.addWidget(self.status_label)
        layout.addWidget(self.error_label)
        layout.addWidget(news_frame, 1)
        
        self.setLayout(layout)
        
    def update_news(self, ticker: str, company_name: str = None):
        """
        Update the news list with articles for the given ticker and company
        
        Args:
            ticker (str): Stock ticker symbol
            company_name (str, optional): Company name for better search results
        """
        self.current_ticker = ticker
        self.current_company = company_name
        self.news_list.clear()
        self.error_label.hide()
        
        if not ticker:
            self.status_label.setText("Please enter a stock symbol")
            return
            
        self.status_label.setText(f"Loading news for {ticker}...")
        
        try:
            articles = get_news(ticker, company_name)
            
            if not articles:
                self.status_label.setText(f"No news found for {ticker}")
                self.error_label.setText("Tip: Make sure you have set the NEWS_API_KEY environment variable and have an active NewsAPI subscription.")
                self.error_label.show()
                return
                
            company_display = f"{company_name} ({ticker})" if company_name else ticker
            self.status_label.setText(f"Latest News for {company_display}")
            
            # Add articles to the list widget
            for article in articles:
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 5, 0, 5)
                container_layout.setSpacing(8)
                
                # Title label with link styling
                title_label = QLabel()
                title_label.setText(f'<a href="{article["url"]}" style="color: #58a6ff; text-decoration: none; font-size: 16px; font-weight: bold;">{article["title"]}</a>')
                title_label.setOpenExternalLinks(True)
                title_label.setWordWrap(True)
                container_layout.addWidget(title_label)
                
                # Source and date
                meta_label = QLabel(f'<span style="color: #388bfd;">{article["source"]}</span> · <span style="color: #8b949e;">{article["published_date"]}</span>')
                container_layout.addWidget(meta_label)
                
                # Description
                desc_label = QLabel(article["description"])
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("color: #c9d1d9; margin-top: 8px;")
                container_layout.addWidget(desc_label)
                
                # Create list item and set the custom widget
                item = QListWidgetItem(self.news_list)
                item.setSizeHint(container.sizeHint())
                self.news_list.setItemWidget(item, container)
                
        except NewsAPIException as e:
            self.status_label.setText(f"Error fetching news for {ticker}")
            self.error_label.setText(str(e) + "\n\nTo fix this:\n1. Get an API key from newsapi.org\n2. Set the NEWS_API_KEY environment variable")
            self.error_label.show()
            
    def open_article(self, item):
        """Open the article URL in the default web browser"""
        url = item.data(Qt.ItemDataRole.UserRole)
        QDesktopServices.openUrl(QUrl(url)) 