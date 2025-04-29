import sys
import time
from PyQt6.QtWidgets import QApplication, QStyleFactory, QSplashScreen
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap, QColor
from stock_predictor_app.ui.main_window import MainWindow
from stock_predictor_app.ui.splash_screen import SplashScreen
from stock_predictor_app import config  # Import config to set up environment variables

def main():
    app = QApplication(sys.argv)
    
    # Set Fusion style for a modern look
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # Create a simple splash screen
    splash = SplashScreen()
    splash.show()
    
    # Process events to make sure splash is displayed
    app.processEvents()
    
    # Create main window but don't show it yet
    window = MainWindow()
    
    # Start loading process
    splash.update_progress_step(0)
    app.processEvents()
    
    # Step 1: Initialize app components
    time.sleep(0.3)  # Short delay for visual effect
    splash.update_progress_step(1)  # Loading market data
    app.processEvents()
    
    # Step 2: Load prediction models
    time.sleep(0.3)  # Short delay for visual effect
    splash.update_progress_step(2)
    app.processEvents()
    
    # Step 3: Prepare UI components
    time.sleep(0.3)  # Short delay for visual effect
    splash.update_progress_step(3)
    app.processEvents()
    
    # Step 4: Load initial data
    time.sleep(0.3)  # Short delay for visual effect
    window.prepare_initial_data()
    splash.update_progress_step(4)  # Ready to launch
    app.processEvents()
    
    # Final delay for visual effect
    time.sleep(1.0)
    
    # Update progress to 100%
    splash.update_progress(100)
    app.processEvents()
    time.sleep(1.0)
    # Directly transition to main window
    splash.close()
    window.show()
    
    # Schedule market data refresh after a delay
    QTimer.singleShot(500, window.refresh_market_overview)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 