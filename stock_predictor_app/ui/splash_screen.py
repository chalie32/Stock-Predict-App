from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient

class SplashScreen(QSplashScreen):
    """Custom splash screen with progress bar and animation"""
    finished = pyqtSignal()
    
    def __init__(self):
        # Create a pixmap for the splash screen background
        pixmap = QPixmap(800, 500)
        pixmap.fill(QColor("#0d1117"))
        super().__init__(pixmap)
        
        # Status message and current progress
        self.status_text = "Initializing..."
        self.current_progress = 0
        self.loading_steps = [
            "Initializing application...",
            "Loading market data...",
            "Setting up prediction models...",
            "Preparing user interface...",
            "Ready to launch!"
        ]
        self.current_step = 0
    
    def drawContents(self, painter):
        """Custom drawing of splash screen contents"""
        # Draw title
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 42, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "\n\nStock Predictor")
        
        # Draw subtitle
        painter.setPen(QColor("#8b949e"))
        painter.setFont(QFont("Arial", 18))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "\n\n\n\nAdvanced stock market analysis platform")
        
        # Draw status text
        painter.setPen(QColor("#58a6ff"))
        painter.setFont(QFont("Arial", 14))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, self.status_text + "\n\n\n")
        
        # Draw progress bar
        progress_rect = self.rect()
        progress_rect.setTop(progress_rect.bottom() - 50)
        progress_rect.setLeft(progress_rect.left() + 100)
        progress_rect.setRight(progress_rect.right() - 100)
        progress_rect.setHeight(6)
        
        # Background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#30363d"))
        painter.drawRoundedRect(progress_rect, 3, 3)
        
        # Progress chunk
        if self.current_progress > 0:
            chunk_rect = progress_rect
            chunk_rect.setWidth(int(progress_rect.width() * (self.current_progress / 100.0)))
            painter.setBrush(QColor("#1f6feb"))
            painter.drawRoundedRect(chunk_rect, 3, 3)
    
    def update_progress(self, value):
        """Update the progress bar with a specific value (0-100)"""
        self.current_progress = min(value, 100)
        print(f"Splash screen progress: {self.current_progress}%")
        self.repaint()
        
        # For compatibility, still emit the signal at 100%
        if self.current_progress >= 100:
            print("Emitting finished signal, but transition will be handled directly")
            self.finished.emit()
    
    def update_progress_step(self, step_index):
        """Update progress to show a specific initialization step"""
        # Validate step index
        if step_index < 0 or step_index >= len(self.loading_steps):
            return
            
        # Update step and message
        self.current_step = step_index
        self.status_text = self.loading_steps[step_index]
        
        # Calculate progress based on step
        # Each step represents 20% of the total progress (5 steps total)
        self.current_progress = (self.current_step * 20)
        print(f"Splash step {step_index}: {self.status_text} ({self.current_progress}%)")
        
        # Force redraw
        self.repaint() 