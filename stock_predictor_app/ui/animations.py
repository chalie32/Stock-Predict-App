from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QPoint, QSize, QSequentialAnimationGroup, QParallelAnimationGroup, QTimer, Qt
from PyQt6.QtWidgets import QWidget

class FadeAnimation:
    """Create fade in/out animations for widgets"""
    
    @staticmethod
    def fade_in(widget, duration=300, callback=None):
        """Fade in a widget gradually"""
        widget.setWindowOpacity(0.0)
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        if callback:
            animation.finished.connect(callback)
        animation.start()
        return animation
    
    @staticmethod
    def fade_out(widget, duration=300, callback=None):
        """Fade out a widget gradually"""
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.InCubic)
        if callback:
            animation.finished.connect(callback)
        animation.start()
        return animation

class SlideAnimation:
    """Create slide animations for widgets"""
    
    @staticmethod
    def slide_in(widget, direction="right", duration=350, callback=None):
        """Slide in a widget from a direction"""
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        start_pos = widget.pos()
        if direction == "right":
            off_pos = QPoint(widget.width(), start_pos.y())
        elif direction == "left":
            off_pos = QPoint(-widget.width(), start_pos.y())
        elif direction == "up":
            off_pos = QPoint(start_pos.x(), widget.height())
        elif direction == "down":
            off_pos = QPoint(start_pos.x(), -widget.height())
        
        animation.setStartValue(off_pos)
        animation.setEndValue(start_pos)
        
        if callback:
            animation.finished.connect(callback)
        
        animation.start()
        return animation
    
    @staticmethod
    def slide_out(widget, direction="right", duration=350, callback=None):
        """Slide out a widget to a direction"""
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.Type.InCubic)
        
        start_pos = widget.pos()
        if direction == "right":
            off_pos = QPoint(widget.width(), start_pos.y())
        elif direction == "left":
            off_pos = QPoint(-widget.width(), start_pos.y())
        elif direction == "up":
            off_pos = QPoint(start_pos.x(), -widget.height())
        elif direction == "down":
            off_pos = QPoint(start_pos.x(), widget.height())
        
        animation.setStartValue(start_pos)
        animation.setEndValue(off_pos)
        
        if callback:
            animation.finished.connect(callback)
        
        animation.start()
        return animation

class PulseAnimation:
    """Create pulse/highlight animations for widgets"""
    
    @staticmethod
    def pulse(widget, scale_factor=1.05, duration=300, callback=None):
        """Create a subtle pulse animation for a widget"""
        # Animation for growing
        grow_anim = QPropertyAnimation(widget, b"size")
        grow_anim.setDuration(duration // 2)
        grow_anim.setStartValue(widget.size())
        grow_anim.setEndValue(QSize(int(widget.width() * scale_factor), 
                                  int(widget.height() * scale_factor)))
        grow_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        # Animation for shrinking back
        shrink_anim = QPropertyAnimation(widget, b"size")
        shrink_anim.setDuration(duration // 2)
        shrink_anim.setStartValue(QSize(int(widget.width() * scale_factor), 
                                     int(widget.height() * scale_factor)))
        shrink_anim.setEndValue(widget.size())
        shrink_anim.setEasingCurve(QEasingCurve.Type.InQuad)
        
        # Combine animations
        sequence = QSequentialAnimationGroup()
        sequence.addAnimation(grow_anim)
        sequence.addAnimation(shrink_anim)
        
        if callback:
            sequence.finished.connect(callback)
        
        sequence.start()
        return sequence

class CardLoadingAnimation:
    """Create loading animation for dashboard cards"""
    
    @staticmethod
    def animate_cards(cards, delay_between=80):
        """Animate a list of cards one after another with delay"""
        animations = QParallelAnimationGroup()
        
        for i, card in enumerate(cards):
            # Start with reduced opacity and slight offset
            card.setWindowOpacity(0.0)
            original_pos = card.pos()
            card.move(original_pos.x(), original_pos.y() + 20)
            
            # Create fade in animation
            fade_anim = QPropertyAnimation(card, b"windowOpacity")
            fade_anim.setDuration(350)
            fade_anim.setStartValue(0.0)
            fade_anim.setEndValue(1.0)
            fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            
            # Create move up animation
            move_anim = QPropertyAnimation(card, b"pos")
            move_anim.setDuration(350)
            move_anim.setStartValue(QPoint(original_pos.x(), original_pos.y() + 20))
            move_anim.setEndValue(original_pos)
            move_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            
            # Group individual card animations
            card_anim_group = QParallelAnimationGroup()
            card_anim_group.addAnimation(fade_anim)
            card_anim_group.addAnimation(move_anim)
            
            # Add delay based on card index
            QTimer.singleShot(i * delay_between, lambda group=card_anim_group: group.start())
            
            animations.addAnimation(card_anim_group)
        
        return animations

class PageTransition:
    """Create transitions between pages in stacked widget"""
    
    @staticmethod
    def slide_transition(stacked_widget, next_index, direction="left", duration=300):
        """Slide transition between pages in a stacked widget"""
        # Get current and next widgets
        current_widget = stacked_widget.currentWidget()
        stacked_widget.setCurrentIndex(next_index)
        next_widget = stacked_widget.currentWidget()
        
        # Capture current widget position
        current_pos = current_widget.pos()
        
        # Set initial positions
        if direction == "left":
            # Next widget starts from right
            next_widget.move(stacked_widget.width(), 0)
            end_pos_current = QPoint(-current_widget.width(), 0)
        elif direction == "right":
            # Next widget starts from left
            next_widget.move(-next_widget.width(), 0)
            end_pos_current = QPoint(stacked_widget.width(), 0)
        
        # Animation for current widget
        current_anim = QPropertyAnimation(current_widget, b"pos")
        current_anim.setDuration(duration)
        current_anim.setStartValue(current_pos)
        current_anim.setEndValue(end_pos_current)
        current_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Animation for next widget
        next_anim = QPropertyAnimation(next_widget, b"pos")
        next_anim.setDuration(duration)
        next_anim.setStartValue(next_widget.pos())
        next_anim.setEndValue(QPoint(0, 0))
        next_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # Run animations in parallel
        anim_group = QParallelAnimationGroup()
        anim_group.addAnimation(current_anim)
        anim_group.addAnimation(next_anim)
        anim_group.start()
        
        return anim_group 