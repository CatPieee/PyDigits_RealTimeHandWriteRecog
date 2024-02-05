from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QPaintEvent, QPen, QPainter, QPixmap

class PainterWidget(QWidget):
    """A widget where user can draw with their mouse

    The user draws on a QPixmap which is itself paint from paintEvent()

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(680, 480)
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.white)

        self.previous_pos = None
        self.painter = QPainter()
        self.pen = QPen()
        self.pen.setWidth(10)
        self.pen.setCapStyle(Qt.RoundCap)
        self.pen.setJoinStyle(Qt.RoundJoin)

    def paintEvent(self, event: QPaintEvent):
        """Override method from QWidget

        Paint the Pixmap into the widget

        """
        with QPainter(self) as painter:
            painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, event: QMouseEvent):
        """Override from QWidget

        Called when user clicks on the mouse

        """
        self.previous_pos = event.position().toPoint()
        QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Override method from QWidget

        Called when user moves and clicks on the mouse

        """
        current_pos = event.position().toPoint()
        self.painter.begin(self.pixmap)
        self.painter.setRenderHints(QPainter.Antialiasing, True)
        self.painter.setPen(self.pen)
        self.painter.drawLine(self.previous_pos, current_pos)
        self.painter.end()

        self.previous_pos = current_pos
        self.update()

        QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Override method from QWidget

        Called when user releases the mouse

        """
        self.previous_pos = None
        QWidget.mouseReleaseEvent(self, event)

    def save(self, filename: str):
        """ save pixmap to filename """
        self.pixmap.save(filename)

    def load(self, filename: str):
        """ load pixmap from filename """
        self.pixmap.load(filename)
        self.pixmap = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        self.update()

    def clear(self):
        """ Clear the pixmap """
        self.pixmap.fill(Qt.white)
        self.update()

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = PainterWidget()
    window.show()
    sys.exit(app.exec_())