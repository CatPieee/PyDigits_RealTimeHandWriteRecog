from PySide6.QtWidgets import QWidget, QMainWindow, QApplication, QFileDialog, QStyle, QColorDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Slot, QStandardPaths
from PySide6.QtGui import QMouseEvent, QPaintEvent, QPen, QAction, QPainter, QColor, QPixmap, QIcon, QKeySequence

from widgets import PainterWidget

class MainWindow(QMainWindow):
    """An Application example to draw using a pen """

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.painter_widget = PainterWidget()
        self.bar = self.addToolBar("Menu")
        self.bar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._save_action = self.bar.addAction(
            QApplication.style().standardIcon(QStyle.SP_DialogSaveButton), "Save", self.on_save
        )
        self._save_action.setShortcut(QKeySequence.Save)
        self._open_action = self.bar.addAction(
            QApplication.style().standardIcon(QStyle.SP_DialogOpenButton), "Open", self.on_open
        )
        self._open_action.setShortcut(QKeySequence.Open)
        self.bar.addAction(
            QApplication.style().standardIcon(QStyle.SP_DialogResetButton),
            "Clear",
            self.painter_widget.clear,
        )
        self.bar.addSeparator()

        self.color_action = QAction(self)
        self.color_action.triggered.connect(self.on_color_clicked)
        self.bar.addAction(self.color_action)

        self.setCentralWidget(self.painter_widget)

        self.color = Qt.black
        self.set_color(self.color)

        self.mime_type_filters = ["image/png", "image/jpeg"]

    @Slot()
    def on_save(self):

        dialog = QFileDialog(self, "Save File")
        dialog.setMimeTypeFilters(self.mime_type_filters)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setDefaultSuffix("png")
        dialog.setDirectory(
            QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        )

        if dialog.exec() == QFileDialog.Accepted:
            if dialog.selectedFiles():
                self.painter_widget.save(dialog.selectedFiles()[0])

    @Slot()
    def on_open(self):

        dialog = QFileDialog(self, "Save File")
        dialog.setMimeTypeFilters(self.mime_type_filters)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setDefaultSuffix("png")
        dialog.setDirectory(
            QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        )

        if dialog.exec() == QFileDialog.Accepted:
            if dialog.selectedFiles():
                self.painter_widget.load(dialog.selectedFiles()[0])

    @Slot()
    def on_color_clicked(self):

        color = QColorDialog.getColor(self.color, self)

        if color:
            self.set_color(color)

    def set_color(self, color: QColor = Qt.black):

        self.color = color
        # Create color icon
        pix_icon = QPixmap(32, 32)
        pix_icon.fill(self.color)

        self.color_action.setIcon(QIcon(pix_icon))
        self.painter_widget.pen.setColor(self.color)
        self.color_action.setText(QColor(self.color).name())

class MyMainWindow(MainWindow):
    """New customized main window class"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create and set up PainterWidget
        self.painter_widget = PainterWidget()

        # Create and set up the label
        self.pushButton = QPushButton("hello")
        self.pushButton.clicked.connect(self.hello)
        self.label = QLabel("Right Panel")

        # Create layout for the main window
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.painter_widget)

        # Create layout for the right panel
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.pushButton)
        right_layout.addWidget(self.label)

        # Add the right panel layout to the main layout
        main_layout.addLayout(right_layout)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)

        # Set the central widget of the main window
        self.setCentralWidget(central_widget)
    
    def hello(self):
        self.label.setText("Hello World")

if __name__ == "__main__":
    app = QApplication([])
    w = MyMainWindow()
    w.show()
    app.exec()