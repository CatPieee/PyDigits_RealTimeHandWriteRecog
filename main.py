import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QPlainTextEdit, QFileDialog, QMessageBox, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt
from ui.Ui_mainwindow import Ui_MainWindow
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from net import MyNet

class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 为按钮添加槽函数
        self.pushButton.clicked.connect(self.readImage)
        self.pushButton_2.clicked.connect(self.recognize)
        self.pushButton_3.clicked.connect(self.writeNumber)
        self.pushButton_4.clicked.connect(self.recognize_handwrite)

        # 深度学习推理准备
        self.image = None
        self.network = MyNet()
        self.network.load_state_dict(torch.load('model\\model.pth'))
        self.network.eval()

    # 弹出文件对话框，选择图片
    def readImage(self):
        print('准备读取图片')
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec():
            file_name = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(file_name)
            self.label = QLabel(self)
            self.label.setGeometry(10, 10, 361, 361)
            self.label.setPixmap(pixmap.scaled(self.label.size()))
            self.label.setScaledContents(True)
            self.image = pixmap.toImage()
        # 在垂直布局中显示label
        # self.label.show()
        # 将label铺满整个垂直布局
        self.verticalLayout.addWidget(self.label)

    # 识别图片，显示在label_2中
    def recognize(self):
        if self.image == None:
            QMessageBox.information(self, '提示', '没有加载图片')
            return
        self.image = self.qimage2tensor(self.image)
        pred = self.predict(self.image)
        self.label_2.setText('预测结果：' + str(pred.item()))

    # 将QImage转换为Tensor
    def qimage2tensor(self, qimage):
        pImg = Image.fromqpixmap(qimage)
        pImg = pImg.convert('L')        # 转换为灰度图
        pImg = pImg.resize((28, 28))    # 调整大小为28*28
        tensor = ToTensor()(pImg)
        tensor = tensor.unsqueeze(0)
        return tensor
    
    # 完成一次预测
    def predict(self, tensor):
        with torch.no_grad():
            output = self.network(tensor)
            pred = output.argmax(dim=1, keepdim=True)
            return pred
    
    def writeNumber(self):
        self.label_2.setText('写数字')
        self.label_2.setStyleSheet('color: red')

        self.label = PainterLabel(self)
        self.label.setGeometry(10, 10, 361, 361)
        self.label.setStyleSheet("border: 2px solid red")
        self.label.show()

        self.label_2.setText('请在左边的画板上写一个数字')
        self.label_2.setStyleSheet('color: blue')
    
    def recognize_handwrite(self):
        if self.label.pixmap.isNull():
            QMessageBox.information(self, '提示', '没有写数字')
            return
        self.image = self.label.pixmap.toImage()
        self.image = self.qimage2tensor(self.image)
        pred = self.predict(self.image)
        self.label_2.setText('预测结果：' + str(pred.item()))
        self.label_2.setStyleSheet('color: red')

class PainterLabel(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    def __init__(self, parent):
        super(PainterLabel,self).__init__(parent)
        print('初始化PainterLabel')
        self.pixmap = QPixmap(361, 361)
        self.pixmap.fill(Qt.black)
        self.setStyleSheet("border: 2px solid red")
        self.Color = Qt.white
        self.penwidth = 15

    def paintEvent(self, event):
        painter = QPainter(self.pixmap)
        painter.setPen(QPen(self.Color, self.penwidth, Qt.SolidLine))
        painter.drawLine(self.x0, self.y0, self.x1, self.y1)

        Label_painter = QPainter(self)
        Label_painter.drawPixmap(2, 2, self.pixmap)
    
    def mousePressEvent(self, event):
        self.x1 = event.x()       # deprecate
        self.y1 = event.y()
        self.flag = True
    
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x0 = self.x1
            self.y0 = self.y1
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec())
