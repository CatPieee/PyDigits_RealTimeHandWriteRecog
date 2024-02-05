# Copyright (C) 2022 The Qt Company Ltd.
# SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

from PySide6.QtWidgets import QApplication
import sys

from mainwindows import MyMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)                # sys.argv是一个命令行参数列表，第一个参数是程序本身，随后是传递给程序的参数
    w = MyMainWindow()
    w.show()
    sys.exit(app.exec())
