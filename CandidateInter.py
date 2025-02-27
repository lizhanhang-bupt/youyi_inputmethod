import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QHBoxLayout, QPushButton

class CandidateWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('输入法候选词窗口')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        label = QLabel('候选词列表', self)
        layout.addWidget(label)

        self.listWidget = QListWidget(self)
        self.listWidget.addItem('你好')
        self.listWidget.addItem('世界')
        self.listWidget.addItem('输入法')
        layout.addWidget(self.listWidget)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    candidateWindow = CandidateWindow()
    candidateWindow.show()

    sys.exit(app.exec_())