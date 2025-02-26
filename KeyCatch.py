import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt5.QtCore import Qt

class KeyboardCaptureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('键盘捕获示例')
        self.setGeometry(100, 100, 400, 200)

        self.layout = QVBoxLayout()

        self.label = QLabel('请按下任意键...', self)
        self.layout.addWidget(self.label)

        self.textEdit = QTextEdit(self)
        self.textEdit.setReadOnly(True)
        self.layout.addWidget(self.textEdit)

        self.setLayout(self.layout)

    def keyPressEvent(self, event):
        key = event.key()
        key_name = self.getKeyName(key)
        self.label.setText(f'按下的键: {key_name}')
        self.textEdit.insertPlainText(f'{key_name}')

    def getKeyName(self, key):
        if key == Qt.Key_A:
            return 'A'
        elif key == Qt.Key_B:
            return 'B'
        # 添加其他键的处理逻辑
        else:
            return f'Key {key}'

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = KeyboardCaptureWindow()
    window.show()

    sys.exit(app.exec_())