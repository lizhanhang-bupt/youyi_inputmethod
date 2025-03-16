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
        key_map = {
            Qt.Key_A: 'A',
            Qt.Key_B: 'B',
            Qt.Key_C: 'C',
            Qt.Key_D: 'D',
            Qt.Key_E: 'E',
            Qt.Key_F: 'F',
            Qt.Key_G: 'G',
            Qt.Key_H: 'H',
            Qt.Key_I: 'I',
            Qt.Key_J: 'J',
            Qt.Key_K: 'K',
            Qt.Key_L: 'L',
            Qt.Key_M: 'M',
            Qt.Key_N: 'N',
            Qt.Key_O: 'O',
            Qt.Key_P: 'P',
            Qt.Key_Q: 'Q',
            Qt.Key_R: 'R',
            Qt.Key_S: 'S',
            Qt.Key_T: 'T',
            Qt.Key_U: 'U',
            Qt.Key_V: 'V',
            Qt.Key_W: 'W',
            Qt.Key_X: 'X',
            Qt.Key_Y: 'Y',
            Qt.Key_Z: 'Z',
            Qt.Key_0: '0',
            Qt.Key_1: '1',
            Qt.Key_2: '2',
            Qt.Key_3: '3',
            Qt.Key_4: '4',
            Qt.Key_5: '5',
            Qt.Key_6: '6',
            Qt.Key_7: '7',
            Qt.Key_8: '8',
            Qt.Key_9: '9',
        }
        return key_map.get(key)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = KeyboardCaptureWindow()
    window.show()

    sys.exit(app.exec_())