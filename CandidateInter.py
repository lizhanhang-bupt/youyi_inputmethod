import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QLineEdit, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal
import win32gui
from pynput.keyboard import Controller, Key
from KeyCatch import KeyCatch
import pinyin
import predict
import time

class PinyinWorker(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        tokens = get_pinyin_words(self.text)
        self.result_ready.emit(tokens)

def get_pinyin_words(pinyin_text):
    user_id = "user.example.com"
    converter = pinyin.PinyinConverter(user_id=user_id)
    sorted_dict = converter.convert(pinyin_text)
    return [item['text'] for item in sorted_dict]

def get_predict_words(predict_text):
    all_tokens = []
    for text in predict_text:
        results = predict.predict_with_finetuned(text)
        for res in results:
            all_tokens.extend(res['tokens'])
    return all_tokens

class KeyCatchThread(QThread):
    key_pressed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.key_catch = KeyCatch()
        self.key_catch.key_pressed.connect(self.key_pressed.emit)

    def run(self):
        self.key_catch.start()

    def stop(self):
        self.key_catch.stop()

class CandidateWindow(QWidget):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.key_catch_thread = KeyCatchThread()
        self.key_catch_thread.key_pressed.connect(self.handle_key_press)
        self.foreground_window = None
        self.keyboard_controller = Controller()
        self.current_pinyin = ""  # 用于记录当前输入的拼音
        self.initUI()

    def initUI(self):
        self.setWindowTitle('输入法候选词窗口')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        layout.addWidget(QLabel('输入文本:', self))
        self.textInput = QLineEdit(self)
        layout.addWidget(self.textInput)

        self.listWidget = QListWidget(self)
        self.listWidget.itemClicked.connect(self.insert_to_foreground_window)
        layout.addWidget(self.listWidget)

        self.processButton = QPushButton('处理文本', self)
        self.processButton.clicked.connect(self.processText)
        layout.addWidget(self.processButton)

        self.setLayout(layout)
        self.key_catch_thread.start()

    def closeEvent(self, event):
        self.key_catch_thread.stop()
        self.key_catch_thread.wait()
        super().closeEvent(event)

    def handle_key_press(self, char):
        self.foreground_window = win32gui.GetForegroundWindow()
        self.current_pinyin += char  # 记录输入的拼音
        self.textInput.setText(self.textInput.text() + char)

    def processText(self):
        text = self.textInput.text()
        self.listWidget.clear()
        if self.mode == 'pinyin':
            self.worker = PinyinWorker(text)
            self.worker.result_ready.connect(self.displayResults)
            self.worker.start()
        elif self.mode == 'predict':
            self.displayResults(get_predict_words([text]))

    def displayResults(self, tokens):
        for token in tokens:
            self.listWidget.addItem(token)

    def insert_to_foreground_window(self, item):
        if self.foreground_window:
            text = item.text()  # 获取点击的 Item 的文本
            win32gui.SetForegroundWindow(self.foreground_window)

            self.key_catch_thread.stop()
            # 删除拼音
            for _ in range(len(self.current_pinyin)):
                self.keyboard_controller.press(Key.backspace)
                self.keyboard_controller.release(Key.backspace)
                time.sleep(0.01)

            # 插入汉字
            for char in text:
                self.keyboard_controller.type(char)

            # 清空当前拼音记录
            self.current_pinyin = ""
            self.textInput.clear()
            self.key_catch_thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    candidateWindow = CandidateWindow(mode='pinyin')  # 或者 'predict'
    candidateWindow.show()
    sys.exit(app.exec_())