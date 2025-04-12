import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QLineEdit, QListWidgetItem, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon
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
        self.candidate_items = []  # 用于存储所有候选词
        self.page_size = 7  # 每页显示的候选词数量
        self.current_page = 0  # 当前页码
        self.setWindowIcon(QIcon('InputmethodLogo.ico'))
        self.initUI()

    def initUI(self):
        self.setWindowTitle('邮易输入法')
        self.setGeometry(800, 300, 400, 500)  # 调整窗口高度以容纳按钮

        layout = QVBoxLayout()

        layout.addWidget(QLabel('输入文本:', self))
        self.textInput = QLineEdit(self)
        layout.addWidget(self.textInput)

        self.listWidget = QListWidget(self)
        self.listWidget.itemClicked.connect(self.insert_to_foreground_window)
        layout.addWidget(self.listWidget)

        # 添加翻页按钮
        self.prevButton = QPushButton('上一页', self)
        self.prevButton.clicked.connect(self.go_to_previous_page)
        layout.addWidget(self.prevButton)

        self.nextButton = QPushButton('下一页', self)
        self.nextButton.clicked.connect(self.go_to_next_page)
        layout.addWidget(self.nextButton)

        self.setLayout(layout)
        self.key_catch_thread.start()

    def go_to_previous_page(self):
        """翻到上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_candidates()

    def go_to_next_page(self):
        """翻到下一页"""
        max_page = (len(self.candidate_items) - 1) // self.page_size
        if self.current_page < max_page:
            self.current_page += 1
            self.update_candidates()

    def handle_key_press(self, char):
        self.foreground_window = win32gui.GetForegroundWindow()

        if char == '\b':  # 检测 Backspace 键
            if self.current_pinyin:
                self.current_pinyin = self.current_pinyin[:-1]  # 删除最后一个字符
        elif char == '\000':
            index = 0
            if 0 <= index < len(self.candidate_items):
                self.insert_to_foreground_window(self.candidate_items[index])
                return
        elif char.isdigit():  # 检测数字键
            index = int(char) - 1  # 将数字转换为索引（从 0 开始）
            if 0 <= index < len(self.candidate_items):  # 检查索引是否有效
                self.insert_to_foreground_window(self.candidate_items[index])
                return
        else:
            self.current_pinyin += char  # 记录输入的拼音

        self.textInput.setText(self.current_pinyin)  # 更新输入框内容

        # 实时处理拼音并更新候选词
        self.update_candidates()
        self.update_window_position()
        self.set_window_on_top(True)

    def update_candidates(self):
        """实时更新候选词"""
        # 调用拼音处理函数生成候选词
        if self.mode == 'pinyin':
            self.candidate_items = get_pinyin_words(self.current_pinyin)
        elif self.mode == 'predict':
            self.candidate_items = get_predict_words([self.current_pinyin])
        else:
            self.candidate_items = []

        # 更新当前页的候选词
        self.listWidget.clear()
        start_index = self.current_page * self.page_size
        end_index = start_index + self.page_size
        page_items = self.candidate_items[start_index:end_index]  # 获取当前页的候选词
        for i, token in enumerate(page_items):
            item_text = f"{(start_index + i) % 7 + 1}. {token}"  # 为候选词添加编号
            item = QListWidgetItem(item_text)
            self.listWidget.addItem(item)

    def insert_to_foreground_window(self, item_or_text):
        if isinstance(item_or_text, QListWidgetItem):
            text = item_or_text.text().split('. ', 1)[-1]  # 提取候选词文本
        else:
            text = item_or_text  # 如果直接传入候选词文本

        if self.foreground_window:
            win32gui.SetForegroundWindow(self.foreground_window)

            self.key_catch_thread.stop()
            # 删除拼音
            for _ in range(len(self.current_pinyin) + 1):
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

    def update_window_position(self):
        if self.foreground_window:
            caret_pos = win32gui.GetCaretPos()  # 光标相对窗口的位置
            rect = win32gui.GetWindowRect(self.foreground_window)  # 获取窗口的绝对位置
            x = rect[0] + caret_pos[0]
            y = rect[1] + caret_pos[1] + 240
            self.move(x, y)  # 移动候选词窗口到指定位置

    def set_window_on_top(self, on_top):
        if on_top:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()  # 重新显示窗口以应用更改
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    candidateWindow = CandidateWindow(mode='pinyin')  # 或者 'predict'
    candidateWindow.show()
    sys.exit(app.exec_())