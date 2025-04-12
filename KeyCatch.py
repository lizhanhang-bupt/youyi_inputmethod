from pynput import keyboard
from PyQt5.QtCore import pyqtSignal, QObject
import threading

class KeyCatch(QObject):
    key_pressed = pyqtSignal(str)  # 定义信号，用于传递捕获的键盘输入

    def __init__(self):
        super().__init__()
        self.listener = None  # 初始化为 None
        self.inputLock = threading.Lock()
        self.is_inserting = False

    def on_keyboard_event(self, key):
        if self.is_inserting:
            return True  # 如果正在插入字符，忽略键盘事件

        try:
            if hasattr(key, 'char') and key.char is not None:
                char = key.char
                self.key_pressed.emit(char)  # 通过信号传递捕获的字符  
            elif key == keyboard.Key.backspace:  # 捕获 Backspace 键
                self.key_pressed.emit('\b')  # 传递 '\b' 表示 Backspace
            elif key == keyboard.Key.space:
                self.key_pressed.emit('\000')
            elif key == keyboard.Key.enter:
                self.key_pressed.emit('\r')
            elif key == keyboard.Key.page_up:
                self.key_pressed.emit('\x21')
            elif key == keyboard.Key.page_down:
                self.key_pressed.emit('\x22')
            elif key == keyboard.Key.home:
                self.key_pressed.emit('\x24')
            elif key == keyboard.Key.end:
                self.key_pressed.emit('\x23')
            return True
        except AttributeError:
            pass

        return True  # 继续监听

    def start(self):
        # 每次启动时创建新的 Listener 实例
        self.listener = keyboard.Listener(on_press=self.on_keyboard_event)
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None  # 停止后将 listener 设置为 None