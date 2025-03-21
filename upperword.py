from pynput import keyboard
import time

class InputMethod:
    def __init__(self):
        self.listener = keyboard.Listener(on_press=self.on_keyboard_event)
        self.controller = keyboard.Controller()
        self.is_inserting = False
        self.listener.start()

    def on_keyboard_event(self, key):
        if self.is_inserting:
            return True  # 如果正在插入字符，忽略键盘事件

        try:
            if key == keyboard.Key.esc:
                # ESC键退出
                return False

            if hasattr(key, 'char') and key.char is not None:
                char = key.char
                # 示例：将输入的字母转换为大写
                if char.isalpha():
                    char = char.upper()
                    self.insert_text(char)
                    return True  # 阻止默认输入

        except AttributeError:
            pass

        return True  # 继续监听

    def insert_text(self, text):
        self.is_inserting = True  # 标记正在插入字符
        self.listener.stop()  # 停止监听器
        # 删除已经输入的小写字母
        self.controller.press(keyboard.Key.backspace)
        self.controller.release(keyboard.Key.backspace)  
        # 使用 keyboard.Controller 进行输入
        for char in text:
            self.controller.press(char)
            self.controller.release(char)
        self.is_inserting = False  # 插入完成，取消标记
        self.listener = keyboard.Listener(on_press=self.on_keyboard_event)  # 重新启动监听器
        self.listener.start()

    def start(self):
        # 进入消息循环
        while True:
            self.listener.join()

if __name__ == "__main__":
    input_method = InputMethod()
    input_method.start()