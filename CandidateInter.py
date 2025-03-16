import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QLineEdit, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal
import pinyin
import predict

class PinyinWorker(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        tokens = get_pinyin_words(self.text)
        self.result_ready.emit(tokens)

def get_pinyin_words(pinyin_text):
    user_id = "user@example.com"

    if pinyin.PrivacyController.create_consent_dialog():
        converter = pinyin.PinyinConverter(user_id=user_id)
    else:
        converter = pinyin.PinyinConverter(user_id=None)

    split_text = converter._split_pinyin(pinyin_text)
    emisssion_dict = converter.hmm_params.emission_dict.get(split_text[0], {})
    sorted_dict = sorted(emisssion_dict.items(), key=lambda item: -item[1])

    all_tokens = []
    for token, prob in sorted_dict:
        all_tokens.append(token)
    return all_tokens

def get_predict_words(predict_text):
    all_tokens = []
    for text in predict_text:
        results = predict.predict_with_finetuned(text)
        for res in results:
            for token, prob in zip(res['tokens'], res['probs']):
                all_tokens.append(token)
    return all_tokens

class CandidateWindow(QWidget):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.initUI()

    def initUI(self):
        self.setWindowTitle('输入法候选词窗口')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        label = QLabel('输入文本:', self)
        layout.addWidget(label)

        self.textInput = QLineEdit(self)
        layout.addWidget(self.textInput)

        self.listWidget = QListWidget(self)
        layout.addWidget(self.listWidget)

        self.processButton = QPushButton('处理文本', self)
        self.processButton.clicked.connect(self.processText)
        layout.addWidget(self.processButton)

        self.setLayout(layout)

    def processText(self):
        text = self.textInput.text()
        self.listWidget.clear()
        if self.mode == 'pinyin':
            self.worker = PinyinWorker(text)
            self.worker.result_ready.connect(self.displayResults)
            self.worker.start()
        elif self.mode == 'predict':
            predicted_tokens = get_predict_words([text])
            self.displayResults(predicted_tokens)

    def displayResults(self, tokens):
        for token in tokens:
            self.listWidget.addItem(token)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    candidateWindow = CandidateWindow(mode='pinyin')  # 或者 'predict'
    candidateWindow.show()

    sys.exit(app.exec_())