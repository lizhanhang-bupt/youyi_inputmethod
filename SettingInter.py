import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QListWidget, QHBoxLayout, QPushButton, QDialog, QFormLayout, QLineEdit, QComboBox

class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('输入法状态栏')
        self.setGeometry(100, 300, 300, 50)

        layout = QHBoxLayout()

        self.statusLabel = QLabel('当前状态: 中文', self)
        layout.addWidget(self.statusLabel)

        self.toggleButton = QPushButton('切换', self)
        self.toggleButton.clicked.connect(self.toggleStatus)
        layout.addWidget(self.toggleButton)

        self.setLayout(layout)

    def toggleStatus(self):
        if self.statusLabel.text() == '当前状态: 中文':
            self.statusLabel.setText('当前状态: 英文')
        else:
            self.statusLabel.setText('当前状态: 中文')

class SettingsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('输入法设置')
        self.setGeometry(200, 200, 400, 300)

        layout = QFormLayout()

        self.skinComboBox = QComboBox(self)
        self.skinComboBox.addItems(['默认皮肤', '黑色皮肤', '蓝色皮肤'])
        layout.addRow('选择皮肤:', self.skinComboBox)

        self.shortcutEdit = QLineEdit(self)
        layout.addRow('快捷键:', self.shortcutEdit)

        self.userDictEdit = QLineEdit(self)
        layout.addRow('用户词库路径:', self.userDictEdit)

        self.saveButton = QPushButton('保存', self)
        self.saveButton.clicked.connect(self.saveSettings)
        layout.addRow(self.saveButton)

        self.setLayout(layout)

    def saveSettings(self):
        # 保存设置逻辑
        print('设置已保存')
        self.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    statusBar = StatusBar()
    statusBar.show()

    settingsDialog = SettingsDialog()
    settingsDialog.exec_()

    sys.exit(app.exec_())