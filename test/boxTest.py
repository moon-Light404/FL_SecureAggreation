import sys

from PyQt5.QtWidgets import QMessageBox, QWidget, QApplication

app = QApplication(sys.argv)
w = QWidget()
box = QMessageBox.question(w, '噶我国的娃去韩国文化我和war哈尔和然而我', '保存实验数据？', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

# self.box.setIcon(1)
# self.box.setGeometry(1000, 600, 0, 0)
# self.box.show()
if box == QMessageBox.Yes:
    print("退出")
else:
    print('不退出')