# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1196, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1200, 800))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setGeometry(QtCore.QRect(20, 120, 291, 541))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 60, 54, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(10, 100, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(10, 140, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(10, 170, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(10, 210, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(20, 250, 54, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(10, 290, 54, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(10, 330, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.groupBox)
        self.label_11.setGeometry(QtCore.QRect(10, 370, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        self.label_12.setGeometry(QtCore.QRect(10, 410, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(150, 20, 91, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_2.setGeometry(QtCore.QRect(150, 60, 91, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox.setGeometry(QtCore.QRect(150, 140, 81, 22))
        self.spinBox.setProperty("value", 30)
        self.spinBox.setObjectName("spinBox")
        self.spinBox_2 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_2.setGeometry(QtCore.QRect(150, 170, 71, 22))
        self.spinBox_2.setProperty("value", 3)
        self.spinBox_2.setObjectName("spinBox_2")
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_3.setGeometry(QtCore.QRect(150, 210, 81, 22))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(150, 250, 81, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_2.setGeometry(QtCore.QRect(150, 290, 81, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit_3.setGeometry(QtCore.QRect(150, 330, 81, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setGeometry(QtCore.QRect(150, 370, 81, 21))
        self.checkBox.setChecked(False)
        self.checkBox.setObjectName("checkBox")
        self.comboBox_4 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_4.setGeometry(QtCore.QRect(150, 410, 67, 22))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.spinBox_3 = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_3.setGeometry(QtCore.QRect(150, 100, 81, 22))
        self.spinBox_3.setProperty("value", 15)
        self.spinBox_3.setObjectName("spinBox_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(70, 450, 101, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.selection = QtWidgets.QGroupBox(self.tab)
        self.selection.setGeometry(QtCore.QRect(30, 50, 201, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.selection.setFont(font)
        self.selection.setAutoFillBackground(False)
        self.selection.setFlat(True)
        self.selection.setObjectName("selection")
        self.radioButton_1 = QtWidgets.QRadioButton(self.selection)
        self.radioButton_1.setGeometry(QtCore.QRect(30, 30, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_1.setFont(font)
        self.radioButton_1.setObjectName("radioButton_1")
        self.radioButton_2 = QtWidgets.QRadioButton(self.selection)
        self.radioButton_2.setGeometry(QtCore.QRect(130, 30, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setAutoFillBackground(True)
        self.radioButton_2.setChecked(True)
        self.radioButton_2.setObjectName("radioButton_2")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(30, 20, 221, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: red;")
        self.label.setObjectName("label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setGeometry(QtCore.QRect(320, 150, 271, 151))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton.setGeometry(QtCore.QRect(10, 30, 161, 21))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QtCore.QRect(170, 30, 101, 21))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setGeometry(QtCore.QRect(10, 70, 131, 31))
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_5.setGeometry(QtCore.QRect(170, 70, 101, 21))
        self.radioButton_5.setObjectName("radioButton_5")
        self.radioButton_11 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_11.setGeometry(QtCore.QRect(10, 110, 131, 31))
        self.radioButton_11.setObjectName("radioButton_11")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setGeometry(QtCore.QRect(640, 160, 351, 141))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.radioButton_6 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_6.setGeometry(QtCore.QRect(10, 30, 86, 21))
        self.radioButton_6.setObjectName("radioButton_6")
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_7.setGeometry(QtCore.QRect(140, 30, 201, 21))
        self.radioButton_7.setObjectName("radioButton_7")
        self.radioButton_8 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_8.setGeometry(QtCore.QRect(10, 110, 251, 21))
        self.radioButton_8.setObjectName("radioButton_8")
        self.radioButton_9 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_9.setGeometry(QtCore.QRect(140, 70, 151, 21))
        self.radioButton_9.setObjectName("radioButton_9")
        self.radioButton_10 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_10.setGeometry(QtCore.QRect(10, 70, 86, 21))
        self.radioButton_10.setObjectName("radioButton_10")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(350, 330, 91, 31))
        self.pushButton.setAutoFillBackground(False)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab)
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setGeometry(QtCore.QRect(270, 80, 101, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_6.setEnabled(False)
        self.groupBox_6.setGeometry(QtCore.QRect(330, 400, 511, 231))
        self.groupBox_6.setObjectName("groupBox_6")
        self.progressBar = QtWidgets.QProgressBar(self.groupBox_6)
        self.progressBar.setGeometry(QtCore.QRect(10, 190, 381, 16))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.label_14 = QtWidgets.QLabel(self.groupBox_6)
        self.label_14.setEnabled(False)
        self.label_14.setGeometry(QtCore.QRect(10, 110, 261, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.groupBox_6)
        self.label_15.setGeometry(QtCore.QRect(10, 150, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.groupBox_6)
        self.label_16.setGeometry(QtCore.QRect(10, 60, 281, 21))
        self.label_16.setObjectName("label_16")
        self.label_17 = QtWidgets.QLabel(self.groupBox_6)
        self.label_17.setGeometry(QtCore.QRect(10, 25, 291, 21))
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.groupBox_6)
        self.label_18.setGeometry(QtCore.QRect(280, 90, 211, 31))
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.groupBox_6)
        self.label_19.setGeometry(QtCore.QRect(350, 65, 121, 21))
        self.label_19.setObjectName("label_19")
        self.pushButton_4 = QtWidgets.QPushButton(self.tab)
        self.pushButton_4.setGeometry(QtCore.QRect(500, 330, 81, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.groupBox_4 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_4.setGeometry(QtCore.QRect(40, 180, 421, 121))
        self.groupBox_4.setObjectName("groupBox_4")
        self.radioButton_12 = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_12.setGeometry(QtCore.QRect(20, 40, 171, 21))
        self.radioButton_12.setObjectName("radioButton_12")
        self.radioButton_13 = QtWidgets.QRadioButton(self.groupBox_4)
        self.radioButton_13.setGeometry(QtCore.QRect(20, 80, 131, 21))
        self.radioButton_13.setObjectName("radioButton_13")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_4.setGeometry(QtCore.QRect(220, 80, 121, 20))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_13 = QtWidgets.QLabel(self.groupBox_4)
        self.label_13.setGeometry(QtCore.QRect(220, 40, 191, 31))
        self.label_13.setObjectName("label_13")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_5.setGeometry(QtCore.QRect(40, 40, 231, 71))
        self.groupBox_5.setObjectName("groupBox_5")
        self.radioButton_14 = QtWidgets.QRadioButton(self.groupBox_5)
        self.radioButton_14.setGeometry(QtCore.QRect(20, 30, 101, 21))
        self.radioButton_14.setObjectName("radioButton_14")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_5.setGeometry(QtCore.QRect(490, 260, 75, 23))
        self.pushButton_5.setObjectName("pushButton_5")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1196, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setGeometry(QtCore.QRect(158, 114, 144, 146))
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        MainWindow.setMenuBar(self.menubar)
        self.actionUpload = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.actionUpload.setFont(font)
        self.actionUpload.setObjectName("actionUpload")
        self.actioninitial = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.actioninitial.setFont(font)
        self.actioninitial.setObjectName("actioninitial")
        self.action_2 = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.action_2.setFont(font)
        self.action_2.setObjectName("action_2")
        self.actiontest = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.actiontest.setFont(font)
        self.actiontest.setObjectName("actiontest")
        self.actionfunc = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.actionfunc.setFont(font)
        self.actionfunc.setObjectName("actionfunc")
        self.actionauthor = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        self.actionauthor.setFont(font)
        self.actionauthor.setObjectName("actionauthor")
        self.actionkrum = QtWidgets.QAction(MainWindow)
        self.actionkrum.setObjectName("actionkrum")
        self.actionmkrum = QtWidgets.QAction(MainWindow)
        self.actionmkrum.setObjectName("actionmkrum")
        self.actionFedavg = QtWidgets.QAction(MainWindow)
        self.actionFedavg.setObjectName("actionFedavg")
        self.actiontrimmed_mean = QtWidgets.QAction(MainWindow)
        self.actiontrimmed_mean.setObjectName("actiontrimmed_mean")
        self.actionAFA = QtWidgets.QAction(MainWindow)
        self.actionAFA.setObjectName("actionAFA")
        self.menu.addAction(self.actionUpload)
        self.menu.addAction(self.actioninitial)
        self.menu.addSeparator()
        self.menu.addAction(self.action_2)
        self.menu.addSeparator()
        self.menu.addAction(self.actiontest)
        self.menu_2.addAction(self.actionfunc)
        self.menu_2.addAction(self.actionauthor)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "参数"))
        self.label_2.setText(_translate("MainWindow", "model_name"))
        self.label_3.setText(_translate("MainWindow", "type"))
        self.label_4.setText(_translate("MainWindow", "no_clients"))
        self.label_5.setText(_translate("MainWindow", "global_epoch"))
        self.label_6.setText(_translate("MainWindow", "local_epoch"))
        self.label_7.setText(_translate("MainWindow", "batch_size"))
        self.label_8.setText(_translate("MainWindow", "lr"))
        self.label_9.setText(_translate("MainWindow", "frac"))
        self.label_10.setText(_translate("MainWindow", "mal_frac"))
        self.label_11.setText(_translate("MainWindow", "attack"))
        self.label_12.setText(_translate("MainWindow", "poison_label"))
        self.comboBox.setItemText(0, _translate("MainWindow", "CNN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "MLP"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "mnist"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "cifar10"))
        self.comboBox_3.setCurrentText(_translate("MainWindow", "64"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "64"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "32"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "64"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "96"))
        self.comboBox_3.setItemText(4, _translate("MainWindow", "128"))
        self.lineEdit.setText(_translate("MainWindow", "0.01"))
        self.lineEdit_2.setText(_translate("MainWindow", "0.6"))
        self.lineEdit_3.setText(_translate("MainWindow", "0.4"))
        self.checkBox.setText(_translate("MainWindow", "True"))
        self.comboBox_4.setCurrentText(_translate("MainWindow", "0"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "0"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "1"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "2"))
        self.comboBox_4.setItemText(3, _translate("MainWindow", "3"))
        self.comboBox_4.setItemText(4, _translate("MainWindow", "4"))
        self.comboBox_4.setItemText(5, _translate("MainWindow", "5"))
        self.comboBox_4.setItemText(6, _translate("MainWindow", "6"))
        self.comboBox_4.setItemText(7, _translate("MainWindow", "7"))
        self.comboBox_4.setItemText(8, _translate("MainWindow", "8"))
        self.comboBox_4.setItemText(9, _translate("MainWindow", "9"))
        self.pushButton_2.setText(_translate("MainWindow", "Sure"))
        self.selection.setTitle(_translate("MainWindow", "选项"))
        self.radioButton_1.setText(_translate("MainWindow", "Yes"))
        self.radioButton_2.setText(_translate("MainWindow", "No"))
        self.label.setText(_translate("MainWindow", "Do you have ConfigFile? Yes or No"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Attack"))
        self.radioButton.setText(_translate("MainWindow", "Symbol_Flip"))
        self.radioButton_3.setText(_translate("MainWindow", "Arbitra"))
        self.radioButton_4.setText(_translate("MainWindow", "Label_Flip"))
        self.radioButton_5.setText(_translate("MainWindow", "Scale"))
        self.radioButton_11.setText(_translate("MainWindow", "Byzantine"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Defend"))
        self.radioButton_6.setText(_translate("MainWindow", "Krum"))
        self.radioButton_7.setText(_translate("MainWindow", "Trimmed_mean"))
        self.radioButton_8.setText(_translate("MainWindow", "FedAvg"))
        self.radioButton_9.setText(_translate("MainWindow", "Mkrum"))
        self.radioButton_10.setText(_translate("MainWindow", "AFA"))
        self.pushButton.setText(_translate("MainWindow", "Reday"))
        self.pushButton_3.setText(_translate("MainWindow", "上传配置"))
        self.groupBox_6.setTitle(_translate("MainWindow", "训练中......."))
        self.label_14.setText(_translate("MainWindow", "当前训练轮数： "))
        self.label_15.setText(_translate("MainWindow", "当前轮的进度："))
        self.label_16.setText(_translate("MainWindow", "恶意客户端个数"))
        self.label_17.setText(_translate("MainWindow", "正常客户端个数"))
        self.label_19.setText(_translate("MainWindow", "ACC_Test"))
        self.pushButton_4.setText(_translate("MainWindow", "Stop"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "FL"))
        self.groupBox_4.setTitle(_translate("MainWindow", "差分隐私"))
        self.radioButton_12.setText(_translate("MainWindow", "拉普拉斯分布"))
        self.radioButton_13.setText(_translate("MainWindow", "高斯分布"))
        self.lineEdit_4.setText(_translate("MainWindow", "0.1"))
        self.label_13.setText(_translate("MainWindow", "<html><head/><body><p>噪声参数标准差<span style=\" font-family:\'KaTeX_Math\'; font-size:16px; color:#4d4d4d; background-color:#ffffff;\">σ</span></p></body></html>"))
        self.groupBox_5.setTitle(_translate("MainWindow", "同态加密"))
        self.radioButton_14.setText(_translate("MainWindow", "paillier"))
        self.pushButton_5.setText(_translate("MainWindow", "Reset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "encrypt"))
        self.menu.setTitle(_translate("MainWindow", "选项"))
        self.menu_2.setTitle(_translate("MainWindow", "帮助"))
        self.actionUpload.setText(_translate("MainWindow", "上传数据"))
        self.actioninitial.setText(_translate("MainWindow", "初始化"))
        self.action_2.setText(_translate("MainWindow", "客户数据分布"))
        self.action_2.setIconText(_translate("MainWindow", "数据分布"))
        self.actiontest.setText(_translate("MainWindow", "测试集"))
        self.actionfunc.setText(_translate("MainWindow", "功能说明"))
        self.actionauthor.setText(_translate("MainWindow", "作者与版权"))
        self.actionkrum.setText(_translate("MainWindow", "Krum"))
        self.actionmkrum.setText(_translate("MainWindow", "Mkrum"))
        self.actionFedavg.setText(_translate("MainWindow", "Fedavg"))
        self.actiontrimmed_mean.setText(_translate("MainWindow", "Trimmed_mean"))
        self.actionAFA.setText(_translate("MainWindow", "AFA"))
