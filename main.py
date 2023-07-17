import os
import sys

import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QWidget

import utils
from attack.Byzan_attack import Byzantine_Attack
from attack.Scale_attack import Scale_Attack
from attack.attack import Attack
from test import Ui_MainWindow

import get_datasets
from Server import Server
from User import User
from attack.flipping_attack import Flipp_Attack
from Nets import *
import numpy as np
import copy
import warnings
from figureTest import PlotCanvasRes, PlotCanvas_train, PlotCanvas_test
import time

warnings.filterwarnings('ignore')


class DP:
    def __init__(self, dp_mode, dp_val):
        # 1或2
        self.dp_mode = dp_mode
        # 标准差sigma
        self.dp_val = dp_val


class Federate_Learn_thread(QThread):
    # 声明子线程传递到父线程的信号，指定类型
    Epoch_signal = pyqtSignal(int)
    # 显示每一轮的进度
    user_signal = pyqtSignal(int)
    fish_signal_list = pyqtSignal(list)
    fish_signal_time = pyqtSignal(float)
    finsh_signal_acc = pyqtSignal(float)

    def __init__(self, dataset_test, dataset_train, dict_users,
                 attack, defend, conf, poly_mode, dp, parent=None):
        super(Federate_Learn_thread, self).__init__(parent)
        self.dataset_test = dataset_test
        self.dataset_train = dataset_train
        self.dict_users = dict_users
        self.attack = attack
        self.defend_method = defend
        self.conf = conf
        # 聚合训练模式
        self.poly_mode = poly_mode
        self.dp = dp

    def __del__(self):
        self.wait()

    def get_attack(self):
        if self.attack == "Symbol_Flip":
            return Flipp_Attack()
        elif self.attack == "Arbitra":
            return Attack()
        elif self.attack == "Scale":
            return Scale_Attack()
        elif self.attack == "Byzantine":
            return Byzantine_Attack()

    # 联邦学习入口
    def run(self):
        print(self.conf)
        server = None
        server = Server(CNNMnist().cuda(), self.conf, self.dataset_test)
        if self.conf["type"] == "mnist":
            if self.conf["model_name"] == "MLP":
                server = Server(MLP().cuda(), self.conf, self.data_test)
            else:
                server = Server(CNNMnist().cuda(), self.conf, self.dataset_test)
        elif self.conf["type"] == "cifar10":
            server = Server(CNNCifar().cuda(), self.conf, self.dataset_test)
        # if self.conf["model_name"] == "MLP":
        #     server = Server(MLP().cuda(), self.conf, self.data_test)
        # elif self.conf["model_name"] == "CNN":
        #     if self.conf["type"] == "mnist":
        #         server = Server(CNNMnist().cuda(), self.conf, self.data_test)
        #     else:
        #         server = Server(CNNCifar().cuda(), self.conf, self.data_test)
        # 计算恶意客户端数量和每一次训练选取的客户端数量
        mal_count = int(self.conf["mal_frac"] * self.conf["no_clients"])
        avg_count = int(self.conf["frac"] * self.conf["no_clients"])
        # 随机选择恶意客户端
        mal_user_id = np.random.choice(range(self.conf["no_clients"]), mal_count, replace=False)
        attacker = None
        if self.conf["attack"] and self.attack != 'Label_Flip':
            # 获取攻击类型
            attacker = self.get_attack()
        users = []
        for usr_id in range(self.conf["no_clients"]):
            if usr_id in mal_user_id:
                is_mal = True
            else:
                is_mal = False
            users.append(User(model=copy.deepcopy(server.net).cuda(), conf=self.conf, dataset=self.dataset_train,
                              idxs=self.dict_users[usr_id], id=usr_id, is_mallicious=is_mal))
        mallicious_users = [u for u in users if u.is_mallicious]
        loss_train = []
        loss_test, acc_test = [], []
        # 全局模型
        w_glob = server.net.state_dict()
        w_locals = []
        # 同态加密，客户端加密模型，时间较长
        if self.poly_mode == 2:
            w_locals = [users[0].get_net_enc() for i in range(self.conf["no_clients"])]
        else:
            w_locals = [w_glob for i in range(self.conf["no_clients"])]

        loading = True
        # 计算训练时间
        sum_time = 0

        for epoch in range(self.conf["global_epochs"]):
            time_start = time.time()
            # 发送当前训练到第几轮的消息
            self.Epoch_signal.emit(epoch)
            loss_locals = []
            m = max(avg_count, 1)
            select_users = np.random.choice(range(self.conf["no_clients"]), m, replace=False)
            w, loss = 0, 0
            cnt = 0
            for idx in select_users:
                cnt += 1
                # 传递信号给父线程，显示进度条
                self.user_signal.emit(cnt)
                print("==================User %d开始训练=================" % idx)
                if users[idx].is_mallicious:
                    print("用户{}是恶意客户端".format(idx))
                    # 标签翻转攻击
                    if self.attack == 'Label_Flip':
                        w, loss = users[idx].local_train_label_mal(net_dict=copy.deepcopy(server.net.state_dict()),
                                                                   poly_mode=self.poly_mode, dp=self.dp,
                                                                   loading=loading)
                    else:
                        w, loss = users[idx].local_train(net_dict=copy.deepcopy(server.net.state_dict()),
                                                         poly_mode=self.poly_mode, dp=self.dp, loading=loading)
                else:
                    print("用户{}是正常用户端".format(idx))
                    w, loss = users[idx].local_train(net_dict=copy.deepcopy(server.net.state_dict()),
                                                     poly_mode=self.poly_mode, dp=self.dp, loading=loading)
                # w, loss = users[idx].local_train(net_dict=copy.deepcopy(server.net.state_dict()))
                w_locals[idx] = copy.deepcopy(w)
                loss_locals.append(copy.deepcopy(loss))
            if self.conf["attack"] and self.attack != 'Label_Flip':
                attacker.attack(mallicious_users)  # 恶意客户端进行攻击
                for u in mallicious_users:
                    w_locals[u.user_id] = copy.deepcopy(u.local_net.state_dict())
            # 服务器进行密文聚合
            # w_param = server.FedAvg_enc(w_locals).decrypt()
            # row_into_parameters(w_param, server.net.parameters())

            # 求每一轮的平均损失(大Epoch)
            loss_avg = round(sum(loss_locals) / len(loss_locals), 3)
            print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
            loss_train.append(loss_avg)
            # 服务器进行聚合
            if self.poly_mode == 2:
                # 服务器进行聚合,此时w_glob和w_locals是加密态
                w_glob = server.FedAvg_enc(w_locals)
                server.net.load_state_dict(copy.deepcopy(utils.get_glob_dec(w_glob, server.get_shape())),
                                           strict=True)
            # poly_mode = 1 or 3 正常聚合
            else:
                # 服务器进行聚合,此时w_glob是明文
                try:
                    w_glob = server.defend(self.defend_method, w_locals, self.conf["no_clients"], mal_count)
                    server.net.load_state_dict(w_glob)
                except:
                    print("无法聚合")
                    break

            loading = False  # 同态加密第一轮是明文，现在标志位改变

            time_end = time.time()
            # 总时间加和
            sum_time += time_end - time_start
            # if epoch % 2 == 0 or epoch == self.conf["global_epochs"] - 1:
            acc, loss = server.model_test()
            self.finsh_signal_acc.emit(acc)
            acc_test.append(acc)
            loss_test.append(loss)
            print("Testing accuracy: {:.3f}".format(acc))
            print("Testing loss: {:.2f}".format(loss))
        signal_list = []
        # 向父线程传递信息
        signal_list.append(loss_train)
        signal_list.append(acc_test)
        self.fish_signal_list.emit(signal_list)
        self.fish_signal_time.emit(sum_time)
        # plot = PlotCanvasRes(loss_train=loss_train, acc_test=acc_test)
        # # plot.show()


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.box = None
        self.conf = {}
        self.file_directory = ""
        self.attack = ""
        self.defend_method = ""
        self.dataset_train = None
        self.dataset_test = None
        # 聚合模式，明文、加密、差分(1、 2、 3)
        self.poly_mode = 1
        self.dict_users = {}
        self.thread_1 = None
        # 训练结果
        self.result = []
        # 数据初始化状态
        self.initial_state = False

        self.setupUi(self)  # 渲染页面控件
        self.initUI()

    def initUI(self):
        self.lineEdit_4.setText('0.1')
        self.conf["momentum"] = 0.9
        self.setGeometry(600, 600, 1200, 800)
        self.setWindowTitle('Federate Learning-Attack And Defend')
        self.radioButton_8.setChecked(True)
        self.actionUpload.triggered.connect(self.action_upload_file)
        # 数据初始化
        self.actioninitial.triggered.connect(self.action_initial_data)
        # 训练集按钮
        self.action_2.triggered.connect(self.action_distribute)
        # 测试集按钮
        self.actiontest.triggered.connect(self.action_test_distribute)

        # 帮助按钮
        self.actionfunc.triggered.connect(self.get_help_info)

        self.radioButton_1.toggled.connect(self.up_config_state)
        self.radioButton_2.toggled.connect(self.up_config_state)

        self.radioButton_14.toggled.connect(self.poly_mode_state)
        self.radioButton_12.toggled.connect(self.poly_mode_state)
        self.radioButton_13.toggled.connect(self.poly_mode_state)

        # 开启训练线程
        self.pushButton.clicked.connect(self.startClick)

        # 停止训练线程
        self.pushButton_4.clicked.connect(self.stopClick)
        self.pushButton_2.clicked.connect(self.showDialog1)
        # 上传配置文件
        self.pushButton_3.clicked.connect(self.btn_upload_config)

        self.pushButton_5.clicked.connect(self.clear)

        self.radioButton.toggled.connect(self.choose_attack)
        self.radioButton_11.toggled.connect(self.choose_attack)
        self.radioButton_3.toggled.connect(self.choose_attack)
        self.radioButton_4.toggled.connect(self.choose_attack)
        self.radioButton_5.toggled.connect(self.choose_attack)

        self.radioButton_6.toggled.connect(self.choose_defend)
        self.radioButton_7.toggled.connect(self.choose_defend)
        self.radioButton_8.toggled.connect(self.choose_defend)
        self.radioButton_9.toggled.connect(self.choose_defend)
        self.radioButton_10.toggled.connect(self.choose_defend)
        self.groupBox_6.setVisible(False)

    # 获取聚合模式：明文、加密、扰动
    def poly_mode_state(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            if radioBtn.text() == 'paillier':
                self.poly_mode = 2
                print(self.poly_mode)
                self.radioButton_8.setChecked(True)
                self.radioButton_13.setAutoExclusive(False)
                self.radioButton_12.setAutoExclusive(False)
                self.radioButton_12.setChecked(False)
                self.radioButton_13.setChecked(False)
                self.radioButton_13.setAutoExclusive(True)
                self.radioButton_12.setAutoExclusive(True)
            else:
                self.poly_mode = 3
                self.radioButton_13.setAutoExclusive(False)
                self.radioButton_14.setChecked(False)
                self.radioButton_13.setAutoExclusive(True)
                print(self.poly_mode)

        # print(self.poly_mode)

    def showDialog1(self):
        ok = QMessageBox.question(self, "确认参数", "确认保存参数吗 ? ", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ok == 16384:
            self.conf["model_name"] = self.comboBox.currentText()
            self.conf["type"] = self.comboBox_2.currentText()
            self.conf["no_clients"] = int(self.spinBox_3.text())
            self.conf["global_epochs"] = int(self.spinBox.text())
            self.conf["local_epochs"] = int(self.spinBox_2.text())
            self.conf["batch_size"] = int(self.comboBox_3.currentText())
            self.conf["lr"] = float(self.lineEdit.text())
            self.conf["frac"] = float(self.lineEdit_2.text())
            self.conf["mal_frac"] = float(self.lineEdit_3.text())
            self.conf["attack"] = self.checkBox.isChecked()
            self.conf["poison_label"] = int(self.comboBox_4.currentText())
            if not self.checkBox.isChecked():
                self.groupBox_2.setEnabled(False)
            else:
                self.groupBox_2.setEnabled(True)
            print(self.conf)
            # self.pushButton_2.setEnabled(False)
            # 保存字典信息
        else:
            print(0)

    # 帮助提示信息框
    def get_help_info(self):
        QMessageBox.information(self, "帮助", "model_name:模型  type:数据集\n"
                                            " no_clients:客户端数量  global_epoch:全局训练轮次\n"
                                            "local_epoch:本地训练轮次  batch_size:训练批次大小\n"
                                            "lr: SGD优化器学习率   frac:每轮选取比例\n"
                                            "mal_frac:恶意比例    attack:是否攻击\n"
                                            "Symbol_Flip:取反    Arbitra:清零\n"
                                            "Label_Flip:标签翻转   Scale:权重放大\n"
                                            "ByZantine:加噪"
                                )

    # 是否上传配置文件按钮
    def btn_upload_config(self):
        try:
            filename, filetype = QFileDialog.getOpenFileName(self, "选择config文件", "*.json")
            self.conf = get_datasets.get_config(filename)
            if self.conf["model_name"] == "cnn":
                self.comboBox.setCurrentIndex(0)
            else:
                self.comboBox.setCurrentIndex(1)

            if self.conf["type"] == "mnist":
                self.comboBox_2.setCurrentIndex(0)
            else:
                self.comboBox_2.setCurrentIndex(1)

            self.spinBox_3.setValue(self.conf["no_clients"])
            self.spinBox.setValue(self.conf["global_epochs"])
            self.spinBox_2.setValue(self.conf["local_epochs"])
            self.comboBox_3.setCurrentIndex(32 / self.conf["batch_size"])
            self.lineEdit.setText(str(self.conf["lr"]))
            self.lineEdit_2.setText(str(self.conf["frac"]))
            self.lineEdit_3.setText(str(self.conf["mal_frac"]))
            self.checkBox.setChecked(self.conf["attack"])
            self.comboBox_4.setCurrentIndex(self.conf["poison_label"])
        except:
            print("没有选择配置文件")

    # 接受子线程数据，更新当前轮次
    def update_train_data_1(self, msg):
        self.label_14.setText("当前训练轮数： {} / {}".format(msg + 1, self.conf["global_epochs"]))

    # 接受子线程数据，更新进度条
    def update_train_data_2(self, cnt):
        avg_count = int(self.conf["no_clients"] * self.conf["frac"])
        self.progressBar.setValue(100.0 * cnt / avg_count)

    # 上传数据目录
    def action_upload_file(self):
        try:
            self.file_directory = QFileDialog.getExistingDirectory(None, "选择数据目录", "")
            print(self.file_directory)
        except:
            print("没有选择目录")

    # 数据初始化
    def action_initial_data(self):
        try:
            self.dataset_train, self.dataset_test = get_datasets.get_dataset(self.file_directory, self.conf["type"])
            if self.conf["type"] == 'mnist':
                self.dict_users = get_datasets.mnist_iid(self.dataset_train, self.conf["no_clients"])
            else:
                self.dict_users = get_datasets.cifar_iid(self.dataset_train, self.conf["no_clients"])
            print("数据初始化完成")
            self.initial_state = True  # 标记初始化完成
        except:
            print("文件不存在或者其他问题")

    # 查看数据分布情况
    def action_distribute(self):
        try:
            class_count = []
            if self.conf["type"] == 'mnist':
                class_count = [_ for _ in range(10)]
            elif self.conf["type"] == 'cifar10':
                class_count = ['plane', 'car', 'bird', 'cat',
                               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            plot = PlotCanvas_train(dataset=self.dataset_train, dict_users=self.dict_users, class_count=class_count)
            plot.show()
        except:
            print("还没初始化数据")

    # 查看测试集分布情况
    def action_test_distribute(self):
        try:
            class_count = []
            if self.conf["type"] == 'mnist':
                class_count = [_ for _ in range(10)]
            elif self.conf["type"] == 'cifar10':
                class_count = ['plane', 'car', 'bird', 'cat',
                               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            plot = PlotCanvas_test(dataset=self.dataset_test, class_count=class_count)
            plot.show()
        except:
            print("还没初始化数据")

    # 上传配置文件按钮状态
    def up_config_state(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            if radioBtn.text() == 'Yes':
                self.pushButton_3.setEnabled(True)
                self.groupBox.setEnabled(False)
            else:
                self.pushButton_3.setEnabled(False)
                self.groupBox.setEnabled(True)

    # 选择攻击方式
    def choose_attack(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.attack = radioBtn.text()
            print(self.attack)

    # 选择防御方式
    def choose_defend(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.defend_method = radioBtn.text()
            print(self.defend_method)

    # 保存文件至目录
    def save_to_file(self, sum_time):
        try:
            filepath = "F:/pythonProject/federated-learning-master/TestResult/"
            path_list = os.listdir(filepath)
            path_list.sort(key=lambda x: int(x))
            file_last = int(path_list[-1]) + 1
            new_filepath = filepath + str(file_last)
            os.mkdir(new_filepath)
            f = open(new_filepath + '/info.txt', 'a', encoding='utf-8')
            mode_str = ''
            dp_mode = ''
            dp_value = 0.0

            if self.poly_mode == 1:
                mode_str = '明文'
            elif self.poly_mode == 2:
                mode_str = '同态加密'
            else:
                mode_str = '差分'
                if self.radioButton_12.isChecked():
                    dp_mode = '拉普拉斯'
                elif self.radioButton_13.isChecked():
                    dp_mode = '高斯'
                dp_value = float(self.lineEdit_4.text())

            f.write("---------------模式----------------\n")
            f.write("        " + mode_str + "\n")
            if self.poly_mode == 3:
                f.write(dp_mode + "   " + str(dp_value))
            f.write("---------------参数-------------------\n")
            for key, val in self.conf.items():
                f.write(key + ":" + str(val) + "\n")
            f.write("攻击： " + self.attack + "\n")
            f.write("防御： " + self.defend_method + "\n")

            loss_list = self.result[0]
            acc_list = self.result[1]

            f.write("-------------------模型分类测试准确率---------------\n")
            for acc in acc_list:
                f.write(str(acc) + " ")

            f.write("\n-------------------训练损失变化---------------\n")
            for loss in loss_list:
                f.write(str(loss) + " ")

            f.write("\n--------总用时-----------：{}s \n".format(round(sum_time, 1)))
            f.write("--------平均用时---------：{}s \n".format(round(sum_time / self.conf["global_epochs"], 1)))
            f.close()

            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.title("全局测试准确率变化图")
            plt.xticks(range(len(acc_list)))
            plt.xlabel("Epoch(轮)")
            plt.ylabel("ACC OF TEST (%)")
            plt.plot(acc_list, 'o-r')
            plt.grid()
            plt.savefig(new_filepath + '/准确率.png')
            plt.clf()

            plt.title("全局训练损失变化图")
            plt.xticks(range(len(loss_list)))
            plt.xlabel("Epoch(轮)")
            plt.ylabel("LOSS OF TRAIN")
            plt.plot(loss_list, 'o-b')
            plt.grid()
            plt.savefig(new_filepath + '/损失.png')

            print("保存成功")
        except:
            print("保存失败!")

    # 绘制结果图
    def plot_result(self, res_list):
        self.result = res_list
        plot = PlotCanvasRes(loss_train=res_list[0], acc_test=res_list[1])
        # plot.show()

    # 显示一轮的准确率
    def get_acc(self, acc):
        self.label_18.setText('当前准确率：{:.2f}%'.format(acc))

    # 获取时间
    def get_time(self, sum_time):
        w = QWidget()
        avg_time = sum_time / self.conf["global_epochs"]
        self.box = QMessageBox.question(w, '总用时:{:.1f}s/平均:{:.1f}s--------------'.format(sum_time,
                                                                                         avg_time),
                                        '保存实验数据？', QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

        # self.box.setIcon(1)
        # self.box.setGeometry(1000, 600, 0, 0)
        # self.box.show()
        if self.box == QMessageBox.Yes:
            self.save_to_file(sum_time)
        else:
            print('不退出')

    # 停止线程
    def stopClick(self):
        try:
            if self.thread_1:
                self.thread_1.terminate()
                self.thread_1 = None
                self.groupBox_6.setVisible(False)
            else:
                print("0")
        except:
            print("00000")

    # 清除选项
    def clear(self):
        print("清除成功")
        self.radioButton_12.setChecked(False)
        self.radioButton_13.setChecked(False)
        self.radioButton_14.setChecked(False)

    # 启动线程
    def startClick(self):
        if not self.initial_state:
            print("还未初始化数据!")
        else:
            ok = QMessageBox.question(self, "ready", "攻击: " + self.attack + "  防御: " + self.defend_method,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ok == 16384:
                print("ok正在训练中......")
                # 恶意和正常的客户端数量
                mal_count = int(self.conf["mal_frac"] * self.conf["no_clients"])
                avg_count = int(self.conf["frac"] * self.conf["no_clients"])

                # 设置训练窗口的数据
                self.groupBox_6.setVisible(True)
                self.label_16.setText("恶意客户端个数：%d" % mal_count)
                self.label_17.setText("正常客户端个数：%d" % avg_count)
                attack_str = ''
                if not self.checkBox:
                    attack_str = 'No_attack'
                else:
                    attack_str = self.attack

                poly_mode_str = ''
                if self.radioButton_12.isChecked() == False and self.radioButton_13.isChecked() == False and self.radioButton_14.isChecked() == False:
                    self.poly_mode = 1
                    print(self.poly_mode)

                if self.poly_mode == 1:
                    poly_mode_str = '明文'
                elif self.poly_mode == 2:
                    poly_mode_str = '同态加密'
                else:
                    poly_mode_str = '差分'

                self.groupBox_6.setTitle("攻击： {} 防御： {} 模式：{}".format(attack_str, self.defend_method, poly_mode_str))
                print("fswagffsf")
                dp = None
                dp_sigma = float(self.lineEdit_4.text())  # 获取噪声参数标准差sigma
                # 只有选中了才会生成DP对象
                if self.radioButton_12.isChecked():
                    dp = DP(1, dp_sigma)  # 拉普拉斯
                elif self.radioButton_13.isChecked():
                    dp = DP(0, dp_sigma)  # 高斯分布

                # 训练函数线程入口
                self.thread_1 = Federate_Learn_thread(self.dataset_test, self.dataset_train, self.dict_users,
                                                      self.attack, self.defend_method, self.conf, self.poly_mode,
                                                      dp)
                # 更新label和进度条,传递信号
                self.thread_1.Epoch_signal.connect(self.update_train_data_1)
                self.thread_1.user_signal.connect(self.update_train_data_2)

                self.thread_1.fish_signal_time.connect(self.get_time)
                self.thread_1.finsh_signal_acc.connect(self.get_acc)
                self.thread_1.fish_signal_list.connect(self.plot_result)
                self.thread_1.start()
            else:
                print(0)
        # federate_learn(conf=self.conf, file_directory=self.file_directory, attack=self.attck, defend=self.defend)


def main():
    app = QApplication(sys.argv)
    window = Window()
    # app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette()))
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
