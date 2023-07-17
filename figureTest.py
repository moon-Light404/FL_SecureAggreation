import sys

import matplotlib
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QSizePolicy, QApplication
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# matplotlib.use('agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from torchvision import transforms, datasets

import get_datasets
from get_datasets import DatasetSplit


class PlotCanvas_train(FigureCanvas):

    def __init__(self, parent=None, dataset=None, dict_users=None, class_count=None):
        fig = Figure(figsize=(15, 6), dpi=100)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.dataset = dataset
        self.dict_users = dict_users
        self.class_count = class_count
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setWindowTitle("训练集客户端数据分布")
        self.plot(dict_users, class_count)


    def get_count_lit(self, idxs):
        dataset = DatasetSplit(self.dataset, idxs)
        class_count = [0 for _ in range(10)]
        for item in dataset:
            img, label = item
            class_count[label] += 1
        return class_count

    def plot(self, dist_users, class_count):
        # plt.style.use('ggplot')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        for i in range(10):
            ax = self.figure.add_subplot(2, 5, i + 1)
            count = self.get_count_lit(dist_users[i])
            ax.set_title("user %d的数据分布" % i)
            ax.set_xlabel("class")
            ax.set_ylabel("num of each class")
            ax.bar(class_count, count)
        self.figure.tight_layout()


class PlotCanvas_test(FigureCanvas):

    def __init__(self, parent=None, dataset=None, class_count=None):
        fig = Figure(figsize=(6, 3), dpi=100)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.dataset = dataset
        self.class_count = class_count
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setWindowTitle("测试集数据分布")
        self.plot_test(dataset, class_count)

    def plot_test(self, dataset, class_count):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

        count = [0 for _ in range(10)]
        for item in dataset:
            img, lab = item
            count[lab] += 1
        ax1 = self.figure.add_subplot(1, 1, 1)
        ax1.set_title("服务器测试集分布")
        ax1.set_xlabel('class')
        ax1.set_ylabel('num of class')
        ax1.bar(class_count, count)
        self.figure.tight_layout()



class PlotCanvasRes(FigureCanvas):
    def __init__(self, parent=None, loss_train=None, acc_test=None):
        fig = Figure(figsize=(12, 5), dpi=100)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.loss_train = loss_train
        self.acc_test = acc_test
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setWindowTitle("训练结果")
        self.plot(loss_train, acc_test)

    def plot(self, loss_train, acc_test):
        print(loss_train)
        print(acc_test)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax1.set_title("全局训练损失变化图")
        ax1.set_xlabel("Epoch(轮)")
        ax1.set_ylabel("LOSS OF TRAIN")
        ax1.set_xticks(range(len(loss_train)))
        ax1.plot(loss_train, 'o-b')
        ax1.grid()

        ax2 = self.figure.add_subplot(1, 2, 2)
        ax2.set_title("全局测试准确率变化图")
        ax2.set_xlabel("Epoch(轮)")
        ax2.set_ylabel("ACC OF TEST (%)")
        ax2.set_xticks(range(len(acc_test)))
        ax2.plot(acc_test, 'o-r')
        ax2.grid()

        self.figure.tight_layout()
        self.show()


if __name__ == "__main__":
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    class_count = [_ for _ in range(10)]
    dictusers = get_datasets.mnist_iid(dataset_train, 10)

    app = QApplication(sys.argv)
    # window = PlotCanvasRes(loss_train=[1, 2, 3], acc_test=[3, 4, 5])
    window = PlotCanvas_test(dataset=dataset_test, class_count=classes)
    window.show()
    sys.exit(app.exec_())
