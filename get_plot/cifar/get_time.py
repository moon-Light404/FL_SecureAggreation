#使用numpy
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
font = {
    'family':'Times New Roman',
    'weight':'bold',
    'size':11
}
matplotlib.rc("font", **font)

names = ['', 'mnist','', '', '', 'cifar10', '']
label = ['FedAvg', 'Krum', 'Trimmed_mean', 'Mkrum', 'AFA']



# data1 = [0.120, 5.210, 1.308, 6.171, 1.227]
# data2 = [0.172, 13.225, 10.838, 15.763, 2.537]

data1 = [0.120, 0.172]
data2 = [5.210, 13.225]
data3 = [1.308, 10.838]
data4 = [6.171, 15.763]
data5 = [1.227, 2.537]




x = np.arange(2)

total_width, n = 0.8, 5
width = total_width / n
x = x - (total_width - width) / 2

tick = [-0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25]


plt.bar(x, data1,  width=width, label=label[0])
plt.bar(x + width, data2, width=width, label=label[1])
plt.bar(x + 2 * width, data3, width=width, label=label[2])
plt.bar(x + 3 * width, data4, width=width, label=label[3])
plt.bar(x + 4 * width, data5, width=width, label=label[4])
plt.ylabel("time(s)")
plt.xlabel("data")
plt.title("aggregate time")
plt.grid()
plt.xticks(tick, names)

plt.legend()
plt.show()