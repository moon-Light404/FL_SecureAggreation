import numpy as np
import matplotlib.pyplot as plt

# 不同攻击方式对FedAvg的影响

# 1
y = [12.54,15.76,10.69,13.51,9.74,10.45,8.45,9.82,9.81,9.77,10.29,10.09,10.06,10.20,10.32,10.10,10.28,10.28,10.38,9.80,9.81,10.28,10.32,9.74,10.30,10.30,10.31,10.31,9.86,10.31]
# 5
y1 = [64.93,87.72,90.71,90.92,90.48,90.62,86.77,90.20,89.63,92.84,93.88,92.71,93.42,93.14,93.60,93.66,93.01,92.02,91.48,92.30,93.92,92.35,91.79,93.22,92.69,92.19,92.81,92.89,93.05,92.53]
# 10
y2 = [84.20, 90.79, 93.94, 94.23, 95.18, 95.16, 95.5, 95.70, 95.70, 95.94, 95.94, 96.18, 96.33, 96.31, 96.43, 96.45, 96.35, 96.56, 96.86, 96.52, 96.76, 96.78, 96.80, 96.66, 96.85, 96.87, 96.88, 97.08, 97.94, 97.87]
# 100
y3 = [86.39,92.50,94.33,95.12,95.56,95.98,96.24,96.39,96.65,96.88,97.10,97.20,97.27,97.36,97.49,97.52,97.69,97.76,97.77,97.85,97.89,98.00,97.96,97.93,98.07,98.14,98.11,98.15,98.16,98.21]
# Scale
y4 = [33.00,  40.68,  42.84,  17.43,  10.34,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00,  10.00]
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15
        }
plt.rcParams['font.sans-serif'] = ['SimHei']
labels = ['k=1', 'k=5', 'k=10', 'k=100']
plt.figure()

plt.ylabel('ACC OF TEST (DP) .%', font)
plt.xlabel('Epoch', font)
plt.plot(y,'-r', label=labels[0])
plt.plot(y1,'-y', label=labels[1])
plt.plot(y2,'-g', label=labels[2])
plt.plot(y3, '-b', label=labels[3])
plt.legend()
plt.title("差分隐私(加噪)下的准确率曲线")
plt.grid()
plt.show()