import matplotlib.pyplot as plt

# label_Flip攻击下不同聚合算法的表现, mal=0.4

# FedAvg
y1 = [10.00,  27.66,  10.23,  28.74,  13.07,  18.16,  13.79,  27.17,  11.97,  37.50,  11.76,  29.30,  17.60,  20.85,  25.15,  22.65,  25.68,  23.90,  25.21,  23.58,  25.14,  25.71,  27.40,  24.66,  28.39,  25.71,  29.43,  28.23,  31.55,  28.16,  30.25,  28.59,  29.61,  29.90,  29.40]
# Krum
y2 = [30.91,  33.59,  27.95,  38.25,  39.35,  15.46,  33.90,  19.39,  37.02,  20.77,  23.81,  40.88,  44.80,  38.69,  44.90,  50.60,  48.35,  38.93,  46.57,  43.59,  45.46,  45.69,  48.21,  45.89,  46.88,  47.37,  48.55,  45.15,  42.15,  44.00,  49.38,  50.07,  44.44,  44.45,  48.82]
# Trimmed_mean
y3 = [10.00,  17.54,  10.98,  25.96,  17.57,  19.36,  21.45,  25.28,  25.88,  24.89,  26.35,  26.58,  29.38,  29.23,  29.60,  28.13,  29.42,  33.09,  32.69,  32.98,  36.87,  35.58,  33.17,  31.16,  32.38,  34.99,  39.59,  32.57,  35.50,  39.82,  42.37,  45.60,  44.69,  47.82,  46.81]
# Bulyan
y4 = [11.64,  28.41,  28.41,  31.57,  31.57,  40.03,  40.03,  42.28,  43.42,  48.71,  48.71,  49.05,  48.69,  52.39,  54.55,  50.00,  48.88,  48.36,  48.36,  53.57,  53.57,  53.91,  53.49,  53.49,  53.49,  53.49,  53.78,  53.78,  53.57,  53.57,  55.16,  55.16,  55.81,  55.43,  53.40]
# AFA
y5 = [10.00,  32.81,  36.67,  41.30,  10.13,  43.41,  43.90,  14.37,  34.90,  48.27,  50.42,  51.88,  52.27,  53.22,  53.89,  55.33,  55.55,  56.67,  56.63,  57.57,  58.04,  58.41,  58.80,  59.55,  60.42,  61.59,  61.32,  61.78,  62.27,  61.91,  61.44,  61.65,  61.18,  61.50,  61.55]


labels = ['FedAvg', 'Krum', 'Trimmed_mean', 'mKrum', 'AFA']

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 12
        }
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.figure()
plt.ylabel('ACC OF TEST(%)', font)
plt.xlabel('Epoch', font)
plt.plot(y1,'-r', label=labels[0])
plt.plot(y2,'-y', label=labels[1])
plt.plot(y3,'-k', label=labels[2])
plt.plot(y4, '-b', label=labels[3])
plt.plot(y5, '-g', label=labels[4])
plt.legend(prop=font)
plt.title("标签翻转攻击下使用不同聚合算法的分类准确率")
plt.grid()
plt.show()