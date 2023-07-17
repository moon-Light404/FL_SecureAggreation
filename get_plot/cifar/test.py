from matplotlib import pyplot as plt


loss_list = [1.4829,  0.8268,  0.5146,  0.4229,  0.3740,  0.3466,  0.3137,  0.3028,  0.2905,  0.2845 , 0.2731 , 0.2782,  0.2734 , 0.2587 , 0.2500 , 0.2438 , 0.2298 , 0.2237  ,0.2201,  0.2197 , 0.2067 , 0.2119  ,0.2035,  0.1972 , 0.1913,  0.1917,  0.1954 , 0.1878 , 0.1846 , 0.1846    ]
acc_list = [79.950,  94.070,  94.730 , 95.360  ,95.790,  96.360,  96.550 , 96.720 , 96.790,  96.860 , 96.930 , 96.970 , 97.220,  97.270 , 97.570 , 97.680 , 97.730 , 97.810,  97.850 , 97.850 , 97.880 , 98.010 , 98.090,  97.990,  98.090,  98.110 , 98.020,  98.060 , 98.160,  98.310 ]

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.title("全局测试准确率变化图")
plt.xticks(range(len(acc_list)))
plt.xlabel("Epoch(轮)")
plt.ylabel("ACC OF TEST (%)")
plt.plot(acc_list, 'o-r')
plt.grid()
plt.savefig("准确率.png")
plt.clf()

plt.title("全局训练损失变化图")
plt.xticks(range(len(loss_list)))
plt.xlabel("Epoch(轮)")
plt.ylabel("LOSS OF TRAIN")
plt.plot(loss_list, 'o-b')
plt.grid()
plt.savefig("损失.png")