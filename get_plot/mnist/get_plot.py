import numpy as np
import matplotlib.pyplot as plt

# 不同攻击方式对FedAvg的影响

# No_attack
y = [86.660,  92.270,  94.120,  94.980,  95.610,  95.790,  96.120,  96.450,  96.770,  96.940,  97.210,  97.260,  97.240,  97.370,  97.390,  97.530,  97.580,  97.720,  97.770,  97.750,  97.860,  97.910,  97.920,  97.960,  98.030,  98.090,  98.160,  98.170,  98.180,  98.190 ]
# Symbol_flip
y1 = [29.270, 51.500, 82.370,  88.710,  88.560,  91.390,  63.000,  91.870,  80.610,  92.010,  87.810,  68.760,  91.780,  92.520,  92.540,  77.830,  93.090,  93.040,  92.830,  92.240,  93.900,  94.070,  91.470,  86.960,  93.030,  88.070,  93.800,  93.760,  94.370,  90.000]
# Label_flip
y2 = [9.800,  79.030,  19.370,  81.800,  21.210,  68.080,  16.070,  87.770,  36.390,  78.550 , 26.850,  75.490,  40.080,  80.700,  19.170 , 85.080 , 19.110 , 83.840,  28.280,  80.300,  19.730,  87.230,  17.940,  82.570,  44.210,  79.250,  38.240,  74.750,  38.150,  76.470 ]
# Byzantine
y3 = [14.310, 14.920,  23.660,  20.870 , 14.380,  11.030,  12.170,  8.660 , 11.910,  10.220,  10.210,  10.610 , 10.300  ,9.630 , 10.200,  9.690 , 9.740,  10.080 , 10.130 , 9.670 , 10.300,  10.090,  11.390,  9.740,  9.580,  9.540,  11.360,  10.100 , 11.350,  11.350 ]
# Scale
y4 = [88.000,  91.790,  81.280,  27.970,  14.330 , 11.350,  11.350,  11.350,  11.350,  11.350 , 11.350,  11.350,  11.350,  11.350,  11.350,  11.350 , 11.350,  11.350, 11.350 , 11.350,  11.350,  11.350,  11.350 , 11.350,  11.350 , 11.350 , 11.350,  11.350 , 11.350,  11.350 ]
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15
        }
plt.ylabel('ACC OF TEST .%', font)
plt.xlabel('Epoch', font)
plt.plot(y3, marker='*')
plt.title("Byzantine", font)
plt.grid()
plt.show()