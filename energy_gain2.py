import numpy as np
from matplotlib import pyplot as plt

# 在程序1中不改变天线阵列的物理位置的条件下，需要使得PI/3 方向的接收端接受的信号能量增益最大
if __name__ == '__main__':
    N = 8  # 线性阵列天线个数
    lamda = 10  # 波长
    d = 3  # 相邻阵元间距
    theta0 = np.pi / 3
    psi0 = 2 * np.pi * d * np.cos(theta0) / lamda
    theta = np.arange(0.0001, 2 * np.pi - 0.0001, 0.01)  # 序列
    psi = 2 * np.pi * d * np.cos(theta) / lamda
    r = np.abs(np.sin(N * (psi - psi0) / 2) / np.sin((psi - psi0) / 2)) / N  # 能量增益
    plt.figure()
    plt.polar(theta, r)
    plt.show()
