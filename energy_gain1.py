import numpy as np
from matplotlib import pyplot as plt

# 基本图像 需要满足d/lamda<=1/2 保证在【0，pi】中有一个极大点
if __name__ == '__main__':
    N = 16  # 线性阵列天线个数
    lamda = 10  # 波长
    d = 10  # 相邻阵元间距
    theta = np.arange(0.0001, 2 * np.pi - 0.0001, 0.01)  # 序列
    psi = 2 * np.pi * d * np.cos(theta) / lamda
    r = np.abs(np.sin(N * psi / 2) / np.sin(psi / 2)) / N  # 能量增益
    plt.figure()
    plt.polar(theta, r)

    plt.figure()
    plt.plot(theta/np.pi,r)
    plt.show()
