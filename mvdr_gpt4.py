import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# 线性麦克风 波束成形
"""matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
"""
#mac
plt.rcParams['font.sans-serif']= ['Heiti TC']#防止中文乱码
plt.rcParams['axes.unicode_minus']=False#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
if __name__ == '__main__':

    # 环境参数
    c = 343  # 声速
    SNR = 20  # 信噪比
    SL = 140  # 信号能量
    #r = 7000  # 距离
    f = 2000  # 频率
    lambda_ = c / f  # 波长
    angle = 30  # 入射角

    # 水平阵参数
    M = 9  # 阵元数
    d = c / (2 * f)  # 阵元间距
    angles = np.arange(-90, 90.1, 0.1)  # 检测角度范围
    Nsnapshot = 15  # 快拍数

    # 阵列响应向量
    v = np.sqrt(M) * np.exp(-1j * 2 * np.pi * (d * np.sin(np.radians(angle)) / lambda_) * np.arange(-(M-1)/2, (M-1)/2 + 1))
    # 发射信号
    s = np.sqrt(10**(SL/10)) * np.exp(1j * 2 * np.pi * f * np.arange(1, Nsnapshot + 1))
    # 高斯白噪声
    n = np.sqrt(10**((SL-SNR)/10)) * (np.random.randn(M, Nsnapshot) + 1j * np.random.randn(M, Nsnapshot)) / np.sqrt(2)
    # 接收信号
    x = np.sqrt(M) * v[:, None] * s + n
    print(np.shape(x))
    # 计算响应向量和波束形成响应
    Rx = np.dot(x, x.conj().T) / Nsnapshot
    # 驾驶向量
    c = np.sqrt(M) * np.exp(-1j * 2 * np.pi * np.arange(-(M-1)/2, (M-1)/2 + 1)[:, None] * (d * np.sin(np.radians(angles)) / lambda_))
    # 直接求解闭式解
    inv_Rx = np.linalg.inv(Rx)
    Cmvdr = np.zeros_like(c, dtype=complex)
    for i in range(c.shape[1]):
        steering_vector = c[:, i]
        Cmvdr[:, i] = np.linalg.inv(Rx).dot(steering_vector) / (steering_vector.conj().T.dot(np.linalg.inv(Rx)).dot(steering_vector))
    y1 = np.abs(np.diag(Cmvdr.conj().T.dot(Rx).dot(Cmvdr)))
    y1 = y1 / np.max(y1)

    # 绘图
    plt.figure()
    plt.plot(angles, 10 * np.log10(y1), 'k', linewidth=2)
    plt.xlabel('Angle (deg)', fontsize=15)
    plt.ylabel('Power Response (dB)', fontsize=15)
    plt.title('单目标MVDR波束形成（实际信号）', fontsize=20)
    plt.show()
