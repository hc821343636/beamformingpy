import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']= ['Heiti TC']#防止中文乱码
plt.rcParams['axes.unicode_minus']=False#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
def mvdr_beamforming(c, SNR, SL, f, angle, M, diameter, Nsnapshot):
    # 环境参数
    lambda_ = c / f  # 波长

    # 圆形阵列参数
    radius = diameter / 2  # 半径 (米)
    angles = np.linspace(0, 2 * np.pi, 360)  # 检测角度范围 (弧度)

    # 计算圆形阵列中每个麦克风的位置
    theta_mics = np.linspace(0, 2 * np.pi, M, endpoint=False)  # 麦克风的角度
    x_mics = radius * np.cos(theta_mics)  # x坐标
    y_mics = radius * np.sin(theta_mics)  # y坐标

    # 阵列响应向量
    v = np.exp(-1j * (2 * np.pi / lambda_) * (x_mics * np.cos(angle) + y_mics * np.sin(angle)))

    # 发射信号
    s = np.sqrt(10**(SL/10)) * np.exp(1j * 2 * np.pi * f * np.arange(Nsnapshot))

    # 高斯白噪声
    noise = np.sqrt(10**((SL-SNR)/10)) * (np.random.randn(M, Nsnapshot) + 1j * np.random.randn(M, Nsnapshot)) / np.sqrt(2)

    # 接收信号
    x = np.sqrt(M) * v[:, None] * s + noise

    # 计算协方差矩阵
    Rx = np.dot(x, x.conj().T) / Nsnapshot

    # 驾驶向量 (针对圆形阵列的修改)
    c_matrix = np.zeros((M, len(angles)), dtype=complex)
    for i, ang in enumerate(angles):
        phase_diff = (2 * np.pi / lambda_) * (x_mics * np.cos(ang) + y_mics * np.sin(ang))
        c_matrix[:, i] = np.exp(-1j * phase_diff)

    # 直接求解闭式解
    Cmvdr = np.zeros_like(c_matrix, dtype=complex)
    for i in range(c_matrix.shape[1]):
        steering_vector = c_matrix[:, i]
        Cmvdr[:, i] = np.linalg.inv(Rx).dot(steering_vector) / (steering_vector.conj().T.dot(np.linalg.inv(Rx)).dot(steering_vector))

    y1 = np.abs(np.diag(Cmvdr.conj().T.dot(Rx).dot(Cmvdr)))
    y1 = y1 / np.max(y1)

    return angles, y1

def main():
    # 环境参数
    c = 343  # 声速 (m/s)
    SNR = 20  # 信噪比
    SL = 140  # 信号能量
    f = 5000  # 频率 (Hz)
    angle = np.radians(358)  # 入射角 (弧度)

    # 圆形阵列参数
    M = 6  # 阵元数
    diameter = 0.05  # 直径 (米)
    Nsnapshot = 15  # 快拍数



    angles, y1 = mvdr_beamforming(c, SNR, SL, f, angle, M, diameter, Nsnapshot)

    # 极坐标绘图
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, 10 * np.log10(y1))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xlabel('角度', fontsize=15)
    ax.set_ylabel('功率响应 (dB)', fontsize=15)
    ax.set_title('MVDR波束成形与圆形麦克风阵列', fontsize=20)
    plt.show()




if __name__ == "__main__":
    main()
