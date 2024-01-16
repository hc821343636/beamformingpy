import numpy as np
import matplotlib.pyplot as plt
import librosa
## 波束成形算法 ，估计有问题
def mvdr_beamforming_from_coords(mic_coords, audio_data, sr, SNR, SL, f, angle, Nsnapshot):
    M = mic_coords.shape[0]  # 麦克风数量
    lambda_ = 343 / f  # 波长

    # 计算阵列响应向量
    v = np.exp(-1j * (2 * np.pi / lambda_) * (mic_coords[:, 0] * np.cos(angle) + mic_coords[:, 1] * np.sin(angle)))

    # 确保音频数据是多通道的，如果不是，则警告
    if audio_data.ndim == 1 or audio_data.shape[0] != M:
        raise ValueError("音频数据必须是多通道的，且通道数与麦克风数量相同")

    # 截取音频数据以匹配快拍数
    if audio_data.shape[1] > Nsnapshot:
        audio_data = audio_data[:, :Nsnapshot]
    elif audio_data.shape[1] < Nsnapshot:
        raise ValueError("音频数据的快拍数少于所需的快拍数")

    # 高斯白噪声
    noise = np.sqrt(10 ** ((SL - SNR) / 10)) * (
                np.random.randn(M, Nsnapshot) + 1j * np.random.randn(M, Nsnapshot)) / np.sqrt(2)

    # 接收信号（使用实际的音频数据）
    x = audio_data + noise

    # 协方差矩阵
    Rx = np.dot(x, x.conj().T) / Nsnapshot

    # 角度范围
    angles = np.linspace(0, 2 * np.pi, 360)

    # 驾驶向量
    c_matrix = np.zeros((M, len(angles)), dtype=complex)
    for i, ang in enumerate(angles):
        phase_diff = (2 * np.pi / lambda_) * (mic_coords[:, 0] * np.cos(ang) + mic_coords[:, 1] * np.sin(ang))
        c_matrix[:, i] = np.exp(-1j * phase_diff)

    # MVDR
    Cmvdr = np.zeros_like(c_matrix, dtype=complex)
    for i in range(c_matrix.shape[1]):
        steering_vector = c_matrix[:, i]
        Cmvdr[:, i] = np.linalg.inv(Rx).dot(steering_vector) / (
            steering_vector.conj().T.dot(np.linalg.inv(Rx)).dot(steering_vector))

    y1 = np.abs(np.diag(Cmvdr.conj().T.dot(Rx).dot(Cmvdr)))
    y1 = y1 / np.max(y1)

    return angles, y1


def plot_beam_pattern(angles, y1):
    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, 10 * np.log10(y1))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.show()

# 示例用法
if __name__ == '__main__':

    audio_path = 'your_audio_file.wav'  # 替换为你的音频文件路径

    audio_data, sr = librosa.load(audio_path, sr=None, mono=False)  # 加载音频数据

    # 环境参数
    SNR = 20  # 信噪比
    SL = 140  # 信号能量
    f = 5000  # 频率 (Hz)
    angle = np.radians(358)  # 入射角 (弧度)
    Nsnapshot = 15  # 快拍数
    M = 6  # 麦克风数量
    diameter = 0.05  # 直径（米）
    radius = diameter / 2  # 半径（米）

    # 计算每个麦克风的角度（弧度）
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    
    # 计算麦克风的笛卡尔坐标
    mic_coords = np.zeros((M, 2))  # 初始化坐标数组
    for i in range(M):
        mic_coords[i, 0] = radius * np.cos(angles[i])  # x坐标
        mic_coords[i, 1] = radius * np.sin(angles[i])  # y坐标
    # 调用 MVDR 波束成形函数
    angles, y1 = mvdr_beamforming_from_coords(mic_coords, audio_data, sr, SNR, SL, f, angle, Nsnapshot)

    # 绘制波束成形图像
    plot_beam_pattern(angles, y1)

