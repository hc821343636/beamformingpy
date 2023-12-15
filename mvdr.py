import matplotlib.pyplot as plt
import numpy as np


def compute_linear_array_response_single_angle(angle_degrees, d, num_microphones):
    """
    计算线性麦克风阵列的阵列响应函数。

    参数：
    angle_degrees: float
        入射角度，以度数表示。
    d: float
        麦克风之间的间隔。
    num_microphones: int
        麦克风的数量。

    返回：
    response: complex
        阵列响应函数值。
    """
    angle_radians = np.radians(angle_degrees)
    response = np.exp(-1j * 2 * np.pi * d * np.arange(num_microphones) * np.sin(angle_radians))
    return response


# 示例用法：

def compute_time_delays(angle_degrees, microphone_spacing, num_microphones, v=343):
    """
    计算线性麦克风阵列中每个麦克风的时间延迟。

    参数：
    angle_degrees: float
        入射角度，以度数表示。
    microphone_spacing: float
        麦克风之间的间隔。
    num_microphones: int
        麦克风的数量。
    v: int
        采样频率。

    返回：
    delays: numpy数组
        包含每个麦克风的时间延迟的数组。
    """
    angle_radians = np.radians(angle_degrees)
    delays = np.arange(0, num_microphones ) * microphone_spacing * np.sin(angle_radians) / v
    return delays




def source_signal(frequency, duration, sampling_rate):
    """
    生成简单的正弦波声源信号。

    参数：
    frequency: float
        正弦波的频率。
    duration: float
        信号的持续时间（秒）。
    sampling_rate: float
        采样频率。

    返回：
    signal: numpy数组
        生成的声源信号。
    """
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * frequency * t)
    print(len(signal))
    return t,signal


def microphone_array_signal_with_noise(source_signal, delays, num_microphones, sampling_rate, noise_level=0):
    """
    模拟麦克风阵列接收到声源信号并加入噪声的过程。

    参数：
    source_signal: numpy数组
        声源的声音信号。
    delays: numpy数组
        包含每个麦克风的时间延迟的数组。
    num_microphones: int
        麦克风的数量。
    sampling_rate: float
        采样频率。
    noise_level: float, 可选
        噪声水平，范围在0到1之间，默认为0.1。

    返回：
    microphone_signals: numpy数组
        包含每个麦克风接收到的信号的矩阵。
    """
    microphone_signals = np.zeros((num_microphones, len(source_signal)))

    for mic_index in range(1, num_microphones + 1):
        delay_samples = int(delays[mic_index - 1] * sampling_rate)
        shifted_signal = np.roll(source_signal, delay_samples)

        # 生成相应长度的高斯噪声
        noise = noise_level * np.random.normal(0, 1, len(source_signal))

        # 将噪声加入信号
        noisy_signal = shifted_signal + noise

        microphone_signals[mic_index - 1, :] = noisy_signal

    return microphone_signals

def plot_microphone_signals(microphone_signals, delays, sampling_rate):
    """
    绘制麦克风接收信号的波形图。

    参数：
    microphone_signals: numpy数组
        包含每个麦克风接收到的信号的矩阵。
    delays: numpy数组
        包含每个麦克风的时间延迟的数组。
    sampling_rate: float
        采样频率。
    """
    time = np.arange(len(microphone_signals[0])) / sampling_rate
    plt.figure(figsize=(10, 6))

    for mic_index in range(len(microphone_signals)):
        plt.plot(time, microphone_signals[mic_index],
                 label=f'Microphone {mic_index + 1}, Delay = {delays[mic_index]:.4f}s')

    plt.title('Microphone Signals with Delays')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
def compute_correlation_matrix(microphone_signal):
    # Compute the correlation matrix R
    R = (1 / microphone_signal.shape[1]) * (microphone_signal @ microphone_signal.T).conj()  # Dimensions: (M, M)

    return R
def compute_mvdr_weights(R, a_theta_d):
    """
    计算MVDR权重向量。

    参数：
    R: numpy数组
        麦克风信号的自相关矩阵，大小为 (M, M)。
    a_theta_d: numpy数组
        阵列响应向量，大小为 (M, 1)。

    返回：
    w_mvdr: numpy数组
        MVDR权重向量，大小为 (M, 1)。
    """
    R_inv = np.linalg.inv(R)
    w_mvdr = R_inv @ a_theta_d / (a_theta_d.T @ R_inv @ a_theta_d)
    return w_mvdr
if __name__ == '__main__':
    # 示例用法：
    single_angle_degrees = 30.0  # 假设入射角度为30度
    microphone_spacing = 1  # 假设麦克风间隔为0.5
    num_microphones = 8  # 假设麦克风数量为8
    sampling_rate = 8000  # 假设采样频率为8000 Hz
    frequency = 10  # 信号频率
    duration = 1  # 持续时间

    t,signal = source_signal(frequency=frequency, duration=duration, sampling_rate=sampling_rate)
    plt.plot(t,signal)
    # 麦克风响应函数
    array_response_at_single_angle = compute_linear_array_response_single_angle(single_angle_degrees,
                                                                                microphone_spacing, num_microphones)
    print(array_response_at_single_angle)

    # 麦克风时延
    delays_at_single_angle = compute_time_delays(single_angle_degrees, microphone_spacing, num_microphones)
    print(delays_at_single_angle)
    # 麦克风每个接受的信号
    microphone_signals = microphone_array_signal_with_noise(source_signal=signal, delays=delays_at_single_angle,
                                                 num_microphones=num_microphones, sampling_rate=sampling_rate)
    print(np.shape(microphone_signals))
    plot_microphone_signals(microphone_signals, delays=delays_at_single_angle, sampling_rate=sampling_rate)
    r=compute_correlation_matrix(microphone_signals)
    print(r)
    MVDRweights=compute_mvdr_weights(R=r,a_theta_d=array_response_at_single_angle)

    print(MVDRweights)


