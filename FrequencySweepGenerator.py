import numpy as np
from scipy.io.wavfile import write

# 设置采样率
sampling_rate = 44100  # 44100样本/秒


# 生成扫描信号的函数
def generate_sweep_signal(freq_start, freq_end, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * ((freq_end - freq_start) / (2 * duration) * t ** 2 + freq_start * t))
    return signal


if __name__ == '__main__':
    low_frequency = 20
    high_frequency = 100
    # 定义不同的持续时间
    durations = [1, 2, 5, 10]  # 持续时间为1秒、2秒、5秒、10秒

    # 生成并保存声音文件
    for duration in durations:
        signal = generate_sweep_signal(low_frequency, high_frequency, duration)
        scaled = np.int16(signal / np.max(np.abs(signal)) * 32767)
        write(f'./data/sweep_signal_from{low_frequency}_to_{high_frequency}_{duration}s.wav', sampling_rate, scaled)

    print("声音文件生成完毕。")
