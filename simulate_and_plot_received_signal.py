import numpy as np
import matplotlib.pyplot as plt

def simulate_received_signal(theta_d, M, N, sigma_squared):
    d = np.sin(theta_d * np.arange(M))
    A = np.random.randn(M, N)  # 干扰信号
    n = np.random.randn(M, N) * np.sqrt(sigma_squared)  # 加性白噪声
    x = d.reshape(-1, 1) + A + n
    return x

def plot_received_signal_samples(x, num_samples):
    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        plt.plot(x[:, i], label=f'Sample {i+1}')

    plt.title('Received Signal Samples')
    plt.xlabel('Array Element Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def simulate_and_plot_received_signal():
    M = 8  # 阵列阵元数量
    N = 5  # 快拍数量
    sigma_squared = 0.1

    theta_d_true = np.random.uniform(0, np.pi)  # 随机选择一个入射角
    x = simulate_received_signal(theta_d_true, M, N, sigma_squared)
    plot_received_signal_samples(x, N)

if __name__ == "__main__":
    simulate_and_plot_received_signal()
