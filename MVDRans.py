import numpy as np
import matplotlib.pyplot as plt

def simulate_received_signal(theta_d, M, N, sigma_squared):
    d = np.sin(theta_d * np.arange(M))
    A = np.random.randn(M, N)  # 干扰信号
    n = np.random.randn(M, N) * np.sqrt(sigma_squared)  # 加性白噪声
    x = d.reshape(-1, 1) + A + n
    return x

def mvdr_beamformer(x, theta_d):
    R = np.cov(x)
    a_theta_d = np.sin(theta_d * np.arange(x.shape[0])).reshape(-1, 1)
    w_mvdr = np.linalg.inv(R) @ a_theta_d / (a_theta_d.T @ np.linalg.inv(R) @ a_theta_d)
    return w_mvdr

def main():
    theta_d_true = 80 * np.pi / 180
    M = 8
    N = 1000
    sigma_squared = 0.1

    x = simulate_received_signal(theta_d_true, M, N, sigma_squared)
    w_mvdr = mvdr_beamformer(x, theta_d_true)

    angles = np.linspace(0.001, 180, 181) * np.pi / 180
    beam_pattern_mvdr = np.abs(np.sin(angles.reshape(-1, 1) * np.arange(M)) @ w_mvdr)

    plt.plot(angles * 180 / np.pi, 20 * np.log10(beam_pattern_mvdr))
    ans=np.argmax(20 * np.log10(beam_pattern_mvdr))
    plt.axvline(x=angles[ans] * 180 / np.pi, color='r', linestyle='--', label='True Angle')
    plt.title('MVDR Beamforming Pattern')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Magnitude (dB)')
    plt.show()

if __name__ == "__main__":
    main()
