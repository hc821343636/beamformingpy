from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#坐标有问题
def generate_signal(
        frequency: float,
        duration: float,
        sampling_rate: int,
        angle: float,
        microphone_array: np.ndarray,
        speed_of_sound: float = 343
) -> Tuple[np.ndarray, np.ndarray]:
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)
    signal = np.zeros((len(microphone_array), num_samples))

    for i, mic_position in enumerate(microphone_array):
        distance = mic_position * np.sin(np.radians(angle))
        delay = distance / speed_of_sound
        signal[i, :] = np.sin(2 * np.pi * frequency * (time - delay))

    return time, signal


def estimate_angle(
        signal: np.ndarray,
        microphone_array: np.ndarray,
        frequency: float,
        d: float,
        sampling_rate: int,
        speed_of_sound: float = 343,
        delta: float = 0.01
) -> Tuple[float, np.ndarray]:
    R = np.cov(signal) + delta * np.identity(len(microphone_array))
    angles = np.linspace(-90, 90, 360)
    output_power = np.zeros_like(angles)

    for idx, theta in enumerate(angles):
        a = np.exp(-1j * 2 * np.pi * frequency * np.sin(np.radians(theta)) * np.arange(
            len(microphone_array)) * d / speed_of_sound)
        output_power[idx] = 1 / np.abs(np.conj(a).T @ np.linalg.inv(R) @ a)

    estimated_angle = angles[np.argmax(output_power)]

    return estimated_angle, output_power


def main() -> None:
    frequency = 2000  # Frequency of the signal in Hz
    duration = 1.0  # Duration of the signal in seconds
    sampling_rate = 44100  # Sampling rate in Hz
    N = 6  # Number of microphones in the array
    d = 0.033  # Distance between adjacent microphones in meters
    microphone_array = np.arange(N) * d
    angle_of_arrival = 30 # Expected angle of arrival in degrees

    time, signal = generate_signal(frequency, duration, sampling_rate, angle_of_arrival, microphone_array)

    # Modify the range of angles for estimation to cover 90 to 270 degrees
    angles = np.linspace(0, 360, 360)
    estimated_angle, output_power = estimate_angle(signal, microphone_array, frequency, d, sampling_rate)

    # Convert angles from degrees to radians for plotting
    angles_rad = np.radians(angles)

    # Plotting in polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles_rad, output_power)
    ax.set_theta_zero_location('E')  # Set 0 degrees to the right (East)
    ax.set_theta_direction(-1)  # Counterclockwise
    plt.show()
    # If you want to save the figure, uncomment the next line
    # plt.savefig('/mnt/data/mvdr_polar_plot.png')
    plt.close()




if __name__ == "__main__":
    main()
