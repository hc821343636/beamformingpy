from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_signal(
    frequency: float, 
    duration: float, 
    sampling_rate: int, 
    angle: float, 
    microphone_array: np.ndarray, 
    speed_of_sound: float = 343
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the signal received by each microphone in the array.

    :param frequency: Frequency of the signal in Hz.
    :param duration: Duration of the signal in seconds.
    :param sampling_rate: Sampling rate in Hz.
    :param angle: Angle of arrival of the signal in degrees.
    :param microphone_array: Array containing the positions of the microphones.
    :param speed_of_sound: Speed of sound in m/s (default is 343 m/s).
    :return: Tuple of time array and generated signal.
    """
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
    """
    Estimate the incident angle using the MVDR algorithm.

    :param signal: Signal received by the microphone array.
    :param microphone_array: Array containing the positions of the microphones.
    :param frequency: Frequency of the signal in Hz.
    :param d: Distance between adjacent microphones in the array.
    :param sampling_rate: Sampling rate in Hz.
    :param speed_of_sound: Speed of sound in m/s (default is 343 m/s).
    :param delta: Regularization term to avoid singularity (default is 0.01).
    :return: Tuple of estimated angle in degrees and output power array.
    """
    R = np.cov(signal) + delta * np.identity(len(microphone_array))
    angles = np.linspace(-90, 90, 360)
    output_power = np.zeros_like(angles)

    for idx, theta in enumerate(angles):
        a = np.exp(-1j * 2 * np.pi * frequency * np.sin(np.radians(theta)) * np.arange(len(microphone_array)) * d / speed_of_sound)
        output_power[idx] = 1 / np.abs(np.conj(a).T @ np.linalg.inv(R) @ a)

    estimated_angle = angles[np.argmax(output_power)]

    return estimated_angle, output_power

def main() -> None:
    """
    Main function to estimate the angle of arrival of a signal using MVDR beamforming.
    """
    frequency = 2000  # Frequency of the signal in Hz
    duration = 1.0    # Duration of the signal in seconds
    sampling_rate = 441000  # Sampling rate in Hz
    N = 6             # Number of microphones in the array
    d = 0.033         # Distance between adjacent microphones in meters
    microphone_array = np.arange(N) * d
    angle_of_arrival = 40  # Expected angle of arrival in degrees

    time, signal = generate_signal(frequency, duration, sampling_rate, angle_of_arrival, microphone_array)
    estimated_angle, output_power = estimate_angle(signal, microphone_array, frequency, d, sampling_rate)

    plt.plot(np.linspace(-90, 90, 360), output_power)
    plt.title(f"Estimated Angle: {estimated_angle} degrees")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Output Power")
    plt.show()

if __name__ == "__main__":
    main()
