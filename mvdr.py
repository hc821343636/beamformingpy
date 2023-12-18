from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


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
        distance = mic_position * np.cos(np.radians(angle))
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
    R = np.cov(signal)
    angles = np.linspace(0, 180, 360)
    output_power = np.zeros_like(angles)

    for idx, theta in enumerate(angles):
        a = np.exp(-1j * 2 * np.pi * frequency * np.cos(np.radians(theta)) * np.arange(
            len(microphone_array)) * d / speed_of_sound)
        output_power[idx] = 1 / np.abs(a.T @ np.linalg.inv(R) @ a)

    estimated_angle = angles[np.argmax(output_power)]

    return estimated_angle, output_power


def main() -> None:
    """
    Main function to estimate the angle of arrival of a signal using MVDR beamforming.
    """
    frequency = 3000  # Frequency of the signal in Hz   高于5200hz就会有旁瓣
    duration = 0.005  # Duration of the signal in seconds
    sampling_rate = 44100  # Sampling rate in Hz
    N = 6  # Number of microphones in the array
    d = 0.001  # Distance between adjacent microphones in meters
    microphone_array = np.arange(N) * d
    angle_of_arrival = 39  # Expected angle of arrival in degrees

    time, signal = generate_signal(frequency, duration, sampling_rate, angle_of_arrival, microphone_array)
    estimated_angle, output_power = estimate_angle(signal, microphone_array, frequency, d, sampling_rate)

    """plt.figure(figsize=(15, 10))
    for i in range(N):
        plt.plot(time, signal[i, :], label=f'Microphone {i + 1}')  # Offset each signal for clarity"""

    """ plt.title('Signals received by each microphone due to incident angle of 40 degrees')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    #plt.savefig(f'image/8Signals of {angle_of_arrival} degrees.png')
    plt.show()"""

    plt.plot(np.linspace(0, 180, 360), output_power)
    plt.title(f"Estimated Angle: {estimated_angle} degrees")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Output Power")
    plt.show()


if __name__ == "__main__":
    main()
