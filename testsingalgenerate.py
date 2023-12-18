import numpy as np
import matplotlib.pyplot as plt

def generate_signal_circular_array(
    frequency: float,
    duration: float,
    sampling_rate: int,
    angle: float,
    num_mics: int,
    radius: float,
    speed_of_sound: float = 343
) -> np.ndarray:
    """
    Generate the signal received by each microphone in a circular array.
    """
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)
    signal = np.zeros((num_mics, num_samples))
    mic_angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)

    # Plot the signals for verification
    plt.figure(figsize=(15, 7))

    for i, mic_angle in enumerate(mic_angles):
        mic_x = radius * np.cos(mic_angle)
        mic_y = radius * np.sin(mic_angle)
        source_x = radius * np.cos(np.radians(angle))
        source_y = radius * np.sin(np.radians(angle))
        distance = np.sqrt((mic_x - source_x)**2 + (mic_y - source_y)**2)
        delay = distance / speed_of_sound
        signal[i, :] = np.sin(2 * np.pi * frequency * (time - delay))

        plt.plot(time, signal[i, :], label=f'Microphone {i+1}')

    plt.title('Signals received by each microphone')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    return np.array([time, signal])
if __name__ == '__main__':
    frequency = 3000  # Frequency of the signal in Hz
    duration = 0.002  # Duration of the signal in seconds
    sampling_rate = 44100  # Sampling rate in Hz
    num_mics = 6  # Number of microphones in the array
    radius = 0.025  # Radius of the circular microphone array in meters
    angle_of_arrival = 40  # Expected angle of arrival in degrees
    time, signal = generate_signal_circular_array(frequency, duration, sampling_rate, angle_of_arrival, num_mics, radius)