import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def generate_signal(frequency, duration, sampling_rate, angle, microphone_array, speed_of_sound=343):
    """
    Generate the signal received by each microphone in the array.
    """
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)
    k = 2 * np.pi * frequency / speed_of_sound
    signal = np.zeros((len(microphone_array), num_samples))

    for i, mic_position in enumerate(microphone_array):
        distance = mic_position * np.sin(np.radians(angle))
        delay = distance / speed_of_sound
        signal[i, :] = np.sin(2 * np.pi * frequency * (time - delay))

    return time, signal

def estimate_angle(signal, microphone_array, sampling_rate, speed_of_sound=343):
    """
    Estimate the incident angle using the MVDR algorithm.
    """
    # Calculate the covariance matrix
    R = np.cov(signal)

    # Assuming the direction of arrival is in the range of -90 to 90 degrees
    angles = np.linspace(-90, 90, 360)
    output_power = np.zeros_like(angles, dtype=np.float64)

    for idx, theta in enumerate(angles):
        # Steering vector
        a = np.exp(-1j * 2 * np.pi * frequency * np.sin(np.radians(theta)) * np.arange(len(microphone_array)) * d / speed_of_sound)
        output_power[idx] = 1 / (np.conj(a).T @ np.linalg.inv(R) @ a).real

    # Find the peak in the output power pattern
    peaks, _ = find_peaks(output_power)
    estimated_angle = angles[peaks[0]]

    return estimated_angle, output_power

# Parameters
frequency = 1000  # Hz
duration = 1.0    # seconds
sampling_rate = 8000  # Hz
N = 4             # Number of microphones
d = 0.05          # Distance between microphones in meters
microphone_array = np.arange(N) * d
angle_of_arrival = 30  # degrees

# Generate signal
time, signal = generate_signal(frequency, duration, sampling_rate, angle_of_arrival, microphone_array)

# Estimate the angle
estimated_angle, output_power = estimate_angle(signal, microphone_array, sampling_rate)

# Plotting
plt.plot(np.linspace(-90, 90, 360), output_power)
plt.title(f"Estimated Angle: {estimated_angle} degrees")
plt.xlabel("Angle (degrees)")
plt.ylabel("Output Power")
plt.show()
