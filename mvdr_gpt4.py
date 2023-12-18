import numpy as np
import matplotlib.pyplot as plt

# 远场效应处理
def generate_signal_circular_array(frequency, duration, sampling_rate, angle, num_mics, radius, speed_of_sound=343):
    num_samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, num_samples)
    signal = np.zeros((num_mics, num_samples))
    source_angle_rad = np.radians(angle)
    mic_angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)

    for i, mic_angle in enumerate(mic_angles):
        # Calculate the angle difference between the source and microphone position
        angle_diff = mic_angle - source_angle_rad

        # Calculate the distance based on the arc length (radius * angle_diff)
        distance = radius * angle_diff

        # Calculate the time delay
        delay = distance / speed_of_sound

        # Generate the signal with the calculated delay
        signal[i, :] = np.sin(2 * np.pi * frequency * (time - delay))

    return signal

def estimate_angle_circular_array(signal, frequency, radius, num_mics, speed_of_sound=343):
    R = np.cov(signal)
    angles = np.linspace(0, 2 * np.pi, 360)
    output_power = np.zeros_like(angles)

    for idx, theta in enumerate(angles):
        steering_vector = np.array([np.exp(-1j * 2 * np.pi * frequency * radius / speed_of_sound *
                                          np.cos(mic_angle - theta)) for mic_angle in np.linspace(0, 2 * np.pi, num_mics, endpoint=False)])
        output_power[idx] = 1 / np.abs(np.conj(steering_vector).T @ np.linalg.inv(R) @ steering_vector)

    estimated_angle = np.degrees(angles[np.argmax(output_power)])
    return estimated_angle, angles, output_power

def main():
    frequency = 10000  # Hz
    duration = 1.0    # seconds
    sampling_rate = 44100  # Hz
    num_mics = 6
    radius = 0.025    # meters
    angle_of_arrival = 40  # degrees

    signal = generate_signal_circular_array(frequency, duration, sampling_rate, angle_of_arrival, num_mics, radius)
    estimated_angle, angles, output_power = estimate_angle_circular_array(signal, frequency, radius, num_mics)

    # Plotting in polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(angles, output_power)
    ax.set_theta_zero_location('N')  # Set 0 degrees to the top
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_title(f"Estimated Angle: {estimated_angle} degrees")
    plt.show()

if __name__ == "__main__":
    main()

