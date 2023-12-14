import numpy as np
import matplotlib.pyplot as plt


def mvdr_beamformer(R, a_theta_d):
    # R: Covariance matrix of the received signals
    # a_theta_d: Steering vector corresponding to the desired direction

    # Calculate the MVDR weight vector
    mu = 1 / (a_theta_d.conj().T @ np.linalg.inv(R) @ a_theta_d)
    w_mvdr = (np.linalg.inv(R) @ a_theta_d) * mu

    return w_mvdr


def compute_mvdr_beam_pattern(R, angles, theta_d):
    beam_pattern = []

    # Function to compute the steering vector for a specific angle
    def a_theta_d(angle):
        return np.exp(1j * 2 * np.pi * np.sin(np.radians(angle)) * np.arange(M))

    for angle in angles:
        a_theta = a_theta_d(angle)
        w_mvdr = mvdr_beamformer(R, a_theta_d(theta_d))
        beam_pattern.append(np.abs(np.dot(w_mvdr.conj().T, a_theta)))

    return np.array(beam_pattern)


# Number of array elements
M = 8

# Generate a sample covariance matrix (you need to replace this with your actual covariance matrix)
R = np.eye(M)

# Generate angles for beam pattern plot (in degrees)
angles = np.linspace(-90, 90, 180)

# Assume the desired direction is 0 degrees
theta_d = 20

# Compute MVDR beam pattern
beam_pattern_mvdr = compute_mvdr_beam_pattern(R, angles, theta_d)

# Plot the MVDR beam pattern
plt.figure(figsize=(10, 6))
plt.plot(angles, 20 * np.log10(beam_pattern_mvdr / np.max(beam_pattern_mvdr)), label='MVDR')
plt.title('MVDR Beam Pattern')
plt.xlabel('Angle (degrees)')
plt.ylabel('Gain (dB)')
plt.legend()
plt.grid(True)
plt.show()