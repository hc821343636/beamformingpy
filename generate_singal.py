import numpy as np
import scipy.io.wavfile as wav
if __name__ == '__main__':

    # Parameters for the sweep signal
    frequency_start = 2000  # Starting frequency of 2 kHz
    frequency_end = 5000    # Ending frequency of 5 kHz
    duration = 1            # Duration of 1 second
    sample_rate = 44100     # Sampling rate in Hz

    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Generating the sweep signal
    sweep_signal = np.sin(2 * np.pi * ((frequency_end - frequency_start) / (2 * duration) * t**2 + frequency_start * t))

    # Normalize to 16-bit range
    sweep_signal_normalized = np.int16(sweep_signal / np.max(np.abs(sweep_signal)) * 32767)

    # Save to a .wav file
    filename = 'data/sweep_signal_2kHz_to_5kHz.wav'
    wav.write(filename, sample_rate, sweep_signal_normalized)


