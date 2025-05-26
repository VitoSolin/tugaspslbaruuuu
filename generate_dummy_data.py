import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def generate_dummy_data(filename="dummy_accel_data.txt",
                          sampling_rate=200.0, # Hz, increased to capture up to 40Hz properly
                          duration=10.0,    # seconds
                          amp_pass_band=1.0,  # Amplitude for 2-10 Hz components
                          amp_block_band=0.8, # Amplitude for 15-40 Hz components
                          random_noise_std=0.15,
                          plot_output_dir="dummy_data_plots"):
    """Generates dummy accelerometer data, saves it, and plots its characteristics."""

    print(f"Generating dummy data: {duration}s at {sampling_rate}Hz...")
    os.makedirs(plot_output_dir, exist_ok=True)

    # --- Time Vector ---
    num_samples = int(sampling_rate * duration)
    dt = 1.0 / sampling_rate
    cumulative_time = np.linspace(0, duration - dt, num_samples)

    # Simulate slight variations in sample time like original data
    time_intervals = np.random.normal(loc=dt, scale=dt*0.05, size=num_samples) # Reduced variation scale
    time_intervals[time_intervals <= 0] = dt # Prevent non-positive intervals
    time_intervals[0] = dt # Start interval isn't well-defined, use dt
    simulated_cumulative_time = np.cumsum(time_intervals)
    simulated_cumulative_time -= simulated_cumulative_time[0] # Start at 0
    
    t = cumulative_time # Use the ideal time for calculations

    # --- Signal Components (Pass Band: 2-10 Hz) ---
    xaccel_pass = np.zeros(num_samples)
    yaccel_pass = np.zeros(num_samples)
    pass_freqs = [3.0, 6.0, 9.0] # Example frequencies in 2-10 Hz range
    for i, freq in enumerate(pass_freqs):
        xaccel_pass += (amp_pass_band / len(pass_freqs)) * np.sin(2 * np.pi * freq * t + i * np.pi/3)
        yaccel_pass += (amp_pass_band / len(pass_freqs)) * np.cos(2 * np.pi * freq * t + i * np.pi/4) 

    # --- Signal Components (Block Band: 15-40 Hz) ---
    xaccel_block = np.zeros(num_samples)
    yaccel_block = np.zeros(num_samples)
    block_freqs = [18.0, 25.0, 35.0] # Example frequencies in 15-40 Hz range
    for i, freq in enumerate(block_freqs):
        xaccel_block += (amp_block_band / len(block_freqs)) * np.sin(2 * np.pi * freq * t + i * np.pi/5)
        yaccel_block += (amp_block_band / len(block_freqs)) * np.cos(2 * np.pi * freq * t + i * np.pi/6)

    # --- Random Noise ---
    random_noise_x = np.random.normal(0, random_noise_std, num_samples)
    random_noise_y = np.random.normal(0, random_noise_std, num_samples)

    # --- Combine Components ---
    xaccel = xaccel_pass + xaccel_block + random_noise_x
    yaccel = yaccel_pass + yaccel_block + random_noise_y

    # --- Placeholder Position Data ---
    x_pos = np.zeros(num_samples)
    y_pos = np.zeros(num_samples)

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'x': x_pos,
        'y': y_pos,
        'time': time_intervals, # Store the varied time intervals
        'xaccel': xaccel,
        'yaccel': yaccel,
        'cumulative_time': simulated_cumulative_time # Use the cumulative time from varied intervals
    })

    # Adjust first row time interval (it's often different in real data)
    # df.loc[0, 'time'] = df.loc[1, 'time'] if len(df) > 1 else dt
    # Ensure first cumulative_time is based on the first time interval
    if len(df) > 0:
        df.loc[0, 'cumulative_time'] = df.loc[0, 'time']
    
    # Ensure correct column order
    df = df[['x', 'y', 'time', 'xaccel', 'yaccel', 'cumulative_time']]

    # --- Save to File ---
    try:
        df.to_csv(filename, sep='\t', index=False, float_format='%.9f')
        print(f"Dummy data successfully saved to: {filename}")
    except Exception as e:
        print(f"Error saving dummy data: {e}")

    # --- Plotting Generated Data ---
    print("Plotting generated data...")

    # 1. Time Domain Plot
    fig_time, axs_time = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs_time[0].plot(df['cumulative_time'], df['xaccel'], label='X Accel')
    axs_time[0].set_title('Generated X Acceleration (Time Domain)')
    axs_time[0].set_ylabel('Acceleration')
    axs_time[0].grid(True)
    axs_time[0].legend()

    axs_time[1].plot(df['cumulative_time'], df['yaccel'], label='Y Accel', color='orange')
    axs_time[1].set_title('Generated Y Acceleration (Time Domain)')
    axs_time[1].set_xlabel('Time (s)')
    axs_time[1].set_ylabel('Acceleration')
    axs_time[1].grid(True)
    axs_time[1].legend()

    plt.tight_layout()
    time_plot_filename = os.path.join(plot_output_dir, "generated_data_time_domain.png")
    plt.savefig(time_plot_filename)
    print(f"Time domain plot saved to: {time_plot_filename}")
    # plt.show() # Optionally show plots interactively
    plt.close(fig_time)

    # 2. FFT Plot
    N = num_samples
    fs = sampling_rate
    yf_x = fft(df['xaccel'].to_numpy())
    yf_y = fft(df['yaccel'].to_numpy())
    xf = fftfreq(N, 1 / fs)[:N//2]

    magnitude_x = 2.0/N * np.abs(yf_x[0:N//2])
    magnitude_y = 2.0/N * np.abs(yf_y[0:N//2])

    fig_fft, axs_fft = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axs_fft[0].plot(xf, magnitude_x, label='FFT of X Accel')
    axs_fft[0].set_title('FFT of Generated X Acceleration')
    axs_fft[0].set_ylabel('Magnitude')
    axs_fft[0].set_xlim(0, fs / 2)
    axs_fft[0].grid(True)
    axs_fft[0].legend()

    axs_fft[1].plot(xf, magnitude_y, label='FFT of Y Accel', color='orange')
    axs_fft[1].set_title('FFT of Generated Y Acceleration')
    axs_fft[1].set_xlabel('Frequency (Hz)')
    axs_fft[1].set_ylabel('Magnitude')
    axs_fft[1].set_xlim(0, fs / 2)
    axs_fft[1].grid(True)
    axs_fft[1].legend()

    plt.tight_layout()
    fft_plot_filename = os.path.join(plot_output_dir, "generated_data_fft.png")
    plt.savefig(fft_plot_filename)
    print(f"FFT plot saved to: {fft_plot_filename}")
    # plt.show() # Optionally show plots interactively
    plt.close(fig_fft)

    print("Data generation and plotting complete.")

if __name__ == "__main__":
    generate_dummy_data() 