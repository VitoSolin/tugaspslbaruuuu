import numpy as np
import pandas as pd
import os

def generate_dummy_data(filename="dummy_accel_data.txt", 
                          sampling_rate=40.0, # Hz (similar to original)
                          duration=10.0,    # seconds
                          signal_freq_x=1.0, # Hz
                          signal_amp_x=2.0,
                          signal_freq_y=1.5, # Hz
                          signal_amp_y=1.5,
                          noise_freq=15.0,  # Hz (high frequency noise)
                          noise_amp=0.5,
                          random_noise_std=0.2):
    """Generates dummy accelerometer data and saves it to a file."""

    print(f"Generating dummy data: {duration}s at {sampling_rate}Hz...")
    
    # --- Time Vector ---
    num_samples = int(sampling_rate * duration)
    dt = 1.0 / sampling_rate
    cumulative_time = np.linspace(0, duration - dt, num_samples)
    
    # Simulate slight variations in sample time like original data
    time_intervals = np.random.normal(loc=dt, scale=dt*0.1, size=num_samples)
    time_intervals[0] = dt # Start interval isn't well-defined, use dt
    # Ensure cumulative time calculated from intervals matches approx
    simulated_cumulative_time = np.cumsum(time_intervals)
    simulated_cumulative_time -= simulated_cumulative_time[0] # Start at 0
    # Use the linearly spaced time for calculations, but save the varied intervals

    # --- Signal Components ---
    t = cumulative_time # Use the ideal time for calculations
    signal_x = signal_amp_x * np.sin(2 * np.pi * signal_freq_x * t)
    signal_y = signal_amp_y * np.cos(2 * np.pi * signal_freq_y * t) # Use cosine for y
    
    noise_signal_x = noise_amp * np.sin(2 * np.pi * noise_freq * t)
    noise_signal_y = noise_amp * np.cos(2 * np.pi * noise_freq * t + np.pi/4) # Phase shift noise y
    
    random_noise_x = np.random.normal(0, random_noise_std, num_samples)
    random_noise_y = np.random.normal(0, random_noise_std, num_samples)

    # --- Combine Components ---
    xaccel = signal_x + noise_signal_x + random_noise_x
    yaccel = signal_y + noise_signal_y + random_noise_y

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
    df.loc[0, 'time'] = df.loc[1, 'time'] if len(df) > 1 else dt 
    df.loc[0, 'cumulative_time'] = df.loc[0, 'time'] # First cumulative is just first interval

    # Ensure correct column order
    df = df[['x', 'y', 'time', 'xaccel', 'yaccel', 'cumulative_time']]

    # --- Save to File ---
    try:
        df.to_csv(filename, sep='\t', index=False, float_format='%.9f')
        print(f"Dummy data successfully saved to: {filename}")
    except Exception as e:
        print(f"Error saving dummy data: {e}")

if __name__ == "__main__":
    generate_dummy_data() 