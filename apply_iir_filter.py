import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

def calculate_sampling_rate(df):
    """Calculates the average sampling rate from the 'time' column."""
    # Use time intervals directly, skip header row if needed
    time_intervals = df['time'].iloc[1:] 
    if time_intervals.empty or time_intervals.mean() == 0:
        print("Warning: Could not determine sampling rate from 'time' column.")
        # Fallback or default if needed, e.g., estimate from cumulative time
        time_diffs = np.diff(df['cumulative_time'])
        if len(time_diffs) > 0 and time_diffs.mean() > 0:
             avg_interval = time_diffs.mean()
             print(f"Estimating sampling rate from 'cumulative_time'. Average interval: {avg_interval:.4f} s")
             return 1.0 / avg_interval
        else:
            print("Error: Cannot determine sampling rate. Please specify manually.")
            return None # Or raise an error
    else:
        avg_interval = time_intervals.mean()
        print(f"Average sampling interval from 'time' column: {avg_interval:.4f} s")
        return 1.0 / avg_interval

def plot_fft(data, fs, column_name):
    """Calculates and plots the FFT of the data."""
    N = len(data)
    if N == 0 or fs is None or fs == 0:
        print(f"Skipping FFT plot for {column_name}: Invalid data or sampling rate.")
        return
        
    yf = fft(data.to_numpy())
    xf = fftfreq(N, 1 / fs)

    # Plot positive frequencies only
    positive_freq_indices = np.where(xf >= 0)
    xf_pos = xf[positive_freq_indices]
    yf_pos = np.abs(yf[positive_freq_indices])

    plt.figure(figsize=(10, 5))
    plt.plot(xf_pos, yf_pos)
    # plt.yscale('log') # Often helpful to see low-amplitude noise floor
    plt.title(f'FFT Analysis for {column_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    print(f"Displaying FFT plot for {column_name}. Close the plot window to continue.")
    plt.show() # Show plot and wait for user to close it

def main():
    input_file = 'dummy_accel_data.txt'
    output_dir = 'irr'
    columns_to_filter = ['xaccel', 'yaccel']
    filter_order = 4 # Butterworth filter order, common choice

    # --- 1. Load Data ---
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t', engine='python')
        print("Data loaded successfully.")
        print("Columns:", df.columns.tolist())
        print("First 5 rows:\n", df.head())
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Ensure columns exist
    if not all(col in df.columns for col in columns_to_filter):
        print(f"Error: One or more columns {columns_to_filter} not found in the file.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # --- 2. Calculate Sampling Rate ---
    print("Calculating sampling rate...")
    fs = calculate_sampling_rate(df)
    if fs is None:
        return # Stop if sampling rate couldn't be determined
    print(f"Calculated average sampling rate: {fs:.2f} Hz")
    nyquist_freq = fs / 2.0
    print(f"Nyquist Frequency: {nyquist_freq:.2f} Hz")

    # --- 3. Plot FFT ---
    print("\n--- FFT Analysis ---")
    print("Plotting FFT to help choose the cutoff frequency.")
    for col in columns_to_filter:
        if col in df.columns:
            # Handle potential NaNs or Infs before FFT
            signal_data = df[col].dropna() 
            if np.isinf(signal_data).any():
                print(f"Warning: Infinite values found in {col}, replacing with NaN.")
                signal_data = signal_data.replace([np.inf, -np.inf], np.nan).dropna()

            if not signal_data.empty:
                 plot_fft(signal_data, fs, col)
            else:
                 print(f"Skipping FFT for {col}: No valid data after cleaning.")
        else:
            print(f"Warning: Column {col} not found for FFT.")


    # --- 4. Get Cutoff Frequency from User ---
    cutoff_hz = None
    while cutoff_hz is None:
        try:
            cutoff_input = input(f"\nEnter the desired low-pass cutoff frequency (Hz) based on the FFT plots (must be > 0 and < {nyquist_freq:.2f}): ")
            cutoff_hz = float(cutoff_input)
            if not (0 < cutoff_hz < nyquist_freq):
                print(f"Error: Cutoff frequency must be between 0 and {nyquist_freq:.2f} Hz.")
                cutoff_hz = None # Reset to loop again
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"Using cutoff frequency: {cutoff_hz:.2f} Hz")

    # --- 5. Design Filter ---
    print(f"Designing Butterworth low-pass filter (order={filter_order}, cutoff={cutoff_hz} Hz)...")
    # Normalize cutoff frequency Wn = cutoff / (fs/2)
    normalized_cutoff = cutoff_hz / nyquist_freq
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    print("Filter coefficients (b, a) calculated.")

    # --- 6. Apply Filter ---
    print("Applying filter to columns...")
    for col in columns_to_filter:
         if col in df.columns:
            output_col_name = f"{col}_filtered_iir"
            # Apply filtfilt for zero phase distortion
            # Handle potential NaNs in the column before filtering
            signal_data = df[col].to_numpy()
            nan_mask = np.isnan(signal_data)
            
            # Basic NaN handling: interpolate or skip filtering if too many NaNs
            # For simplicity here, we'll filter only the non-NaN parts if few, 
            # or skip if mostly NaN. A better approach might involve interpolation.
            if nan_mask.any():
                 print(f"Warning: NaNs found in column {col}. Applying filter only to non-NaN segments.")
                 # A more robust solution might interpolate NaNs first
                 # Simple approach: Filter contiguous non-NaN blocks if possible or just skip
                 # For now, let's try replacing NaNs with 0 for filtering, but this can distort results
                 signal_data_nonan = np.nan_to_num(signal_data) 
                 filtered_signal_nonan = filtfilt(b, a, signal_data_nonan)
                 # Put NaNs back
                 filtered_signal = np.where(nan_mask, np.nan, filtered_signal_nonan)
            else:
                 filtered_signal = filtfilt(b, a, signal_data)
                 
            df[output_col_name] = filtered_signal
            print(f"  - Filtered data added as '{output_col_name}'")
         else:
             print(f"Warning: Column {col} not found for filtering, skipping.")

    # --- 7. Save Results ---
    print("Saving results...")
    try:
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        output_file = os.path.join(output_dir, f"dummy_filtered_iir_cutoff_{cutoff_hz:.1f}Hz.txt")
        df.to_csv(output_file, sep='\t', index=False, float_format='%.9f')
        print(f"Filtered data saved successfully to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    main() 