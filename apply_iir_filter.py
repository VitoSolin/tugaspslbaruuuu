import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

def calculate_sampling_rate(df):
    """Calculates the average sampling rate from the 'time' or 'cumulative_time' column."""
    if 'time' in df.columns and not df['time'].iloc[1:].empty:
        # Check if 'time' looks like cumulative timestamps or intervals
        # If the first value is small and it's generally increasing, it's likely cumulative.
        # A more robust check might be needed for diverse data formats.
        is_cumulative_time_column = df['time'].iloc[0] < (df['time'].iloc[-1] / len(df['time'])) if len(df['time']) > 1 else False
        # Heuristic: if 'time' column values are much larger than their average diff, it's cumulative
        time_col_values = df['time'].dropna()
        if len(time_col_values) > 1:
            avg_diff = np.mean(np.diff(time_col_values))
            if avg_diff > 0 and time_col_values.mean() / avg_diff > 10: # Arbitrary ratio, adjust if needed
                is_cumulative_time_column = True
            else:
                is_cumulative_time_column = False # Likely intervals
        elif len(time_col_values) == 1:
            is_cumulative_time_column = False # Cannot determine, assume interval or single point
        else: # Empty after dropna
            is_cumulative_time_column = False

        if is_cumulative_time_column:
            print("'time' column appears to be cumulative timestamps. Calculating sampling rate from differences.")
            time_diffs = np.diff(time_col_values)
            if len(time_diffs) > 0 and time_diffs.mean() > 0:
                avg_interval = time_diffs.mean()
                print(f"Average interval from 'time' column differences: {avg_interval:.6f} s")
                return 1.0 / avg_interval
            else:
                print("Warning: Could not determine sampling rate from 'time' column differences.")
        else: # Assume 'time' column contains intervals (or it's ambiguous and we try this path)
            print("'time' column appears to contain time intervals. Calculating sampling rate directly.")
            # Use time intervals directly, skip header row if needed by iloc[1:] if first row is not a valid interval
            # For resampled_data, the first value is a timestamp, so diff is better.
            # For dummy_data, it's an interval, but using its mean is fine.
            time_intervals = df['time'].iloc[1:] # This was the original logic for interval-like 'time' col
            if not time_intervals.empty and time_intervals.mean() > 0:
                avg_interval = time_intervals.mean()
                print(f"Average sampling interval from 'time' column (treated as intervals): {avg_interval:.6f} s")
                return 1.0 / avg_interval
            else:
                print("Warning: Could not determine sampling rate from 'time' column (treated as intervals).")

    # Fallback to 'cumulative_time' if 'time' didn't yield a result or is missing
    if 'cumulative_time' in df.columns:
        print("Falling back to 'cumulative_time' for sampling rate calculation.")
        time_diffs = np.diff(df['cumulative_time'].dropna())
        if len(time_diffs) > 0 and time_diffs.mean() > 0:
             avg_interval = time_diffs.mean()
             print(f"Average interval from 'cumulative_time': {avg_interval:.6f} s")
             return 1.0 / avg_interval
        else:
            print("Warning: Could not determine sampling rate from 'cumulative_time'.")
    
    print("Error: Cannot determine sampling rate from available time columns. Please specify manually.")
    return None

def plot_fft(data, fs, column_name, plot_output_dir=None, save_filename=None):
    """Calculates and plots the FFT of the data, optionally saves the plot."""
    N = len(data)
    if N == 0 or fs is None or fs == 0:
        print(f"Skipping FFT plot for {column_name}: Invalid data or sampling rate.")
        return
        
    yf = fft(data.to_numpy())
    xf = fftfreq(N, 1 / fs)

    positive_freq_indices = np.where(xf >= 0)
    xf_pos = xf[positive_freq_indices]
    yf_pos = np.abs(yf[positive_freq_indices])

    fig = plt.figure(figsize=(10, 5))
    plt.plot(xf_pos, 2.0/N * yf_pos) # Normalized magnitude
    plt.title(f'FFT Analysis for {column_name}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0, fs / 2) # Show up to Nyquist frequency

    if plot_output_dir and save_filename:
        if not os.path.exists(plot_output_dir):
            os.makedirs(plot_output_dir)
        full_save_path = os.path.join(plot_output_dir, save_filename)
        plt.savefig(full_save_path)
        print(f"FFT plot saved to: {full_save_path}")
    
    print(f"Displaying FFT plot for {column_name}. Close the plot window to continue.")
    plt.show()
    plt.close(fig) # Close the figure after showing/saving

def main():
    input_file = 'resampled_data.txt'
    output_dir = 'iir_filtered_output'
    plot_output_dir_iir = "iir_filter_analysis_plots"
    os.makedirs(plot_output_dir_iir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    columns_to_filter = ['xaccel', 'yaccel']
    filter_order = 4

    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t', engine='python')
        print("Data loaded successfully.")
        # Ensure 'cumulative_time' column exists, if not, create it from 'time' if 'time' is cumulative
        if 'cumulative_time' not in df.columns and 'time' in df.columns:
            # Heuristic check similar to calculate_sampling_rate to see if 'time' is cumulative
            time_col_values = df['time'].dropna()
            is_cumulative_candidate = False
            if len(time_col_values) > 1:
                avg_diff_check = np.mean(np.diff(time_col_values))
                if avg_diff_check > 0 and time_col_values.mean() / avg_diff_check > 10: 
                    is_cumulative_candidate = True
            
            if is_cumulative_candidate:
                print("'cumulative_time' column not found. Creating it from 'time' column.")
                df['cumulative_time'] = df['time']
            # If 'time' is not cumulative-like, 'calculate_sampling_rate' will handle it or fail.

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    if not all(col in df.columns for col in columns_to_filter + ['time', 'cumulative_time']):
        print(f"Error: One or more required columns ({columns_to_filter + ['time', 'cumulative_time']}) not found.")
        return

    print("Calculating sampling rate...")
    fs = calculate_sampling_rate(df)
    if fs is None:
        return
    print(f"Calculated average sampling rate: {fs:.2f} Hz")
    nyquist_freq = fs / 2.0
    print(f"Nyquist Frequency: {nyquist_freq:.2f} Hz")

    print("\n--- FFT Analysis of Original Data ---")
    for col in columns_to_filter:
        if col in df.columns:
            signal_data = df[col].dropna()
            if not signal_data.empty:
                 plot_fft(signal_data, fs, f"{col} (Original)", 
                          plot_output_dir=plot_output_dir_iir, 
                          save_filename=f"original_fft_{col}.png")
            else:
                 print(f"Skipping FFT for {col}: No valid data.")

    cutoff_hz = None
    while cutoff_hz is None:
        try:
            cutoff_input = input(f"\nEnter the desired low-pass cutoff frequency (Hz) (must be > 0 and < {nyquist_freq:.2f}): ")
            cutoff_hz = float(cutoff_input)
            if not (0 < cutoff_hz < nyquist_freq):
                print(f"Error: Cutoff frequency must be between 0 and {nyquist_freq:.2f} Hz.")
                cutoff_hz = None
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"Using cutoff frequency: {cutoff_hz:.2f} Hz")

    print(f"Designing Butterworth low-pass filter (order={filter_order}, cutoff={cutoff_hz} Hz)..." )
    normalized_cutoff = cutoff_hz / nyquist_freq
    b, a = butter(filter_order, normalized_cutoff, btype='low', analog=False)
    print("Filter coefficients (b, a) calculated.")

    # Plot IIR Filter Frequency Response
    w, h = freqz(b, a, worN=8000, fs=fs)
    fig_resp = plt.figure(figsize=(10, 5))
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title(f'IIR Filter Frequency Response (Cutoff: {cutoff_hz:.1f} Hz)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.axvline(cutoff_hz, color='red', linestyle='--', label=f'Cutoff Freq ({cutoff_hz:.1f} Hz)')
    plt.axhline(-3, color='green', linestyle='--', label='-3dB (Half-power)') # Common reference for cutoff
    plt.ylim(-60, 5) # Adjust Y limit for better view of passband and stopband
    plt.grid(True)
    plt.legend()
    resp_filename = os.path.join(plot_output_dir_iir, f"iir_filter_response_cutoff_{cutoff_hz:.1f}Hz.png")
    plt.savefig(resp_filename)
    print(f"Filter response plot saved to: {resp_filename}")
    plt.show()
    plt.close(fig_resp)

    print("Applying filter to columns...")
    df_filtered_iir = df.copy() # Work on a copy for filtered data
    for col in columns_to_filter:
         if col in df.columns:
            output_col_name = f"{col}_filtered_iir"
            signal_data = df[col].to_numpy()
            nan_mask = np.isnan(signal_data)
            if nan_mask.any():
                 signal_data_nonan = np.nan_to_num(signal_data)
                 filtered_signal_nonan = filtfilt(b, a, signal_data_nonan)
                 filtered_signal = np.where(nan_mask, np.nan, filtered_signal_nonan)
            else:
                 filtered_signal = filtfilt(b, a, signal_data)
            df_filtered_iir[output_col_name] = filtered_signal
            print(f"  - Filtered data added as '{output_col_name}'")

    # Plot Time-Domain Comparison
    fig_time, axs_time = plt.subplots(len(columns_to_filter), 1, figsize=(12, 5 * len(columns_to_filter)), sharex=True)
    if len(columns_to_filter) == 1:
        axs_time = [axs_time] # Ensure axs_time is iterable for single column
    for i, col in enumerate(columns_to_filter):
        original_signal = df[col]
        filtered_signal_col = f"{col}_filtered_iir"
        if filtered_signal_col in df_filtered_iir.columns:
            filtered_signal_data = df_filtered_iir[filtered_signal_col]
            axs_time[i].plot(df['cumulative_time'], original_signal, label=f'Original {col}', alpha=0.7)
            axs_time[i].plot(df['cumulative_time'], filtered_signal_data, label=f'Filtered {col} (IIR Cutoff: {cutoff_hz:.1f}Hz)', linewidth=1.5)
            axs_time[i].set_title(f'{col}: Original vs Filtered (IIR)')
            axs_time[i].set_ylabel('Acceleration')
            axs_time[i].legend()
            axs_time[i].grid(True)
    axs_time[-1].set_xlabel('Cumulative Time (s)')
    plt.tight_layout()
    time_comp_filename = os.path.join(plot_output_dir_iir, f"iir_time_domain_comparison_cutoff_{cutoff_hz:.1f}Hz.png")
    plt.savefig(time_comp_filename)
    print(f"Time domain comparison plot saved to: {time_comp_filename}")
    plt.show()
    plt.close(fig_time)

    print("Saving filtered data results...")
    try:
        output_file = os.path.join(output_dir, f"resampled_filtered_iir_cutoff_{cutoff_hz:.1f}Hz.txt")
        df_filtered_iir.to_csv(output_file, sep='\t', index=False, float_format='%.9f')
        print(f"Filtered data saved successfully to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

    print("\n--- FFT Analysis of Filtered Data ---")
    if fs is not None:
        for col_original in columns_to_filter:
            filtered_col_name = f"{col_original}_filtered_iir"
            if filtered_col_name in df_filtered_iir.columns:
                signal_data_filtered = df_filtered_iir[filtered_col_name].dropna()
                if not signal_data_filtered.empty:
                     plot_fft(signal_data_filtered, fs, f"{filtered_col_name} (Cutoff: {cutoff_hz:.1f} Hz)", 
                              plot_output_dir=plot_output_dir_iir, 
                              save_filename=f"filtered_fft_{col_original}_cutoff_{cutoff_hz:.1f}Hz.png")
                else:
                     print(f"Skipping FFT for {filtered_col_name}: No valid data after cleaning.")
    else:
        print("Skipping FFT of filtered data because sampling rate (fs) is not available.")

if __name__ == "__main__":
    main() 