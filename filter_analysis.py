# filter_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz
from scipy.fft import fft, fftfreq
import os 

# Define the output directory for plots
plot_output_dir = "filter_analysis_plots"

try:
    data = pd.read_csv('resampled_data.txt', sep='\t')
    print("Data loaded successfully.")
    print(data.head())
except FileNotFoundError:
    print("Error: resampled_data.txt not found.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

required_columns = ['time', 'xaccel', 'yaccel']
if not all(col in data.columns for col in required_columns):
    print(f"Error: Missing required columns. Found: {data.columns.tolist()}")
    exit()

time = data['time'].values
xaccel = data['xaccel'].values
yaccel = data['yaccel'].values


time_diffs = np.diff(time)
sampling_interval = np.mean(time_diffs)
fs = 1.0 / sampling_interval
print(f"\n--- Filter Parameters ---")
print(f"Calculated Sampling Frequency (fs): {fs:.2f} Hz")

N = len(time)
yf_x = fft(xaccel)
yf_y = fft(yaccel)
xf = fftfreq(N, sampling_interval)[:N//2] 

magnitude_x = 2.0/N * np.abs(yf_x[0:N//2])
magnitude_y = 2.0/N * np.abs(yf_y[0:N//2])

fig_fft_orig = plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1)
plt.plot(xf, magnitude_x)
plt.title('FFT of X Acceleration (Original)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(0, fs / 2) 

plt.subplot(1, 2, 2)
plt.plot(xf, magnitude_y)
plt.title('FFT of Y Acceleration (Original)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim(0, fs / 2)

plt.tight_layout()
plt.suptitle("Frequency Analysis of Original Data", y=1.02)
os.makedirs(plot_output_dir, exist_ok=True) # Ensure directory exists
plt.savefig(os.path.join(plot_output_dir, "fft_original_analysis.png")) # << SAVE PLOT 1
print(f"\nSaved plot: {os.path.join(plot_output_dir, 'fft_original_analysis.png')}")
plt.show()


cutoff_freq = 12.0  
print(f"Chosen Cutoff Frequency (fc): {cutoff_freq:.2f} Hz")

transition_width = 5.0  
print(f"Chosen Transition Bandwidth: {transition_width:.2f} Hz")


print(f"Passband: 0 Hz to {cutoff_freq:.2f} Hz")

stopband_start = cutoff_freq + transition_width
print(f"Stopband starts at: {stopband_start:.2f} Hz")


numtaps = int(np.ceil(3.3 * fs / transition_width))
if numtaps % 2 == 0: 
    numtaps += 1
print(f"Estimated Filter Order (numtaps): {numtaps}")

fir_coeffs = firwin(numtaps, cutoff_freq, window='hamming', fs=fs, pass_zero='lowpass')
print(f"Designed FIR filter with {len(fir_coeffs)} coefficients.")


fig_filter_resp = plt.figure(figsize=(10, 5))
w, h = freqz(fir_coeffs, worN=8000, fs=fs)
plt.plot(w, 20 * np.log10(np.abs(h))) 
plt.title('Designed FIR Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.axvline(cutoff_freq, color='r', linestyle='--', label=f'Cutoff Freq ({cutoff_freq} Hz)')
plt.axvline(stopband_start, color='g', linestyle='--', label=f'Stopband Start ({stopband_start} Hz)')
plt.ylim(-100, 5) 
plt.grid(True)
plt.legend()
os.makedirs(plot_output_dir, exist_ok=True) # Ensure directory exists
plt.savefig(os.path.join(plot_output_dir, "filter_frequency_response.png")) # << SAVE PLOT 2
print(f"Saved plot: {os.path.join(plot_output_dir, 'filter_frequency_response.png')}")
plt.show()


print("\nApplying FIR filter using numpy.convolve (manual convolution)...")
xaccel_filtered = np.convolve(xaccel, fir_coeffs, mode='same')
yaccel_filtered = np.convolve(yaccel, fir_coeffs, mode='same')
print("Manual convolution applied to xaccel and yaccel data.")

            

fig_time_comp = plt.figure(figsize=(12, 8)) 

plt.subplot(2, 1, 1)
plt.plot(time, xaccel, label='Original X Accel', alpha=0.7)
plt.plot(time, xaccel_filtered, label=f'Filtered X Accel (fc={cutoff_freq} Hz, Manual Conv)', linewidth=2)
plt.title('X Acceleration: Original vs Filtered (Manual Convolution)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, yaccel, label='Original Y Accel', alpha=0.7)
plt.plot(time, yaccel_filtered, label=f'Filtered Y Accel (fc={cutoff_freq} Hz, Manual Conv)', linewidth=2)
plt.title('Y Acceleration: Original vs Filtered (Manual Convolution)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend()
plt.grid(True)

plt.tight_layout()
os.makedirs(plot_output_dir, exist_ok=True) # Ensure directory exists
plt.savefig(os.path.join(plot_output_dir, "time_domain_comparison.png"))
print(f"Saved plot: {os.path.join(plot_output_dir, 'time_domain_comparison.png')}")
plt.show()


yf_x_filtered = fft(xaccel_filtered)
yf_y_filtered = fft(yaccel_filtered)
magnitude_x_filtered = 2.0/N * np.abs(yf_x_filtered[0:N//2])
magnitude_y_filtered = 2.0/N * np.abs(yf_y_filtered[0:N//2])

fig_fft_filt = plt.figure(figsize=(12, 6)) 
plt.subplot(1, 2, 1)
plt.plot(xf, magnitude_x, label='Original', alpha=0.6)
plt.plot(xf, magnitude_x_filtered, label='Filtered', linewidth=1.5)
plt.title('FFT of X Acceleration (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.axvline(cutoff_freq, color='r', linestyle='--', label=f'Cutoff Freq ({cutoff_freq} Hz)')
plt.legend()
plt.grid(True)
plt.xlim(0, fs / 2)
plt.ylim(bottom=0) 

plt.subplot(1, 2, 2)
plt.plot(xf, magnitude_y, label='Original', alpha=0.6)
plt.plot(xf, magnitude_y_filtered, label='Filtered', linewidth=1.5)
plt.title('FFT of Y Acceleration (Filtered)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.axvline(cutoff_freq, color='r', linestyle='--', label=f'Cutoff Freq ({cutoff_freq} Hz)')
plt.legend()
plt.grid(True)
plt.xlim(0, fs / 2)
plt.ylim(bottom=0)

plt.tight_layout()
plt.suptitle("Frequency Analysis After Filtering", y=1.02)
os.makedirs(plot_output_dir, exist_ok=True) # Ensure directory exists
plt.savefig(os.path.join(plot_output_dir, "fft_filtered_analysis.png")) # << SAVE PLOT 4
print(f"Saved plot: {os.path.join(plot_output_dir, 'fft_filtered_analysis.png')}")
plt.show()

print("\nAnalysis and filtering complete. Plots generated and saved.")