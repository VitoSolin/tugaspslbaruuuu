import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, freqz, buttord
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import os

def calculate_sampling_rate(df):
    """Menghitung rata-rata sampling rate dari kolom 'time' atau 'cumulative_time'."""
    if 'time' in df.columns and not df['time'].iloc[1:].empty:
        # Periksa apakah kolom 'time' berisi timestamp kumulatif atau interval waktu.
        # Heuristik: jika nilai 'time' umumnya lebih besar dari rata-rata selisihnya, diasumsikan kumulatif.
        time_col_values = df['time'].dropna()
        is_cumulative_time_column = False
        if len(time_col_values) > 1:
            avg_diff = np.mean(np.diff(time_col_values))
            if avg_diff > 0 and time_col_values.mean() / avg_diff > 10: # Rasio heuristik.
                is_cumulative_time_column = True
        elif len(time_col_values) == 1:
            pass # Tidak dapat ditentukan, akan dicoba jalur lain.

        if is_cumulative_time_column:
            print("Kolom 'time' terdeteksi sebagai timestamp kumulatif. Menghitung sampling rate dari selisih nilai.")
            time_diffs = np.diff(time_col_values)
            if len(time_diffs) > 0 and time_diffs.mean() > 0:
                avg_interval = time_diffs.mean()
                print(f"Rata-rata interval dari selisih kolom 'time': {avg_interval:.6f} s")
                return 1.0 / avg_interval
            else:
                print("Peringatan: Tidak dapat menentukan sampling rate dari selisih kolom 'time'.")
        else: # Asumsikan kolom 'time' berisi interval.
            print("Kolom 'time' diasumsikan berisi interval waktu. Menghitung sampling rate secara langsung.")
            time_intervals = df['time'].iloc[1:] 
            if not time_intervals.empty and time_intervals.mean() > 0:
                avg_interval = time_intervals.mean()
                print(f"Rata-rata interval sampling dari kolom 'time' (sebagai interval): {avg_interval:.6f} s")
                return 1.0 / avg_interval
            else:
                print("Peringatan: Tidak dapat menentukan sampling rate dari kolom 'time' (sebagai interval).")

    # Fallback ke 'cumulative_time' jika 'time' gagal atau tidak ada.
    if 'cumulative_time' in df.columns:
        print("Mencoba menggunakan 'cumulative_time' untuk menghitung sampling rate.")
        time_diffs = np.diff(df['cumulative_time'].dropna())
        if len(time_diffs) > 0 and time_diffs.mean() > 0:
             avg_interval = time_diffs.mean()
             print(f"Rata-rata interval dari 'cumulative_time': {avg_interval:.6f} s")
             return 1.0 / avg_interval
        else:
            print("Peringatan: Tidak dapat menentukan sampling rate dari 'cumulative_time'.")
    
    print("Error: Tidak dapat menentukan sampling rate. Mohon tentukan secara manual.")
    return None

def plot_fft(data, fs, column_name, plot_output_dir=None, save_filename=None):
    """Menghitung dan menampilkan plot FFT. Dapat juga menyimpan plot ke file."""
    N = len(data)
    if N == 0 or fs is None or fs == 0:
        print(f"Melewati plot FFT untuk {column_name}: Data tidak valid atau sampling rate tidak sesuai.")
        return
        
    yf = fft(data.to_numpy())
    xf = fftfreq(N, 1 / fs)

    positive_freq_indices = np.where(xf >= 0)
    xf_pos = xf[positive_freq_indices]
    yf_pos = np.abs(yf[positive_freq_indices])

    fig = plt.figure(figsize=(10, 5))
    plt.plot(xf_pos, 2.0/N * yf_pos) # Magnitudo ternormalisasi.
    plt.title(f'Analisis FFT untuk {column_name}')
    plt.xlabel('Frekuensi (Hz)')
    plt.ylabel('Magnitudo')
    plt.grid(True)
    plt.xlim(0, fs / 2) # Tampilkan hingga frekuensi Nyquist.

    if plot_output_dir and save_filename:
        if not os.path.exists(plot_output_dir):
            os.makedirs(plot_output_dir)
        full_save_path = os.path.join(plot_output_dir, save_filename)
        plt.savefig(full_save_path)
        print(f"Plot FFT disimpan ke: {full_save_path}")
    
    print(f"Menampilkan plot FFT untuk {column_name}. Tutup jendela plot untuk melanjutkan.")
    plt.show()
    plt.close(fig) # Tutup gambar setelah ditampilkan/disimpan.

def main():
    # --- Konfigurasi Awal ---
    input_file = 'resampled_data.txt' # File input.
    output_dir = 'iir_filtered_output' # Direktori output untuk data terfilter IIR.
    plot_output_dir_iir = "iir_filter_analysis_plots" # Direktori output untuk plot analisis IIR.
    os.makedirs(plot_output_dir_iir, exist_ok=True) # Buat direktori jika belum ada.
    os.makedirs(output_dir, exist_ok=True)      # Buat direktori jika belum ada.

    columns_to_filter = ['xaccel', 'yaccel'] # Kolom yang akan difilter.

    # --- 1. Pemuatan Data ---
    print(f"Memuat data dari {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t', engine='python')
        print("Data berhasil dimuat.")
        # Pastikan kolom 'cumulative_time' ada; jika tidak, buat dari 'time' jika 'time' kumulatif.
        if 'cumulative_time' not in df.columns and 'time' in df.columns:
            # Heuristik untuk memeriksa apakah 'time' kumulatif.
            time_col_values = df['time'].dropna()
            is_cumulative_candidate = False
            if len(time_col_values) > 1:
                avg_diff_check = np.mean(np.diff(time_col_values))
                if avg_diff_check > 0 and time_col_values.mean() / avg_diff_check > 10: 
                    is_cumulative_candidate = True
            
            if is_cumulative_candidate:
                print("Kolom 'cumulative_time' tidak ditemukan. Membuat dari kolom 'time'.")
                df['cumulative_time'] = df['time']
            # Jika 'time' tidak kumulatif, 'calculate_sampling_rate' akan menanganinya.

    except FileNotFoundError:
        print(f"Error: File input '{input_file}' tidak ditemukan.")
        return
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return
    
    # Periksa keberadaan kolom yang dibutuhkan.
    required_cols_check = columns_to_filter + ['time', 'cumulative_time']
    if not all(col in df.columns for col in required_cols_check):
        print(f"Error: Kolom yang dibutuhkan ({required_cols_check}) tidak ditemukan.")
        return

    # --- 2. Perhitungan Sampling Rate ---
    print("Menghitung sampling rate...")
    fs = calculate_sampling_rate(df)
    if fs is None:
        return # Tidak dapat melanjutkan jika sampling rate tidak diketahui.
    print(f"Rata-rata sampling rate: {fs:.2f} Hz")
    nyquist_freq = fs / 2.0
    print(f"Frekuensi Nyquist: {nyquist_freq:.2f} Hz")

    # --- 3. Analisis FFT Data Asli ---
    print("\n--- Analisis FFT Data Asli ---")
    for col in columns_to_filter:
        if col in df.columns:
            signal_data = df[col].dropna()
            if not signal_data.empty:
                 plot_fft(signal_data, fs, f"{col} (Asli)", 
                          plot_output_dir=plot_output_dir_iir, 
                          save_filename=f"original_fft_{col}.png")
            else:
                 print(f"Melewati FFT untuk {col}: Data kosong.")

    # --- Penentuan Parameter dan Orde Filter IIR ---
    desired_cutoff_hz = 12.0 # Frekuensi cutoff yang diinginkan tetap 12 Hz
    wp_hz = desired_cutoff_hz   # Tepi passband di frekuensi cutoff
    ws_hz = wp_hz + 5.0         # Tepi stopband (transisi 5 Hz)
    gpass_db = 1.0              # Maksimum loss di passband (dB)
    gstop_db = 40.0             # Minimum atenuasi di stopband (dB)

    print("\n--- Spesifikasi untuk Perhitungan Orde Filter IIR ---")
    print(f"  Frekuensi tepi passband (wp): {wp_hz:.2f} Hz")
    print(f"  Frekuensi tepi stopband (ws): {ws_hz:.2f} Hz")
    print(f"  Atenuasi maks. passband (gpass): {gpass_db:.1f} dB")
    print(f"  Atenuasi min. stopband (gstop): {gstop_db:.1f} dB")

    # Normalisasi frekuensi untuk buttord
    wp_norm = wp_hz / nyquist_freq
    ws_norm = ws_hz / nyquist_freq

    order_calculated, wn_norm_buttord = buttord(wp_norm, ws_norm, gpass_db, gstop_db, analog=False)
    
    print(f"Orde filter IIR yang dihitung: {order_calculated}")
    # wn_norm_buttord adalah frekuensi ternormalisasi (-gpass dB) yang direkomendasikan buttord.
    # Kita akan tetap menggunakan desired_cutoff_hz untuk mendesain filter.
    
    print(f"Menggunakan frekuensi cutoff yang diinginkan: {desired_cutoff_hz:.2f} Hz")
    normalized_desired_cutoff = desired_cutoff_hz / nyquist_freq

    print(f"Mendesain filter Butterworth low-pass (orde={order_calculated}, cutoff={desired_cutoff_hz:.2f} Hz)..." )
    b, a = butter(order_calculated, normalized_desired_cutoff, btype='low', analog=False)
    print("Koefisien filter (b, a) telah dihitung.")

    # Plot IIR Filter Frequency Response (gunakan desired_cutoff_hz untuk plotting)
    cutoff_hz_for_plot = desired_cutoff_hz 
    w, h = freqz(b, a, worN=8000, fs=fs)
    fig_resp = plt.figure(figsize=(10, 5))
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title(f'Respons Frekuensi Filter IIR (Orde: {order_calculated}, Cutoff: {cutoff_hz_for_plot:.1f} Hz)')
    plt.xlabel('Frekuensi (Hz)')
    plt.ylabel('Gain (dB)')
    plt.axvline(cutoff_hz_for_plot, color='red', linestyle='--', label=f'Target Cutoff ({cutoff_hz_for_plot:.1f} Hz)')
    plt.axvline(wp_hz, color='purple', linestyle=':', label=f'wp ({wp_hz:.1f} Hz)')
    plt.axvline(ws_hz, color='cyan', linestyle=':', label=f'ws ({ws_hz:.1f} Hz)')
    plt.axhline(-gpass_db, color='orange', linestyle='--', label=f'-{gpass_db:.1f}dB (gpass)')
    plt.axhline(-gstop_db, color='brown', linestyle='--', label=f'-{gstop_db:.1f}dB (gstop)')
    plt.ylim(-max(gstop_db + 20, 60), 5) 
    plt.grid(True)
    plt.legend()
    resp_filename = os.path.join(plot_output_dir_iir, f"iir_filter_response_order{order_calculated}_cutoff{cutoff_hz_for_plot:.1f}Hz.png")
    plt.savefig(resp_filename)
    print(f"Plot respons filter disimpan ke: {resp_filename}")
    plt.show()
    plt.close(fig_resp)

    # --- 6. Aplikasi Filter pada Data ---
    print("Menerapkan filter pada data...")
    df_filtered_iir = df.copy() # Proses pada salinan DataFrame.
    for col in columns_to_filter:
         if col in df.columns:
            output_col_name = f"{col}_filtered_iir"
            signal_data = df[col].to_numpy()
            nan_mask = np.isnan(signal_data)
            if nan_mask.any(): # Penanganan nilai NaN.
                 signal_data_nonan = np.nan_to_num(signal_data) # Ganti NaN dengan 0 sementara.
                 filtered_signal_nonan = filtfilt(b, a, signal_data_nonan)
                 filtered_signal = np.where(nan_mask, np.nan, filtered_signal_nonan) # Kembalikan NaN.
            else:
                 filtered_signal = filtfilt(b, a, signal_data) # Proses jika tidak ada NaN.
            df_filtered_iir[output_col_name] = filtered_signal
            print(f"  - Data terfilter ditambahkan sebagai '{output_col_name}'")

    # --- 6a. Perbandingan Data Asli vs Filtered (Domain Waktu) ---
    fig_time, axs_time = plt.subplots(len(columns_to_filter), 1, figsize=(12, 5 * len(columns_to_filter)), sharex=True)
    if len(columns_to_filter) == 1: # Pastikan axs_time iterable.
        axs_time = [axs_time]
    for i, col in enumerate(columns_to_filter):
        original_signal = df[col]
        filtered_signal_col = f"{col}_filtered_iir"
        if filtered_signal_col in df_filtered_iir.columns:
            filtered_signal_data = df_filtered_iir[filtered_signal_col]
            axs_time[i].plot(df['cumulative_time'], original_signal, label=f'{col} Asli', alpha=0.7)
            axs_time[i].plot(df['cumulative_time'], filtered_signal_data, label=f'{col} Filtered (IIR Cutoff: {cutoff_hz_for_plot:.1f}Hz)', linewidth=1.5)
            axs_time[i].set_title(f'{col}: Asli vs Filtered (IIR)')
            axs_time[i].set_ylabel('Akselerasi')
            axs_time[i].legend()
            axs_time[i].grid(True)
    axs_time[-1].set_xlabel('Waktu Kumulatif (s)')
    plt.tight_layout()
    time_comp_filename = os.path.join(plot_output_dir_iir, f"iir_time_domain_comp_order{order_calculated}_cutoff{cutoff_hz_for_plot:.1f}Hz.png")
    plt.savefig(time_comp_filename)
    print(f"Plot perbandingan domain waktu disimpan ke: {time_comp_filename}")
    plt.show()
    plt.close(fig_time)

    # --- 7. Penyimpanan Hasil Filter ke File ---
    print("Menyimpan hasil data terfilter...")
    try:
        output_file = os.path.join(output_dir, f"resampled_filtered_iir_order{order_calculated}_cutoff_{cutoff_hz_for_plot:.1f}Hz.txt") 
        df_filtered_iir.to_csv(output_file, sep='\t', index=False, float_format='%.9f')
        print(f"Data terfilter berhasil disimpan ke: {output_file}")
    except Exception as e:
        print(f"Error saat menyimpan hasil: {e}")

    # --- 8. Analisis FFT Data Terfilter ---
    print("\n--- Analisis FFT Data Terfilter ---")
    if fs is not None: # Pastikan fs tersedia.
        for col_original in columns_to_filter:
            filtered_col_name = f"{col_original}_filtered_iir"
            if filtered_col_name in df_filtered_iir.columns:
                signal_data_filtered = df_filtered_iir[filtered_col_name].dropna()
                if not signal_data_filtered.empty:
                     plot_fft(signal_data_filtered, fs, f"{filtered_col_name} (Orde: {order_calculated}, Cutoff: {cutoff_hz_for_plot:.1f} Hz)", 
                              plot_output_dir=plot_output_dir_iir, 
                              save_filename=f"filtered_fft_{col_original}_order{order_calculated}_cutoff_{cutoff_hz_for_plot:.1f}Hz.png")
                else:
                     print(f"Melewati FFT untuk {filtered_col_name}: Data kosong setelah dibersihkan.")
    else:
        print("Melewati FFT data terfilter karena sampling rate (fs) tidak tersedia.")

if __name__ == "__main__":
    main() 