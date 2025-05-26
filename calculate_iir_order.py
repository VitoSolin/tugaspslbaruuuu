import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttord
import os

def visualize_filter_specs(wp_hz, ws_hz, gpass_db, gstop_db, fs, order, wn_hz, plot_filename="iir_order_calculation_specs.png"):
    """Memvisualisasikan spesifikasi yang digunakan untuk perhitungan orde filter IIR."""
    fig, ax = plt.subplots(figsize=(10, 6))

    nyquist = fs / 2.0

    # Area Passband
    ax.axvspan(0, wp_hz, alpha=0.2, color='green', label=f'Passband (0 - {wp_hz:.1f} Hz)')
    ax.hlines(-gpass_db, 0, wp_hz, colors='green', linestyles='--', label=f'-{gpass_db:.1f} dB (gpass)')

    # Area Stopband
    ax.axvspan(ws_hz, nyquist, alpha=0.2, color='red', label=f'Stopband ({ws_hz:.1f} - {nyquist:.1f} Hz)')
    ax.hlines(-gstop_db, ws_hz, nyquist, colors='red', linestyles='--', label=f'-{gstop_db:.1f} dB (gstop)')

    # Area Transition Band
    ax.axvspan(wp_hz, ws_hz, alpha=0.2, color='yellow', label=f'Transition ({wp_hz:.1f} - {ws_hz:.1f} Hz)')

    # Garis vertikal untuk wp dan ws
    ax.axvline(wp_hz, color='darkgreen', linestyle=':', linewidth=2, label=f'wp: {wp_hz:.1f} Hz')
    ax.axvline(ws_hz, color='darkred', linestyle=':', linewidth=2, label=f'ws: {ws_hz:.1f} Hz')
    
    # Garis untuk Wn (cutoff -3dB yang direkomendasikan buttord, jika gpass=3 atau mendekati)
    # Untuk Butterworth, Wn dari buttord adalah frekuensi di mana gain adalah -gpass dB.
    # Frekuensi -3dB aktual dari filter yang didesain akan sangat dekat dengan wn_hz jika gpass=3dB.
    # Jika gpass != 3dB, wn_hz adalah frekuensi -gpass dB.
    ax.axvline(wn_hz, color='blue', linestyle='-.', linewidth=2, label=f'Wn (Cutoff -{gpass_db:.1f}dB): {wn_hz:.2f} Hz\n(Orde Filter: {order})')

    ax.set_xlabel("Frekuensi (Hz)")
    ax.set_ylabel("Atenuasi (dB Konseptual)")
    ax.set_title("Spesifikasi Desain Filter IIR Butterworth & Hasil Perhitungan Orde")
    ax.set_ylim(-max(gstop_db + 20, 70), 5) # Atur batas y agar lebih representatif
    ax.set_xlim(0, nyquist)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Plot visualisasi spesifikasi disimpan ke: {plot_filename}")
    plt.show()
    plt.close(fig)

def main():
    # --- Spesifikasi Filter yang Ditetapkan ---
    fs = 50.0  # Frekuensi Sampling (Hz) - Contoh untuk resampled_data.txt
    # Anda bisa mengubah fs ini atau membuatnya sebagai input jika perlu.
    # fs = 200.0 # Contoh untuk dummy_accel_data.txt

    desired_cutoff_hz = 12.0 # Target frekuensi cutoff -3dB
    wp_hz = desired_cutoff_hz   # Frekuensi tepi passband (Hz)
    # Kita set wp sama dengan target cutoff kita. gpass akan menentukan seberapa besar pelemahan di titik ini.
    
    transition_width_hz = 5.0 # Lebar pita transisi yang diinginkan (Hz)
    ws_hz = wp_hz + transition_width_hz # Frekuensi tepi stopband (Hz)
    
    gpass_db = 1.0              # Atenuasi maksimum di passband (dB) - Harus positif
    gstop_db = 40.0             # Atenuasi minimum di stopband (dB) - Harus positif

    print("--- Perhitungan Orde Filter IIR Butterworth ---")
    print(f"Frekuensi Sampling (fs): {fs:.1f} Hz")
    print(f"Frekuensi Nyquist: {(fs/2.0):.1f} Hz")
    print(f"\nSpesifikasi Filter:")
    print(f"  Frekuensi tepi passband (wp): {wp_hz:.2f} Hz")
    print(f"  Frekuensi tepi stopband (ws): {ws_hz:.2f} Hz")
    print(f"  Atenuasi maks. passband (gpass): {gpass_db:.1f} dB")
    print(f"  Atenuasi min. stopband (gstop): {gstop_db:.1f} dB")

    # Normalisasi frekuensi untuk buttord
    nyquist_freq = fs / 2.0
    wp_norm = wp_hz / nyquist_freq
    ws_norm = ws_hz / nyquist_freq

    # Hitung orde dan frekuensi cutoff alami (Wn) yang ternormalisasi
    # Wn yang dikembalikan buttord adalah frekuensi di mana gain adalah -gpass dB
    order, wn_norm_calculated = buttord(wp_norm, ws_norm, gpass_db, gstop_db, analog=False)
    wn_hz_calculated = wn_norm_calculated * nyquist_freq

    print(f"\nHasil Perhitungan dari buttord:")
    print(f"  Orde Filter (N) yang Dihitung: {order}")
    print(f"  Frekuensi Cutoff Alami (Wn) Ternormalisasi (-{gpass_db}dB): {wn_norm_calculated:.4f}")
    print(f"  Frekuensi Cutoff Alami (Wn) dalam Hz (-{gpass_db}dB): {wn_hz_calculated:.2f} Hz")
    print("\nCatatan: Wn adalah frekuensi di mana gain filter mencapai -gpass_db.")
    print("         Jika filter Butterworth didesain dengan orde ini dan Wn ini,")
    print("         maka spesifikasi gpass pada wp dan gstop pada ws akan terpenuhi.")
    print("         Frekuensi -3dB aktual mungkin sedikit berbeda jika gpass_db bukan 3dB.")
    print("         Dalam apply_iir_filter.py, kita menggunakan 'order' ini dengan cutoff -3dB tetap 12Hz.")

    # Visualisasi
    visualize_filter_specs(wp_hz, ws_hz, gpass_db, gstop_db, fs, order, wn_hz_calculated)

if __name__ == "__main__":
    main() 