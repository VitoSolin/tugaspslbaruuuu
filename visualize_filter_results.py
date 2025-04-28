import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison(df, time_col, original_col, filtered_col):
    """Plots original vs filtered data against time."""
    
    if not all(col in df.columns for col in [time_col, original_col, filtered_col]):
        print(f"Error: Missing one or more required columns: {time_col}, {original_col}, {filtered_col}")
        print(f"Available columns: {df.columns.tolist()}")
        return False # Indicate failure

    plt.figure(figsize=(12, 6))
    plt.plot(df[time_col], df[original_col], label='Original', alpha=0.7)
    plt.plot(df[time_col], df[filtered_col], label='Filtered (IIR 7.0Hz Cutoff)', linewidth=2)
    
    plt.title(f'Comparison: {original_col} vs {filtered_col}')
    plt.xlabel('Cumulative Time (s)')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True)
    return True # Indicate success

def main():
    input_file = os.path.join('irr', 'dummy_filtered_iir_cutoff_7.0Hz.txt')
    time_column = 'cumulative_time'
    original_x_col = 'xaccel'
    filtered_x_col = 'xaccel_filtered_iir'
    original_y_col = 'yaccel'
    filtered_y_col = 'yaccel_filtered_iir'

    # --- Load Data ---
    print(f"Loading filtered data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t', engine='python')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Filtered data file '{input_file}' not found.")
        print("Please ensure you have run 'generate_dummy_data.py' and then 'apply_iir_filter.py' on the dummy data.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Create Plots ---
    print("Generating comparison plots...")
    
    plot_x_success = plot_comparison(df, time_column, original_x_col, filtered_x_col)
    plot_y_success = plot_comparison(df, time_column, original_y_col, filtered_y_col)

    if plot_x_success or plot_y_success:
        print("Displaying plots. Close the plot windows to exit.")
        plt.show()
    else:
        print("Could not generate plots due to missing columns.")

if __name__ == "__main__":
    main() 