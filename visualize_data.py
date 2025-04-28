import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np # Import numpy for calculations

# Define input file path
input_file = 'databaru_processed.txt'
# path_plot_file = 'path_plot.png' # Removed as we focus on time-domain plots now
# timeseries_plot_file = 'timeseries_plot.png' # Removed as we show plot directly


def plot_data(input_file):
    """
    Reads data, calculates acceleration from position, and plots:
    1. Position vs Time
    2. Sensor Acceleration vs Derived Acceleration vs Time
    """
    try:
        # Use sep='\s+' to handle potential multiple spaces/tabs as delimiters
        data = pd.read_csv(input_file, sep='\s+', engine='python')

        # Verify columns - print them to help debug if needed
        # print("Columns in DataFrame:", data.columns)
        # print("First 5 rows:\n", data.head())

        # Check for required columns
        required_columns = ['cumulative_time', 'x', 'y', 'xaccel', 'yaccel']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            print(f"Available columns: {', '.join(data.columns)}")
            # Attempt to rename if the header was read incorrectly (e.g., shifted)
            # This is a guess based on the typical structure
            if len(data.columns) == len(required_columns) + 1: # Check if there's an extra unnamed column
                print("Attempting to fix column names assuming first column is index...")
                try:
                    # Check if the first column looks like an index (0, 1, 2...)
                    if data.columns[0].startswith('Unnamed'): 
                        data = pd.read_csv(input_file, sep='\s+', header=0, names=required_columns, index_col=0, engine='python')
                        print("Reloaded data with explicit column names.")
                    else:
                        data = pd.read_csv(input_file, sep='\s+', header=0, names=required_columns, engine='python')
                        print("Reloaded data with explicit column names.")
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    if missing_columns:
                        print(f"Error persists after attempting rename: Missing {', '.join(missing_columns)}")
                        sys.exit(1)
                except Exception as e:
                    print(f"Failed to automatically fix columns: {e}")
                    sys.exit(1)
            else:
                print("Cannot automatically fix column names. Please check the file's header row.")
                sys.exit(1)

        # Convert columns to numeric, coercing errors (replace non-numeric with NaN)
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with NaN values in essential columns that might result from coercion
        data.dropna(subset=required_columns, inplace=True)

        # Check if data is empty after cleaning
        if data.empty:
            print("Error: No valid data remaining after cleaning. Check the file format and content.")
            sys.exit(1)
        
        # Sort data by time just in case it's not ordered
        data = data.sort_values(by='cumulative_time').reset_index(drop=True)

        # --- Calculate Acceleration from Position ---
        # Calculate time difference (dt)
        dt = data['cumulative_time'].diff().fillna(0) 
        # Avoid division by zero if dt is 0 (e.g., first point or duplicate times)
        dt[dt == 0] = np.nan # Replace 0 dt with NaN to avoid division errors

        # Calculate velocity (first derivative of position)
        vx = data['x'].diff() / dt
        vy = data['y'].diff() / dt

        # Calculate acceleration (first derivative of velocity)
        # Use numpy's gradient for potentially smoother derivative calculation
        # Note: np.gradient might handle endpoints better than simple diff
        time_points = data['cumulative_time'].values
        vx_filled = vx.fillna(0) # Fill NaN for gradient calculation (e.g., first point)
        vy_filled = vy.fillna(0)
        
        ax_derived = np.gradient(vx_filled, time_points, edge_order=2)
        ay_derived = np.gradient(vy_filled, time_points, edge_order=2)
        
        # Add derived accelerations to the DataFrame for plotting
        data['ax_derived'] = ax_derived
        data['ay_derived'] = ay_derived
        
        # Remove the first row where derivatives might be unreliable
        # data = data.iloc[1:].copy()
        # Option: Keep first row but derived accel will be based on fillna(0) or gradient edge handling

        # --- Visualization --- 
        # Increase figure width for better readability
        fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True) # Adjusted figsize

        # Subplot 1: Position vs Time (remains the same)
        axs[0].plot(data['cumulative_time'], data['x'], label='Position X', marker='.', linestyle='-')
        axs[0].plot(data['cumulative_time'], data['y'], label='Position Y', marker='.', linestyle='-')
        axs[0].set_ylabel('Position')
        axs[0].set_title('Robot Position vs. Time')
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Sensor Acceleration vs Derived Acceleration
        axs[1].plot(data['cumulative_time'], data['xaccel'], label='Sensor Accel X', marker='.', linestyle='-', alpha=0.7)
        axs[1].plot(data['cumulative_time'], data['yaccel'], label='Sensor Accel Y', marker='.', linestyle='-', alpha=0.7)
        axs[1].plot(data['cumulative_time'], data['ax_derived'], label='Derived Accel X (from Pos)', marker='', linestyle='--', linewidth=2)
        axs[1].plot(data['cumulative_time'], data['ay_derived'], label='Derived Accel Y (from Pos)', marker='', linestyle='--', linewidth=2)
        axs[1].set_xlabel('Cumulative Time (s)')
        axs[1].set_ylabel('Acceleration')
        axs[1].set_title('Comparison: Sensor Acceleration vs. Derived Acceleration (from Position)')
        axs[1].legend()
        axs[1].grid(True)
        # Optional: Set Y-axis limits if noise is too high and hides derived signal
        # ylim_max = data['ax_derived'].abs().max() * 2 # Example limit based on derived
        # axs[1].set_ylim(-ylim_max, ylim_max)

        # Adjust layout and display the plot
        plt.tight_layout()
        plt.show()

        print("Plot generated successfully.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1) # Exit script if file not found
    except ImportError as e:
        print(f"Error: Required library not found. {e}. Please install it.")
        print("Try: pip install pandas matplotlib numpy")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # You might want to print the full traceback for debugging complex issues
        # import traceback
        # traceback.print_exc()
        sys.exit(1) # Exit script on other errors

if __name__ == "__main__":
    plot_data(input_file) 