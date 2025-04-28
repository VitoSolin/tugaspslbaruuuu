import pandas as pd

# Define input and output file paths
input_file = 'databaru.txt'
output_file = 'databaru_processed.txt'

# Read the tab-separated data file
try:
    df = pd.read_csv(input_file, sep='\t')

    # Check if 'time' column exists
    if 'time' not in df.columns:
        print(f"Error: 'time' column not found in {input_file}")
    else:
        # Calculate cumulative time
        df['cumulative_time'] = df['time'].cumsum()

        # Sort the DataFrame by cumulative time
        df_sorted = df.sort_values(by='cumulative_time')

        # Select and reorder columns for the output file
        # Keeping original columns + cumulative time at the end
        output_columns = ['x', 'y', 'time', 'xaccel', 'yaccel', 'cumulative_time']
        # Ensure all expected columns exist before selecting
        existing_output_columns = [col for col in output_columns if col in df_sorted.columns]
        df_output = df_sorted[existing_output_columns]

        # Save the processed data to a new file
        df_output.to_csv(output_file, sep='\t', index=False, float_format='%.9f')

        print(f"Data successfully processed and saved to {output_file}")
        print("Columns in the new file:")
        print(df_output.columns.tolist())
        print("\nFirst 5 rows of the processed data:")
        print(df_output.head().to_string(index=False))


except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found.")
except Exception as e:
    print(f"An error occurred: {e}") 