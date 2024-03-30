import pandas as pd
import os
import re
import sys

# Check if a directory path was passed as an argument
if len(sys.argv) != 2:
    print("Usage: python script.py /path/to/your/csv/files")
    sys.exit(1)

# The first command line argument is the script name, so we take the second one
directory = sys.argv[1]

# Pattern to match and exclude the variable part of the filename
pattern = re.compile(r'_\d+_of_')

# Function to normalize filenames by removing the variable part
def normalize_filename(filename):
    return re.sub(pattern, '', filename)

# Function to calculate weighted average for a dataframe, including batch_size average
def weighted_average(df):
    # Calculate the weighted average for each column except 'batch_size'
    weighted_avg = {}
    for column in df.columns.difference(['batch_size']):
        total_weight = df['batch_size'].sum()
        weighted_sum = (df[column] * df['batch_size']).sum()
        weighted_avg[column] = weighted_sum / total_weight
    # Also calculate the simple average of the 'batch_size' itself
    weighted_avg['batch_size'] = df['batch_size'].mean()
    return weighted_avg

# Dictionary to hold aggregated data
aggregated_data = {}

# Iterate through files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Normalize the filename to aggregate data
        normalized_name = normalize_filename(filename)
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        
        # Aggregate data
        if normalized_name in aggregated_data:
            aggregated_data[normalized_name].append(df)
        else:
            aggregated_data[normalized_name] = [df]

# Dictionary to hold the final weighted averages
weighted_averages = {}

# Iterate over the aggregated data to calculate weighted averages
for name, dataframes in aggregated_data.items():
    combined_df = pd.concat(dataframes)
    avg_df = weighted_average(combined_df)
    weighted_averages[name] = avg_df

# Print out the results
for name, avg in weighted_averages.items():
    print(f"Results for {name}:")
    for column, value in avg.items():
        print(f"{column}: {value}")
    print("\n---\n")
