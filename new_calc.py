import pandas as pd
import glob
import os

# Set the path to the directory containing your CSV files
path = os.path.expanduser('~/TropicalNN/new_attack_results/cifar10/VGG16/trop/yes/')
all_files = glob.glob(path + "/*_cw_spsa.csv")  # Only files ending with '_cw_spsa.csv'

# List to hold all the DataFrames
dfs = []

# Loop through all files and read them into a DataFrame
for filename in all_files:
    df = pd.read_csv(filename)
    dfs.append(df)

# Concatenate all the DataFrames into one
concatenated_df = pd.concat(dfs, ignore_index=True)

# Save the concatenated DataFrame to a new CSV file
concatenated_df.to_csv(path + 'concatenated.csv', index=False)
