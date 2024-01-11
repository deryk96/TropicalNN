import pandas as pd
import os

# Define the directory where your files are located
directory = 'attack_results\\mnist_no_adv\\relu'  # Replace with the path to your files

# File pattern (assuming all files follow a similar naming pattern)
file_pattern = "CH_TropConv3LayerLogits_mnist_0.1_"
file_pattern = "CH_ReluConv3Layer_mnist_0.1_"
#file_pattern = "CH_MaxoutConv3Layer_mnist_0.1_"

#CH_TropConv3LayerLogits_mnist_0.1_0_of_79
# Initialize an empty list to store the dataframes
dataframes = []

# Loop through all files in the directory
for i in range(79):
    # Construct file name based on the pattern
    file_name = f"{file_pattern}{i}_of_79.csv"
    file_path = os.path.join(directory, file_name)
    
    # Check if the file exists
    if os.path.isfile(file_path):
        # Read the file and append to the list
        df = pd.read_csv(file_path)
        dataframes.append(df)
    else:
        print(f"File not found: {file_path}")

# Concatenate all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv(os.path.join(directory, 'combined_file.csv'), index=False)

print("All files have been concatenated and saved as 'combined_file.csv'")
