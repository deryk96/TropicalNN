import pandas as pd
import os

dataset = 'cifar'
# Define the directory where your files are located
directory = f'attack_results\\{dataset}_trop_no'  # Replace with the path to your files

# File pattern (assuming all files follow a similar naming pattern)
file_patterns = [f"CH_TropConv3Layer_{dataset}_0.03137254901960784_",
                #f"CH_ReluConv3Layer_{dataset}_0.03137254901960784_",
                #f"CH_MaxoutConv3Layer_{dataset}_0.03137254901960784_",
                #f"CH_MMRReluConv3Layer_{dataset}_0.03137254901960784_",
                #f"CH_TropConv3Layer_no_adv_mnist_0.2_",
                #f"CH_TropConv3Layer_yes_adv_mnist_0.2_",
                #"CH_ReluConv3Layer_mnist_0.2_",
                #"CH_MaxoutConv3Layer_mnist_0.2_",
                #"CH_MMRReluConv3Layer_mnist_0.2_",
                
                ]
#CH_MaxoutConv3Layer_svhn_0.03137254901960784_0_of_204_no_cw_spsa
#CH_MaxoutConv3Layer_cifar_0.03137254901960784_2_of_200_just_cw_spsa
for file_pattern in file_patterns:
    # Initialize an empty list to store the dataframes
    dataframes = []

    # Loop through all files in the directory
    for i in range(200):
        # Construct file name based on the pattern
        file_name = f"{file_pattern}{i}_of_200_just_cw_spsa.csv"

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
    combined_df.to_csv(os.path.join(directory, f'{file_pattern}combined_file.csv'), index=False)

    print(f"All files have been concatenated and saved as '{file_pattern}combined_file.csv'")
