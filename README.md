# Tropical CNNs

This README is designed to provide basic instructions needed to run the experiments we conducted for the results reported in the paper entitled ["Tropical Decision Boundaries for Neural Networks Are Robust Against Adversarial Attacks"](https://arxiv.org/abs/2402.00576)

## Step 1. Build the models.
Main file used is cleverhans_model_builds.py

### Load required packages
Code was run in a python 3.8.11
pip install -r $HOME/TropicalNN/requirements.txt
pip install tensorflow_datasets  # likely redundant from requirements.txt
pip install easydict # likely redundant from requirements.txt
pip install cleverhans # likely redundant from requirements.txt

### Arguments for cleverhans_model_builds.py
argument 1: Adversarial training (yes/no)
argument 2: Dataset (mnist, svhn, cifar)
argument 3: Batch size (integer)

### Model Save Location
Models with save in saved_models directory as tf style. 

### Build MNIST models
- normal training (ReLU, Maxout, and Tropical) `python cleverhans_model_builds.py no mnist 128`
- adversarial training (ReLU+AT, Maxout+AT, and tropical+AT) `python cleverhans_model_builds.py yes mnist 128`
- MMR model `python cleverhans_model_builds_with_MMR_reg.py no mnist`

### Build SVHN models
- normal training (ReLU, Maxout, and Tropical) `python cleverhans_model_builds.py no svhn 128`
- adversarial training (ReLU+AT, Maxout+AT, and tropical+AT) `python cleverhans_model_builds.py yes svhn 128`
- MMR model `python cleverhans_model_builds_with_MMR_reg.py no svhn`

### Build CIFAR-10 models
- normal training (ReLU, Maxout, and Tropical) `python cleverhans_model_builds.py no cifar 128`
- adversarial training (ReLU+AT, Maxout+AT, and tropical+AT) `python cleverhans_model_builds.py yes cifar 128`
- MMR model *N/A, no CIFAR MMR model built*

## Step 2. Attack the models. 
Main file used is cleverhans_attacks.py

#### Output save location
Batch results written to CSV files in the attack_results directory. **Please note that summary statistics are not automated, but computed in excel. If multiple batchs run, the CSVs will need to be combined before analysis conducted.**

#### Arguments for cleverhans_attacks.py
argument 1: Batch chunk (integer). *The number of the batch, can be an integer from 0 to (argument 2) - 1. Allows for the user to run the attacks in batches*
argument 2: Total batch chunks (integer). *The total number of batches you want to split the data into. Allows for user to run the attacks in batches*
argument 3: Batch size (integer). *The number of input data attacked at once. Does not have to relate to argument 1 or 2, but ought to be less than total number of examples in dataset / argument 2. 
argument 4: Dataset (mnist, svhn, cifar)
argument 5: Attack the adversarial trained versions of models (yes/no)

### Attack MNIST models
- normal training (ReLU, Maxout, Tropical, and MMR) `python cleverhans_attacks.py 0 1 32 mnist no`
- adversarial training (ReLU+AT, Maxout+AT, and tropical+AT) `python cleverhans_attacks.py 0 1 32 mnist yes`

### Attack SVHN models
- normal training (ReLU, Maxout, Tropical, and MMR) `python cleverhans_attacks.py 0 1 32 svhn no`
- adversarial training (ReLU+AT, Maxout+AT, and tropical+AT) `python cleverhans_attacks.py 0 1 32 svhn yes`

### Attack CIFAR-10 models
- normal training (ReLU, Maxout, and Tropical) `python cleverhans_attacks.py 0 1 32 cifar no`
- adversarial training (ReLU+AT, Maxout+AT, and tropical+AT) `python cleverhans_attacks.py 0 1 32 cifar yes`

