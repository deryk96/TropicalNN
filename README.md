# Tropical CNNs

This README is designed to provide and overview of this repo and basic instructions needed to run the experiments we conducted for the results reported in the paper entitled ["Tropical Decision Boundaries for Neural Networks Are Robust Against Adversarial Attacks"](https://arxiv.org/abs/2402.00576). In short, the experiment involved training neural network models, attacking the test set using 6 attacks, and then evaluating accuracy of model against clean and attacked test set. We also reported model parameter counts and training times to showcase low/no cost of the tropical method relative to other adversarial defense methods.

## Overview of files:

- Training Scripts
  - [train.py](https://github.com/KurtPask/TropicalNN/blob/main/train.py) : Code for building all models except MMR regularized models. Very crude dictionary that serves as config for which models to build in here. Might update for external config at some point.  
  - [train_with_MMR_regularizer.py](https://github.com/KurtPask/TropicalNN/blob/main/train_with_MMR_regularizer.py) : Builds either the MNIST or SVHN MMR-Universal regularized Custom LeNet5 model. Training structure was sufficiently different enough to include separate training file from others. 

- Attacking Scripts : Please note, the config for which models to attack given the dataset argument currently housed in "functions/utils.py". Might update for external config at some point, but the dictionary in the utils file is for both attack scripts below.
  - [attack_pgd_algorithms.py](https://github.com/KurtPask/TropicalNN/blob/main/attack_pgd_algorithms.py) : Code used to perform the PGD-based attacks on target models given a dataset argument by user. Designed to be able to run in batches if one has access to an HPC. 
  - [attack_cw_spsa_algorithms.py](https://github.com/KurtPask/TropicalNN/blob/main/attack_cw_spsa_algorithms.py) : Code used to perform the the Carlini and Wagner as well as the SPSA attacks on target models given a dataset argument by user. Designed to be able to run in batches if one has access to an HPC. 

- Helper Scripts
  - [attack_batch_results_calcs.py](https://github.com/KurtPask/TropicalNN/blob/main/attack_batch_results_calcs.py) : Script useful for getting results from a batch run of attacks. Results are stored in CSV's in nested directory.
  - [test_top5_acc.py](https://github.com/KurtPask/TropicalNN/blob/main/test_top5_acc.py) : Allows one to get top5 accuracy for given models. Only used in experiment for CIFAR-100, but could expand basic structure to quickly get top1 accuracy for clean data too.
  - [custom_layers/tropical_layers.py](https://github.com/KurtPask/TropicalNN/blob/main/custom_layers/tropical_layers.py) : Holds the key part of this experiment, the tropical embedding layer. There are also multiple expansions not used in the experiment including two asymmetric distance metrics possible for use in the tropical embedding layer, two untested regularizers, a tropical convolution layer, and a tropical embedding layer with the top 2 metric. 
  - [custom_layers/mmr_regularizer.py](https://github.com/KurtPask/TropicalNN/blob/main/custom_layers/mmr_regularizer.py) : Holds the code required to compute the MMR regularizer penalty. Adapted and updated from [Croce's code](https://github.com/max-andr/provable-robustness-max-linear-regions/blob/master/regularizers.py). 
  - [updated_cleverhans/updated_carlini_wagner_l2.py](https://github.com/KurtPask/TropicalNN/blob/main/updated_cleverhans/updated_carlini_wagner_l2.py) : The Carlini and Wagner attack in the Cleverhans repo had some bugs I made adjustments in a new file to ensure transparency that I didn't meaningfully alter the Cleverhans version.
  - [updated_cleverhans/updated_spsa.py](https://github.com/KurtPask/TropicalNN/blob/main/updated_cleverhans/updated_spsa.py) : The SPSA attack in the Cleverhans repo had some bugs I made adjustments in a new file to ensure transparency that I didn't meaningfully alter the Cleverhans version.
  - [functions/models.py](https://github.com/KurtPask/TropicalNN/blob/main/functions/models.py) : Holds the structured models we use for the experiment. We built a custom class that enables us to add "top layers" easily to whatever base model's we include, making the code modular and easily expandable to new base models.
  - [functions/attacks.py](https://github.com/KurtPask/TropicalNN/blob/main/functions/attacks.py) : Holds the functions for the SLIDE algorithm and the l_2 PGD algorithm. SLIDE was not implemented in Cleverhans so we adapted from [Tramer's code](https://github.com/ftramer/MultiRobustness). PGD l_2 was not working right for me from Cleverhans, so I built a simple, custom version in this file to use. 
  - [functions/utils.py](https://github.com/KurtPask/TropicalNN/blob/main/functions/utils.py) : Most importantly it holds the attack config dictionary. Otherwise holds a lot of helper functions for executing the code, loading data, loading and finding files, etc.


## Required packages
All code was built and run in a python 3.11.2 environment. Below are dependencies for all files that are outside basic packages like numpy.

- pip install tensorflow
- pip install torch
- pip install cleverhans
- pip install tensorflow_datasets
- pip install easydict

## Experiment Overview

We trained and attacked many models to evaluate the performance of the tropical embedding layer used as the final, output layer for 6 kinds of neural network architectures across 4 different datasets.

### Training
Using a single GPU (NVIDIA V100) in an HPC environment, we built all 68 files using a batch script and the `train.py` and the `train_with_MMR_regularizer.py` files and managing the config dictionary within. We occassionally had to adjust optimization schemes between Adam and SGD to effectively train, but otherwise the same methodology was used throughout.

### Attacking
When models were built, we would use the two attack files `attack_pgd_algorithms.py` and `attack_cw_spsa_algorithms.py` to perturb the test set and evaluate the accuracy performance of the model on the perturbations. We distributed the attacks across many small batchs using sometimes several hundred CPU's within an HPC environment. At the end, the batches were combined for the final result using the `attack_batch_results_calcs.py` script.

## Final Note

This experiment was largely managed by Kurt Pasque while studying Operations Research. I am not a computer scientist, so I apologize for any weird formatting and bad practices. I think this methodology has potential and this is a small window into potential performance of tropical embedding layers within neural networks. 
