"""
File name: train.py
Description:
    Code for building all models except MMR regularized models. Very crude dictionary that serves as config
    for which models to build in here. Might update for external config at some point.

    Use: python train.py [model_num] [learning_rate] [optimizer_name]
"""

import math
import time
import sys
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import tqdm  # Used for progress bar

# Dependencies for PyTorch
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from absl import app
from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import load_models, load_data, find_model, model_choice
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

# Packages needed for attacks using Foolbox
from foolbox import PyTorchModel, accuracy
from foolbox.attacks import (L1ProjectedGradientDescentAttack,
                             L2ProjectedGradientDescentAttack,
                             LinfProjectedGradientDescentAttack)


# TODO: Modularize some of this code to make it more readable
def main(_):
    # Set seed for reproducibility
    torch.manual_seed(0)

    model_num = sys.argv[1]
    lr = float(sys.argv[2])
    optimizer_name = sys.argv[3]

    if model_num == "all":
        boo_run_all = True
    else:
        boo_run_all = False
        model_num = int(model_num)
        print(f"{model_num = }, {type(model_num) = }")

    # Set optimizer based on user input
    optimizer_class = {
        "adam": optim.Adam,
        "sgd": optim.SGD
    }.get(optimizer_name, None)

    # Ensure optimizer is properly set
    if not optimizer_class:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Make dictionary with model settings
    dict_settings = {
        "mnist" : {"LeNet5" :           {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}},
                    "ModifiedLeNet5" :  {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}}},
        "svhn" :  {"LeNet5" :           {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}},
                    "ModifiedLeNet5" :  {"maxout" : {"yes" : 0, "no" : 0}, 
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}},
                    "MobileNet" :       {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}}},
        "cifar10" : {"ResNet50" :       {"trop":    {"yes" : 0, "no" : 0}, 
                                        "relu" :    {"yes" : 0, "no" : 0}, 
                                        "maxout" :  {"yes" : 0, "no" : 0}},
                    "VGG16" :           {"maxout" : {"yes" : 0, "no" : 0}, 
                                        "relu" :    {"yes" : 0, "no" : 0}, 
                                        "trop" :    {"yes" : 0, "no" : 0}},
                    "EfficientNetB4" :  {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}},
                    "CifarCnnModel" :   {"relu" :   {"yes" : 0, "no" : 0},
                                         "trop" :   {"yes" : 0, "no" : 1}}},
        "cifar100" : {"ResNet50" :      {"maxout" : {"yes" : 0, "no" : 0}, 
                                        "relu" :    {"yes" : 0, "no" : 0}, 
                                        "trop" :    {"yes" : 0, "no" : 0}},
                    "VGG16" :           {"maxout" : {"yes" : 0, "no" : 0}, 
                                        "relu" :    {"yes" : 0, "no" : 0}, 
                                        "trop" :    {"yes" : 0, "no" : 0}},
                    "EfficientNetB4" :  {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}}},
    }

    # Initialize other parameters
    models = load_models(config=dict_settings)
    old_dataset_name = None
    model_counter = -1  # Un-smart way to start a counter  # TODO: Fix whatever this is
    batch_size = 128  # Training batch size
    eps_iter_portion = 0.2  # Scale of epsilon iterations for attack steps if adversarial training
    att_steps = 10  # Number of PGD/SLIDE attack steps if adversarial training
    early_stopping_patience = 5   # Number of epochs to wait for improvement
    min_delta = 0.001   # Minimum change to qualify as an improvement
    min_epochs = 10  # Min epochs
    # max_epochs = 300  # Max epochs  # TODO: Undo epoch change after debugging
    max_epochs = 100

    # Set device to be used
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available.")
    elif torch.backends.mps.is_available():  # MPS is for Macs with an M series GPU
        device = torch.device("mps")
        print("MPS available.")
    else:
        device = torch.device("cpu")
        print("No GPU or MPS available, using CPU.")

    # Iterate through models
    for name, model in models.items():
        # Determine if we are running a given model (for use in batch runs)
        print(f"Model name: {name}")
        model_counter += 1
        if boo_run_all == False and model_num != model_counter:
            print(f'Skipping model {model_counter}.')
            continue
            
        # Get key information from model name
        name_split = name.split("_")
        dataset_name = name_split[0]
        base_model = name_split[1]
        top_layer = name_split[2]
        adv_train = name_split[3]
        print(f"{name = }, {adv_train = }")
    
        # Load dataset
        if old_dataset_name == None or old_dataset_name != dataset_name:
            _, eps, input_elements, data, info, input_shape, _, num_channels = load_data(dataset_name, batch_size)

            # Extract and load training/validation sets
            tot_size = len(data["train"])
            val_size = int(tot_size * 0.1)  # 10% for validation
            train_size = tot_size - val_size
            generator = torch.Generator().manual_seed(0)  # Set seed for train/val split
            data_train, data_val = random_split(data["train"], [train_size, val_size], generator=generator)
            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

        # Save old dataset name
        old_dataset_name = dataset_name

        # Epsilon values
        eps_l2 = math.sqrt((eps**2)*input_elements)
        eps_l1 = 2 * eps_l2    

        # Send model to device chosen earlier
        model.to(device)

        # Initialize loss function
        loss_function = nn.CrossEntropyLoss()  # PyTorch version of SparseCategoricalCrossentropy
                                               # Expects raw inputs (logits, not probabilities)
                                               # See: https://stackoverflow.com/questions/72622202/why-is-the-tensorflow-and-pytorch-crossentropy-loss-returns-different-values-for/72622279

        # Initialize optimizer
        optimizer = optimizer_class(model.parameters(), lr=lr)

        # Initiate counters specific to model
        best_val_accuracy = -np.inf  # Initialize to negative infinity
        patience_counter = 0  # Counts epochs without improvement
        lr_reduced_counter = 0
        boo_adv_train = adv_train != "no"
        boo_update_weights = top_layer in {"maxout", "trop"} and not boo_adv_train

        # Training loop
        for epoch in range(max_epochs):
            model.train()  # Set model in training mode
            epoch_loss = 0
            correct = 0
            total = 0
            epoch_counter = epoch + 1

            # TODO: Make progress bar with TQDM

            # Print progress
            print(f"\nEpoch {epoch}, adv_train: {adv_train}, boo_adv_train: {boo_adv_train}, patience_counter: {patience_counter}")

            # Iterate through training data
            start = time.perf_counter()
            for i, (x, y) in enumerate(train_loader):
                # x is the input; y is the label
                x, y = x.to(device), y.to(device)

                # Perturb data if we are doing adversarial training
                if boo_adv_train:
                    # y_pre_att = model(x).argmax(1)

                    # Create Foolbox model
                    # Adjust bounds to match normalization
                    fmodel = PyTorchModel(model, bounds=(-1, 1))

                    # Check accuracy of model before attack
                    acc_pre_att = accuracy(model, x, y)

                    # List of attacks
                    attacks = [L1ProjectedGradientDescentAttack(),
                               L2ProjectedGradientDescentAttack(),
                               LinfProjectedGradientDescentAttack()]

                    # Conduct attacks and print results
                    attack_success = np.zeros((len(attacks), 1, len(x)), dtype=np.bool)
                    for i, attack in enumerate(attacks):
                        _, _, success = attack(fmodel, x, y, epsilon=eps_l1, steps=att_steps)
                        assert success.shape == (1, len(x))
                        success_ = success.numpy()
                        assert success_.dtype == np.bool
                        attack_success[i] = success_
                        print(f'{attack = }')
                        print(f'** Success: {1.0 - success_.mean(axis=-1):.3f}')

                    # Calculate robust accuracy (accuracy of model after attack)
                    # Uses best attack per sample
                    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
                    print('-'*50)
                    print(f'Worst case (best attack per sample): {rubust_accuracy:.3f}')

                    # Old: from Kurt's custom classes
                    # x_l1 = l1_projected_gradient_descent(model,
                    #                                      x,
                    #                                      y_pre_att,
                    #                                      steps=att_steps,
                    #                                      epsilon=eps_l1,
                    #                                      eps_iter=eps_iter_portion * eps_l1,
                    #                                      loss_object=loss_function,
                    #                                      x_min=-1.0,
                    #                                      x_max=1.0,
                    #                                      perc=99)
                    # x_l2 = l2_projected_gradient_descent(model,
                    #                                      x,
                    #                                      y_pre_att,
                    #                                      steps=att_steps,
                    #                                      epsilon=eps_l2,
                    #                                      eps_iter=eps_iter_portion * eps_l2,
                    #                                      loss_object=loss_function,
                    #                                      x_min=-1.0,
                    #                                      x_max=1.0,
                    #                                      perc=99)
                    # x_linf = projected_gradient_descent(model_fn = model,
                    #                                     x = x,
                    #                                     eps = eps,
                    #                                     eps_iter = eps_iter_portion * eps,
                    #                                     nb_iter = att_steps,
                    #                                     norm = np.inf,
                    #                                     loss_fn = None,
                    #                                     clip_min = -1.0,
                    #                                     clip_max = 1.0,
                    #                                     y = y_pre_att,
                    #                                     targeted = False,
                    #                                     rand_init = True,
                    #                                     rand_minmax = eps,
                    #                                     sanity_checks=False)
                    x = torch.cat([x_l1, x_l2, x_linf], dim=0)
                    y = torch.cat([y, y, y], dim=0)

                # Get predictions and losses from model
                predictions = model(x)
                loss = loss_function(predictions, y)
                optimizer.zero_grad()  # Set gradients to zero before backpropagation
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                correct += (predictions.argmax(1) == y).type(torch.float).sum().item()
                total += y.size(0)

                # Initialize weights as the ReLU model's weights
                if boo_update_weights:
                    starting_model_path = find_model(dataset_name, base_model, "relu")
                    # relu_model = load_models({base_model: {"activation": "relu"}})[base_model]
                    relu_model = model_choice(dataset_name, base_model, "relu", adv_train)
                    relu_model.to(device)
                    relu_model.load_state_dict(torch.load(starting_model_path, map_location=device, weights_only=True))

                    # Transfer weights layer by layer
                    for relu_layer, target_layer in zip(relu_model.children(), model.children()):
                        if hasattr(target_layer, "weight") and target_layer.weight is not None:
                            target_layer.weight.data = relu_layer.weight.data.clone()
                        if hasattr(target_layer, "bias") and target_layer.bias is not None:
                            target_layer.bias.data = relu_layer.bias.data.clone()

                    boo_update_weights = False

            # Calculate training accuracy
            train_accuracy = correct / total
            print(f"\tEpoch Loss: {epoch_loss:.4f}, "
                  f"Training Accuracy: {train_accuracy:.4f}")

            # Check validation set for improvement
            model.eval()  # Set model in eval mode
            correct = 0
            total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    predictions = model(x_val)
                    correct += (predictions.argmax(1) == y_val).type(torch.float).sum().item()
                    total += y_val.size(0)

            # Calculate validation accuracy
            val_accuracy = correct / total
            print(f"\tValidation Accuracy: {val_accuracy:.4f}")

            # Check if model has improved
            if val_accuracy > best_val_accuracy + min_delta:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            # Kill training if conditions are met
            if patience_counter >= early_stopping_patience and epoch >= min_epochs - 1:
                lr /= 10
                optimizer = optimizer_class(model.parameters(), lr=lr)
                lr_reduced_counter += 1
                patience_counter = 0
                if lr_reduced_counter > 3:
                    break
                print(f"\t**** Updating learning rate from {lr*10} to {lr} ****")

        # Print training metrics
        elapsed = timedelta(seconds=time.perf_counter() - start)
        print(f'\nTraining time per epoch (H:MM:SS.UUUUUU) = {elapsed/epoch_counter}.')
        print(f'Training time total (H:MM:SS.UUUUUU)     = {elapsed}')
        summary(model)

        # Save model
        formatted_date = time.localtime().strftime("%d%b%y", current_time)    # Get date
        os.makedirs('new_master_models', exist_ok=True)                 # Make folder (will not overwrite)
        file_path = Path(f'new_master_models/{name}_{formatted_date}.pth')    # Put together file path
        torch.save(model.state_dict(), file_path)                             # Save model state_dict
        print(f'Model saved: {file_path}')


if __name__ == "__main__":
    print(f"########## Number of GPUs Available: {torch.cuda.device_count()}")
    app.run(main)