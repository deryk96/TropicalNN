"""
File name: train.py
Description:
    Code for building all models except MMR regularized models. Very crude dictionary that serves as config
    for which models to build in here. Might update for external config at some point.
"""

import math
import time
import sys
import numpy as np
import tensorflow as tf  # TODO: Delete once works
import os
from PathLib import Path

# Dependencies for PyTorch
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from absl import app
from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import load_models, load_data, find_model
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

def main(_):

    model_num = sys.argv[1]
    lr = float(sys.argv[2])
    optimizer_name = sys.argv[3]

    if model_num == "all":
        boo_run_all = True
    else:
        boo_run_all = False
        model_num = int(model_num)
        print(f"{model_num = }, {type(model_num) = }")

    # TODO: Delete once works
    '''
    # Old optimizer code  
    if optimizer_name == "adam":
        optimizer = tf.optimizers.Adam(learning_rate = lr)
    elif optimizer_name == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate = lr)#, momentum = 0.5)
    '''

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
                                        "trop" :    {"yes" : 0, "no" : 1}}},
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
                                        "trop" :    {"yes" : 0, "no" : 0}}},
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
    model_counter = -1  # Unsmart way to start a counter  # TODO: Fix whatever this is
    batch_size = 128  # Training batch size that is.
    eps_iter_portion = 0.2  # Scale of epsilon iterations for attack steps if adversarially training
    att_steps = 10  # Number of PGD/SLIDE attack steps if adversarially training
    early_stopping_patience = 5   # Number of epochs to wait for improvement
    min_delta = 0.001   # Minimum change to qualify as an improvement
    min_epochs = 10  # Min epochs
    max_epochs = 300  # Max epochs

    # Set device to be used (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate through models
    for name, model in models.items():
        # Determine if we are running a given model (for use in batch runs)
        print(f"Model name: {name}")
        model_counter += 1
        if boo_run_all == False and model_num != model_counter:
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
            _, eps, input_elements, data, info, _, _ = load_data(dataset_name, batch_size)

            # Extract and load training/validation sets
            train_size = int(info["train_size"] * 0.9)
            val_size = info["train_size"] - train_size
            data_train, data_val = random_split(data["train"], [train_size, val_size])
            train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

        # Save old dataset name
        old_dataset_name = dataset_name

        # TODO: Delete once works
        '''
        # Placed inside if statement, no need to do every time  
        total_size = info.splits['train'].num_examples
        val_size = int((total_size * 0.1) // batch_size) # 10% for validation
        data_train = data.train.skip(val_size)  # Skip the first X% for training
        data_val = data.train.take(val_size)  # Take the first X% for validation
        '''

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
        best_val_accuracy = 0 
        patience_counter = 0  # Counts epochs without improvement
        lr_reduced_counter = 0
        boo_adv_train = adv_train != "no"
        boo_update_weights = top_layer in {"maxout", "trop"} and not boo_adv_train

        # TODO: Delete once works
        '''
        # Not an efficient way to do this,  
        if (top_layer == "maxout" and adv_train == "no") or (top_layer == "trop" and adv_train == "no"):
            boo_update_weights = True
        if  (adv_train == "no"):
            boo_adv_train = False
        else:
            boo_adv_train = True

        # --- initiate tensorflow objects ---
        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)#, reduction=tf.keras.losses.Reduction.NONE)
        train_loss = tf.metrics.Mean(name="train_loss")
        train_acc = tf.metrics.SparseCategoricalAccuracy()
        validation_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        '''

        # Training loop
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            epoch_counter = epoch + 1

            # TODO: Make progress bar?

            # Print progress
            print(f"\nEpoch {epoch}, adv_train: {adv_train}, "
                  f"boo_adv_train: {boo_adv_train}")

            start = time.time()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # Perturb data if we are doing adversarial training
                if boo_adv_train:
                    y_pre_att = model(x).argmax(1)
                    x_l1 = l1_projected_gradient_descent(model,
                                                         x,
                                                         y_pre_att,
                                                         steps=att_steps,
                                                         epsilon=eps_l1,
                                                         eps_iter=eps_iter_portion * eps_l1,
                                                         loss_object=loss_function,
                                                         x_min=-1.0,
                                                         x_max=1.0,
                                                         perc=99)
                    x_l2 = l2_projected_gradient_descent(model,
                                                         x,
                                                         y_pre_att,
                                                         steps=att_steps,
                                                         epsilon=eps_l2,
                                                         eps_iter=eps_iter_portion * eps_l2,
                                                         loss_object=loss_function,
                                                         x_min=-1.0,
                                                         x_max=1.0,
                                                         perc=99)
                    x_linf = projected_gradient_descent(model_fn = model,
                                                        x = x,
                                                        eps = eps,
                                                        eps_iter = eps_iter_portion * eps,
                                                        nb_iter = att_steps,
                                                        norm = np.inf,
                                                        loss_fn = None,
                                                        clip_min = -1.0,
                                                        clip_max = 1.0,
                                                        y = y_pre_att,
                                                        targeted = False,
                                                        rand_init = True,
                                                        rand_minmax = eps,
                                                        sanity_checks=False)
                    x = torch.cat([x_l1, x_l2, x_linf], dim=0)
                    y = torch.cat([y, y, y], dim=0)

                # Get predictions and losses from model
                optimizer.zero_grad()  # Set gradients to zero before backpropagation
                predictions = model(x)
                loss = loss_function(predictions, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                correct += (predictions.argmax(1) == y).sum().item()
                total += y.size(0)

                # Initialize weights as the ReLU model's weights
                if boo_update_weights:
                    starting_model_path = find_model(dataset_name, base_model, "relu")
                    relu_model = load_models({base_model: {"activation": "relu"}})[base_model]
                    relu_model.load_state_dict(torch.load(starting_model_path, map_location=device))
                    relu_model.to(device)

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
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    predictions = model(x_val)
                    correct += (predictions.argmax(1) == y_val).sum().item()
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
                optiizer = optimizer_class(model.parameters(), lr=lr)
                lr_reduced_counter += 1
                patience_counter = 0
                if lr_reduced_counter > 3:
                    break
                print(f"\t**** Updating learning rate from {lr*10} to {lr} ****")

        # Print training metrics
        elapsed = time.time() - start
        print(f'\nTraining time per epoch = {elapsed/epoch_counter} seconds | {elapsed/60/epoch_counter} minutes')
        print(f'Training time total = {elapsed} seconds | {elapsed/60} minutes')
        model.summary()

        # Save model
        current_time = time.localtime()
        formatted_date = time.strftime("%d%b%y", current_time)
        os.makedirs('new_master_models', exist_ok=True)
        file_path = Path(f'new_master_models/{name}_{formatted_date}_model.pth')
        torch.save(model.state_dict(), file_path)
        print(f'Model saved: {file_path}')



        # TODO: Delete once works
    '''
        # --- define training step ---
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = loss_object(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_acc(y, predictions)

        start = time.time()
        for epoch in range(max_epochs):
            # --- initialize some tracking variables ---
            validation_acc.reset_state()
            progress_bar_train = tf.keras.utils.Progbar(info.splits['train'].num_examples - val_size*batch_size)
            epoch_counter = epoch + 1
            simple_counter = 0
            progress_count = 1

            print(f"---- epoch {epoch}, adv_train: {adv_train}, boo_adv_train: {boo_adv_train} ----")

            for (x, y) in data_train:
                # --- perturb data if we are doing adversarial training ---
                if boo_adv_train:
                    y_pre_att = tf.argmax(model(x, training=False), 1)
                    x_l1 = l1_projected_gradient_descent(model,
                                                    x,
                                                    y_pre_att,
                                                    steps = att_steps,
                                                    epsilon = eps_l1,
                                                    eps_iter = eps_iter_portion * eps_l1,
                                                    loss_object = loss_object,
                                                    x_min = -1.0,
                                                    x_max = 1.0,
                                                    perc = 99)
                    x_l2 = l2_projected_gradient_descent(model,
                                                    x,
                                                    y_pre_att,
                                                    steps=att_steps,
                                                    epsilon= eps_l2,
                                                    eps_iter = eps_iter_portion * eps_l2,
                                                    loss_object = loss_object,
                                                    x_min = -1.0,
                                                    x_max = 1.0)
                    x_linf = projected_gradient_descent(model_fn = model,
                                                    x = x,
                                                    eps = eps,
                                                    eps_iter = eps_iter_portion * eps,
                                                    nb_iter = att_steps,
                                                    norm = np.inf,
                                                    loss_fn = None,
                                                    clip_min = -1.0,
                                                    clip_max = 1.0,
                                                    y = y_pre_att,
                                                    targeted = False,
                                                    rand_init = True,
                                                    rand_minmax = eps,
                                                    sanity_checks=False)
                    x = tf.concat([x_l1, x_l2, x_linf], axis=0)
                    y = tf.concat([y, y, y], axis=0)

                # --- training step my friends ---
                train_step(x, y)

                # --- initialize weights as the relu model's (not elegant I know...) ---
                if boo_update_weights:
                    starting_model_path = find_model(dataset_name, base_model, "relu")
                    trained_model = tf.keras.models.load_model(starting_model_path)
                    for layer, new_layer in zip(trained_model.base_layers.layers, model.base_layers.layers):
                        new_layer.set_weights(layer.get_weights())
                    boo_update_weights = False

                # --- update progress bar ---
                simple_counter += 1
                if simple_counter == progress_count:
                    progress_bar_train.add(batch_size*progress_count, values=[("loss", train_loss.result()), ("acc", train_acc.result())])
                    simple_counter = 0

            # --- check validation set for improvement ---
            for (x_val, y_val) in data_val:
                predictions = model(x_val, training=False)
                validation_acc.update_state(y_val, predictions)
            val_accuracy = validation_acc.result().numpy()
            if val_accuracy > best_val_accuracy + min_delta:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            print(f'---- epoch {epoch}, Validation Accuracy {val_accuracy}, Best: {best_val_accuracy} ----') #Validation Loss: {val_loss}, Best: {best_val_loss},

            # --- kill training if conditions are met
            if patience_counter >= early_stopping_patience and epoch >= min_epochs - 1:
                patience_counter = 0
                current_lr = optimizer.learning_rate.numpy()
                optimizer.learning_rate.assign(current_lr/10)
                lr_reduced_counter += 1
                if lr_reduced_counter > 3:
                    break
                print(f"**** Updating learning rate from {current_lr} to {current_lr/10} ****")

        # --- printout training metrics ---
        elapsed = time.time() - start
        print(f'##### training time per epoch = {elapsed/epoch_counter} seconds | {elapsed/60/epoch_counter} minutes')
        print(f'##### training time total = {elapsed} seconds | {elapsed/60} minutes')
        model.summary()

        # --- save model ---
        current_time = time.localtime()
        formatted_date = time.strftime("%d%b%y", current_time)
        if not os.path.exists('new_master_models'):  # Check if directory doesn't exist
            os.makedirs('new_master_models')
        model.save(f'new_master_models/{name}_{formatted_date}.keras')#, save_format='tf')
        print(f'Model saved here: master_models/{name}_{formatted_date}.keras')
    '''

if __name__ == "__main__":
    print(f"########## Number of GPUs Available: {torch.cuda.device_count()}")
    app.run(main)