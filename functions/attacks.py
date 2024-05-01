# attacks.py
# Description: This file contains a collection of functions for attacking input data given a neural network model.
# Author: Kurt Pasque
# Date: October 25, 2023

'''
Module: attacks.py

This module provides a collection of functions for attacking input data given a neural network model.

Functions:
- fgsm_attack : Fast-gradient-sign-method attack on 1 input vector given label, loss object, and model
- pgd_attack : Projected-gradient-descent attack on 1 input vector given label, loss object, and model.
- attackTestSet : Attack a whole set of data 1-by-1 given loss object, model, data, and type of attack.
- pgd_attack_batch : Projected-gradient-descent attack on batch of input vectors given label, loss object, and model.
- attackTestSetBatch : Attack a whole set of data batch-by-batch given loss object, model, data, and type of attack.
'''

# - imports - 
from tensorflow import GradientTape, random, math, identity, sign, clip_by_value, clip_by_norm, convert_to_tensor, float32, shape, cast
import numpy as np
import time

def uniform_weights(n_attacks, n_samples):
    x = np.random.uniform(size=(n_attacks, n_samples))
    y = np.maximum(-np.log(x), 1e-8)
    return y / np.sum(y, axis=0, keepdims=True)

def l1_projected_gradient_descent(model, x, y, steps, epsilon, eps_iter, loss_object, x_min, x_max, perc):
    norm_weights = uniform_weights(1, len(x))
    eps_w = epsilon * norm_weights[0]
    eps = eps_w.reshape(len(x), 1, 1, 1)
    r = np.random.laplace(size=x.shape)
    norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1, ord=1).reshape(-1, 1, 1, 1)
    old_delta = (r / norm) * eps
    x_adv = np.clip(x + old_delta, x_min, x_max)
    for _ in range(steps):
        x_adv = convert_to_tensor(x_adv)
        with GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv, training=False)
            loss = loss_object(y, logits)
        gradients = tape.gradient(loss, x_adv) # Compute gradients
        _, h, w, ch = gradients.shape
        a = x_max # gamma
        perc = perc # percentile
        bad_pos = ((x_adv == x_max) & (gradients > 0)) | ((x_adv == x_min) & (gradients < 0)) # boolean mask for points at edge of allowable max/min with gradient pointing out of allowable edge
        negated_bad_pos = math.logical_not(bad_pos)
        negated_bad_pos = cast(negated_bad_pos, float32)
        gradients = gradients * negated_bad_pos
        abs_grad = np.abs(gradients) 
        sign = np.sign(gradients)

        if isinstance(perc, list):
            perc_low, perc_high = perc
            perc = np.random.uniform(low=perc_low, high=perc_high)

        max_abs_grad = np.percentile(abs_grad, perc, axis=(1, 2, 3), keepdims=True)
        tied_for_max = (abs_grad >= max_abs_grad).astype(np.float32)
        num_ties = np.sum(tied_for_max, (1, 2, 3), keepdims=True)
        optimal_perturbation = sign * tied_for_max / num_ties
        new_delta = old_delta + a * optimal_perturbation
        l1 = np.sum(np.abs(new_delta), axis=(1, 2, 3))
        to_project = l1 > eps_w
        if np.any(to_project):
            n = np.sum(to_project)
            d = new_delta[to_project].reshape(n, -1)  # n * N (N=h*w*ch)
            abs_d = np.abs(d)  # n * N
            mu = -np.sort(-abs_d, axis=-1)  # n * N
            cumsums = mu.cumsum(axis=-1)  # n * N
            eps_d = eps_w[to_project]
            js = 1.0 / np.arange(1, h * w * ch + 1)
            temp = mu - js * (cumsums - np.expand_dims(eps_d, -1))
            rho = np.argmin(temp > 0, axis=-1)
            theta = 1.0 / (1 + rho) * (cumsums[range(n), rho] - eps_d)
            sgn = np.sign(d)
            d = sgn * np.maximum(abs_d - np.expand_dims(theta, -1), 0)
            new_delta[to_project] = d.reshape(-1, h, w, ch)

        new_delta = np.clip(new_delta, x_min - (x_adv - old_delta), x_max - (x_adv - old_delta))
        old_delta = new_delta
        x_adv = np.clip(x + new_delta, x_min, x_max)
    return x_adv

def l2_projected_gradient_descent(model, x, y, steps, epsilon, eps_iter, loss_object, x_min, x_max):
    # Initialize adversarial examples with the original input
    x_adv = identity(x)
    
    # Create a random perturbation
    random_perturbation = random.uniform(shape(x), minval=-epsilon, maxval=epsilon)
    
    # Initialize adversarial examples with the original input plus random perturbation
    x_adv = x_adv + random_perturbation
    x_adv = clip_by_value(x_adv, x_min, x_max) 


    for _ in range(steps):
        with GradientTape() as tape:
            tape.watch(x_adv)
            logits = model(x_adv, training=False)
            loss = loss_object(y, logits)

        # Compute gradients
        gradients = tape.gradient(loss, x_adv)

        # Apply gradient ascent to maximize the loss
        x_adv = x_adv + eps_iter * sign(gradients)

        # Project back into the epsilon ball and clip to valid image range
        x_adv = x + clip_by_norm(x_adv - x, epsilon, axes=[1, 2, 3])
        x_adv = clip_by_value(x_adv, x_min, x_max)  

    return x_adv


def fgsm_attack(model, input_data, target_label, loss_object, epsilson = 8/255):
    '''
    This function implements the Fast Gradient Signed Method to perturbate an input image based on a given model and true data label. 
    The method employs (almost exactly) the implementation of FGSM articulated in the Tensorflow docs here: 
    
    https://www.tensorflow.org/tutorials/generative/adversarial_fgsm 

    This method adjusts the image by taking 1 simple (using just sign of gradient, ignoring magnitude) step in the direction that will maximize loss.

    Parameters
    ----------
    model : tensorflow model object
        trained tensorflow model
    input_image : tensorflow tensor object
        tensor of input data to attack
    target_label : tensorflow tensor object
        tensor of the target label for the given input_image
    loss_object : tensorflow loss object
        loss object from tensorflow such as binary or categorical cross entropy 
    epsilson : float
        our "adversarial budget", i.e. how far we can deviate from the original data

    Returns
    -------
    perturbed_input : tensorflow tensor object
        A perturbated version of the input image
    '''
    # - make copy - 
    id_input = identity(input_data) # Return a Tensor with the same shape and contents as input.
    
    # - calculate gradient image given model - 
    with GradientTape() as tape: # Record operations for automatic differentiation.
        tape.watch(input_data) # Ensures that tensor is being traced by this tape.
        prediction = model(input_data, training=False) # predicts class of input data using model
        loss = loss_object(target_label, prediction) # calculates loss based on loss object, true label, and predicted label
    gradient = tape.gradient(loss, input_data) # Computes the gradient using operations recorded in context of this tape.

    # - take signs from gradient and compute 1 epsilon step - 
    signed_grad = sign(gradient) # take sign of gradients: will be either a -1, 0, or 1
    perturbed_input = id_input + epsilson*signed_grad # add an epsilon-step using signed gradient and add to image

    return perturbed_input


def pgd_attack(model, input_image, target_label, loss_object, epsilon = 8/255, num_steps = 5, step_size = 0.01):
    '''
    This function implements the Projected Gradient Descent Method to perturbate an input image based on a given model and true data label. 
    The method takes multiple (num_steps) of FGSM, however it differes in that each step has a specified step size (step_size). After each step,
    the perturbation is projected back inside the epsilon ball and/or projected back inside the given range of the pixel data (in our case [-0.5, 0.5]).

    Parameters
    ----------
    model : tensorflow model object
        trained tensorflow model
    input_image : tensorflow tensor object
        tensor of input data to attack
    target_label : tensorflow tensor object
        tensor of the target label for the given input_image
    loss_object : tensorflow loss object
        loss object from tensorflow such as binary or categorical cross entropy 
    epsilson : float
        our "adversarial budget", i.e. how far we can deviate from the original data
    num_steps : int
        the number of gradient steps we take to maximize the image's loss relative to the input model
    step_size : float
        the step size taken at each step

    Returns
    -------
    perturbed_image : tensorflow tensor object
        A perturbated version of the input image
    '''
    perturbed_image = identity(input_image)  # Create a copy of the input image
    for _ in range(num_steps): # loop num_steps times
        with GradientTape() as tape: # Record operations for automatic differentiation.
            tape.watch(perturbed_image) # Ensures that tensor is being traced by this tape.
            prediction = model(perturbed_image, training=False) # predicts class of input data using model
            loss = loss_object(target_label, prediction) # calculates loss based on loss object, true label, and predicted label
        gradient = tape.gradient(loss, perturbed_image) # Computes the gradient using operations recorded in context of this tape.
        signed_grad = sign(gradient) # take sign of gradients: will be either a -1, 0, or 1
        perturbed_image = perturbed_image + step_size * signed_grad # add a step_size step using signed gradient and add to image
        perturbed_image = clip_by_value(perturbed_image, input_image - epsilon, input_image + epsilon) # Ensure no RGB value is more than epsilon different from original value
        perturbed_image = clip_by_value(perturbed_image, -0.5, 0.5)  # Ensure pixel values are in [-0.5, 0.5] range
    return perturbed_image


def attackTestSet(model, x_test, y_test, loss_object, modelName, attack='pgd', epsilon = 8/255, num_steps = 5, step_size = 0.01):
    '''
    The method attacks a set of data 1-by-1 so that the loss is mazimized relative to a given model using the other attack hyperparameters above.

    Parameters
    ----------
    model : tensorflow model object
        trained tensorflow model
    x_test : numpy array
        set of numpy arrays of input data to attack
    y_test : numpy array
        set of numpy arrays of the target label for the given input_images
    loss_object : tensorflow loss object
        loss object from tensorflow such as binary or categorical cross entropy 
    modelName : str
        just used for clear printouts as attack progresses
    attack : string
        user can specify if wanting to do a pgd or fgsm attack
    epsilson : float
        our "adversarial budget", i.e. how far we can deviate from the original data
    num_steps : int
        the number of gradient steps we take to maximize the image's loss relative to the input model
    step_size : float
        the step size taken at each step

    Returns
    -------
    x_test_attacked : tensorflow tensor object
        A perturbated version of the input image
    '''
    start_time = time.time()    
    x_test_attacked = np.copy(x_test)
    numberToAttack = len(x_test_attacked)
    num_predictors = x_test.shape[1]
    y_dim = y_test[0].shape[0]

    for i in range(len(x_test_attacked)):
        xtensor = convert_to_tensor(x_test_attacked[i].reshape(1,num_predictors), dtype=float32)
        ytensor = convert_to_tensor(y_test[i].reshape(1,y_dim), dtype=float32)
        if attack == 'pgd':
            x_test_attacked[i] = pgd_attack(model, xtensor, ytensor, loss_object, epsilon, num_steps, step_size)
        elif attack == 'fgsm':
            x_test_attacked[i] = fgsm_attack(model, xtensor, ytensor, loss_object, epsilon)
        
        if i % round(numberToAttack/10) == 0:
            print(f'{modelName} model {round(i/numberToAttack*100)}% attacked.')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{modelName} attack complete. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time/60:.2f} minutes.')
    return x_test_attacked


def pgd_attack_batch(model, 
                     input_images, 
                     target_labels, 
                     loss_object, 
                     epsilon=8/255, 
                     num_steps=5, 
                     step_size=0.01, 
                     already_attacked_input_images = None,
                     random_start = True):
    '''
    This function implements the Projected Gradient Descent Method in batches to improve computation time. The method perturbates input images based 
    on a given model and true data label. The method takes multiple (num_steps) of FGSM, however it differes in that each step has a specified step size 
    (step_size). After each step, the perturbation is projected back inside the epsilon ball and/or projected back inside the given range of the pixel 
    data (in our case [-0.5, 0.5]).

    Parameters
    ----------
    model : tensorflow model object
        trained tensorflow model
    input_images : numpy array
        set of numpy arrays of input data to attack
    target_label : numpy array
        set of numpy arrays of the target label for the given input_images
    loss_object : tensorflow loss object
        loss object from tensorflow such as binary or categorical cross entropy 
    epsilson : float
        our "adversarial budget", i.e. how far we can deviate from the original data
    num_steps : int
        the number of gradient steps we take to maximize the image's loss relative to the input model
    step_size : float
        the step size taken at each step

    Returns
    -------
    perturbed_images : tensorflow tensor object
        A perturbated version of the input image
    '''
    if already_attacked_input_images is not None:
        perturbed_images = identity(already_attacked_input_images)  # Create a copy of the input images
    else:
        perturbed_images = identity(input_images)  # Create a copy of the input images

    if random_start:
        # Generating random perturbation
        random_perturbation = random.uniform(shape=input_images.shape, minval=-epsilon, maxval=epsilon)
        perturbed_images = perturbed_images + random_perturbation  # Add random perturbation to input images
    
    for _ in range(num_steps): # loop num_steps times
        with GradientTape() as tape: # Record operations for automatic differentiation.
            tape.watch(perturbed_images) # Ensures that tensor is being traced by this tape.
            predictions = model(perturbed_images, training=False) # predicts classes of input data using model
            loss = loss_object(convert_to_tensor(target_labels), predictions) # calculates losses based on loss object, true label, and predicted label
        gradients = tape.gradient(loss, perturbed_images) # Computes the gradient using operations recorded in context of this tape.
        signed_gradients = sign(gradients) # take sign of gradients: will be either a -1, 0, or 1
        perturbed_images = perturbed_images + step_size * signed_gradients # add a step_size step using signed gradient and add to images
        perturbed_images = clip_by_value(perturbed_images, input_images - epsilon, input_images + epsilon) # Clip/project pixel values so they are no more than epsilon different from original value
        perturbed_images = clip_by_value(perturbed_images, -0.5, 0.5) # Clip/project pixel values are in [-0.5, 0.5] range
    return perturbed_images


def attackTestSetBatch(model, x_test, y_test, loss_object, modelName, epsilon=8/255, num_steps=5, step_size=0.01, batch_size=32, already_attacked_input_images=None):
    '''
    Attacks a set of data in batches to maximize the loss relative to a given model using the PGD attack method.

    Parameters
    ----------
    model : tensorflow model object
        Trained TensorFlow model.
    x_test : numpy array
        Set of numpy arrays of input data to attack.
    y_test : numpy array
        Set of numpy arrays of the target label for the given input images.
    loss_object : tensorflow loss object
        Loss object from TensorFlow such as binary or categorical cross-entropy.
    modelName : str
        Used for clear printouts as the attack progresses.
    epsilon : float, optional
        Our "adversarial budget", i.e. how far we can deviate from the original data.
    num_steps : int, optional
        The number of gradient steps taken to maximize the image's loss relative to the input model.
    step_size : float, optional
        The step size taken at each step.
    batch_size : int, optional
        Size of each batch for the attack.
    already_attacked_input_images : numpy array, optional
        Input images that have already been attacked.

    Returns
    -------
    x_test_attacked : numpy array
        A perturbated version of the input images.
    '''
    start_time = time.time()  # Record the starting time of the attack process.

    if already_attacked_input_images is not None:  # Check if there are previously attacked images available.
        x_test_attacked = np.copy(already_attacked_input_images)  # Create a copy of previously attacked images.
        random_start = False
    else:
        x_test_attacked = np.copy(x_test)  # Otherwise, create a copy of the original input images.
        random_start = True

    numberToAttack = len(x_test_attacked)  # Get the total number of images to attack.
    num_print_update = round(numberToAttack / 25)  # Calculate the number of images for each progress update.

    for i in range(0, len(x_test_attacked), batch_size):  # Loop through the images in batches.
        if already_attacked_input_images is not None:
            # Extract the corresponding batches of already attacked and original input images.
            batch_x_already_attacked = x_test_attacked[i:i + batch_size]
            batch_x = x_test[i:i + batch_size]
        else:
            batch_x_already_attacked = None
            batch_x = x_test_attacked[i:i + batch_size]

        batch_y = y_test[i:i + batch_size]  # Extract the corresponding batch of target labels.
        # Perform PGD attack on the current batch of images.
        batch_x_attack = pgd_attack_batch(
            model,
            batch_x,
            batch_y,
            loss_object,
            epsilon,
            num_steps,
            step_size,
            already_attacked_input_images=batch_x_already_attacked,
            random_start = random_start
        )

        x_test_attacked[i:i + batch_size] = batch_x_attack  # Update the attacked images with the perturbed batch.

        if i % num_print_update == 0:  # Print progress updates based on the number of images processed.
            print(f'{modelName} model {round(i / numberToAttack * 100)}% attacked.')

    end_time = time.time()  # Record the ending time of the attack process.
    elapsed_time = end_time - start_time  # Calculate the elapsed time for the attack.
    print(f'{modelName} attack complete. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time / 60:.2f} minutes.')

    return x_test_attacked  # Return the perturbed images.
