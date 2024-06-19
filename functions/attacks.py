# attacks.py
# Description: This file contains a collection of functions for attacking input data given a neural network model.
# Author: Kurt Pasque
# Date: October 25, 2023
# Last Update June 19, 2024

'''
Module: attacks.py

This module provides a collection of functions for attacking input data given a neural network model.

Functions:
- uniform_weights : Helper function for l1 PGD attack
- l1_projected_gradient_descent : 
- l2_projected_gradient_descent : 
'''

# - imports - 
from tensorflow import GradientTape, random, math, identity, sign, clip_by_value, clip_by_norm, convert_to_tensor, float32, shape, cast
import numpy as np

def uniform_weights(n_attacks, n_samples):
    x = np.random.uniform(size=(n_attacks, n_samples))
    y = np.maximum(-np.log(x), 1e-8)
    return y / np.sum(y, axis=0, keepdims=True)

def l1_projected_gradient_descent(model, x, y, steps, epsilon, loss_object, x_min, x_max, perc):
    '''
    Sparse L1 Descent Attack (SLIDE) from Tramer and Boneh (2019): https://arxiv.org/abs/1904.13000. 
    GitHub repo from research here: https://github.com/ftramer/MultiRobustness.
    '''
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
    '''
    L2 constrained projected gradient descent adapted from FGSM tutorial here: https://www.tensorflow.org/tutorials/generative/adversarial_fgsm.
    Native Cleverhans L2 PGD appears broken and not behaving as expected, causing need to write this algorithm.
    '''
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
