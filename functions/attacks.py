from tensorflow import GradientTape, identity, sign, clip_by_value, convert_to_tensor, float32
import numpy as np
import time


def fgsm_attack(model, input_image, target_label, loss_object, epsilson = 8/255):
  image = identity(input_image)
  with GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(target_label, prediction)
  gradient = tape.gradient(loss, input_image)
  signed_grad = sign(gradient)
  new_image = image + epsilson*signed_grad
  return new_image


def pgd_attack(model, input_image, target_label, loss_object, epsilon = 8/255, num_steps = 5, step_size = 0.01):
    perturbed_image = identity(input_image)  # Create a copy of the input image
    for _ in range(num_steps):
        with GradientTape() as tape:
            tape.watch(perturbed_image)
            prediction = model(perturbed_image)
            loss = loss_object(target_label, prediction)
        gradient = tape.gradient(loss, perturbed_image)
        signed_grad = sign(gradient)
        perturbed_image = perturbed_image + step_size * signed_grad
        perturbed_image = clip_by_value(perturbed_image, input_image - epsilon, input_image + epsilon) # Ensure no RGB value is more than epsilon different from original value
        perturbed_image = clip_by_value(perturbed_image, 0.0, 1.0)  # Ensure pixel values are in [0, 1] range
    return perturbed_image


def attackTestSet(model, x_test, y_test, loss_object, modelName, attack='pgd', epsilon = 8/255, num_steps = 5, step_size = 0.01):
    start_time = time.time()    
    x_test_attacked = np.copy(x_test)
    numberToAttack = len(x_test_attacked)
    num_predictors = x_test.shape[1]

    for i in range(len(x_test_attacked)):
        xtensor = convert_to_tensor(x_test_attacked[i].reshape(1,num_predictors), dtype=float32)
        ytensor = convert_to_tensor(y_test[i].reshape(1,1), dtype=float32)
        if attack == 'pgd':
            x_test_attacked[i] = pgd_attack(model, xtensor, ytensor, loss_object, epsilon, num_steps, step_size)
        elif attack == 'fgsm':
            x_test_attacked[i] = fgsm_attack(model, xtensor, ytensor, loss_object, epsilon)
        
        if i % round(numberToAttack/10) == 0:
            print(f'{modelName} model {round(i/numberToAttack*100)}% attacked.')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'{modelName} attack complete. Elapsed time: {elapsed_time:.2f} seconds /// {elapsed_time/60:.2f} minutes.')
    return x_test_attacked