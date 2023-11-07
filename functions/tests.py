from functions.load_data import shuffle_data
from functions.attacks import attackTestSetBatch
from tensorflow.keras import losses
import numpy as np
import csv
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # Modify this path to your Graphviz installation directory

def evaluate_attack(model, x_test, y_test, x_test_att):
    # - before attack -
    pre_trop_preds = np.argmax(model.predict(x_test), axis=1)
    y_tester = np.argmax(y_test, axis=1)
    pre_correct_index = np.equal(pre_trop_preds, y_tester)
    pre_loss, pre_acc = model.evaluate(x_test, y_test)

    # - after attack - 
    post_trop_preds = np.argmax(model.predict(x_test_att), axis=1)
    post_loss, post_acc = model.evaluate(x_test_att, y_test)

    # - number maintaining prediction - 
    maintained_correct_pred = np.average(np.equal(post_trop_preds[pre_correct_index], y_tester[pre_correct_index]))
    return pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred


def test_pdg_steps(file_path, x_test, y_test, 
                   min_steps = 1, max_steps = 41, step_size = 2, eps=8/255, 
                   trop_model=None, relu_model=None,
                   loss_object=losses.CategoricalCrossentropy()):
    
    if not Path(file_path).is_file():
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['pre_loss', 'pre_acc', 'post_loss', 'post_acc', 'maintained_correct_pred', 'model_type', 'attack'])

    if trop_model:
        for i in range(min_steps, max_steps+1, step_size):
            print(f'\n\n{i}\n\n')
            if i == min_steps:
                trop_x_test_att = attackTestSetBatch(trop_model, x_test, y_test,  epsilon=eps, loss_object=loss_object,modelName='Tropical', num_steps=i)
            else:
                trop_x_test_att = attackTestSetBatch(trop_model, x_test, y_test,  epsilon=eps, loss_object=loss_object,modelName='Tropical', num_steps=step_size, already_attacked_input_images=trop_x_test_att)
            pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred = evaluate_attack(trop_model, x_test, y_test, trop_x_test_att)
            data = [pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred, 'trop', i]
            with open(file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(data)

    if relu_model: 
        for i in range(min_steps, max_steps+1, step_size):
            print(f'\n\n{i}\n\n')
            if i == min_steps:
                relu_x_test_att = attackTestSetBatch(relu_model, x_test, y_test,  epsilon=eps, loss_object=loss_object, modelName='ReLU', num_steps=i)
            else:
                relu_x_test_att = attackTestSetBatch(relu_model, x_test, y_test,  epsilon=eps, loss_object=loss_object,modelName='ReLU', num_steps=step_size, already_attacked_input_images=relu_x_test_att)
            pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred = evaluate_attack(relu_model, x_test, y_test, relu_x_test_att)
            data = [pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred, 'relu', i]
            with open(file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(data)


def test_train_attack_repeat(file_path, x_train, y_train, x_test, y_test, 
                   num_iterations = 1, eps=8/255, 
                   shuffle=True, func_train_trop_model=None, func_train_relu_model=None,
                   loss_object=losses.CategoricalCrossentropy(),
                   trop_save_name = None,relu_save_name = None,
                   func_args = {}):
    if not Path(file_path).is_file():
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['pre_loss', 'pre_acc', 'post_loss', 'post_acc', 'maintained_correct_pred', 'model_type'])

    for i in range(num_iterations):
        print(f'\n\n{i}\n\n')
        # -- Shuffle Data --
        if shuffle:
            x_train, x_test, y_train, y_test = shuffle_data(x_train, x_test, y_train, y_test)

        if func_train_trop_model:
            # -- TROPICAL: attack model -- 
            trop_model = func_train_trop_model(x_train, y_train, func_args=func_args)
            trop_x_test_att = attackTestSetBatch(trop_model, x_test, y_test,  epsilon=eps, loss_object=loss_object,modelName='Tropical', num_steps=i)
            pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred = evaluate_attack(trop_model, x_test, y_test, trop_x_test_att)
            data = [pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred, 'trop', i]
            with open(file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(data)

        if func_train_relu_model:
            # -- RELU: attack model --
            relu_model = func_train_relu_model(x_train, y_train, func_args=func_args)
            relu_x_test_att = attackTestSetBatch(relu_model, x_test, y_test,  epsilon=eps, loss_object=loss_object, modelName='ReLU', num_steps=i)
            pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred = evaluate_attack(relu_model, x_test, y_test, relu_x_test_att)
            data = [pre_loss, pre_acc, post_loss, post_acc, maintained_correct_pred, 'relu', i]
            with open(file_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(data)
            
        if trop_save_name:
            np.save(f'saved_models/{trop_save_name}_{i}.npy', trop_x_test_att)
        if relu_save_name:
            np.save(f'saved_models/{relu_save_name}_{i}.npy', relu_x_test_att)

