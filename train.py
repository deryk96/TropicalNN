import math
import time
import sys
import numpy as np
import tensorflow as tf
import os

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
        print(model_num, type(model_num))

    if optimizer_name == "adam":
        optimizer = tf.optimizers.Adam(learning_rate = lr)
    elif optimizer_name == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate = lr)#, momentum = 0.5)

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

    models = load_models(config=dict_settings)
    old_dataset_name = "not set" # arbitrary
    model_counter = -1 # Unsmart way to start a counter
    batch_size = 128 # Training batch size that is.
    eps_iter_portion = 0.2 # Scale of epsilon iterations for attack steps if adversarially training
    att_steps = 10 # Number of PGD/SLIDE attack steps if adversarially training
    early_stopping_patience = 5  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum change to qualify as an improvement
    min_epochs = 10
    max_epochs = 300 # Max epochs

    for name, model in models.items():
        # --- determine if we are running a given model (for use in batch runs) --- 
        print("model name", name)
        model_counter += 1
        if boo_run_all == False:
            if model_num != model_counter:
                continue
            
        # --- get key information from model name ---
        dataset_name = name.split("_")[0]
        base_model = name.split("_")[1]
        top_layer = name.split("_")[2]
        adv_train = name.split("_")[3]
        
        print(name, adv_train)
    
        # --- data --- 
        if old_dataset_name == "not set" or old_dataset_name != dataset_name:
            _, eps, input_elements, data, info, _, _ = load_data(dataset_name, batch_size)
       
        old_dataset_name = dataset_name
        total_size = info.splits['train'].num_examples
        val_size = int((total_size * 0.1) // batch_size) # 10% for validation
        data_train = data.train.skip(val_size)  # Skip the first X% for training
        data_val = data.train.take(val_size)  # Take the first X% for validation
        

        eps_l2 = math.sqrt((eps**2)*input_elements)
        eps_l1 = 2 * eps_l2    

        # --- initiate counters specific to model ---
        best_val_accuracy = 0 
        patience_counter = 0  # Counts epochs without improvement
        lr_reduced_counter = 0
        boo_adv_train = False
        boo_update_weights = False
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

if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    app.run(main)