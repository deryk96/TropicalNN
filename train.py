import math
import time
import sys
import numpy as np
import tensorflow as tf
import os

from absl import app, flags
from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import load_build_settings, load_models, load_data, find_model
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

#FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#tf.debugging.set_log_device_placement(True)
#tf.config.experimental.set_visible_devices('GPU:0', 'GPU')

def main(_):

    model_num = 0#sys.argv[1]
    
    if model_num == "all":
        boo_run_all = True
    else:
        boo_run_all = False
        model_num = int(model_num)
        print(model_num, type(model_num))
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
                    "MobileNet" :       {"maxout" : {"yes" : 0, "no" : 0},##<
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}}},
        "cifar10" : {"ResNet50" :       {"trop":    {"yes" : 0, "no" : 0}, #here
                                        "relu" :    {"yes" : 0, "no" : 0}, #here
                                        "maxout" :  {"yes" : 0, "no" : 0}},#here
                    "VGG16" :           {"maxout" : {"yes" : 0, "no" : 0}, #here
                                        "relu" :    {"yes" : 0, "no" : 0}, #here
                                        "trop" :    {"yes" : 0, "no" : 0}},#here
                    "EfficientNetB4" :  {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}}},
        "cifar100" : {"ResNet50" :      {"maxout" : {"yes" : 0, "no" : 0}, #here
                                        "relu" :    {"yes" : 0, "no" : 0}, #here
                                        "trop" :    {"yes" : 0, "no" : 0}},#here
                    "VGG16" :           {"maxout" : {"yes" : 0, "no" : 0}, #here
                                        "relu" :    {"yes" : 0, "no" : 0}, #here
                                        "trop" :    {"yes" : 0, "no" : 0}},#here
                    "EfficientNetB4" :  {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}}},
    }

    models = load_models(config=dict_settings)
    old_dataset_name = "not set"
    model_counter = -1
    batch_size = 512
    eps_iter_portion = 0.2
    att_steps = 10
    early_stopping_patience = 5  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum change to qualify as an improvement
    nb_epochs = 300

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
            boo_update_weights = False ### CHANGE BACK TO TRUE
        if  (adv_train == "no"): 
            boo_adv_train = False
        else:
            boo_adv_train = True

        #boo_update_weights = True
        # --- initiate tensorflow objects ---
        lr = 0.5
        #optimizer = tf.optimizers.Adam(learning_rate=lr)#, weight_decay=0.0001, use_ema=True, ema_momentum=0.9)
        optimizer = tf.optimizers.SGD(learning_rate = lr)#, momentum = 0.5)
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
        boo_temporary = True ### get rid of this young man
        for epoch in range(nb_epochs):
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
                
                if epoch == 0 and boo_temporary == True:
                    ####### SETTING UP SMALL SHIT ############
                    tropical_layer = model.top_layer.get_layer(name="tropical")
                    tf.print(tropical_layer.w)
                    corners = 60
                    cross = 25
                    desired_weights = [
                        tf.constant([
                            [4*cross, 0, cross], 
                            [-4*cross, 0, cross], 
                            [-4*cross, 0, -cross], 
                            [4*cross, 0, -cross], 
                            [corners, 0, corners], 
                            [-corners, 0, corners], 
                            [corners, 0, -corners], 
                            [-corners, 0, -corners], 
                            [0, 0, corners + 10],
                            [0, 0, -corners - 10]
                        ], dtype=tf.float32),
                        tf.constant([0.0, 0.0, 0.0,0.0, 0.0, 0.0,0.0, 0.0, 0.0,0.0], dtype=tf.float32)

                    ]
                    tropical_layer.set_weights(desired_weights)
                    tropical_layer.w.trainable = False
                    tropical_layer.bias.trainable = False
                    boo_temporary = False
                    tf.print(tropical_layer.w)
                elif epoch == 2:
                    tf.print(tropical_layer.w)
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
            if patience_counter >= early_stopping_patience and epoch >= 299:
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