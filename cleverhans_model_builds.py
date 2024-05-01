import math
import time
import sys
import numpy as np
import tensorflow as tf
import os

from absl import app, flags
from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import load_build_settings, load_models, load_data
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

#FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#tf.debugging.set_log_device_placement(True)
#tf.config.experimental.set_visible_devices('GPU:0', 'GPU')

def main(_):

    model_num = sys.argv[1]
    
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
                                        "relu" :    {"yes" : 0, "no" : 1},
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
        "cifar10" : {"ResNet50" :       {"trop":    {"yes" : 0, "no" : 0}, #here
                                        "relu" :    {"yes" : 0, "no" : 0}, #here
                                        "maxout" :  {"yes" : 0, "no" : 0}},#here
                    "VGG16" :           {"maxout" : {"yes" : 0, "no" : 0},
                                        "relu" :    {"yes" : 0, "no" : 0},
                                        "trop" :    {"yes" : 0, "no" : 0}},
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
    # Load training and test data
    #input_elements, eps, data, info, models = load_build_settings(arg_dataset, base_model_index, batch_size)
    old_dataset_name = "not set"
    model_counter = -1
    for name, model in models.items():
        model_counter += 1
        #print(model_counter, model_num)
        if boo_run_all == False:
            if model_num != model_counter:
                continue
        dataset_name = name.split("_", 1)[0]
        last_underscore_index = name.rfind("_")
        adv_train = name[last_underscore_index + 1:]
        
        print(name, adv_train)
        
        batch_size = 512
        if old_dataset_name == "not set" or old_dataset_name != dataset_name:
            _, eps, input_elements, data, info, _, _ = load_data(dataset_name, batch_size)
       
        old_dataset_name = dataset_name
        eps_l2 = math.sqrt((eps**2)*input_elements)
        eps_l1 = 2 * eps_l2
        eps_iter_portion = 0.2
        att_steps = 10
        
        # -- just added -- 
        total_size = info.splits['train'].num_examples
        val_size = int((total_size * 0.1) // batch_size) # 10% for validation

        data_train = data.train.skip(val_size)  # Skip the first X% for training
        data_val = data.train.take(val_size)  # Take the first X% for validation

        early_stopping_patience = 4  # Number of epochs to wait for improvement
        min_delta = 0.001  # Minimum change to qualify as an improvement
        nb_epochs = 100
        nb_epochs_no_adv = 5
        # -- just added -- 
        
        best_val_loss = float('inf')  # Best validation loss seen so far
        best_val_accuracy = 0 
        patience_counter = 0  # Counts epochs without improvement

        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)#, reduction=tf.keras.losses.Reduction.NONE)
        lr = 0.001

        #initial_learning_rate = 0.1
        #lr = tf.keras.optimizers.schedules.ExponentialDecay(
            #initial_learning_rate,
            #decay_steps=10000,
            #decay_rate=0.96,
            #staircase=True)

        optimizer = tf.optimizers.Adam(learning_rate=lr)

        # Metrics to track the different accuracies.
        train_loss = tf.metrics.Mean(name="train_loss")
        train_acc = tf.metrics.SparseCategoricalAccuracy()
        validation_acc = tf.keras.metrics.SparseCategoricalAccuracy()

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
        epoch_counter = 0
        boo_making_progress = False
        boo_adv_train = False
        # Train model with adversarial training
        for epoch in range(nb_epochs):
            validation_acc.reset_state() 
            epoch_counter = epoch + 1
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(info.splits['train'].num_examples - val_size*batch_size)
            simple_counter = 0
            progress_count = 1
            
            if ((adv_train == "yes") and (nb_epochs_no_adv >= epoch_counter)) or (adv_train == "no"):
                boo_adv_train = False
            else:
                boo_adv_train = True
            print(f"--epoch {epoch}, adv_train: {adv_train}, boo_adv_train: {boo_adv_train}--")    
                
            for (x, y) in data_train:
                if boo_adv_train:
                    # Replace clean example with adversarial example for adversarial training
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
                train_step(x, y)
                simple_counter += 1
                if simple_counter == progress_count:
                    progress_bar_train.add(batch_size*progress_count, values=[("loss", train_loss.result()), ("acc", train_acc.result())])
                    simple_counter = 0
                
            # -- \/\/just added\/\/ -- 
            val_loss = 0
            for (x_val, y_val) in data_val:
                predictions = model(x_val, training=False)
                v_loss = loss_object(y_val, predictions).numpy()
                val_loss += v_loss  # Sum up validation loss
                validation_acc.update_state(y_val, predictions)

            val_accuracy = validation_acc.result().numpy()
            val_loss /= len(data_val)  # Get average validation loss
            
            # Early Stopping Check
            if val_loss < best_val_loss - min_delta: 
                best_val_loss = val_loss
                boo_making_progress = True
            if val_accuracy > best_val_accuracy + min_delta:
                best_val_accuracy = val_accuracy
                boo_making_progress = True
                
            if boo_making_progress:
                patience_counter = 0
            else:
                patience_counter += 1  # No improvement found
            boo_making_progress = False
                
            print(f'Epoch {epoch}, Validation Loss: {val_loss}, Best: {best_val_loss}, Validation Accuracy {val_accuracy}, Best: {best_val_accuracy}')
            
            if patience_counter >= early_stopping_patience and epoch >= 14:
                print("Stopping early due to lack of improvement.")
                break
            # -- ^^just added^^ --
            
        elapsed = time.time() - start
        print(f'##### training time per epoch = {elapsed/epoch_counter} seconds | {elapsed/60/epoch_counter} minutes')
        print(f'##### training time total = {elapsed} seconds | {elapsed/60} minutes')
        model.summary()
        current_time = time.localtime()

        # Format the time as required
        formatted_date = time.strftime("%d%b%y", current_time)
        if not os.path.exists('master_models'):  # Check if directory doesn't exist
            os.makedirs('master_models')

        model.save(f'master_models/{name}_{formatted_date}.keras')#, save_format='tf')

if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    #flags.DEFINE_float("eps", 0.1, "Total epsilon for FGM and PGD attacks.")
    #flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    #flags.DEFINE_string("dataset", "mnist", "Specifies dataset used to train the model.")
    app.run(main)