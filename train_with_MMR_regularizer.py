import math
import time
import sys
import os
import numpy as np
import tensorflow as tf

from absl import app, flags

from functions.models import MMRModel#, CH_MMRReLU_ResNet50
from functions.load_data import ld_mnist, ld_svhn, ld_cifar10  

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from custom_layers.mmr_regularizer import mmr_cnn

def main(_):
    dataset = "mnist"
    batch_size = 8
    nb_epochs = 1
    # Load training and test data
    if dataset == "mnist":
        eps = 0.2
        data, info = ld_mnist(batch_size = batch_size)
        input_shape = (None,28,28,1)
        models = {'mnist_ModifiedLeNet5_MMRModel_no': MMRModel(num_classes=10)}
        lr = 5e-4
        n_total_hidden_units = 51722

    elif dataset == "svhn":
        eps = 8/255
        data, info = ld_svhn(batch_size = batch_size)
        input_shape = (None,32,32,3)
        models = {'svhn_ModifiedLeNet5_MMRModel_no': MMRModel(num_classes=10)}
        lr = 1e-3
        n_total_hidden_units = 69578
    val_split_size = 0.99
    total_size = info.splits['train'].num_examples
    val_size = int((total_size * val_split_size) // batch_size) # 10% for validation
    data_train = data.train.skip(val_size)  # Skip the first X% for training
    data_val = data.train.take(val_size)  # Take the first X% for validation
    min_delta = 0.001
    early_stopping_patience = 3

    for name, model in models.items():
        model.build(input_shape=input_shape)       
        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        best_val_accuracy = 0 
        patience_counter = 0
        # Metrics to track the different accuracies.
        train_loss = tf.metrics.Mean(name="train_loss")
        train_acc = tf.metrics.SparseCategoricalAccuracy()
        validation_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        gamma_l1 = 1.0
        gamma_linf = 0.15
        lmbd_l1 = 3.0
        lmbd_linf = 12.0
        hyp_start = 0.2
        hyp_end = 0.05
        n_train_ex, height, width, n_col = input_shape
        n_out = 10

        #@tf.function
        def train_step(x, y, frac_reg_tf, n_rb_tf, n_db_tf, boo_first_go = False):
            with tf.GradientTape() as tape:
                predictions, feature_maps = model(x, return_feature_maps=True)
                loss_sce = loss_object(y, predictions)
                if not boo_first_go or boo_first_go:
                    model_weights = model.get_weights()
                    
                    #model_weights = [var for var in model.trainable_variables]
                    reg_details = mmr_cnn(feature_maps, x, y_true = tf.one_hot(y, 10),
                                        model = model, #might have issues if y is not one-hot encoded
                                        n_rb = n_rb_tf, 
                                        n_db = n_db_tf, 
                                        gamma_rb = [gamma_l1, gamma_linf], 
                                        gamma_db = [gamma_l1, gamma_linf], 
                                        bs=x.shape[0], 
                                        q = 'univ',
                                        weights = model_weights)
                    rb_reg_part = frac_reg_tf * (lmbd_l1 * reg_details[0] + lmbd_linf * reg_details[1]) / tf.cast(n_rb_tf, tf.float32)
                    db_reg_part = frac_reg_tf * (lmbd_l1 * reg_details[2] + lmbd_linf * reg_details[3]) / tf.cast(n_db_tf, tf.float32)
                    
                    reg_rb_tower, reg_db_tower = tf.reduce_mean(rb_reg_part), tf.reduce_mean(db_reg_part)
                    
                    loss = loss_sce + reg_rb_tower + reg_db_tower
                else:
                    loss = loss_sce
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_acc(y, predictions)

        start = time.time()
        # Train model with adversarial training
        old_acc = 0
        epoch_counter = 0 
        for epoch in range(nb_epochs):
            epoch_counter = epoch + 1
            epoch_start_reduced_lr = 0.9
            lr_actual = lr / 10 if epoch >= epoch_start_reduced_lr * nb_epochs else lr
            optimizer.learning_rate = lr_actual
            
            frac_reg = min(epoch / 10.0, 1.0)  # from 0 to 1 linearly over the first 10 epochs

            frac_start, frac_end = hyp_start, hyp_end  # decrease the number of linear region hyperplanes from 10% to 2%
            n_db = n_out  # the number of decision boundary hyperplanes is always the same (the number of classes)
            n_rb_start, n_rb_end = int(frac_start * n_total_hidden_units), int(frac_end * n_total_hidden_units)
            n_rb = (n_rb_end - n_rb_start) / nb_epochs * epoch + n_rb_start
            
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(int(info.splits['train'].num_examples*(1-val_split_size)))
            print(f"--epoch {epoch}--")
            batches = 2
            i = 1
            for (x, y) in data_train:
                if epoch==0:
                    boo_first_go = True
                else:
                    boo_first_go = False
                train_step(x, y, frac_reg_tf = frac_reg, n_rb_tf=n_rb, n_db_tf=n_db, boo_first_go = boo_first_go)
                if (i % batches) == 0:
                    progress_bar_train.add(x.shape[0]*batches, values=[("loss", train_loss.result()), ("acc", train_acc.result())])
                i += 1
            print("loss", train_loss.result(), "acc",  train_acc.result())
            
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
            if patience_counter >= early_stopping_patience and epoch >= 19:
                break
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