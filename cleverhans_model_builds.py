import math
import time
import sys
import numpy as np
import tensorflow as tf
import os

from absl import app, flags
from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import load_build_settings
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def main(_):
    if len(sys.argv) > 1:
        adv_train = sys.argv[1]
        arg_dataset = sys.argv[2]
        batch_size = int(sys.argv[3])
        base_model_index = int(sys.argv[4])
        print('argument dataset', arg_dataset, arg_dataset == FLAGS.dataset)
    else:
        adv_train = 'yes'
        arg_dataset = 'mnist'
        batch_size = 128
        base_model_index = 0
    
    if adv_train == 'yes':
        FLAGS.adv_train = True
    else:
        FLAGS.adv_train = False
    
    # Load training and test data
    input_elements, eps, data, info, models = load_build_settings(arg_dataset, base_model_index, batch_size)
    
    FLAGS.eps = eps
    eps_l2 = math.sqrt((FLAGS.eps**2)*input_elements)
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
    # -- just added -- 

    for name, model in models.items():
        best_val_loss = float('inf')  # Best validation loss seen so far
        patience_counter = 0  # Counts epochs without improvement

        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Metrics to track the different accuracies.
        train_loss = tf.metrics.Mean(name="train_loss")
        train_acc = tf.metrics.SparseCategoricalAccuracy()

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
        # Train model with adversarial training
        for epoch in range(FLAGS.nb_epochs):
            epoch_counter = epoch + 1
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(info.splits['train'].num_examples - val_size*batch_size)
            simple_counter = 0
            progress_count = 10
            print(f"--epoch {epoch}--")
            for (x, y) in data_train:
                if FLAGS.adv_train:
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
                                                    eps = FLAGS.eps,
                                                    eps_iter = eps_iter_portion * FLAGS.eps,
                                                    nb_iter = att_steps,
                                                    norm = np.inf,
                                                    loss_fn = None,
                                                    clip_min = -1.0,
                                                    clip_max = 1.0,
                                                    y = y_pre_att,
                                                    targeted = False,
                                                    rand_init = True,
                                                    rand_minmax = FLAGS.eps,
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
            for (x_val, y_val) in data_val:  # Assuming you have a validation dataset
                predictions = model(x_val, training=False)
                v_loss = loss_object(y_val, predictions).numpy()
                val_loss += v_loss  # Sum up validation loss
                
            val_loss /= len(data_val)  # Get average validation loss
            
            # Early Stopping Check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience since we found an improvement
                # You can also save the model here as the current best model
            else:
                patience_counter += 1  # No improvement found
                
            print(f'Epoch {epoch}, Validation Loss: {val_loss}, Best: {best_val_loss}')
            
            if patience_counter >= early_stopping_patience:
                print("Stopping early due to lack of improvement.")
                break
            # -- ^^just added^^ --
            
        elapsed = time.time() - start
        print(f'##### training time per epoch = {elapsed/epoch_counter} seconds | {elapsed/60/epoch_counter} minutes')
        model.summary()
        current_time = time.localtime()

        # Format the time as required
        formatted_date = time.strftime("%d%b%y", current_time)
        if not os.path.exists('master_models'):  # Check if directory doesn't exist
            os.makedirs('master_models')

        model.save(f'master_models/{arg_dataset}_{name}_{adv_train}_{formatted_date}', save_format='tf')

if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    flags.DEFINE_float("eps", 0.1, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    flags.DEFINE_string("dataset", "mnist", "Specifies dataset used to train the model.")
    app.run(main)