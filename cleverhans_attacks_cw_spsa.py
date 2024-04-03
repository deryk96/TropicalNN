import csv
import math
import sys
import os
import numpy as np
import tensorflow as tf
import time
from absl import app, flags

from functions.utils import load_attack_settings, l2
from edited_cleverhans.edited_carlini_wagner_l2 import carlini_wagner_l2
#from cleverhans.carlini_wagner_l2 import carlini_wagner_l2
from edited_cleverhans.edited_spsa import spsa

FLAGS = flags.FLAGS

def main(_):
    # argument parsing
    if len(sys.argv) > 1:
        batch_chunk = int(sys.argv[1])
        total_batch_chunks = int(sys.argv[2])
        batch_size = int(sys.argv[3]) 
        arg_dataset = sys.argv[4]
        #adv_train = sys.argv[5]
        print('argument dataset', arg_dataset, arg_dataset == FLAGS.dataset)
    else:
        batch_chunk = 0
        total_batch_chunks = 1
        batch_size = 128
        arg_dataset = 'mnist'
        #adv_train = 'no'
        
    if arg_dataset != FLAGS.dataset:
        FLAGS.dataset = arg_dataset
        
    num_classes = 10
    cw_targeted = False

    print("starting run")
    # Load data
    eps, data, info, model_paths = load_attack_settings(arg_dataset, batch_size)#, adv_train)
    FLAGS.eps = eps
    

    total_test_examples = info.splits['test'].num_examples
    total_batches = math.ceil(total_test_examples / batch_size)

    batches_per_chunk = total_batches // (total_batch_chunks)
    start_batch = batch_chunk * batches_per_chunk
    
    test_data_subset = data.test.skip(start_batch).take(batches_per_chunk) # Slicing the test dataset by batches
    
    for name, model_path in model_paths.items():
        print(f'attacking {name}, stored in {model_path}')
        model = tf.keras.models.load_model(model_path)

        test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
        test_acc_cw = tf.metrics.SparseCategoricalAccuracy()
        test_acc_spsa = tf.metrics.SparseCategoricalAccuracy()

        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(batches_per_chunk*batch_size)
        l2_x_cw_avgs = []
        current_batch_size = 0
        for x, y in test_data_subset: #data.test:
            time_0 = time.time()
            input_elements = 1
            for i in range(1, len(x.shape)):
                input_elements *= x.shape[i]

            # -- clean --
            y_pred = model(x, training=False)
            test_acc_clean(y, y_pred)
            time_1 = time.time()
            print(f"\nclean done{time_1 - time_0} sec")

            # -- carlini wagner --
            cw_args = {
                        'model_fn' : model, 
                        'x' : x, 
                        'y' : None,
                        'abort_early' : True, 
                        'clip_min' : -1.0, 
                        'max_iterations' : 1000, 
                        'binary_search_steps' : 10,
                        'confidence' : 0,
                        'initial_const' : 10,
                        'learning_rate' : 0.1
                        }
            
            if cw_targeted:
                list_cw_x = []
                for i in range(num_classes):
                    print("cw", i, time.time())
                    y_current = tf.fill(y.shape, i)
                    cw_args['y'] = y_current
                    x_cw_curr = carlini_wagner_l2(**cw_args)
                    list_cw_x.append(x_cw_curr)
                list_cw_l2 = []
                for i in range(num_classes):
                    l2_tensor = l2(x, list_cw_x[i])
                    mask_true_class = tf.equal(y, i)
                    l2_tensor = tf.where(mask_true_class, tf.constant(1000, dtype=tf.float32), l2_tensor)
                    mask_no_example_found = tf.equal(l2_tensor, 0.0)
                    l2_tensor = tf.where(mask_no_example_found, tf.constant(900, dtype=tf.float32), l2_tensor)
                    list_cw_l2.append(l2_tensor)
                combined_x_tensor = tf.stack(list_cw_x)
                argmin_non_zero_values = tf.argmin(tf.stack(list_cw_l2) , axis=0) # Find the indices (dimensions) of the minimum non-zero values
                list_best_x_cw = []
                for i in range(x.shape[0]):
                    perturbation_to_use = combined_x_tensor[argmin_non_zero_values[i], i, :, :, :]
                    list_best_x_cw.append(perturbation_to_use)
                x_cw = tf.stack(list_best_x_cw)
            else:
                x_cw = carlini_wagner_l2(**cw_args)
            l2_x_cw = l2(x, x_cw)
            non_zero_mask = tf.not_equal(l2_x_cw, 0.0)
            non_zero_values = tf.boolean_mask(l2_x_cw, non_zero_mask)
            
            l2_x_cw_avgs.append(tf.reduce_sum(non_zero_values).numpy().item())
            # Check if non_zero_values is empty
            #if tf.size(non_zero_values) == 0:
                # Append 0 when there are no non-zero values
                #l2_x_cw_avgs.append(0.0)
            #else:
                # Otherwise, calculate the mean and append it
                #l2_x_cw_avgs.append(tf.reduce_sum(non_zero_values).numpy().item())
            y_pred_cw = model(x_cw, training=False)
            test_acc_cw(y, y_pred_cw)
            time_2 = time.time()
            print(f"\ncw done {time_2 - time_1} sec")
            

            # -- spsa --
            #x_spsa_list = [] # uncomment if you want to view/store the spsa images for some reason.
            y_pred_spsa_list = []

            for i in range(x.shape[0]):
                #if (i%1)==0:
                    #print("spsa", i, time.time())
                x_spsa_single = spsa(model, 
                                      x=x[i:i+1], 
                                      y=y[i], 
                                      eps=FLAGS.eps, 
                                      nb_iter=100, 
                                      learning_rate=0.01, 
                                      delta=0.01, 
                                      spsa_samples=128, 
                                      spsa_iters=1,
                                      clip_min=-1, 
                                      clip_max=1,
                                      early_stop_loss_threshold = 0.0)
                #x_spsa_list.append(x_spsa_single) # uncomment if you want to view/store the spsa images for some reason.
                y_pred_spsa_list.append(model(x_spsa_single, training=False))
            #x_spsa = tf.concat(x_spsa_list, axis=0) # uncomment if you want to view/store the spsa images for some reason.
            y_pred_spsa = tf.concat(y_pred_spsa_list, axis=0)
            test_acc_spsa(y, y_pred_spsa)
            time_3 = time.time()
            print(f"\nspsa done {time_3 - time_2} sec")

            current_batch_size += x.shape[0]
            progress_bar_test.add(x.shape[0], values = [("clean", test_acc_clean.result()),
                                                        ("CW", test_acc_cw.result()), 
                                                        ("SPSA", test_acc_spsa.result())
                                                        ])

        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        print("test acc on Carlini Wagner adversarial examples (%): {:.3f}".format(test_acc_cw.result() * 100))
        print("test acc on SPSA adversarial examples (%): {:.3f}".format(test_acc_spsa.result() * 100))

        # Assuming you have already calculated these accuracies
        test_acc_clean_value = test_acc_clean.result().numpy()
        test_acc_cw_value = test_acc_cw.result().numpy()
        test_acc_spsa_value = test_acc_spsa.result().numpy()
        non_zero_l2_x_cw_avg = [element for element in l2_x_cw_avgs if element > 0]
        if len(non_zero_l2_x_cw_avg) == 0:
            l2_x_cw_avg = 0.0
        else:
            l2_x_cw_avg = sum(non_zero_l2_x_cw_avg) / ((1 - test_acc_cw_value) * batches_per_chunk * batch_size) #len(non_zero_l2_x_cw_avg)

        # Prepare the data to be written to CSV
        accuracy_data = [test_acc_clean_value, 
                         test_acc_cw_value, 
                         test_acc_spsa_value,
                         l2_x_cw_avg,
                         current_batch_size
                         ]

        # Specify the CSV file name
        if not os.path.exists('attack_results'):  # Check if directory doesn't exist
            os.makedirs('attack_results')
        if not os.path.exists(f'attack_results/{FLAGS.dataset}'):  # Check if directory doesn't exist
            os.makedirs(f'attack_results/{FLAGS.dataset}')
        csv_file = f'attack_results/{FLAGS.dataset}/{name}_{batch_chunk}_of_{total_batch_chunks}_cw_spsa.csv'

        # Write to CSV
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Clean', 
                             'CW', 
                             'SPSA',
                             'CW L2 Distortion Avg',
                             'batch_size',
                             ])
            writer.writerow(accuracy_data)

        print(f"Accuracies written to {csv_file}")


if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    flags.DEFINE_float("eps", 0.1, "Total epsilon for FGSM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    flags.DEFINE_string("dataset", "mnist", "Specifies dataset used to train the model.")
    app.run(main)