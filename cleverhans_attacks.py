import csv
import math
import sys
import os
import numpy as np
import tensorflow as tf
import time
from absl import app

from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import  load_attack_settings, save_location_attack_results

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

def main(_):
    # argument parsing

    batch_chunk = int(sys.argv[1])
    total_batch_chunks = int(sys.argv[2])
    batch_size = int(sys.argv[3]) 
    arg_dataset = sys.argv[4]
    print('argument dataset', arg_dataset)

    eps, data, info, model_paths = load_attack_settings(arg_dataset, batch_size, "master_models")#, adv_train)

    total_test_examples = info.splits['test'].num_examples
    total_batches = math.ceil(total_test_examples / batch_size)

    batches_per_chunk = total_batches // (total_batch_chunks)
    start_batch = batch_chunk * batches_per_chunk
    
    test_data_subset = data.test.skip(start_batch).take(batches_per_chunk) # Slicing the test dataset by batches

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for name, model_path in model_paths.items():
        print(f'attacking {name}, stored in {model_path}')
        model = tf.keras.models.load_model(model_path)

        test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
        test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_inf = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_2 = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_1 = tf.metrics.SparseCategoricalAccuracy()

        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(batches_per_chunk*batch_size)
        current_batch_size = 0
        for x, y in test_data_subset: #data.test:
            input_elements = 1
            for i in range(1, len(x.shape)):
                input_elements *= x.shape[i]
                
            eps_l2 = math.sqrt((eps**2)*input_elements)
            eps_l1 = 2 * eps_l2

            # -- clean --
            y_pred = model(x, training=False)
            test_acc_clean(y, y_pred)
            print("clean done", time.time())
            
            # -- l_1 || projected gradient descent --
            x_pgd_1 = l1_projected_gradient_descent(model, 
                                                     x, 
                                                     y, 
                                                     steps = 100, 
                                                     epsilon = eps_l1, 
                                                     eps_iter = 0.01, 
                                                     loss_object = loss_object, 
                                                     x_min = -1.0, 
                                                     x_max = 1.0,
                                                     perc = 99)
            y_pred_pgd_1 = model(x_pgd_1, training=False)
            test_acc_pgd_1(y, y_pred_pgd_1)
            print("l1 pgd done", time.time())
            
            # -- fast gradient sign method --
            x_fgsm = fast_gradient_method(model_fn = model,
                                                    x = x,
                                                    eps = eps,
                                                    norm = np.inf,
                                                    loss_fn = None,
                                                    clip_min = -1.0,
                                                    clip_max = 1.0,
                                                    y = y,
                                                    targeted = False,
                                                    sanity_checks=False)
            y_pred_fgsm = model(x_fgsm, training=False)
            test_acc_fgsm(y, y_pred_fgsm)
            print("fgsm done", time.time())

            # -- l_inf || projected gradient descent --
            x_pgd_inf = projected_gradient_descent(model_fn = model,
                                                    x = x,
                                                    eps = eps,
                                                    eps_iter = 0.01,
                                                    nb_iter = 100,
                                                    norm = np.inf,
                                                    loss_fn = None,
                                                    clip_min = -1.0,
                                                    clip_max = 1.0,
                                                    y = y,
                                                    targeted = False,
                                                    rand_init = True,
                                                    rand_minmax = eps,
                                                    sanity_checks=False)
            y_pred_pgd_inf = model(x_pgd_inf, training=False)
            test_acc_pgd_inf(y, y_pred_pgd_inf)
            print("linf pgd done", time.time())
            
            # -- l_2 || projected gradient descent --
            x_pgd_2 = l2_projected_gradient_descent(model, 
                                                    x, 
                                                    y, 
                                                    steps=100, 
                                                    epsilon= eps_l2, 
                                                    eps_iter = 0.01, 
                                                    loss_object = loss_object, 
                                                    x_min = -1.0, 
                                                    x_max = 1.0)
            y_pred_pgd_2 = model(x_pgd_2, training=False)
            test_acc_pgd_2(y, y_pred_pgd_2)
            print("l2 pgd done", time.time())
            
            current_batch_size += x.shape[0]
            progress_bar_test.add(x.shape[0], values = [("clean", test_acc_clean.result()),
                                                        ("FGSM", test_acc_fgsm.result()),
                                                        ('PGD L1', test_acc_pgd_1.result()),
                                                        ('PGD L2', test_acc_pgd_2.result()),
                                                        ("PGD Linf", test_acc_pgd_inf.result()), 
                                                        ])

        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        print("test acc on FGSM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
        print("test acc on l_inf PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_inf.result() * 100))
        print("test acc on l_2 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_2.result() * 100))
        print("test acc on l_1 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_1.result() * 100))

        # Assuming you have already calculated these accuracies
        test_acc_clean_value = test_acc_clean.result().numpy()
        test_acc_fgsm_value = test_acc_fgsm.result().numpy()
        test_acc_pgd_inf_value = test_acc_pgd_inf.result().numpy()
        test_acc_pgd_2_value = test_acc_pgd_2.result().numpy()
        test_acc_pgd_1_value = test_acc_pgd_1.result().numpy()

        # Prepare the data to be written to CSV
        accuracy_data = [test_acc_clean_value, 
                         test_acc_pgd_1_value,
                         test_acc_pgd_2_value,
                         test_acc_fgsm_value,
                         test_acc_pgd_inf_value,
                         current_batch_size,
                         ]

        # Specify the CSV file name
        csv_file = save_location_attack_results(arg_dataset,name,batch_chunk,total_batch_chunks,"pgd")

        # Write to CSV
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Clean', 
                             'PGD L1', 
                             'PGD L2', 
                             'FGSM', 
                             'PGD Linf',
                             'batch_size',
                             ])
            writer.writerow(accuracy_data)

        print(f"Accuracies written to {csv_file}")


if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    app.run(main)