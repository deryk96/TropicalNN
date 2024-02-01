import csv
import math
import sys
import numpy as np
import tensorflow as tf
import time
from absl import app, flags

from functions.load_data import ld_mnist, ld_svhn, ld_cifar10
from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import  plot_images_in_grid

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
#from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from edited_cleverhans.edited_carlini_wagner_l2 import carlini_wagner_l2
#from cleverhans.tf2.attacks.spsa import spsa
from edited_cleverhans.edited_spsa import spsa

FLAGS = flags.FLAGS

def l2(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape)))))

def l1(x, y):
    return tf.reduce_sum(tf.abs(x - y), list(range(1, len(x.shape))))

def main(_):
    # argument parsing
    if len(sys.argv) > 1:
        batch_chunk = int(sys.argv[1])
        total_batch_chunks = int(sys.argv[2])
        batch_size = int(sys.argv[3]) 
        arg_dataset = sys.argv[4]
        adv_train = sys.argv[5]
        print('argument dataset', arg_dataset, arg_dataset == FLAGS.dataset)
    else:
        batch_chunk = 0
        total_batch_chunks = 1
        batch_size = 128
        arg_dataset = 'mnist'
        adv_train = 'no'
        
    if arg_dataset != FLAGS.dataset:
        FLAGS.dataset = arg_dataset
        
    num_classes = 10
    cw_targeted = False
    check = False
    print("starting run")
    # Load data
    if FLAGS.dataset == "mnist":
        FLAGS.eps = 0.2
        data, info = ld_mnist(batch_size = batch_size)
        if adv_train == 'yes':
            model_paths = {
                       'CH_TropConv3Layer_yes_adv': 'saved_models/CH_TropConv3Layer_mnist_0.2_100_True',
                       'CH_TropConv3Layer_no_adv': 'saved_models/CH_TropConv3Layer_mnist_0.1_100_False',
                       #'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_mnist_0.2_100_True',
                       #'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_mnist_0.2_100_True',
                        }
        else:
            model_paths = {
                       'CH_MMRReluConv3Layer':'saved_models/CH_MMRReluConv3Layer_mnist_0.1_100_False',
                       'CH_TropConv3Layer': 'saved_models/CH_TropConv3Layer_mnist_0.1_100_False',
                       'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_mnist_0.1_100_False',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_mnist_0.1_100_False',
                        }
    elif FLAGS.dataset == "svhn":
        FLAGS.eps = 8/255
        data, info = ld_svhn(batch_size = batch_size)
        if adv_train == 'yes':
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_TropConv3Layer_svhn_0.03137254901960784_100_True',
                       'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_svhn_0.03137254901960784_100_True',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_svhn_0.03137254901960784_100_True',
                        }
        else:
            model_paths = {
                       'CH_MMRReluConv3Layer': 'saved_models/CH_MMRReluConv3Layer_svhn_0.01568627450980392_100_False',
                       #'CH_TropConv3Layer': 'saved_models/CH_TropConv3Layer_svhn_0.01568627450980392_100_False',
                       #'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_svhn_0.01568627450980392_100_False',
                       #'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_svhn_0.01568627450980392_100_False',
                        }
    else:
        FLAGS.eps = 8/255
        data, info = ld_cifar10(batch_size = batch_size)
        if adv_train == 'yes':
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_Trop_ResNet50_cifar_0.03137254901960784_100_True',
                       'CH_ReluConv3Layer':'saved_models/CH_ReLU_ResNet50_cifar_0.03137254901960784_100_True',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxOut_ResNet50_cifar_0.03137254901960784_100_True',
                        }
        else:
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_Trop_ResNet50_cifar_0.01568627450980392_100_False',
                       #'CH_ReluConv3Layer':'saved_models/CH_ReLU_ResNet50_cifar_0.01568627450980392_100_False',
                       #'CH_MaxoutConv3Layer':'saved_models/CH_MaxOut_ResNet50_cifar_0.01568627450980392_100_False',
                        }
    

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
        test_acc_cw = tf.metrics.SparseCategoricalAccuracy()
        test_acc_spsa = tf.metrics.SparseCategoricalAccuracy()

        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(total_test_examples)
        l2_x_cw_avgs = []
        batch_size = 0
        for x, y in test_data_subset: #data.test:
            input_elements = 1
            for i in range(1, len(x.shape)):
                input_elements *= x.shape[i]
                
            eps_l2 = math.sqrt((FLAGS.eps**2)*input_elements)
            eps_l1 = 2 * eps_l2

            # -- clean --
            y_pred = model(x)
            test_acc_clean(y, y_pred)
            print("clean done", time.time())
            '''
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
            y_pred_pgd_1 = model(x_pgd_1)
            test_acc_pgd_1(y, y_pred_pgd_1)
            print("l1 pgd done", time.time())
            
            # -- fast gradient sign method --
            x_fgsm = fast_gradient_method(model_fn = model,
                                                    x = x,
                                                    eps = FLAGS.eps,
                                                    norm = np.inf,
                                                    loss_fn = None,
                                                    clip_min = -1.0,
                                                    clip_max = 1.0,
                                                    y = y,
                                                    targeted = False,
                                                    sanity_checks=False)
            y_pred_fgsm = model(x_fgsm)
            test_acc_fgsm(y, y_pred_fgsm)
            print("fgsm done", time.time())

            # -- l_inf || projected gradient descent --
            x_pgd_inf = projected_gradient_descent(model_fn = model,
                                                    x = x,
                                                    eps = FLAGS.eps,
                                                    eps_iter = 0.01,
                                                    nb_iter = 100,
                                                    norm = np.inf,
                                                    loss_fn = None,
                                                    clip_min = -1.0,
                                                    clip_max = 1.0,
                                                    y = y,
                                                    targeted = False,
                                                    rand_init = True,
                                                    rand_minmax = FLAGS.eps,
                                                    sanity_checks=False)
            y_pred_pgd_inf = model(x_pgd_inf)
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
            y_pred_pgd_2 = model(x_pgd_2)
            test_acc_pgd_2(y, y_pred_pgd_2)
            print("l2 pgd done", time.time())
            
            # -- carlini wagner --
            '''
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
                for i in range(batch_size):
                    perturbation_to_use = combined_x_tensor[argmin_non_zero_values[i], i, :, :, :]
                    list_best_x_cw.append(perturbation_to_use)
                x_cw = tf.stack(list_best_x_cw)
            else:
                x_cw = carlini_wagner_l2(**cw_args)
            l2_x_cw = l2(x, x_cw)
            non_zero_mask = tf.not_equal(l2_x_cw, 0.0)
            non_zero_values = tf.boolean_mask(l2_x_cw, non_zero_mask)
            
            # Check if non_zero_values is empty
            if tf.size(non_zero_values) == 0:
                # Append 0 when there are no non-zero values
                l2_x_cw_avgs.append(0.0)
            else:
                # Otherwise, calculate the mean and append it
                l2_x_cw_avgs.append(tf.reduce_mean(non_zero_values).numpy().item())
            y_pred_cw = model(x_cw)
            test_acc_cw(y, y_pred_cw)
            print("cw done", time.time())
            

            # -- spsa --
            x_spsa_list = [] # uncomment if you want to view/store the spsa images for some reason.
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
                x_spsa_list.append(x_spsa_single) # uncomment if you want to view/store the spsa images for some reason.
                y_pred_spsa_list.append(model(x_spsa_single))
            x_spsa = tf.concat(x_spsa_list, axis=0) # uncomment if you want to view/store the spsa images for some reason.
            y_pred_spsa = tf.concat(y_pred_spsa_list, axis=0)
            test_acc_spsa(y, y_pred_spsa)
            print("spsa done", time.time())
            
            if check:
                plot_images_in_grid(list_of_xs = [x, x_fgsm, x_pgd_inf, x_pgd_2, x_pgd_1, x_cw, x_spsa], 
                                    row_labels = [y[z] for z in range(10)], 
                                    col_labels = ['Clean','FGSM','PGD Linf','PGD L2','PGD L1','CW', 'SPSA'],
                                    save_path = 'images.jpg',
                                    input_elements = input_elements,
                                    )
                check = False
            batch_size += x.shape[0]
            progress_bar_test.add(x.shape[0], values = [("clean", test_acc_clean.result()),
                                                        #("FGSM", test_acc_fgsm.result()),
                                                        #('PGD L1', test_acc_pgd_1.result()),
                                                        #('PGD L2', test_acc_pgd_2.result()),
                                                        #("PGD Linf", test_acc_pgd_inf.result()), 
                                                        ("CW", test_acc_cw.result()), 
                                                        ("SPSA", test_acc_spsa.result())
                                                        ])

        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        #print("test acc on FGSM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
        #print("test acc on l_inf PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_inf.result() * 100))
        #print("test acc on l_2 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_2.result() * 100))
        #print("test acc on l_1 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_1.result() * 100))
        print("test acc on Carlini Wagner adversarial examples (%): {:.3f}".format(test_acc_cw.result() * 100))
        print("test acc on SPSA adversarial examples (%): {:.3f}".format(test_acc_spsa.result() * 100))

        # Assuming you have already calculated these accuracies
        test_acc_clean_value = test_acc_clean.result().numpy()
        #test_acc_fgsm_value = test_acc_fgsm.result().numpy()
        #test_acc_pgd_inf_value = test_acc_pgd_inf.result().numpy()
        #test_acc_pgd_2_value = test_acc_pgd_2.result().numpy()
        #test_acc_pgd_1_value = test_acc_pgd_1.result().numpy()
        test_acc_cw_value = test_acc_cw.result().numpy()
        test_acc_spsa_value = test_acc_spsa.result().numpy()
        non_zero_l2_x_cw_avg = [element for element in l2_x_cw_avgs if element > 0]
        if len(non_zero_l2_x_cw_avg) == 0:
            l2_x_cw_avg = 0.0
        else:
            l2_x_cw_avg = sum(non_zero_l2_x_cw_avg)/len(non_zero_l2_x_cw_avg)

        # Prepare the data to be written to CSV
        accuracy_data = [test_acc_clean_value, 
                         #test_acc_pgd_1_value,
                         #test_acc_pgd_2_value,
                         #test_acc_fgsm_value,
                         #test_acc_pgd_inf_value,
                         test_acc_cw_value, 
                         test_acc_spsa_value,
                         l2_x_cw_avg,
                         batch_size
                         ]

        # Specify the CSV file name
        csv_file = f'attack_results/{FLAGS.dataset}_{adv_train}_adv/{name}_{FLAGS.dataset}_{FLAGS.eps}_{batch_chunk}_of_{total_batch_chunks}_just_cw_spsa.csv'

        # Write to CSV
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Clean', 
                             #'PGD L1', 
                             #'PGD L2', 
                             #'FGSM', 
                             #'PGD Linf',
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