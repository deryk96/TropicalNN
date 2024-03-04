import csv
import math
import sys
import numpy as np
import tensorflow as tf
from absl import app, flags

from functions.load_data import ld_mnist, ld_svhn, ld_cifar10
from functions.attacks import l1_projected_gradient_descent, l2_projected_gradient_descent
from functions.utils import  plot_images_in_grid

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
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
        print('argument dataset', arg_dataset)
    else:
        batch_chunk = 0
        total_batch_chunks = 1
        batch_size = 128
        arg_dataset = 'mnist'
    if arg_dataset != FLAGS.dataset:
        FLAGS.dataset = arg_dataset
    num_classes = 10
    cw_targeted = True
    check = True

    # Load data
    if FLAGS.dataset == "mnist":
        data, info = ld_mnist(batch_size = batch_size)
    elif FLAGS.dataset == "svhn":
        data, info = ld_svhn(batch_size = batch_size)
    else:
        data, info = ld_cifar10(batch_size = batch_size)
    
    

    total_test_examples = info.splits['test'].num_examples
    total_batches = math.ceil(total_test_examples / batch_size)

    batches_per_chunk = total_batches // (total_batch_chunks)
    start_batch = batch_chunk * batches_per_chunk

    # Slicing the test dataset by batches
    test_data_subset = data.test.skip(start_batch).take(batches_per_chunk)

    model_paths = {
                   #'CH_ReluConv3Layer_ACTIVATED': 'saved_models/CH_ReluConv3Layer_ACTIVATED_mnist_0.1_100_False',
                   #'CH_TropConv3LayerLogits_ACTIVATED': 'saved_models/CH_TropConv3LayerLogits_ACTIVATED_mnist_0.1_100_False',
                   #'CH_TropConv3LayerLogits_new_func3': 'saved_models/CH_TropConv3LayerLogits_new_func2_mnist_0.1_100_False',
                   #'CH_TropConv3LayerLogits': 'saved_models/CH_TropConv3LayerLogits_mnist_0.1_100_False',
                   'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_mnist_0.1_100_False',
                   #'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_mnist_0.1_100_False',
                    }

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for name, model_path in model_paths.items():
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
        for x, y in test_data_subset: #data.test:
            input_elements = 1
            for i in range(1, len(x.shape)):
                input_elements *= x.shape[i]

            # -- clean --
            y_pred = model(x)
            test_acc_clean(y, y_pred)
            
            # -- l_1 || projected gradient descent --
            x_pgd_1 = l1_projected_gradient_descent(model, 
                                                     x, 
                                                     y, 
                                                     steps = 100, 
                                                     epsilon = (FLAGS.eps**2)*input_elements, 
                                                     eps_iter = 0.01, 
                                                     loss_object = loss_object, 
                                                     x_min = -1.0, 
                                                     x_max = 1.0,
                                                     perc = 99)
            y_pred_pgd_1 = model(x_pgd_1)
            test_acc_pgd_1(y, y_pred_pgd_1)
            
            # -- fast gradient sign method --
            x_fgsm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
            y_pred_fgsm = model(x_fgsm)
            test_acc_fgsm(y, y_pred_fgsm)

            # -- l_inf || projected gradient descent --
            x_pgd_inf = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
            y_pred_pgd_inf = model(x_pgd_inf)
            test_acc_pgd_inf(y, y_pred_pgd_inf)
            
            # -- l_2 || projected gradient descent --
            x_pgd_2 = l2_projected_gradient_descent(model, 
                                                    x, 
                                                    y, 
                                                    steps=100, 
                                                    epsilon= math.sqrt((FLAGS.eps**2)*input_elements), 
                                                    eps_iter = 0.01, 
                                                    loss_object = loss_object, 
                                                    x_min = -1.0, 
                                                    x_max = 1.0)
            y_pred_pgd_2 = model(x_pgd_2)
            test_acc_pgd_2(y, y_pred_pgd_2)

            # -- spsa --
            x_spsa_list = [] # uncomment if you want to view/store the spsa images for some reason.
            y_pred_spsa_list = []

            for i in range(x.shape[0]):
                x_spsa_single = spsa(model, x=x[i:i+1], y=y[i], eps=FLAGS.eps, 
                                nb_iter=500, learning_rate=0.01, delta=0.01, 
                                spsa_samples=128, spsa_iters=1,
                                clip_min=-1, clip_max=1)
                x_spsa_list.append(x_spsa_single) # uncomment if you want to view/store the spsa images for some reason.
                y_pred_spsa_list.append(model(x_spsa_single))
            x_spsa = tf.concat(x_spsa_list, axis=0) # uncomment if you want to view/store the spsa images for some reason.
            y_pred_spsa = tf.concat(y_pred_spsa_list, axis=0)
            test_acc_spsa(y, y_pred_spsa)
            
            # -- carlini wagner --
            if cw_targeted:
                list_cw_x = []
                for i in range(num_classes):
                    y_current = tf.fill(y.shape, i)
                    x_cw_curr = carlini_wagner_l2(model, x, y = y_current,
                                            abort_early=True, 
                                            clip_min=-1.0, 
                                            max_iterations=1000, 
                                            binary_search_steps=9,
                                            confidence=0,
                                            initial_const=1e-2,
                                            learning_rate=5e-2)
                    list_cw_x.append(x_cw_curr)
                list_cw_l2 = []
                for i in range(num_classes):
                    l2_tensor = l2(x, list_cw_x[i])
                    mask_true_class = tf.equal(y, i)
                    l2_tensor = tf.where(mask_true_class, tf.constant(10000, dtype=tf.float32), l2_tensor)
                    mask_no_example_found = tf.equal(l2_tensor, 0.0)
                    l2_tensor = tf.where(mask_no_example_found, tf.constant(9000, dtype=tf.float32), l2_tensor)
                    list_cw_l2.append(l2_tensor)
                combined_x_tensor = tf.stack(list_cw_x)
                argmin_non_zero_values = tf.argmin(tf.stack(list_cw_l2) , axis=0) # Find the indices (dimensions) of the minimum non-zero values
                list_best_x_cw = []
                for i in range(batch_size):
                    perturbation_to_use = combined_x_tensor[argmin_non_zero_values[i], i, :, :, :]
                    list_best_x_cw.append(perturbation_to_use)
                x_cw = tf.stack(list_best_x_cw)
            else:
                x_cw = carlini_wagner_l2(model, x, y = None,
                                        abort_early=True, 
                                        clip_min=-1.0, 
                                        max_iterations=10000, 
                                        binary_search_steps=9,
                                        confidence=0,
                                        initial_const=1e-2,
                                        learning_rate=5e-2)
            l2_x_cw = l2(x, x_cw)
            non_zero_mask = tf.not_equal(l2_x_cw, 0.0)
            non_zero_values = tf.boolean_mask(l2_x_cw, non_zero_mask)
            l2_x_cw_avgs.append(tf.reduce_mean(non_zero_values).numpy().item())
            print(sum(l2_x_cw_avgs)/len(l2_x_cw_avgs))
            y_pred_cw = model(x_cw)
            test_acc_cw(y, y_pred_cw)

            if check:
                plot_images_in_grid(list_of_xs = [x, x_fgsm, x_pgd_inf, x_pgd_2, x_pgd_1, x_cw, x_spsa], 
                                    row_labels = [y[z] for z in range(10)], 
                                    col_labels = ['Clean','FGSM','PGD Linf','PGD L2','PGD L1','CW', 'SPSA'],
                                    save_path = 'images.jpg',
                                    input_elements = input_elements,
                                    )
                check = False
            progress_bar_test.add(x.shape[0], values = [("clean", test_acc_clean.result()),
                                                        ("FGSM", test_acc_fgsm.result()),
                                                        ('PGD L1', test_acc_pgd_1.result()),
                                                        ('PGD L2', test_acc_pgd_2.result()),
                                                        ("PGD Linf", test_acc_pgd_inf.result()), 
                                                        ("CW", test_acc_cw.result()), 
                                                        ("SPSA", test_acc_spsa.result())
                                                        ])

        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        print("test acc on FGSM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
        print("test acc on l_inf PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_inf.result() * 100))
        print("test acc on l_2 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_2.result() * 100))
        print("test acc on l_1 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_1.result() * 100))
        print("test acc on Carlini Wagner adversarial examples (%): {:.3f}".format(test_acc_cw.result() * 100))
        print("test acc on SPSA adversarial examples (%): {:.3f}".format(test_acc_spsa.result() * 100))

        # Assuming you have already calculated these accuracies
        test_acc_clean_value = test_acc_clean.result().numpy()
        test_acc_fgsm_value = test_acc_fgsm.result().numpy()
        test_acc_pgd_inf_value = test_acc_pgd_inf.result().numpy()
        test_acc_pgd_2_value = test_acc_pgd_2.result().numpy()
        test_acc_pgd_1_value = test_acc_pgd_1.result().numpy()
        test_acc_cw_value = test_acc_cw.result().numpy()
        test_acc_spsa_value = test_acc_spsa.result().numpy()
        l2_x_cw_avg = sum(l2_x_cw_avgs)/len(l2_x_cw_avgs)

        # Prepare the data to be written to CSV
        accuracy_data = [test_acc_clean_value, 
                         test_acc_fgsm_value, 
                         test_acc_pgd_inf_value, 
                         test_acc_pgd_2_value,
                         test_acc_pgd_1_value, 
                         test_acc_cw_value, 
                         test_acc_spsa_value,
                         l2_x_cw_avg]

        # Specify the CSV file name
        csv_file = f'attack_results/{name}_{FLAGS.dataset}_{FLAGS.eps}_{batch_chunk}_of_{total_batch_chunks}.csv'

        # Write to CSV
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Clean', 
                             'FGSM', 
                             'PGD Linf', 
                             'PGD L2', 
                             'PGD L1',
                             'CW', 
                             'SPSA',
                             'CW L2 Distortion Avg'])
            writer.writerow(accuracy_data)

        print(f"Accuracies written to {csv_file}")


if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    flags.DEFINE_float("eps", 0.1, "Total epsilon for FGSM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    flags.DEFINE_string("dataset", "mnist", "Specifies dataset used to train the model.")
    app.run(main)