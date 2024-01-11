import csv
import math
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling2D, Dense, Flatten

from custom_layers.tropical_layers import TropEmbedMaxMinLogits, ChangeSignLayer, TropEmbedMaxMin
from functions.models import CH_ReluConv3Layer, CH_ReLU_ResNet50, CH_Trop_ResNet50, CH_TropConv3LayerLogits, CH_MaxoutConv3Layer, CH_MaxOut_ResNet50
from functions.load_data import ld_mnist, ld_svhn, ld_cifar10

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
#from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from edited_cleverhans.edited_carlini_wagner_l2 import carlini_wagner_l2
#from cleverhans.tf2.attacks.spsa import spsa
from edited_cleverhans.edited_spsa import spsa

FLAGS = flags.FLAGS


def main(_):
    # Load data
    if FLAGS.dataset == "mnist":
        data, info = ld_mnist()
    elif FLAGS.dataset == "svhn":
        data, info = ld_svhn()
    else:
        data, info = ld_cifar10()
    
    cw_targeted = True

    batch_chunk = int(sys.argv[1])
    total_batch_chunks = int(sys.argv[2])

    batch_size = 128  
    total_test_examples = info.splits['test'].num_examples
    total_batches = math.ceil(total_test_examples / batch_size)

    batches_per_chunk = total_batches // (total_batch_chunks)
    start_batch = batch_chunk * batches_per_chunk

    # Slicing the test dataset by batches
    test_data_subset = data.test.skip(start_batch).take(batches_per_chunk)

    model_paths = {'CH_TropConv3LayerLogits': 'saved_models/CH_TropConv3LayerLogits_mnist_0.1_100_False',
                   'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_mnist_0.1_100_False',
                   'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_mnist_0.1_100_False'}

    for name, model_path in model_paths.items():
        model = tf.keras.models.load_model(model_path)

        test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
        test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_inf = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_2 = tf.metrics.SparseCategoricalAccuracy()
        test_acc_cw = tf.metrics.SparseCategoricalAccuracy()
        test_acc_spsa = tf.metrics.SparseCategoricalAccuracy()

        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(total_test_examples)

        for x, y in test_data_subset: #data.test:
            # -- clean --
            y_pred = model(x)
            test_acc_clean(y, y_pred)

            # -- fast gradient sign method --
            x_fgm = fast_gradient_method(model, x, FLAGS.eps, np.inf)
            y_pred_fgm = model(x_fgm)
            test_acc_fgsm(y, y_pred_fgm)

            # -- l_inf || projected gradient descent --
            x_pgd_inf = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
            y_pred_pgd_inf = model(x_pgd_inf)
            test_acc_pgd_inf(y, y_pred_pgd_inf)
            
            # -- l_2 || projected gradient descent --
            x_pgd_2 = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, 2)
            y_pred_pgd_2 = model(x_pgd_2)
            test_acc_pgd_2(y, y_pred_pgd_2)
           
            # -- spsa --
            #x_spsa_list = []
            y_pred_spsa_list = []

            for i in range(x.shape[0]):
                x_single = x[i:i+1]
                x_spsa_single = spsa(model, x=x_single, y=y[i], eps=FLAGS.eps, 
                                nb_iter=100, learning_rate=0.01, delta=0.01, 
                                spsa_samples=128, spsa_iters=1,
                                clip_min=-1, clip_max=1)
                #x_spsa_list.append(x_spsa_single)
                y_pred_spsa_single = model(x_spsa_single)
                y_pred_spsa_list.append(y_pred_spsa_single)
            #x_spsa = tf.concat(x_spsa_list, axis=0)
            y_pred_spsa = tf.concat(y_pred_spsa_list, axis=0)
            test_acc_spsa(y, y_pred_spsa)
            
            
            # -- carlini wagner --
            if cw_targeted:
                max_class = tf.reduce_max(y)
                random_tensor = tf.random.uniform(shape=y.shape, minval=0, maxval=max_class+1, dtype=tf.int64)
                random_tensor = tf.where(random_tensor == y, (random_tensor + 1) % (max_class+1), random_tensor)
            else:
                random_tensor = None
            x_cw = carlini_wagner_l2(model, x, y = random_tensor,
                                    abort_early=True, 
                                    clip_min=-1.0, 
                                    max_iterations=10000, 
                                    binary_search_steps=9,
                                    confidence=0,
                                    initial_const=1e-2,
                                    learning_rate=1e-2,
                                    loss_add = 50)
            y_pred_cw = model(x_cw)
            test_acc_cw(y, y_pred_cw)
            
            progress_bar_test.add(x.shape[0], values = [("clean", test_acc_clean.result()), ("FGSM", test_acc_fgsm.result()), ("PGD", test_acc_pgd_inf.result()), ("CW", test_acc_cw.result()), ("SPSA", test_acc_spsa.result())])

        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        print("test acc on FGSM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
        print("test acc on l_inf PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_inf.result() * 100))
        print("test acc on l_2 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_2.result() * 100))
        print("test acc on Carlini Wagner adversarial examples (%): {:.3f}".format(test_acc_cw.result() * 100))
        print("test acc on SPSA adversarial examples (%): {:.3f}".format(test_acc_spsa.result() * 100))

        # Assuming you have already calculated these accuracies
        test_acc_clean_value = test_acc_clean.result().numpy()
        test_acc_fgsm_value = test_acc_fgsm.result().numpy()
        test_acc_pgd_inf_value = test_acc_pgd_inf.result().numpy()
        test_acc_pgd_2_value = test_acc_pgd_2.result().numpy()
        test_acc_cw_value = test_acc_cw.result().numpy()
        test_acc_spsa_value = test_acc_spsa.result().numpy()

        # Prepare the data to be written to CSV
        accuracy_data = [test_acc_clean_value, test_acc_fgsm_value, test_acc_pgd_inf_value, 
                        test_acc_pgd_2_value, test_acc_cw_value, test_acc_spsa_value]

        # Specify the CSV file name
        csv_file = f'attack_results/{name}_{FLAGS.dataset}_{FLAGS.eps}_{batch_chunk}_of_{total_batch_chunks}.csv'

        # Write to CSV
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Test Accuracy Clean', 'Test Accuracy FGSM', 'Test Accuracy PGD Inf', 
                            'Test Accuracy PGD 2', 'Test Accuracy CW', 'Test Accuracy SPSA'])
            writer.writerow(accuracy_data)

        print(f"Accuracies written to {csv_file}")


if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    flags.DEFINE_float("eps", 0.1, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    flags.DEFINE_string("dataset", "mnist", "Specifies dataset used to train the model.")
    app.run(main)