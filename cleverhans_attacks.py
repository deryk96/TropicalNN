import math
import argparse
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
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_chunk', type=int, default=0, help='Batch chunk to process')
    parser.add_argument('--total_batch_chunks', type=int, default=1, help='Total number of batch chunks')
    args = parser.parse_args()

    # Load data
    if FLAGS.dataset == "mnist":
        data, info = ld_mnist()
        models = {'CH_ReluConv3Layer': CH_ReluConv3Layer(num_classes=10),
                  'CH_TropConv3LayerLogits': CH_TropConv3LayerLogits(num_classes=10),
                  'CH_MaxoutConv3Layer': CH_MaxoutConv3Layer(num_classes=10)}
    elif FLAGS.dataset == "svhn":
        data, info = ld_svhn()
        models = {'CH_ReluConv3Layer': CH_ReluConv3Layer(num_classes=10),
                  'CH_TropConv3LayerLogits': CH_TropConv3LayerLogits(num_classes=10),
                  'CH_MaxoutConv3Layer': CH_MaxoutConv3Layer(num_classes=10)}
    else:
        data, info = ld_cifar10()
        models = {'CH_ReLU_ResNet50': CH_ReLU_ResNet50(num_classes=10),
                  'CH_Trop_ResNet50': CH_Trop_ResNet50(num_classes=10),
                  'CH_MaxOut_ResNet50': CH_MaxOut_ResNet50(num_classes=10)}
    batch_size = 128  
    total_test_examples = info.splits['test'].num_examples
    total_batches = math.ceil(total_test_examples / batch_size)

    batches_per_chunk = total_batches // args.total_batch_chunks
    start_batch = args.batch_chunk * batches_per_chunk

    # Slicing the test dataset by batches
    test_data_subset = data.test.skip(start_batch).take(batches_per_chunk)

    model_paths = ['saved_models/CH_TropConv3LayerLogits_0.05_1_False',
              'saved_models/CH_ReluConv3Layer_0.05_1_False']

    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)

        test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
        test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_inf = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_2 = tf.metrics.SparseCategoricalAccuracy()
        test_acc_cw = tf.metrics.SparseCategoricalAccuracy()
        test_acc_spsa = tf.metrics.SparseCategoricalAccuracy()

        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(10000)

        for x, y in data.test:
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
            '''
            # -- l_2 || projected gradient descent --
            x_pgd_2 = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, 2)
            y_pred_pgd_2 = model(x_pgd_2)
            test_acc_pgd_2(y, y_pred_pgd_2)
            '''
            # -- spsa --
            #x_spsa_list = []
            y_pred_spsa_list = []

            for i in range(x.shape[0]):
                x_single = x[i:i+1]
                x_spsa_single = spsa(model, x=x_single, y=y[i], eps=FLAGS.eps, 
                                nb_iter=2, learning_rate=0.01, delta=0.01, 
                                spsa_samples=12, spsa_iters=1,
                                clip_min=-1, clip_max=1)
                #x_spsa_list.append(x_spsa_single)
                y_pred_spsa_single = model(x_spsa_single)
                y_pred_spsa_list.append(y_pred_spsa_single)
            #x_spsa = tf.concat(x_spsa_list, axis=0)
            y_pred_spsa = tf.concat(y_pred_spsa_list, axis=0)
            test_acc_spsa(y, y_pred_spsa)
            
            
            # -- carlini wagner --
            x_cw = carlini_wagner_l2(model, x, 
                                    clip_min=-1.0, 
                                    max_iterations=10, 
                                    binary_search_steps=2,
                                    confidence=0,
                                    initial_const=1e-2,
                                    learning_rate=1e-2)
            y_pred_cw = model(x_cw)
            test_acc_cw(y, y_pred_cw)
            
            progress_bar_test.add(x.shape[0], values = [("clean", test_acc_clean.result()), ("FGSM", test_acc_fgsm.result()), ("PGD", test_acc_pgd_inf.result()), ("CW", test_acc_cw.result()), ("SPSA", test_acc_spsa.result())])

        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        print("test acc on FGSM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
        print("test acc on l_inf PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_inf.result() * 100))
        #print("test acc on l_2 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_2.result() * 100))
        print("test acc on Carlini Wagner adversarial examples (%): {:.3f}".format(test_acc_cw.result() * 100))
        print("test acc on SPSA adversarial examples (%): {:.3f}".format(test_acc_spsa.result() * 100))


if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    flags.DEFINE_integer("nb_epochs", 1, "Number of epochs.")
    flags.DEFINE_float("eps", 0.05, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    flags.DEFINE_string("dataset", "mnist", "Specifies dataset used to train the model.")
    app.run(main)