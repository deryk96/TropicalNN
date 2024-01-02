import math
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
from functions.load_data import load_CIFAR_data

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
#from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from edited_carlini_wagner_l2 import carlini_wagner_l2
#from cleverhans.tf2.attacks.spsa import spsa
from edited_spsa import spsa

FLAGS = flags.FLAGS

def ld_cifar10():
    """Load training and test data."""

    def convert_types(image, label):
        image = tf.cast(image, tf.float32)
        image /= 127.5
        image -= 1.0
        return image, label

    dataset, info = tfds.load("cifar10", with_info=True, as_supervised=True)

    def augment_mirror(x):
        return tf.image.random_flip_left_right(x)

    def augment_shift(x, w=4):
        y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode="REFLECT")
        return tf.image.random_crop(y, tf.shape(x))

    cifar10_train, cifar10_test = dataset["train"], dataset["test"]
    # Augmentation helps a lot in CIFAR10
    cifar10_train = cifar10_train.map(
        lambda x, y: (augment_mirror(augment_shift(x)), y)
    )
    cifar10_train = cifar10_train.map(convert_types).shuffle(10000).batch(128)
    cifar10_test = cifar10_test.map(convert_types).batch(128)

    return EasyDict(train=cifar10_train, test=cifar10_test)


def main(_):
    # Load training and test data
    data = ld_cifar10()
    #model = CH_TropConv3LayerLogits(num_classes=10)
    #model = CH_ReLU_ResNet50(num_classes=10)
    #model = CH_Trop_ResNet50(num_classes=10)
    #model = CH_MaxoutConv3Layer(num_classes=10)
    #model = CH_ReluConv3Layer(num_classes=10)
    #CH_MaxOut_ResNet50(num_classes=10), 
    models = [CH_ReluConv3Layer(num_classes=10)]

    for model in models:
        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Metrics to track the different accuracies.
        train_loss = tf.metrics.Mean(name="train_loss")
        train_acc = tf.metrics.SparseCategoricalAccuracy()
        test_acc_clean = tf.metrics.SparseCategoricalAccuracy()
        test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_inf = tf.metrics.SparseCategoricalAccuracy()
        test_acc_pgd_2 = tf.metrics.SparseCategoricalAccuracy()
        test_acc_cw = tf.metrics.SparseCategoricalAccuracy()
        test_acc_spsa = tf.metrics.SparseCategoricalAccuracy()

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = loss_object(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_acc(y, predictions)

        # Train model with adversarial training
        for epoch in range(FLAGS.nb_epochs):
            #train_acc = tf.metrics.SparseCategoricalAccuracy()
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(50000)
            print(f"--epoch {epoch}--")
            for (x, y) in data.train:
                if FLAGS.adv_train:
                    # Replace clean example with adversarial example for adversarial training
                    #x = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
                    x = fast_gradient_method(model, x, FLAGS.eps, np.inf)
                train_step(x, y)
                progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result()), ("acc", train_acc.result())])
        model.summary()
        #model.save(f'CH_MaxoutConv3Layer_{FLAGS.eps}_{FLAGS.nb_epochs}_{FLAGS.adv_train}', save_format='tf')

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
            
            progress_bar_test.add(x.shape[0])

        print("test acc on clean examples (%): {:.3f}".format(test_acc_clean.result() * 100))
        print("test acc on FGM adversarial examples (%): {:.3f}".format(test_acc_fgsm.result() * 100))
        print("test acc on l_inf PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_inf.result() * 100))
        #print("test acc on l_2 PGD adversarial examples (%): {:.3f}".format(test_acc_pgd_2.result() * 100))
        print("test acc on Carlini Wagner adversarial examples (%): {:.3f}".format(test_acc_cw.result() * 100))
        print("test acc on SPSA adversarial examples (%): {:.3f}".format(test_acc_spsa.result() * 100))


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 1, "Number of epochs.")
    flags.DEFINE_float("eps", 0.05, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    app.run(main)