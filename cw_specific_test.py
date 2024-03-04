import math
import time
import numpy as np
import tensorflow as tf

from absl import app, flags

from functions.models import CH_ReluConv3Layer, CH_ReLU_ResNet50, CH_Trop_ResNet50, CH_TropConv3LayerLogits, CH_MaxoutConv3Layer, CH_MaxOut_ResNet50
from functions.load_data import ld_mnist, ld_svhn, ld_cifar10  

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

FLAGS = flags.FLAGS

def main(_):
    # Load training and test data
    if FLAGS.dataset == "mnist":
        data, info = ld_mnist()
        models = {#'CH_ReluConv3Layer': CH_ReluConv3Layer(num_classes=10),
                  'CH_TropConv3LayerLogits_ACTIVATED': CH_TropConv3LayerLogits(num_classes=10),
                  #'CH_MaxoutConv3Layer': CH_MaxoutConv3Layer(num_classes=10)
                  }
    elif FLAGS.dataset == "svhn":
        data, info = ld_svhn()
        models = {'CH_TropConv3LayerLogits_ACTIVATED': CH_TropConv3LayerLogits(num_classes=10),
                  #'CH_ReluConv3Layer': CH_ReluConv3Layer(num_classes=10),
                  #'CH_MaxoutConv3Layer': CH_MaxoutConv3Layer(num_classes=10)
                  }
    else:
        data, info = ld_cifar10()
        models = {#'CH_ReLU_ResNet50': CH_ReLU_ResNet50(num_classes=10),
                  'CH_Trop_ResNet50_ACTIVATED': CH_Trop_ResNet50(num_classes=10),
                  #'CH_MaxOut_ResNet50': CH_MaxOut_ResNet50(num_classes=10)
                  }

    for name, model in models.items():
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
        # Train model with adversarial training
        for epoch in range(FLAGS.nb_epochs):
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(info.splits['train'].num_examples)
            print(f"--epoch {epoch}--")
            for (x, y) in data.train:
                if FLAGS.adv_train:
                    # Replace clean example with adversarial example for adversarial training
                    #x = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
                    x = fast_gradient_method(model, x, FLAGS.eps, np.inf)
                train_step(x, y)
                progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result()), ("acc", train_acc.result())])
        elapsed = time.time() - start
        print(f'##### training time = {elapsed} seconds | {elapsed/60} minutes')
        model.summary()
        model.save(f'saved_models/{name}_{FLAGS.dataset}_{FLAGS.eps}_{FLAGS.nb_epochs}_{FLAGS.adv_train}', save_format='tf')

if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    flags.DEFINE_float("eps", 0.1, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    flags.DEFINE_string("dataset", "mnist", "Specifies dataset used to train the model.")
    app.run(main)