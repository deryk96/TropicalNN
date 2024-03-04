import math
import time
import numpy as np
import tensorflow as tf

from absl import app, flags

from functions.models import CH_MMRReluConv3Layer, CH_MMRReLU_ResNet50
from functions.load_data import ld_mnist, ld_svhn, ld_cifar10  

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method

from custom_layers.mmr_regularizer import mmr_cnn

FLAGS = flags.FLAGS

batch_size = 32

def main(_):
    # Load training and test data
    if FLAGS.dataset == "mnist":
        data, info = ld_mnist(batch_size = batch_size)
        input_shape = (batch_size,28,28,1)
        models = {'CH_MMRReluConv3Layer': CH_MMRReluConv3Layer(num_classes=10)}
        lr = 5e-4
    elif FLAGS.dataset == "svhn":
        data, info = ld_svhn(batch_size = batch_size)
        input_shape = (batch_size,32,32,3)
        models = {'CH_MMRReluConv3Layer': CH_MMRReluConv3Layer(num_classes=10)}
        lr = 1e-3
    else:
        data, info = ld_cifar10(batch_size = batch_size)
        input_shape = (batch_size,32,32,3)
        models = {'CH_MMRReLU_ResNet50': CH_MMRReLU_ResNet50(num_classes=10)}
        lr = 1e-3

    for name, model in models.items():
        model.build(input_shape=(None, 32, 32, 3))       

        loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Metrics to track the different accuracies.
        train_loss = tf.metrics.Mean(name="train_loss")
        train_acc = tf.metrics.SparseCategoricalAccuracy()

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
        for epoch in range(FLAGS.nb_epochs):
            epoch_start_reduced_lr = 0.9
            lr_actual = lr / 10 if epoch >= epoch_start_reduced_lr * FLAGS.nb_epochs else lr

            frac_reg = min(epoch / 10.0, 1.0)  # from 0 to 1 linearly over the first 10 epochs

            frac_start, frac_end = hyp_start, hyp_end  # decrease the number of linear region hyperplanes from 10% to 2%
            n_db = n_out  # the number of decision boundary hyperplanes is always the same (the number of classes)
            n_total_hidden_units = 51648 #MNIST #utils.get_n_total_hidden_units(hps.nn_type, hps.n_hs, hps.height)
            n_rb_start, n_rb_end = int(frac_start * n_total_hidden_units), int(frac_end * n_total_hidden_units)
            n_rb = (n_rb_end - n_rb_start) / FLAGS.nb_epochs * epoch + n_rb_start
            
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(info.splits['train'].num_examples)
            print(f"--epoch {epoch}--")
            for (x, y) in data.train:
                if FLAGS.adv_train:
                    # Replace clean example with adversarial example for adversarial training
                    #x = projected_gradient_descent(model, x, FLAGS.eps, 0.01, 40, np.inf)
                    x = fast_gradient_method(model, x, FLAGS.eps, np.inf)
                if epoch==0:
                    boo_first_go = True
                else:
                    boo_first_go = False
                train_step(x, y, frac_reg_tf = frac_reg, n_rb_tf=n_rb, n_db_tf=n_db, boo_first_go = boo_first_go)
                progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result()), ("acc", train_acc.result())])
        elapsed = time.time() - start
        print(f'##### training time = {elapsed} seconds | {elapsed/60} minutes')
        model.summary()
        model.save(f'saved_models/{name}_{FLAGS.dataset}_{FLAGS.eps}_{FLAGS.nb_epochs}_{FLAGS.adv_train}', save_format='tf')

if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    flags.DEFINE_float("eps", 4/255, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool("adv_train", False, "Use adversarial training (on PGD adversarial examples).")
    flags.DEFINE_string("dataset", "svhn", "Specifies dataset used to train the model.")
    app.run(main)