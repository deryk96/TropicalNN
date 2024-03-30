from custom_layers.tropical_layers import TropConv2D, TropConv2DMax, TropEmbedMaxMinLogits, SoftminLayer, ChangeSignLayer, SoftmaxLayer, TropRegDecreaseDistance
from custom_layers.initializers import BimodalNormalInitializer
from tensorflow.keras import Sequential, Model, initializers, models
from tensorflow.keras.layers import Dense, Concatenate, MaxPooling2D, MaxPooling1D, AveragePooling1D, Activation, Flatten, Conv2D, Dropout, Input, BatchNormalization, ReLU, Add, AveragePooling2D, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow import reduce_sum, reduce_max, subtract, reshape, convert_to_tensor, shape, cond, cast, concat
import time
import tensorflow as tf
import numpy as np

from tensorflow import reshape, expand_dims, reduce_max ,reduce_min, float32, transpose, shape, ones, bool, exp, boolean_mask, zeros, concat, add, fill, constant
from tensorflow.math import top_k, reduce_sum, exp, logical_not, maximum, minimum, scalar_mul
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import repeat_elements
from tensorflow.keras import initializers, regularizers
from tensorflow.image import extract_patches
from tensorflow.linalg import band_part
from keras.constraints import NonNeg

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer
from tensorflow.keras.models import Sequential, Model
import time
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from functions.load_data import ld_mnist, ld_svhn, ld_cifar10  


class TropEmbedMaxMin(Layer):
    '''
    Custom TensorFlow layer implementing Tropical Embedding for max-min distances.
    '''

    def __init__(self, units=2, initializer_w=initializers.random_normal, lam=0.0, axis_for_reduction=2, **kwargs):
        '''
        Initializes the TropEmbedMaxMin layer.

        Parameters
        ----------
        units : int, optional
            Number of output units (default is 2).
        initializer_w : initializer function, optional
            Weight initializer function (default is random_normal).
        lam : float, optional
            Regularization parameter (default is 0.0).
        axis_for_reduction : int, optional
            Axis for reduction in distance calculation (default is 2).
        **kwargs : dict
            Additional keyword arguments.
        '''
        super(TropEmbedMaxMin, self).__init__(**kwargs)
        self.units = units
        self.initializer_w = initializer_w
        self.lam = lam
        self.axis_for_reduction = axis_for_reduction

    def build(self, input_shape):
        input_dim = input_shape[-1]  # Extract the last dimension from input_shape
        self.w = self.add_weight(name='tropical_fw',
                                 shape=(self.units, input_dim),
                                 initializer=self.initializer_w,
                                 kernel_regularizer=TropRegDecreaseDistance(lam=self.lam),
                                 trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer="zeros",
                                    trainable=True)
        super(TropEmbedMaxMin, self).build(input_shape)

    def call(self, x):
        '''
        Performs the forward pass of the TropEmbedMaxMin layer.

        Parameters
        ----------
        x : tensorflow tensor object
            Input tensor.

        Returns
        -------
        trop_distance : tensorflow tensor object
            Output tensor after applying Tropical Embedding for max-min distances.
        '''
        x_reshaped = reshape(x, [-1, 1, self.w.shape[-1]])  # Reshape input data
        x_for_broadcast = repeat_elements(x_reshaped, self.units, 1)  # Repeat input for broadcasting
        result_addition = x_for_broadcast + self.w  # Calculate addition of input and weights
        trop_distance = reduce_max(result_addition, axis=(self.axis_for_reduction)) - reduce_min(result_addition, axis=(self.axis_for_reduction)) + self.bias  # Calculate tropical distances with bias
        return trop_distance

    def get_config(self):
        '''
        Gets the configuration of the layer.

        Returns
        -------
        config : dict
            Configuration of the layer.
        '''
        config = {
            'units': self.units,
            'initializer_w': initializers.serialize(self.initializer_w),
            'lam': self.lam
        }
        base_config = super(TropEmbedMaxMin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        '''
        Creates a layer from its config.

        Parameters
        ----------
        config : dict
            Configuration of the layer.

        Returns
        -------
        cls : TropEmbedMaxMin object
            Instantiated TropEmbedMaxMin object with given configuration.
        '''
        return cls(**config)

class CH_TropConv3LayerLogits(Model):
    def __init__(self, num_classes, initializer_w=initializers.RandomNormal(0.5, 1), lam=1):
        super(CH_TropConv3LayerLogits, self).__init__()
        self.num_classes = num_classes
        self.initializer_w = initializer_w
        self.lam = lam

        self._build_model()

    def _build_model(self):
        self.conv_layers = Sequential([
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten()
        ])

        self.dense_layer = Dense(64, activation='relu', name='dense')
        self.trop_act = TropEmbedMaxMin(self.num_classes, initializer_w=self.initializer_w, lam=self.lam)
        self.change_sign = ChangeSignLayer()
        #self.final_layer = Activation('softmax')

    def call(self, inputs, training=True):
        x = self.conv_layers(inputs)
        x = self.dense_layer(x)
        x = self.trop_act(x)
        logits = self.change_sign(x)
        #logits = self.final_layer(x)
        return logits

    def compile_model(self, loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']):
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit_model(self, x_train, y_train, num_epochs=10, batch_size=64, verbose=1):
        start_time = time.time()
        self.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=verbose)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model built. Elapsed time: {elapsed_time:.2f} seconds | {elapsed_time / 60:.2f} minutes.")

        return self

batch_size = 64
data, info = ld_mnist(batch_size=batch_size)
eps = 0.2 
adv_train = False
nb_epochs = 100
model = CH_TropConv3LayerLogits(num_classes=10, lam=100000000)
name = "Tropicalboi"

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
for epoch in range(nb_epochs):
    # keras like display of progress
    progress_bar_train = tf.keras.utils.Progbar(info.splits['train'].num_examples)
    print(f"--epoch {epoch}--")
    for (x, y) in data.train:
        if adv_train:
            # Replace clean example with adversarial example for adversarial training
            
            #l_inf
            x = projected_gradient_descent(model_fn = model,
                                            x = x,
                                            eps = eps,
                                            eps_iter = 0.01,
                                            nb_iter = 40,
                                            norm = np.inf,
                                            loss_fn = None,
                                            clip_min = -1.0,
                                            clip_max = 1.0,
                                            y = None,
                                            targeted = False,
                                            rand_init = True,
                                            rand_minmax = eps,
                                            sanity_checks=False)
        train_step(x, y)
        progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result()), ("acc", train_acc.result())])
'''
@tf.function
def train_step(x, y, model, loss_object, optimizer, train_loss, train_acc):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)  # Ensure `training=True` is set to enable training-specific behavior
        primary_loss = loss_object(y, predictions)
        # Calculate regularization loss
        regularization_loss = tf.add_n(model.losses)  # This automatically sums all regularization losses in the model
        print(regularization_loss)
    # Compute the total loss
    total_loss = primary_loss + regularization_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(total_loss)
    train_acc(y, predictions)
    return primary_loss, regularization_loss  # Return both losses for logging

# Then, modify your training loop to log the regularization loss
start = time.time()
for epoch in range(nb_epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_acc.reset_states()
    progress_bar_train = tf.keras.utils.Progbar(target=info.splits['train'].num_examples, unit_name='samples')
    print(f"--epoch {epoch}--")
    for (x, y) in data.train:
        primary_loss, reg_loss = train_step(x, y, model, loss_object, optimizer, train_loss, train_acc)
        progress_bar_train.add(x.shape[0], values=[("loss", primary_loss), ("reg_loss", reg_loss), ("acc", train_acc.result())])
'''
        
elapsed = time.time() - start
print(f'##### training time = {elapsed} seconds | {elapsed/60} minutes')

elapsed = time.time() - start
print(f'##### training time = {elapsed} seconds | {elapsed/60} minutes')
model.summary()
model.save(f'saved_models/{name}_MNIST_{nb_epochs}_{adv_train}', save_format='tf')
