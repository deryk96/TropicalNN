from custom_layers.tropLayers import TropEmbedMaxMin
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import time


def buildReLuModel(x_train, 
                    y_train, 
                    num_epochs = 10,
                    first_layer_size = 100, 
                    verbose = 0, 
                    initializer_w = initializers.RandomNormal(mean=0.5, stddev=1., seed=0),
                    second_layer_size = 1,
                    second_layer_activation = 'sigmoid', 
                    clipnorm = None,
                    clipvalue = None,
                    training_loss = 'binary_crossentropy'):
    start_time = time.time()
    num_predictors = x_train.shape[1]
    initializer_out = initializers.RandomNormal(mean=0.5, stddev=1., seed=0)
    model = Sequential([Dense(first_layer_size, input_shape=(num_predictors,), activation="relu",  kernel_initializer=initializer_w),
                        Dense(second_layer_size, activation=second_layer_activation,  kernel_initializer=initializer_out)])
    model.compile(optimizer=Adam(0.1, clipnorm=clipnorm, clipvalue=clipvalue),loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ReLU model built. Elapsed time: {elapsed_time:.2f} seconds /// {elapsed_time/60:.2f} minutes.")
    return model


def buildTropicalModel(x_train, 
                       y_train, 
                       num_epochs = 10,
                       first_layer_size = 100, 
                       verbose = 0, 
                       initializer_w = initializers.random_normal,
                       second_layer_size = 1,
                       second_layer_activation = 'sigmoid', 
                       clipnorm = None,
                       clipvalue = None,
                       training_loss = 'binary_crossentropy'):
    start_time = time.time() 
    num_predictors = x_train.shape[1] #assumes flattened array input
    initializer_out = initializers.random_normal
    model = Sequential([TropEmbedMaxMin(first_layer_size, num_predictors, initializer_w = initializer_w),
                        Dense(second_layer_size, activation=second_layer_activation,  kernel_initializer=initializer_out)])
    model.compile(optimizer=Adam(0.1, clipnorm=clipnorm, clipvalue=clipvalue), loss=training_loss, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tropical model built. Elapsed time: {elapsed_time:.2f} seconds /// {elapsed_time/60:.2f} minutes.")
    return model