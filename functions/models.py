from custom_layers.tropLayers import TropEmbedMaxMin
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time

# - relu - 
def buildReLuModel(x_train, y_train, layer_size = 100, verbose = 0):
    start_time = time.time()
    num_predictors = x_train.shape[1]
    num_epochs = 10
    initializer = RandomNormal(mean=0.5, stddev=1., seed=0)
    model = Sequential([Dense(layer_size, input_shape=(num_predictors,), activation="relu",  kernel_initializer=initializer),
                        Dense(1, activation="sigmoid",  kernel_initializer=initializer)])
    model.compile(optimizer=Adam(0.1),loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ReLU model built. Elapsed time: {elapsed_time:.2f} seconds /// {elapsed_time/60:.2f} minutes.")
    
    return model


# - tropical - 
def buildTropicalModel(x_train, y_train, layer_size = 100, verbose = 0):
    start_time = time.time() 
    num_predictors = x_train.shape[1]
    num_epochs = 10
    initializer = RandomNormal(mean=0.5, stddev=1., seed=0)
    model = Sequential([TropEmbedMaxMin(layer_size, num_predictors),
                        Dense(1, activation="sigmoid",  kernel_initializer=initializer)])
    model.compile(optimizer=Adam(0.1), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=num_epochs, verbose=verbose)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tropical model built. Elapsed time: {elapsed_time:.2f} seconds /// {elapsed_time/60:.2f} minutes.")
    return model