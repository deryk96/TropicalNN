'''from functions.load_data import ld_mnist, ld_svhn, ld_cifar10  
import math 
# Load data
data, info = ld_mnist()
batch_size = 128  # This should match the batch size used in ld_mnist
total_test_examples = info.splits['test'].num_examples
total_batches = math.ceil(total_test_examples / batch_size)
total_batch_chunks = 5 + 1
batch_chunk = 4

batches_per_chunk = total_batches // total_batch_chunks
start_batch = batch_chunk * batches_per_chunk
end_batch = start_batch + batches_per_chunk if batch_chunk != total_batch_chunks - 1 else total_batches
print(total_test_examples, total_batches, start_batch, end_batch, batches_per_chunk)
# Slicing the test dataset by batches

test_data_subset = data.test.skip(start_batch).take(batches_per_chunk)
i = 0
for x,y in test_data_subset:
    print(i)
    i += 1
print(test_data_subset)
'''

'''
import numpy as np
import tensorflow as tf
from functions.load_data import ld_mnist
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from edited_cleverhans.edited_carlini_wagner_l2 import carlini_wagner_l2
model_path = 'saved_models/CH_TropConv3LayerLogits_mnist_0.05_1_False'
model_path = 'saved_models/CH_ReluConv3Layer_mnist_0.05_1_False'
model = tf.keras.models.load_model(model_path)
data, info = ld_mnist()
i =0
for x, y in data.train:
    print('#####', tf.reduce_min(tf.reduce_min(model(x), axis=1), axis=0))
    print('#####', tf.reduce_max(tf.reduce_max(model(x), axis=1), axis=0))
    x_pgd_2 = projected_gradient_descent(model, x, 0.2, 0.01, 50, np.inf)
    print('#####', tf.reduce_min(tf.reduce_min(model(x_pgd_2), axis=1), axis=0))
    print('#####', tf.reduce_max(tf.reduce_max(model(x_pgd_2), axis=1), axis=0))
    x_cw = carlini_wagner_l2(model, x, 
                        clip_min=-1.0, 
                        max_iterations=100, 
                        binary_search_steps=9,
                        confidence=0,
                        initial_const=1e-2,
                        learning_rate=1e-2,
                        loss_add = 0)
    print('#####', tf.reduce_min(tf.reduce_min(model(x_cw), axis=1), axis=0))
    print('#####', tf.reduce_max(tf.reduce_max(model(x_cw), axis=1), axis=0))
    i += 1
    if i != 0:
        break
'''
from functions.load_data import ld_mnist, ld_cifar10, ld_svhn
import tensorflow as tf
data, info = ld_mnist()
i =0
for x, y in data.test:
    if i == 0:
        print(y)
        max_class = tf.reduce_max(y)
        random_tensor = tf.random.uniform(shape=y.shape, minval=0, maxval=max_class+1, dtype=tf.int64)
        random_tensor = tf.where(random_tensor == y, (random_tensor + 1) % (max_class+1), random_tensor)
        print(random_tensor)
        print(y == random_tensor)
    i += 1
print('mnist', i)

data, info = ld_cifar10()
i =0
for x, y in data.test:
    if i == 0:
        print(y.shape)
    i += 1
print('cifar', i)

data, info = ld_svhn()
i =0
for x, y in data.test:
    if i == 0:
        print(y.shape)
    i += 1
print('svhn', i)

