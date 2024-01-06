from functions.load_data import ld_mnist, ld_svhn, ld_cifar10  
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