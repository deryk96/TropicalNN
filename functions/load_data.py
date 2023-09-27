from tensorflow.keras.utils import to_categorical
from scipy.io import loadmat
import numpy as np
import pickle
import os
import struct
from array import array


def shuffle_data(x_train, x_test, y_train, y_test):
    train_size = x_train.shape[0]

    # - combine the data - 
    combined_data = np.vstack((x_train, x_test))
    combined_labels = np.concatenate((y_train, y_test))

    # - shuffle the combined data and labels in the same order -
    shuffled_indices = np.arange(combined_data.shape[0])
    np.random.shuffle(shuffled_indices)

    shuffled_data = combined_data[shuffled_indices]
    shuffled_labels = combined_labels[shuffled_indices]

    # - split the shuffled arrays back into two separate arrays -
    x_train = shuffled_data[:train_size]
    x_test = shuffled_data[train_size:]
    y_train = shuffled_labels[:train_size]
    y_test = shuffled_labels[train_size:]

    return x_train, x_test, y_train, y_test


def filter_arrays(x, y, desired_classes):
    mask = np.isin(y, desired_classes)
    x = x[mask]
    y = y[mask]
    x = np.divide(x, 255)
    
    if len(desired_classes) == 2:
        y[y==desired_classes[0]] = 0
        y[y==desired_classes[1]] = 1
    else:
        y = to_categorical(y, len(desired_classes))

    return x, y


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def load_CIFAR_data(desired_classes = [7,8]):
    '''
    --- CIFAR 10 data pictures ---
    --- sourced from here: http://www.cs.toronto.edu/~kriz/cifar.html  ---
    '''
    script_dir = os.path.dirname(__file__)
    data_file_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'CIFAR'))
    
    # - get training data - 
    batch1 = unpickle(data_file_path + '\\data_batch_1')
    batch2 = unpickle(data_file_path + '\\data_batch_2')
    batch3 = unpickle(data_file_path + '\\data_batch_3')
    batch4 = unpickle(data_file_path + '\\data_batch_4')
    batch5 = unpickle(data_file_path + '\\data_batch_5')
    x_train = np.concatenate((batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']), axis=0)
    y_train = np.array(batch1[b'labels'] + batch2[b'labels'] + batch3[b'labels'] + batch4[b'labels'] + batch5[b'labels'])

    # - get testing data -
    test = unpickle(data_file_path + '\\test_batch')
    x_test = test[b'data']
    y_test = np.array(test[b'labels'])

    # - cleanAndSort the data - 
    x_train, y_train = filter_arrays(x_train, y_train, desired_classes)
    x_test, y_test = filter_arrays(x_test, y_test, desired_classes)
    
    return x_train, x_test, y_train, y_test


def read_mat_files(filePath, desired_classes = [1, 4]):
    f = loadmat(filePath)
    x = np.reshape(f['X'], (32 * 32 * 3, f['X'].shape[3])).T
    y = np.subtract(np.squeeze(f['y']), 2)
    y[y==255] = 9
    print(np.unique(y))
    x, y = filter_arrays(x, y, desired_classes)

    return x, y


def load_Google_Digit_Data(desired_classes = [0,1]):
    '''
    --- Stanford Google Images House Number Digits --- 
    --- sourced from here: http://ufldl.stanford.edu/housenumbers/  ---
    '''
    script_dir = os.path.dirname(__file__)
    data_file_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'SVHN'))
    
    y_train = np.load(data_file_path + f'\\SVHN_y_train.npy')
    y_test = np.load(data_file_path + f'\\SVHN_y_test.npy')

    for i in range(40):
        path_train = data_file_path + f'\\SVHN_x_train{i}.npy'
        path_test = data_file_path + f'\\SVHN_x_test{i}.npy'
        if i == 0:
            x_train = np.load(path_train)
            x_test = np.load(path_test)
        elif i < 20:
            x_train = np.concatenate((x_train, np.load(path_train)))
            x_test = np.concatenate((x_test, np.load(path_test)))
        else:
            x_train = np.concatenate((x_train, np.load(path_train)))

    x_train, y_train = filter_arrays(x_train, y_train, desired_classes)
    x_test, y_test = filter_arrays(x_test, y_test, desired_classes)

    #x_train, y_train = read_mat_files(data_file_path + '\\train_32x32.mat', desired_classes)
    #x_test, y_test = read_mat_files(data_file_path + '\\test_32x32.mat', desired_classes)

    return x_train, x_test, y_train, y_test


def read_images_labels(images_filepath, labels_filepath, desired_classes):        
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())        
    
    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())  

    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
    
    x = np.array(images)
    x = x.reshape(x.shape[0], -1)
    y = np.array(labels)
    x, y = filter_arrays(x, y, desired_classes)

    return x, y

     
def load_MNIST_data(desired_classes = [1,4]):
    '''
    MNIST Dataset
    '''
    script_dir = os.path.dirname(__file__)
    data_file_path = os.path.abspath(os.path.join(script_dir, '..', 'data'))
    training_images_filepath = data_file_path + '\\train-images.idx3-ubyte'
    training_labels_filepath = data_file_path + '\\train-labels.idx1-ubyte'
    test_images_filepath = data_file_path + '\\t10k-images.idx3-ubyte'
    test_labels_filepath = data_file_path + '\\t10k-labels.idx1-ubyte'

    x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath, desired_classes)
    x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath, desired_classes)

    return x_train, x_test, y_train, y_test 