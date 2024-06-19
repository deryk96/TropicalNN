# load_data.py
# Description: This file contains a collection of functions for loading the different datasets in as numpy arrays.
# Author: Kurt Pasque
# Date: October 25, 2023

'''
Module: load_data.py

This module provides a collection of functions for loading the different datasets in as numpy arrays.

Functions:
- shuffle_data : Shuffles data between training and testing sets. Concatenates data then splits, retaining original train/test split size.
- filter_arrays : Internal helper function that filters train and test sets. Allows us to scale our tests if we want to classify 2 up to 10 classes very easily.
- unpickle : Internal helper function to unpickle CIFAR-10 data.
- read_mat_files : (no longer used) Internal helper function to load .mat files from SVHN dataset.
- read_images_labels : Internal helper function to read .idx3-ubyte and .idx1-ubyte data that MNIST is stored in.
- load_CIFAR_data : Reads in training and testing CIFAR-10 data.
- load_SVHN_data : Reads in training and testing SVHN data.
- load_MNIST_data : Reads in training and testing MNIST data.
'''


# - imports -
import numpy as np
import pickle
import os
import struct
from tensorflow.keras.utils import to_categorical
from tensorflow import cast, float32, reduce_any
import tensorflow_datasets as tfds
from easydict import EasyDict
from scipy.io import loadmat
from array import array


'''
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
'''

def convert_types(image, label):
    """Convert image types and normalize to [-1, 1]."""
    image = cast(image, float32)
    image /= 127.5
    image -= 1.0
    return image, label

def filter_classes(dataset, classes):
    """Filter the dataset to include only specified classes."""
    return dataset.filter(lambda image, label: reduce_any([label == cls for cls in classes]))

def ld_mnist(batch_size=128, classes=None):
    """Load MNIST training and test data and filter by specified classes."""
    dataset, info = tfds.load("mnist", with_info=True, as_supervised=True)

    mnist_train, mnist_test = dataset["train"], dataset["test"]
    
    # Apply class filtering if classes are specified
    if classes is not None:
        mnist_train = filter_classes(mnist_train, classes)
        mnist_test = filter_classes(mnist_test, classes)
    
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(batch_size)
    mnist_test = mnist_test.map(convert_types).batch(batch_size)

    return EasyDict(train=mnist_train, test=mnist_test), info

'''
def ld_mnist(batch_size = 128):
    """Load MNIST training and test data."""
    dataset, info = tfds.load("mnist", with_info=True, as_supervised=True)

    mnist_train, mnist_test = dataset["train"], dataset["test"]
    mnist_train = mnist_train.map(convert_types).shuffle(10000).batch(batch_size)
    mnist_test = mnist_test.map(convert_types).batch(batch_size)

    return EasyDict(train=mnist_train, test=mnist_test), info
'''
def ld_svhn(batch_size = 128):
    """Load SVHN training and test data."""
    dataset, info = tfds.load("svhn_cropped", with_info=True, as_supervised=True)

    svhn_train, svhn_test = dataset["train"], dataset["test"]
    svhn_train = svhn_train.map(convert_types).shuffle(10000).batch(batch_size)
    svhn_test = svhn_test.map(convert_types).batch(batch_size)

    return EasyDict(train=svhn_train, test=svhn_test), info

def ld_cifar10(batch_size = 128):
    """Load CIFAR-10 training and test data."""
    dataset, info = tfds.load("cifar10", with_info=True, as_supervised=True)

    cifar10_train, cifar10_test = dataset["train"], dataset["test"]
    cifar10_train = cifar10_train.map(convert_types).shuffle(10000).batch(batch_size)
    cifar10_test = cifar10_test.map(convert_types).batch(batch_size)

    return EasyDict(train=cifar10_train, test=cifar10_test), info

def ld_cifar100(batch_size = 128):
    """Load CIFAR-10 training and test data."""
    dataset, info = tfds.load("cifar100", with_info=True, as_supervised=True)

    cifar100_train, cifar100_test = dataset["train"], dataset["test"]
    cifar100_train = cifar100_train.map(convert_types).shuffle(10000).batch(batch_size)
    cifar100_test = cifar100_test.map(convert_types).batch(batch_size)

    return EasyDict(train=cifar100_train, test=cifar100_test), info


def shuffle_data(x_train, x_test, y_train, y_test):
    '''
    Shuffles data between training and testing sets. Concatenates data then splits, retaining original train/test split size.

    Parameters
    ----------
    x_train : numpy array
        training input data we are to shuffle
    x_test : numpy array
        testing input data we are to shuffle 
    y_train : numpy array
        training label data we are to shuffle
    y_test : numpy array
        testing label data we are to shuffle   

    Returns
    -------
    x_train : numpy array
        shuffled training data
    x_test : numpy array
        shuffled testing data 
    y_train : numpy array
        shuffled training labels
    y_test : numpy array
        shuffled testing labels  
    '''
    train_size = x_train.shape[0] # log size of the training side of the data for re-splitting later

    # - combine the data - 
    combined_data = np.vstack((x_train, x_test)) # stack arrays in sequence vertically (row wise)
    combined_labels = np.concatenate((y_train, y_test)) # join a sequence of arrays along an existing axis

    # - shuffle the combined data and labels in the same order -
    shuffled_indices = np.arange(combined_data.shape[0]) # return evenly spaced values within a given interval. In this case [0, 1, ..., combined_data.shape[0]]
    np.random.shuffle(shuffled_indices) # shuffles numbers from above array build

    shuffled_data = combined_data[shuffled_indices] # using shuffled_indices array as a mask to re-map the training data
    shuffled_labels = combined_labels[shuffled_indices] # using shuffled_indices array as a mask to re-map the label data

    # - split the shuffled arrays back into two separate arrays -
    x_train = shuffled_data[:train_size] # split for new train dataset
    x_test = shuffled_data[train_size:] # split for new test dataset
    y_train = shuffled_labels[:train_size] # split for new train labels
    y_test = shuffled_labels[train_size:] # split for new test label

    return x_train, x_test, y_train, y_test


def filter_arrays(x, y, desired_classes):
    '''
    Filters train and test sets by desired_classes hyperparameter. Allows us to scale our tests if we want to classify 2 up to 10 classes very easily.

    Parameters
    ----------
    x : numpy array
        input data
    y : numpy array
        label data
    desired_classes : list
        list of classes to filter x and y for

    Returns
    -------
    x : numpy array
        filtered input data
    y : numpy array
        filtered label data
    '''
    # - fiter the data -
    mask = np.isin(y, desired_classes) # boolean mask where True's are in desired_classes and False's are not
    x = x[mask] # apply mask to x
    y = y[mask] # apply mask to y
    
    # - re-scale the input data - 
    x = np.divide(x, 255) # divide x data by 255. Puts input data on scale of [0,1]
    x = np.subtract(x, 0.5) # subtract 0.5 to put input on scale of [-.5, .5]

    # - format y based on length of desired_classes - 
    if len(desired_classes) == 2: # if only 2 classes given
        y[y==desired_classes[0]] = 0 # make first class 0 (aribtrary choice for 0 but will be used for logisitic regression calculations)
        y[y==desired_classes[1]] = 1 # make second class 1 (aribtrary choice for 1 but will be used for logisitic regression calculations)
    else: # if > 2 (technically allows for a list of 1... might shore that case up later, but okay for now)
        y = to_categorical(y, len(desired_classes)) # transform label data to categorical tensor

    return x, y


def unpickle(file_path):
    '''
    Simple function to unpickle a file given its file path. Cleans up unpickling of the 6 CIFAR batches. 

    Parameters
    ----------
    file_path : str
        file path for pickle file

    Returns
    -------
    dict_data : dict
        dictionary of the unpickled file data
    '''
    with open(file_path, 'rb') as f: # open file with 'rb' or read binary argument
        dict_data = pickle.load(f, encoding='bytes') # use pickle.load to read data within file

    return dict_data


def read_mat_files(file_path, desired_classes = [1, 4]):
    '''
    Special function to read .mat files. We ended up converting all the .mat files (from SVHN dataset) into binary numpy array files,
    so this funciton is un-used, but I am leaving it because it took a cool minute to figure out initially... so its a just in case.

    Parameters
    ----------
    file_path : str
        file path for mat file

    Returns
    -------
    x : numpy array
        filtered input data
    y : numpy array
        filtered label data
    ''' 
    f = loadmat(file_path) # use special loadmat method to read data. comes in as a dictionary
    x = np.reshape(f['X'], (32 * 32 * 3, f['X'].shape[3])).T # messy line of code that simply reshapes x data into a flat array and turns into numpy array
    y = np.subtract(np.squeeze(f['y']), 2) # squeezes an unneeded dimension out of label data and subtracts 2
    y[y==255] = 9 # there is a quirk where 1 class label comes out as 255 so this auto-adjusts
    print(np.unique(y)) # print out unique labels for validation
    x, y = filter_arrays(x, y, desired_classes) # use filter_arrays to filter x and y

    return x, y


def read_images_labels(images_filepath, labels_filepath, desired_classes):        
    '''
    Helper function to read in MNIST data binary files.

    Parameters
    ----------
    images_filepath : str
        file path for .idx3-ubyte file containing input data
    labels_filepath : str
        file path for .idx3-ubyte file containing label data

    Returns
    -------
    x : numpy array
        filtered input data
    y : numpy array
        filtered label data
    '''
    # -- commenting 2-3 months after first built this function w/ ChatGPT help... comments may be sparse -- 
    # - load label data -
    labels = []
    with open(labels_filepath, 'rb') as file: # open label data file with 'rb' argument to 'read binary' file
        magic, size = struct.unpack(">II", file.read(8)) # don't remember specifics, but its a check on a magic number in file...
        if magic != 2049: # error catch
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read()) # read data as an array (note not numpy array yet)
    
    # - load input data - 
    with open(images_filepath, 'rb') as file: # open input data file with 'rb' argument to 'read binary' file
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16)) # get some metadata from binary like magic value and shape
        if magic != 2051: # error catch
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read()) # read data as an array (note not numpy array yet)

    images = [] #initialize images list
    for i in range(size): # iterate over size of input data
        images.append([0] * rows * cols) # initial a bunch of 0 vectors
    for i in range(size): # iterate over size of input data
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]) # can't remember this one...
        img = img.reshape(28, 28) # single image is (28,28) shaped
        images[i][:] = img # update i's 0 vector w/ image data.
    
    x = np.array(images) # convert input data list to numpy array
    x = x.reshape(x.shape[0], -1) # flatten
    y = np.array(labels) # convery label array into numpy array
    x, y = filter_arrays(x, y, desired_classes) # use filter_arrays w/ training data

    return x, y


def load_CIFAR_data(desired_classes = [7,8]):
    '''
    Loads the saved CIFAR 10 dataset from local directory. CIFAR-10 data sourced from here: http://www.cs.toronto.edu/~kriz/cifar.html 
    
    The function allows the user to specify which classes from CIFAR-10 to load (out of possible 0-9). If desiring a binary classification problem then a 
    list of 2 numbers between 0 and 9 will load only these classes. Please follow link above for list of which classes coordinate with which number. 
    
    Parameters
    ----------
    desired_classes : list
        list of classes user wants from the 10 classes of CIFAR-10 dataset

    Returns
    -------
    x_train : numpy array
        training data
    x_test : numpy array
        testing data 
    y_train : numpy array
        training labels
    y_test : numpy array
        testing labels  
    '''
    # - dynamic file path so users can read in data from their local machines - 
    script_dir = os.path.dirname(__file__) # returns present working directory of this load_data.py file
    data_file_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'CIFAR')) # use .. to go up then back down to data/CIFAR folder
    
    # - unpickle data - 
    batch1 = unpickle(data_file_path + '\\data_batch_1') # unpickle batch 1
    batch2 = unpickle(data_file_path + '\\data_batch_2') # unpickle batch 2
    batch3 = unpickle(data_file_path + '\\data_batch_3') # unpickle batch 3
    batch4 = unpickle(data_file_path + '\\data_batch_4') # unpickle batch 4
    batch5 = unpickle(data_file_path + '\\data_batch_5') # unpickle batch 5
    test = unpickle(data_file_path + '\\test_batch') # unpickle testing batch

    # - convert unpickled data to numpy arrays - 
    x_train = np.concatenate((batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']), axis=0) # concatenate and convert training input data to 1 large numpy array
    y_train = np.array(batch1[b'labels'] + batch2[b'labels'] + batch3[b'labels'] + batch4[b'labels'] + batch5[b'labels']) # concatenate and convert training label data to 1 large numpy array
    x_test = np.array(test[b'data']) # read in and convert testing input data
    y_test = np.array(test[b'labels']) # read in and convert testing label data

    # - filter and re-scale the data - 
    x_train, y_train = filter_arrays(x_train, y_train, desired_classes) # use filter_arrays w/ training data
    x_test, y_test = filter_arrays(x_test, y_test, desired_classes) # use filter_arrays w/ testing data
    
    return x_train, x_test, y_train, y_test


def load_SVHN_data(desired_classes = [0,1]):
    '''
    Loads the Stanford Google Images House Number (SVHN) Digits datset from local directory. Data sourced from here: http://ufldl.stanford.edu/housenumbers/ 

    The function allows the user to specify which classes from SVHN to load (out of possible 0-9). If desiring a binary classification problem then a 
    list of 2 numbers between 0 and 9 will load only these classes. Please follow link above for list of which classes coordinate with which number. 
    
    Parameters
    ----------
    desired_classes : list
        list of classes user wants from the 10 classes of SVHN dataset

    Returns
    -------
    x_train : numpy array
        training data
    x_test : numpy array
        testing data 
    y_train : numpy array
        training labels
    y_test : numpy array
        testing labels  
    '''
    # - dynamic file path so users can read in data from their local machines - 
    script_dir = os.path.dirname(__file__) # returns present working directory of this load_data.py file
    data_file_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'SVHN')) # use .. to go up then back down to data/SVHN folder
    
    # - load saved numpy binary files for label data - 
    y_train = np.load(data_file_path + f'\\SVHN_y_train.npy') # use np.load w/ file path to read .npy file
    y_test = np.load(data_file_path + f'\\SVHN_y_test.npy') # use np.load w/ file path to read .npy file

    # - load input data (big loop because had to save in many batches)
    for i in range(40): # loop through all 40 files
        path_train = data_file_path + f'\\SVHN_x_train{i}.npy' # path to use given i
        path_test = data_file_path + f'\\SVHN_x_test{i}.npy' # path to use given i
        if i == 0: # if first file
            x_train = np.load(path_train) # use np.load w/ file path to read .npy file
            x_test = np.load(path_test) # use np.load w/ file path to read .npy file
        elif i < 20: # only had 20 test files so we need a stop here for that fact
            x_train = np.concatenate((x_train, np.load(path_train))) # use np.load w/ file path to read .npy file, then concatenate with already loaded data
            x_test = np.concatenate((x_test, np.load(path_test))) # use np.load w/ file path to read .npy file, then concatenate with already loaded data
        else: # finish training input data loading
            x_train = np.concatenate((x_train, np.load(path_train))) # use np.load w/ file path to read .npy file, then concatenate with already loaded data

    # - filter and re-scale the data - 
    x_train, y_train = filter_arrays(x_train, y_train, desired_classes) # use filter_arrays w/ training data
    x_test, y_test = filter_arrays(x_test, y_test, desired_classes) # use filter_arrays w/ testing data

    return x_train, x_test, y_train, y_test

     
def load_MNIST_data(desired_classes = [1,4]):
    '''
    Loads the Modified National Institute of Standards and Technology (MNIST) Digits datset from local directory. Data sourced from kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset 

    The function allows the user to specify which classes from MNIST to load (out of possible 0-9). If desiring a binary classification problem then a 
    list of 2 numbers between 0 and 9 will load only these classes. Please follow link above for list of which classes coordinate with which number. 
    
    Parameters
    ----------
    desired_classes : list
        list of classes user wants from the 10 classes of MNIST dataset

    Returns
    -------
    x_train : numpy array
        training data
    x_test : numpy array
        testing data 
    y_train : numpy array
        training labels
    y_test : numpy array
        testing labels  
    '''
    # - dynamic file path so users can read in data from their local machines - 
    script_dir = os.path.dirname(__file__) # returns present working directory of this load_data.py file
    data_file_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'MNIST')) # use .. to go up then back down to data/MNIST folder

    # - define file paths - 
    training_images_filepath = data_file_path + '\\train-images.idx3-ubyte' # training input data path
    training_labels_filepath = data_file_path + '\\train-labels.idx1-ubyte' # training label data path
    test_images_filepath = data_file_path + '\\t10k-images.idx3-ubyte' # testing input data path
    test_labels_filepath = data_file_path + '\\t10k-labels.idx1-ubyte' # testing label data path

    # - use special helper function to read in data - 
    x_train, y_train = read_images_labels(training_images_filepath, training_labels_filepath, desired_classes) # use helper function to load training data w/ given classes
    x_test, y_test = read_images_labels(test_images_filepath, test_labels_filepath, desired_classes) # use helper function to load testing data w/ given classes

    return x_train, x_test, y_train, y_test 