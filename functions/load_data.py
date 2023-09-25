from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle


### RUN THIS CELL FOR CIFAR 10 IMAGE OBJECT DATA AS X AND Y ###
# --- CIFAR 10 data pictures ---
# --- sourced from here: http://www.cs.toronto.edu/~kriz/cifar.html  ---

# -- read in the data -- 
# - function from site on extracting data from binary files -
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# - function to cleanAndSort the data for 2 classes - 
def cleanAndSort(x_arr, y_arr, desired_classes = [0,1], subtractTot = 0):
    # - filter for just 2 -
    y_arr = np.subtract(y_arr, subtractTot)
    filter_mask = np.isin(y_arr, desired_classes)
    x_arr = x_arr[filter_mask]
    y_arr = y_arr[filter_mask]

    # - divide by 255 so the RGB data is on range of [0, 1] -
    x_arr = np.divide(x_arr, 255) 
    
    # - for > 2 categories - 
    if desired_classes != [0,1]:
        num_classes = len(desired_classes)
        y_arr = to_categorical(y_arr, num_classes)
    
    return x_arr, y_arr

def readCIFARData(shuffle = True, desired_classes = [0,1], subtractTot = 7):
    # - get training data - 
    batch1 = unpickle('cifar-10-batches-py\\data_batch_1')
    batch2 = unpickle('cifar-10-batches-py\\data_batch_2')
    batch3 = unpickle('cifar-10-batches-py\\data_batch_3')
    batch4 = unpickle('cifar-10-batches-py\\data_batch_4')
    batch5 = unpickle('cifar-10-batches-py\\data_batch_5')
    x_train = np.concatenate((batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']), axis=0)
    y_train = np.array(batch1[b'labels'] + batch2[b'labels'] + batch3[b'labels'] + batch4[b'labels'] + batch5[b'labels'])

    # - get testing data -
    test = unpickle('cifar-10-batches-py\\test_batch')
    x_test = test[b'data']
    y_test = np.array(test[b'labels'])

    # - cleanAndSort the data - 
    x_train, y_train = cleanAndSort(x_train, y_train, desired_classes = desired_classes, subtractTot = subtractTot)
    x_test, y_test = cleanAndSort(x_test, y_test, desired_classes = desired_classes, subtractTot = subtractTot)
    
    if shuffle:
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

        print(x_train[0])
    return x_train, x_test, y_train, y_test