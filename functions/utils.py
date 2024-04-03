import matplotlib.pyplot as plt
import tensorflow as tf
import os
from functions.load_data import ld_mnist, ld_svhn, ld_cifar10, ld_cifar100
from functions.models import ResNet50Model, ModifiedLeNet5, LeNet5, VGG16Model, MobileNetModel, EfficientNetB4Model

def l2(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), list(range(1, len(x.shape)))))

def l1(x, y):
    return tf.reduce_sum(tf.abs(x - y), list(range(1, len(x.shape))))

def plot_images_in_grid(list_of_xs, row_labels, col_labels, save_path, input_elements):
    num_rows = len(row_labels)
    num_cols = len(col_labels)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    if input_elements == 784:
        cmap = 'gray'
    else:
        cmap = None
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            ax.imshow(list_of_xs[j][i,:,:,:], cmap=cmap)
            ax.axis('off')
            if i == 0:
                ax.set_title(col_labels[j], size='large')
    
    #plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def load_data(dataset_name, batch_size):
    # -- load data and set epsilon --
    if dataset_name == "mnist":
        dataset_category = 0
        eps = 0.2
        input_elements = 28*28*1
        data, info = ld_mnist(batch_size=batch_size)
        num_classes = 10
        input_shape = (28,28,1)
    elif dataset_name == "svhn":
        dataset_category = 0
        eps = 8/255
        input_elements = 32*32*3
        data, info = ld_svhn(batch_size=batch_size)
        num_classes = 10
        input_shape = (32,32,3)
    elif dataset_name == "cifar10":
        dataset_category = 1
        eps = 8/255
        input_elements = 32*32*3
        data, info = ld_cifar10(batch_size=batch_size)
        num_classes = 10
        input_shape = (32,32,3)
    elif dataset_name == "cifar100":
        dataset_category = 1
        eps = 8/255
        input_elements = 32*32*3
        data, info = ld_cifar100(batch_size=batch_size)
        num_classes = 100
        input_shape = (32,32,3)
    else:
        raise ValueError("Invalid dataset name provided. Should be either mnist, svhn, cifar10, cifar100, or imagenet")
    return dataset_category, eps, input_elements, data, info, input_shape, num_classes
    
def load_build_settings(dataset_name, base_model_index, batch_size):
    dataset_category, eps, input_elements, data, info, input_shape, num_classes = load_data(dataset_name, batch_size)

    # -- setup the models --   
    if dataset_category == 0:
        if base_model_index == 0:
            models = {'ModifiedLeNet5_relu': ModifiedLeNet5(num_classes=num_classes, top="relu"),
                    'ModifiedLeNet5_trop': ModifiedLeNet5(num_classes=num_classes, top="trop"),
                    'ModifiedLeNet5_maxout': ModifiedLeNet5(num_classes=num_classes, top="maxout")
                    }
        elif base_model_index == 1:
            models = {'LeNet5_relu': LeNet5(num_classes=num_classes, top="relu"),
                    'LeNet5_trop': LeNet5(num_classes=num_classes, top="trop"),
                    'LeNet5_maxout': LeNet5(num_classes=num_classes, top="maxout")
                    }
        elif base_model_index == 2:
            models = {'MobileNet_relu': MobileNetModel(num_classes=num_classes, top="relu", input_shape=input_shape),
                    'MobileNet_trop': MobileNetModel(num_classes=num_classes, top="trop", input_shape=input_shape),
                    'MobileNet_maxout': MobileNetModel(num_classes=num_classes, top="maxout", input_shape=input_shape)
                    }
        else:
            raise ValueError("Invalid base_model_index provided. Should be either 0, 1, or 2")
    elif dataset_category == 1:
        if base_model_index == 0:
            models = {'ResNet50_relu': ResNet50Model(num_classes=num_classes, top="relu", input_shape=input_shape),
                    'ResNet50_trop': ResNet50Model(num_classes=num_classes, top="trop", input_shape=input_shape),
                    'ResNet50_maxout': ResNet50Model(num_classes=num_classes, top="maxout", input_shape=input_shape)
                    }
        elif base_model_index == 1:
            models = {'VGG16_relu': VGG16Model(num_classes=num_classes, top="relu", input_shape=input_shape),
                    'VGG16_trop': VGG16Model(num_classes=num_classes, top="trop", input_shape=input_shape),
                    'VGG16_maxout': VGG16Model(num_classes=num_classes, top="maxout", input_shape=input_shape)
                    }
        elif base_model_index == 2:
            models = {'EfficientNetB4_relu': EfficientNetB4Model(num_classes=num_classes, top="relu", input_shape=input_shape),
                    'EfficientNetB4_trop': EfficientNetB4Model(num_classes=num_classes, top="trop", input_shape=input_shape),
                    'EfficientNetB4_maxout': EfficientNetB4Model(num_classes=num_classes, top="maxout", input_shape=input_shape)
                    }
        else:
            raise ValueError("Invalid base_model_index provided. Should be either 0, 1, or 2")
    return input_elements, eps, data, info, models

def find_directories_with_keyphrase(root_dir, keyphrase):
    result = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if keyphrase in dirname:
                result[dirname] = os.path.join(dirpath, dirname)
    return result

def load_attack_settings(dataset_name, batch_size, dir_path):
    _, eps, _, data, info, _, _ = load_data(dataset_name, batch_size)

    model_paths = find_directories_with_keyphrase(dir_path, dataset_name)
    #"/home/kurt.pasque/TropicalNN/master_models"
    '''
    if dataset_name == "mnist":
        
        if adv_train == 'yes':
            model_paths = {
                       'mnist_ModifiedLeNet5_trop_yes_27Mar24': 'master_models/mnist_ModifiedLeNet5_trop_yes_27Mar24', #new
                       'mnist_ModifiedLeNet5_relu_yes_27Mar24':'master_models/mnist_ModifiedLeNet5_relu_yes_27Mar24', #new
                       'mnist_ModifiedLeNet5_maxout_yes_27Mar24':'master_models/mnist_ModifiedLeNet5_maxout_yes_27Mar24', #new
                        }
        else:
            model_paths = {
                       'mnist_ModifiedLeNet5_MMR_no':'saved_models/CH_MMRReluConv3Layer_mnist_0.1_100_False', # respectfully not doing again...
                       'mnist_ModifiedLeNet5_trop_no_27Mar24': 'master_models/mnist_ModifiedLeNet5_trop_no_27Mar24/', #new 
                       'mnist_ModifiedLeNet5_relu_no_27Mar24':'master_models/mnist_ModifiedLeNet5_relu_no_27Mar24/', #new 
                       'mnist_ModifiedLeNet5_maxout_no_27Mar24':'master_models/mnist_ModifiedLeNet5_maxout_no_27Mar24/', #new 
                        }
    elif dataset_name == "svhn":
        if adv_train == 'yes':
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_TropConv3Layer_redo_svhn_0.03137254901960784_100_True',
                       'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_redo_svhn_0.03137254901960784_100_True',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_redo_svhn_0.03137254901960784_100_True',
                        }
        else:
            model_paths = {
                       'CH_MMRReluConv3Layer': 'saved_models/CH_MMRReluConv3Layer_svhn_0.01568627450980392_100_False', # respectfully not doing again...
                       #'CH_TropConv3Layer': 'saved_models/CH_TropConv3Layer_svhn_0.01568627450980392_100_False',
                       #'CH_ReluConv3Layer':'saved_models/CH_ReluConv3Layer_svhn_0.01568627450980392_100_False',
                       #'CH_MaxoutConv3Layer':'saved_models/CH_MaxoutConv3Layer_svhn_0.01568627450980392_100_False',
                        }
    elif dataset_name == "cifar10":
        if adv_train == 'yes':
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_Trop_ResNet50_redo_cifar_0.03137254901960784_100_True',
                       'CH_ReluConv3Layer':'saved_models/CH_ReLU_ResNet50_redo_cifar_0.03137254901960784_100_True',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxOut_ResNet50_redo_cifar_0.03137254901960784_100_True',
                        }
        else:
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_Trop_ResNet50_cifar_0.01568627450980392_100_False',
                       'CH_ReluConv3Layer':'saved_models/CH_ReLU_ResNet50_cifar_0.01568627450980392_100_False',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxOut_ResNet50_cifar_0.01568627450980392_100_False',
                        }
    elif dataset_name == "cifar100":
        if adv_train == 'yes':
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_Trop_ResNet50_redo_cifar_0.03137254901960784_100_True',
                       'CH_ReluConv3Layer':'saved_models/CH_ReLU_ResNet50_redo_cifar_0.03137254901960784_100_True',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxOut_ResNet50_redo_cifar_0.03137254901960784_100_True',
                        }
        else:
            model_paths = {
                       'CH_TropConv3Layer': 'saved_models/CH_Trop_ResNet50_cifar_0.01568627450980392_100_False',
                       'CH_ReluConv3Layer':'saved_models/CH_ReLU_ResNet50_cifar_0.01568627450980392_100_False',
                       'CH_MaxoutConv3Layer':'saved_models/CH_MaxOut_ResNet50_cifar_0.01568627450980392_100_False',
                        }
    else:
        raise ValueError("Invalid dataset name provided. Should be either mnist, svhn, cifar10, cifar100, or imagenet")
    '''
    return eps, data, info, model_paths