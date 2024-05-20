import csv
import math
import sys
import os
import numpy as np
import tensorflow as tf
import time
from absl import app

from functions.utils import  load_attack_settings, save_location_attack_results

def main(_):
    # argument parsing

    batch_chunk = int(sys.argv[1])
    total_batch_chunks = int(sys.argv[2])
    batch_size = int(sys.argv[3]) 
    arg_dataset = sys.argv[4]
    print('argument dataset', arg_dataset)

    eps, data, info, model_paths = load_attack_settings(arg_dataset, batch_size, "new_master_models")#, adv_train)

    total_test_examples = info.splits['test'].num_examples
    total_batches = math.ceil(total_test_examples / batch_size)

    batches_per_chunk = total_batches // (total_batch_chunks)
    start_batch = batch_chunk * batches_per_chunk
    
    test_data_subset = data.test.skip(start_batch).take(batches_per_chunk) # Slicing the test dataset by batches

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for name, model_path in model_paths.items():
        print(f'evaluating top5 accuracy on {name}, stored in {model_path}')
        model = tf.keras.models.load_model(model_path)

        test_acc_clean = tf.metrics.TopKCategoricalAccuracy(k=5)


        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(batches_per_chunk*batch_size)
        current_batch_size = 0
        for x, y in test_data_subset: #data.test:
            input_elements = 1
            #print("Original y shape:", y.shape)
            for i in range(1, len(x.shape)):
                input_elements *= x.shape[i]
                # Check if 'y' is one-hot encoded; assuming 'y' has shape [batch_size, num_classes]
            
            #if len(y.shape) > 1 and y.shape[-1] > 1:
                #y = tf.argmax(y, axis=1)
            
            
            # Alternatively, if 'y' has an extra singleton dimension
            #if len(y.shape) > 1:
                #y = tf.squeeze(y)  
                
            num_classes = 100  # Update this based on your dataset specifics

            # Check if 'y' needs to be one-hot encoded
            if len(y.shape) == 1 or (len(y.shape) > 1 and y.shape[-1] != num_classes):
                y = tf.one_hot(y, depth=num_classes)

            #print("Adjusted y shape:", y.shape)   
            y = tf.cast(y, tf.int32)
            eps_l2 = math.sqrt((eps**2)*input_elements)
            eps_l1 = 2 * eps_l2

            # -- clean --
            y_pred = model(x, training=False)
            #print("y_pred shape:", y_pred.shape)
            test_acc_clean(y, y_pred)
            #print("clean done", time.time())
            
            
            current_batch_size += x.shape[0]
            progress_bar_test.add(x.shape[0], values = [("top 5 acc", test_acc_clean.result()),
                                                        ])

        print("top5 acc on clean examples (%): {:.3f}".format(test_acc_clean.result().numpy() * 100))

        # Assuming you have already calculated these accuracies
        test_acc_clean_value = test_acc_clean.result().numpy()

        # Prepare the data to be written to CSV
        accuracy_data = [test_acc_clean_value, 
                         ]

        # Specify the CSV file name
        #csv_file = save_location_attack_results(arg_dataset,name,batch_chunk,total_batch_chunks,"top5_clean")

        # Write to CSV
        '''with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Top5Clean', 
                             ])
            writer.writerow(accuracy_data)

        print(f"Accuracies written to {csv_file}")'''


if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    app.run(main)