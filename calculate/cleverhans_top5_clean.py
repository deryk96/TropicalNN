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
    num_classes = 100 

    eps, data, info, model_paths = load_attack_settings(arg_dataset, batch_size, "new_master_models")#, adv_train)

    total_test_examples = info.splits['test'].num_examples
    total_batches = math.ceil(total_test_examples / batch_size)

    batches_per_chunk = total_batches // (total_batch_chunks)
    start_batch = batch_chunk * batches_per_chunk
    
    test_data_subset = data.test.skip(start_batch).take(batches_per_chunk) # Slicing the test dataset by batches

    for name, model_path in model_paths.items():
        print(f'evaluating top5 accuracy on {name}, stored in {model_path}')
        model = tf.keras.models.load_model(model_path)

        test_acc_clean = tf.metrics.TopKCategoricalAccuracy(k=5)

        # Evaluate on clean and adversarial data
        progress_bar_test = tf.keras.utils.Progbar(batches_per_chunk*batch_size)
        current_batch_size = 0
        for x, y in test_data_subset: #data.test:
            input_elements = 1
            for i in range(1, len(x.shape)):
                input_elements *= x.shape[i]            

            # Check if 'y' needs to be one-hot encoded
            if len(y.shape) == 1 or (len(y.shape) > 1 and y.shape[-1] != num_classes):
                y = tf.one_hot(y, depth=num_classes)
  
            y = tf.cast(y, tf.int32)

            # -- clean --
            y_pred = model(x, training=False)
            test_acc_clean(y, y_pred)            
            
            current_batch_size += x.shape[0]
            progress_bar_test.add(x.shape[0], values = [("top 5 acc", test_acc_clean.result()),
                                                        ])

        print("top5 acc on clean examples (%): {:.3f}".format(test_acc_clean.result().numpy() * 100))

        accuracy_data = [test_acc_clean.result().numpy()]
        csv_file = save_location_attack_results(arg_dataset,name,batch_chunk,total_batch_chunks,"top5_clean")
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Top5Clean', 
                             ])
            writer.writerow(accuracy_data)

        print(f"Accuracies written to {csv_file}")


if __name__ == "__main__":
    print("##########      Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    app.run(main)