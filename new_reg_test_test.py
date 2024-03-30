import tensorflow as tf
from tensorflow.keras import regularizers

class TropRegIncreaseDistance(regularizers.Regularizer):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, weight_matrix):
        reshaped_weights = tf.expand_dims(weight_matrix, 1)
        result_addition = reshaped_weights + tf.transpose(reshaped_weights, perm=[1, 0, 2])
        tropical_distances = tf.reduce_max(result_addition, axis=2) - tf.reduce_min(result_addition, axis=2)
        n = tf.shape(tropical_distances)[0]
        mask = tf.linalg.band_part(tf.ones((n, n), dtype=tf.bool), 0, -1)
        flat_vector = tf.boolean_mask(tropical_distances, tf.logical_not(mask))
        return self.lam * tf.reduce_min(flat_vector)

# Example usage
if __name__ == "__main__":
    # Assuming TensorFlow 2.x
    import numpy as np

    # Initialize the regularizer
    reg = TropRegIncreaseDistance(lam=-10)

    # Create a sample weight matrix (e.g., for a dense layer with 5 units)
    weight_matrix = np.random.rand(50, 50)  # Replace this with actual weights from a layer if needed

    # Convert to TensorFlow tensor
    weight_tensor = tf.convert_to_tensor(weight_matrix, dtype=tf.float32)

    # Calculate the regularization term
    regularization_term = reg(weight_tensor)

    # Print the output
    print(f"Regularization term: {regularization_term.numpy()}")
