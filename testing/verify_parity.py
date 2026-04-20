import os
os.environ["ROCKET_SEED"] = "42"
os.environ["ROCKET_REG_LAMBDA"] = "0"
os.environ["ROCKET_SHUFFLE"] = "0"
os.environ["ROCKET_DROPOUT"] = "0"

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
import sys; sys.path.append("build"); import rocket

# Configurations
input_dim = 16
batch_size = 1200
epochs = 1
keras_lr = 0.01

# Synthetic dataset
X, y = make_classification(
    n_samples=1200, n_features=input_dim, n_informative=10, n_redundant=4,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.2,
    flip_y=0.03, random_state=42
)
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]

# Disable GPU
tf.config.set_visible_devices([], 'GPU')
tf.random.set_seed(42)
np.random.seed(42)

# Build Rocket model
r_dense1 = rocket.DenseLayer(input_dim, 64)
r_relu1 = rocket.ActivationLayer(rocket.ReLU())
r_dense2 = rocket.DenseLayer(64, 32)
r_relu2 = rocket.ActivationLayer(rocket.ReLU())
r_dense3 = rocket.DenseLayer(32, 16)
r_relu3 = rocket.ActivationLayer(rocket.ReLU())
r_dense_out = rocket.DenseLayer(16, 1)

r_input = rocket.InputLayer()
r_model = rocket.Model()
r_model.add(r_input, [])
r_model.add(r_dense1, [r_input])
r_model.add(r_relu1, [r_dense1])
r_model.add(r_dense2, [r_relu1])
r_model.add(r_relu2, [r_dense2])
r_model.add(r_dense3, [r_relu2])
r_model.add(r_relu3, [r_dense3])
r_model.add(r_dense_out, [r_relu3])

r_model.setInputOutputLayers([r_input], [r_dense_out])
r_opt = rocket.Adam(keras_lr)
r_loss = rocket.BCEWithLogits()
r_model.compile(r_loss, r_opt)

# Build Keras model
k_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation=None, name="dense_1"),
    tf.keras.layers.ReLU(name="relu_1"),
    tf.keras.layers.Dense(32, activation=None, name="dense_2"),
    tf.keras.layers.ReLU(name="relu_2"),
    tf.keras.layers.Dense(16, activation=None, name="dense_3"),
    tf.keras.layers.ReLU(name="relu_3"),
    tf.keras.layers.Dense(1, activation="sigmoid", name="dense_out"),
])

k_model.compile(
    loss="binary_crossentropy", 
    optimizer=tf.keras.optimizers.Adam(learning_rate=keras_lr, epsilon=1e-7)
)

# Sync weights
def sync_dense_weights(k_layers, r_layers):
    for k_layer, r_layer in zip(k_layers, r_layers):
        weights, biases = k_layer.get_weights()
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                r_layer.weights.set_val(i, j, float(weights[i, j]))
        for j in range(biases.shape[0]):
            r_layer.biases.set_val(0, j, float(biases[j]))

sync_dense_weights([k_model.get_layer("dense_1"), k_model.get_layer("dense_2"), k_model.get_layer("dense_3"), k_model.get_layer("dense_out")],
                   [r_dense1, r_dense2, r_dense3, r_dense_out])

# Convert training data
def to_rocket_tensors(numpy_array):
    tensors = []
    for row in numpy_array:
        t = rocket.Tensor(1, row.shape[0])
        for i, val in enumerate(row):
            t.set_val(0, i, float(val))
        tensors.append(t)
    return tensors

r_xtrain = to_rocket_tensors(X_train)
r_ytrain = to_rocket_tensors(y_train.reshape(-1, 1))

# Train both for exactly 1 epoch
r_model.train(r_xtrain, r_ytrain, r_xtrain, r_ytrain, 1, batch_size)
k_model.fit(X_train, y_train, epochs=1, batch_size=batch_size, shuffle=False, verbose=0)

# Compare weights
def compare_weights(name, k_layer, r_layer):
    k_weights, k_biases = k_layer.get_weights()
    max_diff_w = 0
    for i in range(k_weights.shape[0]):
        for j in range(k_weights.shape[1]):
            diff = abs(k_weights[i, j] - r_layer.weights.get_val(i, j))
            max_diff_w = max(max_diff_w, diff)
            
    max_diff_b = 0
    for j in range(k_biases.shape[0]):
        diff = abs(k_biases[j] - r_layer.biases.get_val(0, j))
        max_diff_b = max(max_diff_b, diff)
        
    print(f"Layer {name}: max weight diff = {max_diff_w:.8f}, max bias diff = {max_diff_b:.8f}")

compare_weights("dense_1", k_model.get_layer("dense_1"), r_dense1)
compare_weights("dense_2", k_model.get_layer("dense_2"), r_dense2)
compare_weights("dense_3", k_model.get_layer("dense_3"), r_dense3)
compare_weights("dense_out", k_model.get_layer("dense_out"), r_dense_out)
