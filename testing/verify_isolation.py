import os
os.environ["ROCKET_SEED"] = "42"
os.environ["ROCKET_REG_LAMBDA"] = "0"
os.environ["ROCKET_SHUFFLE"] = "0"
os.environ["ROCKET_DROPOUT"] = "0"

import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
import sys
sys.path.append("build")
import rocket

input_dim = 16
batch_size = 1200
X, y = make_classification(
    n_samples=1200, n_features=input_dim, n_informative=10, n_redundant=4,
    n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=1.2,
    flip_y=0.03, random_state=42
)

# Convert to Rocket Tensors
def to_rocket(numpy_array):
    t = rocket.Tensor(numpy_array.shape[0], numpy_array.shape[1])
    for i in range(numpy_array.shape[0]):
        for j in range(numpy_array.shape[1]):
            t.set_val(i, j, float(numpy_array[i, j]))
    return t

def to_numpy(rocket_tensor):
    arr = np.zeros((rocket_tensor.rows, rocket_tensor.cols))
    for i in range(rocket_tensor.rows):
        for j in range(rocket_tensor.cols):
            arr[i, j] = rocket_tensor.get_val(i, j)
    return arr

r_X = to_rocket(X)
r_y = to_rocket(y.reshape(-1, 1))

# Initialize layers
r_dense3 = rocket.DenseLayer(32, 16)
r_relu3 = rocket.ActivationLayer(rocket.ReLU())
r_dense_out = rocket.DenseLayer(16, 1)

# Initialize Keras layers
tf.random.set_seed(42)
np.random.seed(42)
k_dense3 = tf.keras.layers.Dense(16, activation=None)
k_relu3 = tf.keras.layers.ReLU()
k_dense_out = tf.keras.layers.Dense(1, activation="sigmoid")

# Dummy forward pass to initialize weights
dummy_input = tf.random.normal((batch_size, 32))
k_out3 = k_dense3(dummy_input)
k_r3 = k_relu3(k_out3)
k_out = k_dense_out(k_r3)

# Sync weights
def sync(k_layer, r_layer):
    w, b = k_layer.get_weights()
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            r_layer.weights.set_val(i, j, float(w[i, j]))
    for j in range(b.shape[0]):
        r_layer.biases.set_val(0, j, float(b[j]))

sync(k_dense3, r_dense3)
sync(k_dense_out, r_dense_out)

# Forward pass
r_in = to_rocket(dummy_input.numpy())
r_out3 = r_dense3.forward(r_in)
r_r3 = r_relu3.forward(r_out3)
r_out = r_dense_out.forward(r_r3)

# BCE gradient
loss_fn = rocket.BCEWithLogits()
r_grad_out = loss_fn.backward(r_out, r_y)

# Keras Forward and Gradient
with tf.GradientTape(persistent=True) as tape:
    tape.watch(dummy_input)
    k_out3_val = k_dense3(dummy_input)
    k_r3_val = k_relu3(k_out3_val)
    k_preds = k_dense_out(k_r3_val)
    loss = tf.keras.losses.BinaryCrossentropy(reduction="sum_over_batch_size")(y.reshape(-1, 1), k_preds)

k_grad_out = tape.gradient(loss, k_preds)
k_grad_r3 = tape.gradient(loss, k_r3_val)
k_grad_out3 = tape.gradient(loss, k_out3_val)

# Backward pass Rocket
r_grad_r3 = r_dense_out.backward(r_r3, r_grad_out)
r_grad_out3 = r_relu3.backward(r_out3, r_grad_r3)
r_grad_in = r_dense3.backward(r_in, r_grad_out3)

# Compare
def compare(name, k_tensor, r_tensor):
    k_np = k_tensor.numpy() if hasattr(k_tensor, 'numpy') else k_tensor
    r_np = to_numpy(r_tensor)
    diff = np.abs(k_np - r_np).max()
    print(f"{name} diff: {diff:.8e}")

print("--- Forward Pass ---")
compare("dense3 out", k_out3_val, r_out3)
compare("relu3 out", k_r3_val, r_r3)
# Logits vs Sigmoid
r_out_sig = 1 / (1 + np.exp(-to_numpy(r_out)))
compare("dense_out (sigmoid vs prob)", k_preds, r_out_sig)

print("--- Backward Pass ---")
# Keras dL/d(sigmoid) vs Rocket dL/d(logit)
# Actually, tape.gradient(loss, k_r3_val) should be identical to r_grad_r3!
compare("grad_r3", k_grad_r3, r_grad_r3)
compare("grad_out3", k_grad_out3, r_grad_out3)

# Check weight gradients
k_w_grad, k_b_grad = tape.gradient(loss, k_dense_out.trainable_variables)
compare("dense_out w_grad", k_w_grad, r_dense_out.grad_weights)
compare("dense_out b_grad", k_b_grad, r_dense_out.grad_biases)

k_w3_grad, k_b3_grad = tape.gradient(loss, k_dense3.trainable_variables)
compare("dense3 w_grad", k_w3_grad, r_dense3.grad_weights)
compare("dense3 b_grad", k_b3_grad, r_dense3.grad_biases)

del tape
