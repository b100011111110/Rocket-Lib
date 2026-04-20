import os
import sys
import numpy as np

build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.append(build_dir)
sys.path.append(os.path.dirname(__file__))

import rocket
import tensorflow as tf

from compare_keras import to_rocket_tensors, sync_dense_weights

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_batch = np.random.randn(2, 4).astype(np.float32)
    y_batch = np.array([[1.0], [0.0]], dtype=np.float32)

    # Rocket Model
    r_model = rocket.Model()
    r_input = rocket.InputLayer()
    r_dense1 = rocket.DenseLayer(4, 2)
    r_relu1 = rocket.ActivationLayer(rocket.ReLU())
    # No reg/dropout to isolate core math
    r_dense_out = rocket.DenseLayer(2, 1)
    r_out = rocket.ActivationLayer(rocket.Linear())
    
    r_model.add(r_input, [])
    r_model.add(r_dense1, [r_input])
    r_model.add(r_relu1, [r_dense1])
    r_model.add(r_dense_out, [r_relu1])
    r_model.add(r_out, [r_dense_out])
    r_model.setInputOutputLayers([r_input], [r_out])
    
    r_loss = rocket.BCEWithLogits()
    r_opt = rocket.Adam(0.01, 0.9, 0.999, 1e-7)
    r_model.compile(r_loss, r_opt)

    # Keras Model
    k_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(4,)),
        tf.keras.layers.Dense(2, activation=None, name="dense_1"),
        tf.keras.layers.ReLU(name="relu_1"),
        tf.keras.layers.Dense(1, activation=None, name="dense_out"),
    ])
    k_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, epsilon=1e-7))
    
    sync_dense_weights([k_model.get_layer("dense_1"), k_model.get_layer("dense_out")], [r_dense1, r_dense_out])

    # Keras train on 1 batch
    with tf.GradientTape() as tape:
        preds = k_model(X_batch)
        loss = k_model.loss(y_batch, preds)
    grads = tape.gradient(loss, k_model.trainable_weights)

    # Rocket train on 1 batch
    x_rt = to_rocket_tensors(X_batch)
    y_rt = to_rocket_tensors(y_batch)
    
    r_model.train(x_rt, y_rt, 1, 2, r_opt)

    print("Keras Grad dense_out W:")
    print(grads[2].numpy())
    
    print("Rocket Grad dense_out W:")
    for i in range(2):
        print(f"[{r_dense_out.grad_weights.get_val(i, 0):.6f}]")

    print("\nKeras dense_out W updated:")
    k_model.optimizer.apply_gradients(zip(grads, k_model.trainable_weights))
    print(k_model.get_layer("dense_out").get_weights()[0])
    
    print("Rocket dense_out W updated:")
    for i in range(2):
        print(f"[{r_dense_out.weights.get_val(i, 0):.6f}]")

if __name__ == "__main__":
    main()
