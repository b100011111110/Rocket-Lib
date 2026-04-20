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

    r_model = rocket.Model()
    r_input = rocket.InputLayer()
    r_dense1 = rocket.DenseLayer(4, 2)
    r_out = rocket.ActivationLayer(rocket.Linear())
    r_model.add(r_input, [])
    r_model.add(r_dense1, [r_input])
    r_model.add(r_out, [r_dense1])
    r_model.setInputOutputLayers([r_input], [r_out])
    
    r_loss = rocket.BCEWithLogits()
    r_opt = rocket.SGD(0.01)
    r_model.compile(r_loss, r_opt)

    k_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(4,)),
        tf.keras.layers.Dense(2, activation=None, name="dense_1"),
    ])
    
    sync_dense_weights([k_model.get_layer("dense_1")], [r_dense1])

    x_rt = to_rocket_tensors(X_batch)
    y_rt = to_rocket_tensors(y_batch)

    x_sample = x_rt[0]
    y_sample = y_rt[0]
    
    pred_r = r_model.predict([x_sample])[0]
    loss_val_r = r_loss.forward(pred_r, y_sample)
    grad_out_r = r_loss.backward(pred_r, y_sample)
    
    grad_in_r = r_out.backward(r_dense1.output, grad_out_r)
    r_dense1.backward(x_sample, grad_in_r)

    print("Rocket Pred:", pred_r.get_val(0, 0), pred_r.get_val(0, 1))
    print("Rocket Loss:", loss_val_r)
    print("Rocket Grad W:")
    for i in range(4):
        print(f"[{r_dense1.grad_weights.get_val(i, 0):.6f}, {r_dense1.grad_weights.get_val(i, 1):.6f}]")

    with tf.GradientTape() as tape:
        pred_k = k_model(X_batch[0:1])
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_val_k = loss_fn(y_batch[0:1], pred_k)

    grads_k = tape.gradient(loss_val_k, k_model.trainable_weights)
    print("\nKeras Pred:", pred_k.numpy())
    print("Keras Loss:", loss_val_k.numpy())
    print("Keras Grad W:")
    print(grads_k[0].numpy())

if __name__ == "__main__":
    main()
